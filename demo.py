from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import json
from collections import Counter
from typing import Dict, Optional

from interpolate_how2sign_wilor import (
    LINEAR_FIELDS,
    ROTATION_FIELDS,
    build_observed_values,
    fill_linear,
    fill_rotations,
    interpolation_plan,
    select_detections,
)
from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO 
LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)
INTERPOLATED_ORANGE=(1.0, 0.35, 0.05)

def rotmat_to_axis_angle(rotmat):
    """Convert (..., 3, 3) rotation matrices to (..., 3) axis-angle vectors."""
    original_shape = rotmat.shape[:-2]
    flat_rotmat = rotmat.reshape(-1, 3, 3)
    axis_angle = np.stack(
        [cv2.Rodrigues(matrix)[0].reshape(3) for matrix in flat_rotmat],
        axis=0,
    )
    return axis_angle.reshape(*original_shape, 3)


def save_empty_mano_params(output_path, img_path, image_size):
    """Write a shape-stable archive when no hands are detected."""
    np.savez_compressed(
        output_path,
        source_image=np.asarray(str(img_path)),
        global_orient_rotmat=np.empty((0, 1, 3, 3), dtype=np.float32),
        hand_pose_rotmat=np.empty((0, 15, 3, 3), dtype=np.float32),
        global_orient_axis_angle=np.empty((0, 3), dtype=np.float32),
        hand_pose_axis_angle=np.empty((0, 45), dtype=np.float32),
        mano_pose_axis_angle=np.empty((0, 48), dtype=np.float32),
        betas=np.empty((0, 10), dtype=np.float32),
        pred_cam=np.empty((0, 3), dtype=np.float32),
        cam_t_full=np.empty((0, 3), dtype=np.float32),
        bbox_xyxy=np.empty((0, 4), dtype=np.float32),
        detection_confidence=np.empty((0,), dtype=np.float32),
        is_right=np.empty((0,), dtype=np.float32),
        box_center=np.empty((0, 2), dtype=np.float32),
        box_size=np.empty((0,), dtype=np.float32),
        image_size=np.asarray(image_size, dtype=np.float32),
        focal_length=np.empty((0,), dtype=np.float32),
    )


def infer_frame_parameters(
    img_path,
    img_cv2,
    detector,
    model,
    model_cfg,
    device,
    rescale_factor,
    fast,
):
    """Infer every detected hand and return the raw variable-N frame schema."""
    detections = detector(img_cv2, conf=0.3, verbose=False)[0]
    if detections.boxes is None or len(detections.boxes) == 0:
        boxes = np.empty((0, 4), dtype=np.float32)
        right = np.empty((0,), dtype=np.float32)
        confidence = np.empty((0,), dtype=np.float32)
    else:
        boxes = detections.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        right = detections.boxes.cls.detach().cpu().numpy().astype(np.float32)
        confidence = (
            detections.boxes.conf.detach().cpu().numpy().astype(np.float32)
        )

    values = {
        "global_orient_rotmat": [],
        "hand_pose_rotmat": [],
        "betas": [],
        "pred_cam": [],
        "cam_t_full": [],
        "box_center": [],
        "box_size": [],
        "image_size": [],
        "focal_length": [],
    }

    if len(boxes):
        dataset = ViTDetDataset(
            model_cfg,
            img_cv2,
            boxes,
            right,
            rescale_factor=rescale_factor,
            fp16=fast,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
        )
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.inference_mode():
                output = model(batch)

            mano_params = output["pred_mano_params"]
            pred_cam_raw = output["pred_cam"].detach().float()
            pred_cam_adjusted = pred_cam_raw.clone()
            multiplier = 2 * batch["right"].float() - 1
            pred_cam_adjusted[:, 1] *= multiplier

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            image_size = batch["img_size"].float()
            focal_length = (
                model_cfg.EXTRA.FOCAL_LENGTH
                / model_cfg.MODEL.IMAGE_SIZE
                * image_size.max(dim=1).values
            )
            cam_t_full = cam_crop_to_full(
                pred_cam_adjusted,
                box_center,
                box_size,
                image_size,
                focal_length,
            )

            values["global_orient_rotmat"].append(
                mano_params["global_orient"].detach().float().cpu().numpy()
            )
            values["hand_pose_rotmat"].append(
                mano_params["hand_pose"].detach().float().cpu().numpy()
            )
            values["betas"].append(
                mano_params["betas"].detach().float().cpu().numpy()
            )
            values["pred_cam"].append(pred_cam_raw.cpu().numpy())
            values["cam_t_full"].append(
                cam_t_full.detach().float().cpu().numpy()
            )
            values["box_center"].append(box_center.cpu().numpy())
            values["box_size"].append(box_size.cpu().numpy())
            values["image_size"].append(image_size.cpu().numpy())
            values["focal_length"].append(focal_length.cpu().numpy())

    def packed(field, item_shape):
        if not values[field]:
            return np.empty((0, *item_shape), dtype=np.float32)
        return np.concatenate(values[field], axis=0).astype(
            np.float32,
            copy=False,
        )

    return {
        "schema_version": 1,
        "frame_name": img_path.name,
        "source_image": str(img_path),
        "num_hands": int(len(boxes)),
        "global_orient_rotmat": packed(
            "global_orient_rotmat",
            (1, 3, 3),
        ),
        "hand_pose_rotmat": packed("hand_pose_rotmat", (15, 3, 3)),
        "betas": packed("betas", (10,)),
        "pred_cam": packed("pred_cam", (3,)),
        "cam_t_full": packed("cam_t_full", (3,)),
        "bbox_xyxy": boxes,
        "detection_confidence": confidence,
        "is_right": right,
        "box_center": packed("box_center", (2,)),
        "box_size": packed("box_size", ()),
        "image_size": packed("image_size", (2,)),
        "focal_length": packed("focal_length", ()),
    }


def interpolate_sequence_parameters(raw_frames, img_paths):
    """Apply the same fixed left/right interpolation used by the batch tool."""
    selected, observed, confidence, discarded = select_detections(
        raw_frames,
        img_paths,
    )
    previous, following, alpha, valid_side = interpolation_plan(observed)

    linear_outputs = {}
    for field, item_shape in LINEAR_FIELDS.items():
        observed_values = build_observed_values(
            raw_frames,
            img_paths,
            selected,
            observed,
            field,
            item_shape,
        )
        linear_outputs[field] = fill_linear(
            observed_values,
            previous,
            following,
            alpha,
            valid_side,
        )

    rotation_outputs = {}
    for field, item_shape in ROTATION_FIELDS.items():
        observed_values = build_observed_values(
            raw_frames,
            img_paths,
            selected,
            observed,
            field,
            item_shape,
        )
        rotation_outputs[field] = fill_rotations(
            observed_values,
            observed,
            valid_side,
        )

    valid_mask = np.broadcast_to(valid_side, observed.shape).copy()
    return {
        "linear": linear_outputs,
        "rotation": rotation_outputs,
        "observed_mask": observed,
        "interpolated_mask": valid_mask & ~observed,
        "valid_mask": valid_mask,
        "selected_detection_index": selected,
        "discarded_duplicate_count": discarded,
        "selected_confidence": confidence,
        "previous": previous,
        "following": following,
        "alpha": alpha,
    }


def draw_interpolation_status(
    image,
    frame_index,
    raw_num_hands,
    interpolated,
):
    observed = interpolated["observed_mask"][frame_index]
    filled = interpolated["interpolated_mask"][frame_index]
    valid = interpolated["valid_mask"][frame_index]
    bbox = interpolated["linear"]["bbox_xyxy"][frame_index]

    cv2.putText(
        image,
        f"raw detections: {raw_num_hands}",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"raw detections: {raw_num_hands}",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    for side_index, side_name in enumerate(("L", "R")):
        if observed[side_index]:
            status = "observed"
            color = (168, 70, 64)
        elif filled[side_index]:
            status = "interpolated"
            color = (13, 89, 255)
        else:
            status = "invalid"
            color = (128, 128, 128)

        y = 60 + side_index * 30
        cv2.putText(
            image,
            f"{side_name}: {status}",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        if filled[side_index] and valid[side_index]:
            x1, y1, x2, y2 = np.rint(bbox[side_index]).astype(np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)


def run_interpolated_sequence_demo(
    *,
    img_paths,
    out_folder,
    detector,
    model,
    model_cfg,
    renderer,
    device,
    rescale_factor,
    fast,
    output_video_fps,
):
    """Infer the full sequence, fill missing hands, and render the result."""
    raw_frames = []
    for frame_index, img_path in enumerate(img_paths):
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not read input image: {img_path}")
        raw_frames.append(
            infer_frame_parameters(
                img_path,
                image,
                detector,
                model,
                model_cfg,
                device,
                rescale_factor,
                fast,
            )
        )
        if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(img_paths):
            print(f"Inferred {frame_index + 1}/{len(img_paths)} frames")

    interpolated = interpolate_sequence_parameters(raw_frames, img_paths)
    observed_mask = interpolated["observed_mask"]
    filled_mask = interpolated["interpolated_mask"]
    valid_mask = interpolated["valid_mask"]

    video_path = Path(out_folder) / "interpolation_preview.mp4"
    video_writer = None
    mano_dtype = next(model.parameters()).dtype

    for frame_index, (img_path, raw_frame) in enumerate(
        zip(img_paths, raw_frames)
    ):
        image_bgr = cv2.imread(str(img_path))
        height, width = image_bgr.shape[:2]

        if video_writer is None and output_video_fps > 0:
            video_writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_video_fps,
                (width, height),
            )
            if not video_writer.isOpened():
                print(f"Warning: could not create video {video_path}")
                video_writer = False

        active_sides = np.flatnonzero(valid_mask[frame_index])
        if active_sides.size:
            global_orient = torch.from_numpy(
                interpolated["rotation"]["global_orient_rotmat"][
                    frame_index,
                    active_sides,
                ]
            ).to(device=device, dtype=mano_dtype)
            hand_pose = torch.from_numpy(
                interpolated["rotation"]["hand_pose_rotmat"][
                    frame_index,
                    active_sides,
                ]
            ).to(device=device, dtype=mano_dtype)
            betas = torch.from_numpy(
                interpolated["linear"]["betas"][
                    frame_index,
                    active_sides,
                ]
            ).to(device=device, dtype=mano_dtype)

            with torch.inference_mode():
                mano_output = model.mano(
                    global_orient=global_orient,
                    hand_pose=hand_pose,
                    betas=betas,
                    pose2rot=False,
                )
            vertices = mano_output.vertices.detach().float().cpu().numpy()

            vertices_to_render = []
            camera_to_render = []
            sides_to_render = []
            colors_to_render = []
            for local_index, side_index in enumerate(active_sides):
                hand_vertices = vertices[local_index].copy()
                if side_index == 0:
                    hand_vertices[:, 0] *= -1
                vertices_to_render.append(hand_vertices)
                camera_to_render.append(
                    interpolated["linear"]["cam_t_full"][
                        frame_index,
                        side_index,
                    ]
                )
                sides_to_render.append(int(side_index))
                colors_to_render.append(
                    INTERPOLATED_ORANGE
                    if filled_mask[frame_index, side_index]
                    else LIGHT_PURPLE
                )

            focal_length = (
                model_cfg.EXTRA.FOCAL_LENGTH
                / model_cfg.MODEL.IMAGE_SIZE
                * max(width, height)
            )
            camera_view = renderer.render_rgba_multiple(
                vertices_to_render,
                cam_t=camera_to_render,
                render_res=[width, height],
                is_right=sides_to_render,
                mesh_base_color=colors_to_render,
                scene_bg_color=(1, 1, 1),
                focal_length=focal_length,
            )
            input_rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
            output_rgb = (
                input_rgb * (1 - camera_view[:, :, 3:])
                + camera_view[:, :, :3] * camera_view[:, :, 3:]
            )
            output_bgr = np.clip(
                output_rgb[:, :, ::-1] * 255.0,
                0,
                255,
            ).astype(np.uint8)
        else:
            output_bgr = image_bgr.copy()

        draw_interpolation_status(
            output_bgr,
            frame_index,
            raw_frame["num_hands"],
            interpolated,
        )
        output_path = Path(out_folder) / f"{img_path.stem}.jpg"
        cv2.imwrite(str(output_path), output_bgr)
        if video_writer is not None and video_writer is not False:
            video_writer.write(output_bgr)

        if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(img_paths):
            print(f"Rendered {frame_index + 1}/{len(img_paths)} frames")

    if video_writer is not None and video_writer is not False:
        video_writer.release()

    raw_distribution = Counter(
        int(frame["num_hands"]) for frame in raw_frames
    )
    interpolated_frame_indices = np.flatnonzero(filled_mask.any(axis=1))
    summary = {
        "num_frames": len(img_paths),
        "raw_detection_count_distribution": {
            str(key): value for key, value in sorted(raw_distribution.items())
        },
        "observed_hand_slots": int(observed_mask.sum()),
        "interpolated_hand_slots": int(filled_mask.sum()),
        "invalid_hand_slots": int((~valid_mask).sum()),
        "discarded_duplicate_detections": int(
            interpolated["discarded_duplicate_count"].sum()
        ),
        "interpolated_frame_indices": interpolated_frame_indices.tolist(),
        "slot_order": ["left", "right"],
        "observed_color_rgb": list(LIGHT_PURPLE),
        "interpolated_color_rgb": list(INTERPOLATED_ORANGE),
        "video": str(video_path) if video_path.is_file() else None,
    }
    summary_path = Path(out_folder) / "interpolation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"Interpolation visualization complete: "
        f"frames={len(img_paths)}, "
        f"observed_slots={summary['observed_hand_slots']}, "
        f"interpolated_slots={summary['interpolated_hand_slots']}, "
        f"invalid_slots={summary['invalid_hand_slots']}"
    )
    print(f"Summary: {summary_path}")
    if summary["video"]:
        print(f"Video: {summary['video']}")


def main():
    parser = argparse.ArgumentParser(description='WiLoR demo code')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument(
        '--params_only',
        '--save_params_only',
        dest='params_only',
        action='store_true',
        default=False,
        help='Save one MANO/camera parameter NPZ per image; skip rendered images and meshes',
    )
    parser.add_argument(
        '--interpolate_missing_hands',
        '--interpolate_hands',
        dest='interpolate_missing_hands',
        action='store_true',
        default=False,
        help=(
            'Treat img_folder as an ordered sequence, fill missing left/right '
            'hands temporally, and render observed/interpolated meshes'
        ),
    )
    parser.add_argument(
        '--output_video_fps',
        type=float,
        default=25.0,
        help=(
            'FPS for interpolation_preview.mp4; use 0 to disable video output'
        ),
    )
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg'], help='List of file extensions to consider')
    parser.add_argument('--fast',   dest='fast', action='store_true', default=False, help='Use FP16 and layer dropping to accelerate inference')
    args = parser.parse_args()
    if args.params_only and args.save_mesh:
        parser.error('--params_only cannot be combined with --save_mesh')
    if args.interpolate_missing_hands and args.params_only:
        parser.error(
            '--interpolate_missing_hands cannot be combined with --params_only'
        )
    if args.interpolate_missing_hands and args.save_mesh:
        parser.error(
            '--interpolate_missing_hands cannot be combined with --save_mesh'
        )
    if args.output_video_fps < 0:
        parser.error('--output_video_fps must be non-negative')

    # Download and load checkpoints
    model, model_cfg = load_wilor(checkpoint_path = './pretrained_models/wilor_final.ckpt' , cfg_path= './pretrained_models/model_config.yaml')
    if args.fast:     
        torch.set_float32_matmul_precision('high')
        model = model.half()
        if not args.interpolate_missing_hands:
            model.backbone = torch.compile(model.backbone)
        model.backbone.skip_blocks = True 
        
    detector = YOLO('./pretrained_models/detector.pt')
    # Setup the renderer
    renderer = None if args.params_only else Renderer(model_cfg, faces=model.mano.faces)
    
    device   = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model    = model.to(device)
    detector = detector.to(device)
    model.eval()

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = sorted(
        {
            img
            for file_pattern in args.file_type
            for img in Path(args.img_folder).glob(file_pattern)
        }
    )
    if not img_paths:
        raise ValueError(f'No input images found in {args.img_folder}')
    if args.interpolate_missing_hands:
        run_interpolated_sequence_demo(
            img_paths=img_paths,
            out_folder=args.out_folder,
            detector=detector,
            model=model,
            model_cfg=model_cfg,
            renderer=renderer,
            device=device,
            rescale_factor=args.rescale_factor,
            fast=args.fast,
            output_video_fps=args.output_video_fps,
        )
        return

    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        img_fn, _ = os.path.splitext(os.path.basename(img_path))
        detections = detector(img_cv2, conf = 0.3, verbose=False)[0]
        bboxes    = []
        is_right  = []
        detection_confidence = []
        for det in detections: 
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())
            detection_confidence.append(det.boxes.conf.cpu().detach().squeeze().item())
        
        if len(bboxes) == 0:
            if args.params_only:
                output_path = os.path.join(args.out_folder, f'{img_fn}.npz')
                save_empty_mano_params(
                    output_path,
                    img_path,
                    image_size=[img_cv2.shape[1], img_cv2.shape[0]],
                )
                print(f'Saved 0 hand parameter sets to {output_path}')
            continue
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor, fp16=args.fast)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_joints= []
        all_kpts  = []
        all_global_orient = []
        all_hand_pose = []
        all_betas = []
        all_pred_cam = []
        all_box_center = []
        all_box_size = []
        all_img_size = []
        all_focal_length = []
        
        for batch in dataloader: 
            batch = recursive_to(batch, device)
    
            with torch.no_grad():
                out = model(batch) 
                
            multiplier    = (2*batch['right']-1)
            pred_cam_raw  = out['pred_cam'].detach().clone()
            pred_cam      = pred_cam_raw.clone()
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center    = batch["box_center"].float()
            box_size      = batch["box_size"].float()
            img_size      = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            if args.params_only:
                mano_params = out['pred_mano_params']
                all_global_orient.append(mano_params['global_orient'].detach().float().cpu().numpy())
                all_hand_pose.append(mano_params['hand_pose'].detach().float().cpu().numpy())
                all_betas.append(mano_params['betas'].detach().float().cpu().numpy())
                all_pred_cam.append(pred_cam_raw.float().cpu().numpy())
                all_cam_t.append(pred_cam_t_full.astype(np.float32))
                all_box_center.append(box_center.cpu().numpy().astype(np.float32))
                all_box_size.append(box_size.cpu().numpy().astype(np.float32))
                all_img_size.append(img_size.cpu().numpy().astype(np.float32))
                all_focal_length.append(
                    np.full((batch['img'].shape[0],), float(scaled_focal_length), dtype=np.float32)
                )
                continue

            
            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                verts  = out['pred_vertices'][n].detach().cpu().numpy()
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                
                is_right    = batch['right'][n].cpu().numpy()
                verts[:,0]  = (2*is_right-1)*verts[:,0]
                joints[:,0] = (2*is_right-1)*joints[:,0]
                cam_t = pred_cam_t_full[n]
                kpts_2d = project_full_img(verts, cam_t, scaled_focal_length, img_size[n])
                
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_joints.append(joints)
                all_kpts.append(kpts_2d)
                
                
                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_PURPLE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{n}.obj'))

        if args.params_only:
            global_orient = np.concatenate(all_global_orient, axis=0).astype(np.float32)
            hand_pose = np.concatenate(all_hand_pose, axis=0).astype(np.float32)
            betas = np.concatenate(all_betas, axis=0).astype(np.float32)
            global_orient_axis_angle = rotmat_to_axis_angle(global_orient).reshape(-1, 3).astype(np.float32)
            hand_pose_axis_angle = rotmat_to_axis_angle(hand_pose).reshape(-1, 45).astype(np.float32)
            mano_pose_axis_angle = np.concatenate(
                [global_orient_axis_angle, hand_pose_axis_angle],
                axis=1,
            )
            output_path = os.path.join(args.out_folder, f'{img_fn}.npz')
            np.savez_compressed(
                output_path,
                source_image=np.asarray(str(img_path)),
                global_orient_rotmat=global_orient,
                hand_pose_rotmat=hand_pose,
                global_orient_axis_angle=global_orient_axis_angle,
                hand_pose_axis_angle=hand_pose_axis_angle,
                mano_pose_axis_angle=mano_pose_axis_angle,
                betas=betas,
                pred_cam=np.concatenate(all_pred_cam, axis=0).astype(np.float32),
                cam_t_full=np.concatenate(all_cam_t, axis=0).astype(np.float32),
                bbox_xyxy=boxes.astype(np.float32),
                detection_confidence=np.asarray(detection_confidence, dtype=np.float32),
                is_right=right.astype(np.float32),
                box_center=np.concatenate(all_box_center, axis=0),
                box_size=np.concatenate(all_box_size, axis=0),
                image_size=np.concatenate(all_img_size, axis=0),
                focal_length=np.concatenate(all_focal_length, axis=0),
            )
            print(f'Saved {len(global_orient)} hand parameter sets to {output_path}')
            continue

        # Render front view
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}.jpg'), 255*input_img_overlay[:, :, ::-1])

def project_full_img(points, cam_trans, focal_length, img_res): 
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3) 
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:] 
    
    V_2d = (K @ points.T).T 
    return V_2d[..., :-1]

if __name__ == '__main__':
    main()
