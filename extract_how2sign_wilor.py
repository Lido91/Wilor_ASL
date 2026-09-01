#!/usr/bin/env python3
"""Batch-extract per-frame WiLoR MANO parameters from How2Sign directories.

Each input frame produces one pickle:

    <output_root>/wilor_params/<clip_id>/<frame_stem>.pkl

Each pickle is a dictionary whose first dimension is the number of hands in
that frame.  Frames with no detected hands still receive a pickle containing
shape-stable empty arrays.  Writes are atomic and resumable at frame level; a
``.complete`` marker avoids scanning completed clips on later runs.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data._utils.collate import default_collate
from ultralytics import YOLO

from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.utils.renderer import cam_crop_to_full


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_DATASET_ROOT = Path("/data/hwu/how2sign")
DEFAULT_SPLITS = ("test", "val", "train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract packed WiLoR MANO parameters for How2Sign clips."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Directory containing how2sign_images_<split> inputs.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help=(
            "Direct clip-directory root for a non-How2Sign dataset. "
            "Must be used together with --output-root."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Direct output root for --input-root; frame PKLs are written "
            "under <output-root>/wilor_params/<clip-id>."
        ),
    )
    parser.add_argument(
        "--split-name",
        default="custom",
        help="Metadata/sharding name used with direct --input-root mode.",
    )
    parser.add_argument(
        "--clip-id",
        dest="clip_ids",
        action="append",
        default=None,
        metavar="CLIP_ID",
        help=(
            "Only process this clip directory name; may be repeated. "
            "This option is available in direct --input-root mode."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=list(DEFAULT_SPLITS),
        help="Splits to process in order.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of independent workers sharing the clip set.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="This worker's zero-based shard index.",
    )
    parser.add_argument(
        "--detector-batch-size",
        type=int,
        default=32,
        help="Number of full frames passed to YOLO together.",
    )
    parser.add_argument(
        "--wilor-batch-size",
        type=int,
        default=64,
        help="Number of detected hand crops passed to WiLoR together.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="YOLO hand-detection confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="YOLO NMS IoU threshold.",
    )
    parser.add_argument(
        "--rescale-factor",
        type=float,
        default=2.0,
        help="Hand bounding-box padding factor.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use FP16 and WiLoR backbone layer dropping; torch.compile is not used.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Process at most this many assigned clips (smoke testing).",
    )
    parser.add_argument(
        "--max-frames-per-clip",
        type=int,
        default=None,
        help="Process at most this many frames from each clip (smoke testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute clips whose final archive already exists.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrained_models/wilor_final.ckpt"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("pretrained_models/model_config.yaml"),
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=Path("pretrained_models/detector.pt"),
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num-shards)")
    if args.detector_batch_size < 1 or args.wilor_batch_size < 1:
        parser.error("batch sizes must be at least 1")
    if args.max_clips is not None and args.max_clips < 1:
        parser.error("--max-clips must be at least 1")
    if args.max_frames_per_clip is not None and args.max_frames_per_clip < 1:
        parser.error("--max-frames-per-clip must be at least 1")
    if (args.input_root is None) != (args.output_root is None):
        parser.error("--input-root and --output-root must be used together")
    if args.clip_ids is not None:
        if args.input_root is None:
            parser.error("--clip-id requires --input-root")
        if len(args.clip_ids) != len(set(args.clip_ids)):
            parser.error("--clip-id must not contain duplicates")
    return args


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def clip_belongs_to_shard(
    split: str,
    clip_id: str,
    num_shards: int,
    shard_index: int,
) -> bool:
    digest = hashlib.sha1(f"{split}/{clip_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value % num_shards == shard_index


def list_frame_paths(clip_dir: Path, max_frames: int | None) -> list[Path]:
    paths = sorted(
        path
        for path in clip_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return paths if max_frames is None else paths[:max_frames]


def empty_array(shape: tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
    return np.empty(shape, dtype=dtype)


def rotation_matrices_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    if rotmat.shape[0] == 0:
        return empty_array((*rotmat.shape[:-2], 3))
    flat = rotmat.reshape(-1, 3, 3).astype(np.float64, copy=False)
    rotvec = Rotation.from_matrix(flat).as_rotvec().astype(np.float32)
    return rotvec.reshape(*rotmat.shape[:-2], 3)


def save_frame_pickle_atomic(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".tmp.pkl",
        dir=output_path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def build_frame_payload(
    *,
    split: str,
    clip_id: str,
    frame_path: Path,
    record: dict[str, list[np.ndarray]],
    confidence: float,
    iou: float,
    rescale_factor: float,
    fast: bool,
) -> dict:
    def stack(key: str, shape: tuple[int, ...]) -> np.ndarray:
        if not record[key]:
            return empty_array((0, *shape))
        return np.stack(record[key], axis=0).astype(np.float32, copy=False)

    global_orient_rotmat = stack("global_orient_rotmat", (1, 3, 3))
    hand_pose_rotmat = stack("hand_pose_rotmat", (15, 3, 3))
    global_orient_axis_angle = rotation_matrices_to_axis_angle(
        global_orient_rotmat
    ).reshape(-1, 3)
    hand_pose_axis_angle = rotation_matrices_to_axis_angle(
        hand_pose_rotmat
    ).reshape(-1, 45)

    mano_pose_axis_angle = np.concatenate(
        [global_orient_axis_angle, hand_pose_axis_angle],
        axis=1,
    )
    return {
        "schema_version": 1,
        "split": split,
        "clip_id": clip_id,
        "frame_name": frame_path.name,
        "num_hands": int(hand_pose_axis_angle.shape[0]),
        # Short canonical names use MANO axis-angle in radians.
        "hand_pose": hand_pose_axis_angle,
        "global_orient": global_orient_axis_angle,
        "mano_pose": mano_pose_axis_angle,
        # Explicit aliases and original network rotation matrices are retained.
        "global_orient_rotmat": global_orient_rotmat,
        "hand_pose_rotmat": hand_pose_rotmat,
        "global_orient_axis_angle": global_orient_axis_angle,
        "hand_pose_axis_angle": hand_pose_axis_angle,
        "mano_pose_axis_angle": mano_pose_axis_angle,
        "betas": stack("betas", (10,)),
        "pred_cam": stack("pred_cam", (3,)),
        "cam_t_full": stack("cam_t_full", (3,)),
        "bbox_xyxy": stack("bbox_xyxy", (4,)),
        "detection_confidence": stack("detection_confidence", ()),
        "is_right": stack("is_right", ()),
        "box_center": stack("box_center", (2,)),
        "box_size": stack("box_size", ()),
        "image_size": stack("image_size", (2,)),
        "focal_length": stack("focal_length", ()),
        "detector_confidence_threshold": np.float32(confidence),
        "detector_iou_threshold": np.float32(iou),
        "rescale_factor": np.float32(rescale_factor),
        "fast_mode": bool(fast),
    }


def new_frame_record() -> dict[str, list[np.ndarray]]:
    return {
        "global_orient_rotmat": [],
        "hand_pose_rotmat": [],
        "betas": [],
        "pred_cam": [],
        "cam_t_full": [],
        "bbox_xyxy": [],
        "detection_confidence": [],
        "is_right": [],
        "box_center": [],
        "box_size": [],
        "image_size": [],
        "focal_length": [],
    }


def run_wilor_batches(
    *,
    crop_items: list[dict[str, np.ndarray | torch.Tensor]],
    crop_refs: list[tuple[int, int]],
    frame_records: list[dict[str, list[np.ndarray]]],
    model: torch.nn.Module,
    model_cfg,
    device: torch.device,
    batch_size: int,
) -> None:
    for start in range(0, len(crop_items), batch_size):
        stop = min(start + batch_size, len(crop_items))
        batch = default_collate(crop_items[start:stop])
        batch = recursive_to(batch, device)

        with torch.inference_mode():
            output = model(batch)

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

        global_orient = (
            output["pred_mano_params"]["global_orient"]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        hand_pose = (
            output["pred_mano_params"]["hand_pose"]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        betas = (
            output["pred_mano_params"]["betas"]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        pred_cam_np = pred_cam_raw.cpu().numpy()
        cam_t_full_np = cam_t_full.detach().cpu().numpy()
        box_center_np = box_center.cpu().numpy()
        box_size_np = box_size.cpu().numpy()
        image_size_np = image_size.cpu().numpy()
        focal_length_np = focal_length.cpu().numpy()

        for local_index, (frame_index, detection_index) in enumerate(
            crop_refs[start:stop]
        ):
            record = frame_records[frame_index]
            record["global_orient_rotmat"].append(global_orient[local_index])
            record["hand_pose_rotmat"].append(hand_pose[local_index])
            record["betas"].append(betas[local_index])
            record["pred_cam"].append(pred_cam_np[local_index])
            record["cam_t_full"].append(cam_t_full_np[local_index])
            record["box_center"].append(box_center_np[local_index])
            record["box_size"].append(box_size_np[local_index])
            record["image_size"].append(image_size_np[local_index])
            record["focal_length"].append(focal_length_np[local_index])


def process_clip(
    *,
    split: str,
    clip_dir: Path,
    output_dir: Path,
    frame_paths: list[Path],
    detector: YOLO,
    model: torch.nn.Module,
    model_cfg,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[int, int]:
    processed_frames = 0
    detected_hands = 0

    for chunk_start in range(0, len(frame_paths), args.detector_batch_size):
        chunk_paths = frame_paths[
            chunk_start : chunk_start + args.detector_batch_size
        ]
        frame_records = [new_frame_record() for _ in chunk_paths]
        results = detector.predict(
            source=[str(path) for path in chunk_paths],
            conf=args.confidence,
            iou=args.iou,
            device=args.device,
            batch=args.detector_batch_size,
            half=args.fast,
            verbose=False,
            stream=False,
        )

        crop_items: list[dict[str, np.ndarray | torch.Tensor]] = []
        crop_refs: list[tuple[int, int]] = []

        for chunk_index, result in enumerate(results):
            frame_index = chunk_index
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            is_right = result.boxes.cls.detach().cpu().numpy().astype(np.float32)
            scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            dataset = ViTDetDataset(
                model_cfg,
                result.orig_img,
                boxes,
                is_right,
                rescale_factor=args.rescale_factor,
                fp16=args.fast,
            )

            record = frame_records[frame_index]
            for detection_index in range(len(dataset)):
                crop_items.append(dataset[detection_index])
                crop_refs.append((frame_index, detection_index))
                record["bbox_xyxy"].append(boxes[detection_index])
                record["detection_confidence"].append(scores[detection_index])
                record["is_right"].append(is_right[detection_index])

        run_wilor_batches(
            crop_items=crop_items,
            crop_refs=crop_refs,
            frame_records=frame_records,
            model=model,
            model_cfg=model_cfg,
            device=device,
            batch_size=args.wilor_batch_size,
        )

        for frame_path, record in zip(chunk_paths, frame_records):
            payload = build_frame_payload(
                split=split,
                clip_id=clip_dir.name,
                frame_path=frame_path,
                record=record,
                confidence=args.confidence,
                iou=args.iou,
                rescale_factor=args.rescale_factor,
                fast=args.fast,
            )
            output_path = output_dir / f"{frame_path.stem}.pkl"
            save_frame_pickle_atomic(output_path, payload)
            processed_frames += 1
            detected_hands += payload["num_hands"]

    return processed_frames, detected_hands


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full How2Sign extraction")
    if args.device >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {args.device} is unavailable; "
            f"found {torch.cuda.device_count()} device(s)"
        )

    device = torch.device(f"cuda:{args.device}")
    log(
        f"worker shard={args.shard_index}/{args.num_shards}, "
        f"device={device}, fast={args.fast}"
    )
    model, model_cfg = load_wilor(
        checkpoint_path=str(args.checkpoint),
        cfg_path=str(args.model_config),
    )
    if args.fast:
        model = model.half()
        model.backbone.skip_blocks = True
    model = model.to(device)
    model.eval()
    detector = YOLO(str(args.detector))

    processed_clips = 0
    skipped_clips = 0
    failed_clips = 0
    processed_frames = 0
    detected_hands = 0
    started_at = time.monotonic()

    if args.input_root is not None:
        split_jobs = [(args.split_name, args.input_root, args.output_root)]
    else:
        split_jobs = [
            (
                split,
                args.dataset_root / f"how2sign_images_{split}",
                args.dataset_root / f"how2sign_images_{split}_wilor_out",
            )
            for split in args.splits
        ]

    for split, input_root, output_root in split_jobs:
        if not input_root.is_dir():
            raise FileNotFoundError(f"Input split does not exist: {input_root}")

        clip_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
        if args.clip_ids is not None:
            clip_dirs_by_id = {path.name: path for path in clip_dirs}
            missing_clip_ids = [
                clip_id for clip_id in args.clip_ids if clip_id not in clip_dirs_by_id
            ]
            if missing_clip_ids:
                missing = ", ".join(missing_clip_ids)
                raise FileNotFoundError(
                    f"Clip IDs not found under {input_root}: {missing}"
                )
            clip_dirs = [clip_dirs_by_id[clip_id] for clip_id in args.clip_ids]
        assigned = [
            path
            for path in clip_dirs
            if clip_belongs_to_shard(
                split,
                path.name,
                args.num_shards,
                args.shard_index,
            )
        ]
        log(
            f"split={split}: {len(assigned)}/{len(clip_dirs)} clips assigned; "
            f"output={output_root}"
        )

        for clip_dir in assigned:
            if args.max_clips is not None and processed_clips >= args.max_clips:
                log("Reached --max-clips; stopping cleanly")
                return 0

            output_dir = output_root / "wilor_params" / clip_dir.name
            complete_marker = output_dir / ".complete"
            if complete_marker.is_file() and not args.overwrite:
                skipped_clips += 1
                continue

            frame_paths = list_frame_paths(
                clip_dir,
                max_frames=args.max_frames_per_clip,
            )
            if not frame_paths:
                log(f"SKIP empty clip: {split}/{clip_dir.name}")
                skipped_clips += 1
                continue
            if not args.overwrite:
                frame_paths = [
                    frame_path
                    for frame_path in frame_paths
                    if not (output_dir / f"{frame_path.stem}.pkl").is_file()
                ]
                if not frame_paths:
                    if args.max_frames_per_clip is None:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        complete_marker.touch()
                    skipped_clips += 1
                    continue

            clip_started_at = time.monotonic()
            try:
                frames, hands = process_clip(
                    split=split,
                    clip_dir=clip_dir,
                    output_dir=output_dir,
                    frame_paths=frame_paths,
                    detector=detector,
                    model=model,
                    model_cfg=model_cfg,
                    device=device,
                    args=args,
                )
                if args.max_frames_per_clip is None:
                    complete_marker.touch()
            except Exception:
                failed_clips += 1
                log(f"FAIL {split}/{clip_dir.name}\n{traceback.format_exc()}")
                continue

            processed_clips += 1
            processed_frames += frames
            detected_hands += hands
            clip_seconds = time.monotonic() - clip_started_at
            total_seconds = time.monotonic() - started_at
            log(
                f"DONE {split}/{clip_dir.name}: frames={frames}, hands={hands}, "
                f"clip_fps={frames / max(clip_seconds, 1e-6):.2f}; "
                f"worker_total clips={processed_clips}, frames={processed_frames}, "
                f"fps={processed_frames / max(total_seconds, 1e-6):.2f}, "
                f"failed={failed_clips}, skipped={skipped_clips}"
            )

    elapsed = time.monotonic() - started_at
    log(
        f"COMPLETE clips={processed_clips}, frames={processed_frames}, "
        f"hands={detected_hands}, failed={failed_clips}, "
        f"skipped={skipped_clips}, seconds={elapsed:.1f}"
    )
    return 1 if failed_clips else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted")
        raise SystemExit(130)
