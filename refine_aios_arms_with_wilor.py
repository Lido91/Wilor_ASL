#!/usr/bin/env python3
"""Refine fused SMPL-X upper-body pose using WiLoR hand anchors.

The input must already contain fused WiLoR finger rotations and a fixed
clipwise camera. WiLoR MANO wrist and palm joints are projected into normalized
image coordinates. A differentiable SMPL-X fitting stage then adjusts only
spine3, collars, shoulders, elbows, and wrists.

Finger pose, camera, shape, face, root, and lower-body parameters are copied
unchanged. The optimization is clip-level and regularizes both the magnitude
and temporal derivatives of the upper-body correction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import smplx
import torch
from tqdm.auto import tqdm

from fuse_shared_aios_wilor import save_npz_atomic, zero_phase_filter


DEFAULT_FUSED_ROOT = Path("shared_samples/aios_smoothed_wilor_hands_fused")
DEFAULT_WILOR_ROOT = Path("shared_samples/wilor_params_interpolated")
DEFAULT_OUTPUT_ROOT = Path("shared_samples/aios_wilor_arm_refined")
DEFAULT_SMPLX_MODEL_ROOT = Path(
    "/home/student/hwu/Workplace/SOKE/prepare/deps/smpl_models"
)
DEFAULT_MANO_MODEL_PATH = Path("mano_data/MANO_RIGHT.pkl")

FUSION_SCHEMA_VERSION = 4
BODY_JOINT_COUNT = 21
REFINED_BODY_JOINT_INDICES = (8, 12, 13, 15, 16, 17, 18, 19, 20)
REFINED_BODY_JOINT_NAMES = (
    "spine3",
    "left_collar",
    "right_collar",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
SMPLX_HAND_JOINT_INDICES = (
    (20, 25, 28, 31, 34, 37),
    (21, 40, 43, 46, 49, 52),
)
MANO_PALM_JOINT_INDICES = (0, 1, 4, 7, 10, 13)
PRIOR_JOINT_WEIGHTS = (8.0, 4.0, 4.0, 1.5, 1.5, 1.0, 1.0, 0.75, 0.75)
MAX_DELTA_RADIANS = (0.20, 0.30, 0.30, 0.60, 0.60, 0.75, 0.75, 0.60, 0.60)

REQUIRED_FUSED_FIELDS = {
    "fusion_schema_version",
    "clip_id",
    "source_fps",
    "num_frames",
    "frame_names",
    "img_shape",
    "cam_trans",
    "smplx_root_pose",
    "smplx_body_pose",
    "smplx_lhand_pose",
    "smplx_rhand_pose",
    "smplx_jaw_pose",
    "smplx_expr",
    "smplx_shape",
    "wilor_hands_smoothed",
    "camera_fixed_clipwise",
}
REQUIRED_WILOR_FIELDS = {
    "clip_id",
    "num_frames",
    "frame_names",
    "hand_slot_order",
    "global_orient_rotmat",
    "hand_pose_rotmat",
    "betas",
    "cam_t_full",
    "image_size",
    "focal_length",
    "observed_mask",
    "interpolated_mask",
    "valid_mask",
    "detection_confidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-root", type=Path, default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--wilor-root", type=Path, default=DEFAULT_WILOR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--smplx-model-root", type=Path, default=DEFAULT_SMPLX_MODEL_ROOT
    )
    parser.add_argument(
        "--mano-model-path", type=Path, default=DEFAULT_MANO_MODEL_PATH
    )
    parser.add_argument(
        "--clip-id",
        dest="clip_ids",
        action="append",
        help="Process only this clip; may be repeated.",
    )
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:N")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--aios-focal-length", type=float, default=5000.0)
    parser.add_argument("--target-cutoff-hz", type=float, default=5.0)
    parser.add_argument("--wrist-weight", type=float, default=4.0)
    parser.add_argument("--palm-weight", type=float, default=1.0)
    parser.add_argument("--pose-prior-weight", type=float, default=0.08)
    parser.add_argument("--velocity-weight", type=float, default=0.20)
    parser.add_argument("--acceleration-weight", type=float, default=1.0)
    parser.add_argument(
        "--interpolated-weight",
        type=float,
        default=0.25,
        help="Relative weight for a temporally interpolated WiLoR hand.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--no-verts",
        action="store_true",
        help=(
            "Skip SMPL-X vertex computation during fitting. The loss only uses "
            "joints, so this is a speed/memory optimization, not a model change."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input contracts without loading models or optimizing.",
    )
    args = parser.parse_args()

    positive = (
        "iterations",
        "learning_rate",
        "batch_size",
        "aios_focal_length",
        "target_cutoff_hz",
        "wrist_weight",
        "palm_weight",
        "log_every",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    nonnegative = (
        "pose_prior_weight",
        "velocity_weight",
        "acceleration_weight",
        "interpolated_weight",
    )
    for name in nonnegative:
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.interpolated_weight > 1:
        parser.error("--interpolated-weight must not exceed 1")
    if args.max_clips is not None and args.max_clips <= 0:
        parser.error("--max-clips must be positive")
    return args


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested}")
    return device


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"NPZ file does not exist: {path}")
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def require_fields(
    data: dict[str, np.ndarray], required: set[str], path: Path
) -> None:
    missing = sorted(required - data.keys())
    if missing:
        raise KeyError(f"{path}: missing fields {missing}")


def validate_inputs(
    fused: dict[str, np.ndarray],
    wilor: dict[str, np.ndarray],
    fused_path: Path,
    wilor_path: Path,
) -> tuple[str, int, float]:
    require_fields(fused, REQUIRED_FUSED_FIELDS, fused_path)
    require_fields(wilor, REQUIRED_WILOR_FIELDS, wilor_path)
    clip_id = str(fused["clip_id"])
    if clip_id != fused_path.stem or str(wilor["clip_id"]) != clip_id:
        raise ValueError(f"{clip_id}: clip IDs or filename stem disagree")
    input_schema = int(fused["fusion_schema_version"])
    if input_schema < 3:
        raise ValueError(
            f"{clip_id}: arm refinement requires fusion schema v3+, "
            f"got v{input_schema}"
        )
    if not bool(fused["camera_fixed_clipwise"]):
        raise ValueError(f"{clip_id}: camera must be fixed clipwise")

    num_frames = int(fused["num_frames"])
    if num_frames <= 0 or int(wilor["num_frames"]) != num_frames:
        raise ValueError(f"{clip_id}: invalid or mismatched frame count")
    if not np.array_equal(fused["frame_names"], wilor["frame_names"]):
        raise ValueError(f"{clip_id}: fused and WiLoR frame_names differ")
    if tuple(wilor["hand_slot_order"].tolist()) != ("left", "right"):
        raise ValueError(f"{clip_id}: WiLoR hand slots are not [left, right]")

    shapes = {
        "img_shape": (num_frames, 2),
        "cam_trans": (num_frames, 3),
        "smplx_root_pose": (num_frames, 3),
        "smplx_body_pose": (num_frames, 63),
        "smplx_lhand_pose": (num_frames, 45),
        "smplx_rhand_pose": (num_frames, 45),
        "smplx_jaw_pose": (num_frames, 3),
        "smplx_expr": (num_frames, 10),
        "smplx_shape": (num_frames, 10),
    }
    for key, expected in shapes.items():
        if fused[key].shape != expected:
            raise ValueError(
                f"{fused_path}: {key} shape {fused[key].shape}, "
                f"expected {expected}"
            )
    wilor_shapes = {
        "global_orient_rotmat": (num_frames, 2, 1, 3, 3),
        "hand_pose_rotmat": (num_frames, 2, 15, 3, 3),
        "betas": (num_frames, 2, 10),
        "cam_t_full": (num_frames, 2, 3),
        "image_size": (num_frames, 2, 2),
        "focal_length": (num_frames, 2),
        "observed_mask": (num_frames, 2),
        "interpolated_mask": (num_frames, 2),
        "valid_mask": (num_frames, 2),
        "detection_confidence": (num_frames, 2),
    }
    for key, expected in wilor_shapes.items():
        if wilor[key].shape != expected:
            raise ValueError(
                f"{wilor_path}: {key} shape {wilor[key].shape}, "
                f"expected {expected}"
            )
    for path, data in ((fused_path, fused), (wilor_path, wilor)):
        for key, value in data.items():
            array = np.asarray(value)
            if array.dtype.kind in "fc" and not np.isfinite(array).all():
                raise ValueError(f"{path}: {key} contains NaN or Inf")

    fps = float(fused["source_fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{clip_id}: invalid source_fps={fps}")
    return clip_id, num_frames, fps


def discover_clip_paths(
    root: Path, requested: Sequence[str] | None
) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Fused root does not exist: {root}")
    if requested:
        # Clip ids contain dots (timestamps), so Path.stem would truncate at the
        # last one. Strip only a trailing ".npz".
        paths = [
            root / f"{clip_id[:-4] if clip_id.endswith('.npz') else clip_id}.npz"
            for clip_id in requested
        ]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Requested fused clips are missing: {missing}")
        return paths
    return sorted(root.glob("*.npz"))


def make_smplx_model(model_root: Path, device: torch.device) -> torch.nn.Module:
    model_path = model_root / "smplx/SMPLX_NEUTRAL.npz"
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL-X model does not exist: {model_path}")
    layer_args = {
        "create_global_orient": False,
        "create_body_pose": False,
        "create_left_hand_pose": False,
        "create_right_hand_pose": False,
        "create_jaw_pose": False,
        "create_leye_pose": False,
        "create_reye_pose": False,
        "create_betas": False,
        "create_expression": False,
        "create_transl": False,
    }
    return smplx.create(
        str(model_root),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        flat_hand_mean=False,
        use_face_contour=True,
        num_betas=10,
        num_expression_coeffs=10,
        **layer_args,
    ).to(device).eval()


def make_mano_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    if not model_path.is_file():
        raise FileNotFoundError(f"MANO model does not exist: {model_path}")
    legacy_aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in legacy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    from smplx.body_models import MANOLayer

    return MANOLayer(
        str(model_path), is_rhand=True, num_betas=10
    ).to(device).eval()


def build_wilor_targets(
    wilor: dict[str, np.ndarray],
    mano_model: torch.nn.Module,
    device: torch.device,
    fps: float,
    cutoff_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(wilor["valid_mask"], dtype=bool)
    indices = np.argwhere(valid)
    targets = np.zeros((*valid.shape, len(MANO_PALM_JOINT_INDICES), 2))
    if len(indices):
        frame_indices = indices[:, 0]
        side_indices = indices[:, 1]
        with torch.inference_mode():
            global_orient = torch.from_numpy(
                wilor["global_orient_rotmat"][frame_indices, side_indices]
            ).to(device=device, dtype=torch.float32)
            hand_pose = torch.from_numpy(
                wilor["hand_pose_rotmat"][frame_indices, side_indices]
            ).to(device=device, dtype=torch.float32)
            betas = torch.from_numpy(
                wilor["betas"][frame_indices, side_indices]
            ).to(device=device, dtype=torch.float32)
            output = mano_model(
                global_orient=global_orient,
                hand_pose=hand_pose,
                betas=betas,
                return_verts=True,
            )
            joints = output.joints[:, MANO_PALM_JOINT_INDICES].clone()
            left = torch.from_numpy(side_indices == 0).to(device)
            joints[left, :, 0] *= -1.0
            translation = torch.from_numpy(
                wilor["cam_t_full"][frame_indices, side_indices]
            ).to(device=device, dtype=torch.float32)
            joints = joints + translation[:, None, :]
            focal = torch.from_numpy(
                wilor["focal_length"][frame_indices, side_indices]
            ).to(device=device, dtype=torch.float32)
            image_size = torch.from_numpy(
                wilor["image_size"][frame_indices, side_indices]
            ).to(device=device, dtype=torch.float32)
            depth = joints[..., 2].clamp_min(1e-4)
            projected = torch.stack(
                (
                    focal[:, None] * joints[..., 0] / depth,
                    focal[:, None] * joints[..., 1] / depth,
                ),
                dim=-1,
            )
            projected = projected + image_size[:, None, :] / 2.0
            normalized = projected / image_size[:, None, :]
        targets[frame_indices, side_indices] = normalized.cpu().numpy()

    for side in range(2):
        if valid[:, side].any():
            targets[:, side] = zero_phase_filter(
                targets[:, side],
                fps,
                cutoff_hz,
                order=4,
                hampel_window=7,
                hampel_sigmas=3.0,
            )

    observed = np.asarray(wilor["observed_mask"], dtype=bool)
    confidence = np.asarray(wilor["detection_confidence"], dtype=np.float32)
    return targets.astype(np.float32), np.where(observed, confidence, 0.0)


def make_side_weights(
    wilor: dict[str, np.ndarray],
    observed_confidence: np.ndarray,
    interpolated_weight: float,
) -> np.ndarray:
    valid = np.asarray(wilor["valid_mask"], dtype=bool)
    observed = np.asarray(wilor["observed_mask"], dtype=bool)
    weights = np.zeros(valid.shape, dtype=np.float32)
    weights[observed] = np.clip(observed_confidence[observed], 0.25, 1.0)
    weights[valid & ~observed] = interpolated_weight
    return weights


def tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array)).to(
        device=device, dtype=torch.float32
    )


def fused_tensors(
    fused: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "shape": tensor(fused["smplx_shape"], device),
        "root": tensor(fused["smplx_root_pose"], device),
        "body": tensor(fused["smplx_body_pose"], device).view(-1, 21, 3),
        "left_hand": tensor(fused["smplx_lhand_pose"], device),
        "right_hand": tensor(fused["smplx_rhand_pose"], device),
        "jaw": tensor(fused["smplx_jaw_pose"], device),
        "expression": tensor(fused["smplx_expr"], device),
        "camera": tensor(fused["cam_trans"], device),
        "image_shape": tensor(fused["img_shape"], device),
    }


def smplx_palm_joints(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    body_pose: torch.Tensor,
    start: int,
    stop: int,
    return_verts: bool = True,
) -> torch.Tensor:
    count = stop - start
    zeros = torch.zeros((count, 3), device=body_pose.device)
    output = model(
        betas=values["shape"][start:stop],
        global_orient=values["root"][start:stop],
        body_pose=body_pose.reshape(count, -1),
        left_hand_pose=values["left_hand"][start:stop],
        right_hand_pose=values["right_hand"][start:stop],
        jaw_pose=values["jaw"][start:stop],
        leye_pose=zeros,
        reye_pose=zeros,
        expression=values["expression"][start:stop],
        return_verts=return_verts,
    )
    indices = torch.tensor(
        SMPLX_HAND_JOINT_INDICES,
        device=body_pose.device,
        dtype=torch.long,
    )
    return output.joints[:, indices]


def project_smplx_joints(
    joints: torch.Tensor,
    camera: torch.Tensor,
    image_shape: torch.Tensor,
    focal_length: float,
) -> torch.Tensor:
    height = image_shape[:, 0]
    width = image_shape[:, 1]
    depth = (joints[..., 2] + camera[:, None, None, 2]).clamp_min(1e-4)
    x = focal_length * (
        joints[..., 0] - camera[:, None, None, 0]
    ) / depth + width[:, None, None] / 2.0
    y = focal_length * (
        joints[..., 1] + camera[:, None, None, 1]
    ) / depth + height[:, None, None] / 2.0
    return torch.stack(
        (x / width[:, None, None], y / height[:, None, None]), dim=-1
    )


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def data_losses(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    wrist_error = ((predicted[:, :, 0] - targets[:, :, 0]) ** 2).sum(-1)
    wrist_loss = weighted_mean(wrist_error, weights)

    predicted_palm = predicted[:, :, 1:] - predicted[:, :, :1]
    target_palm = targets[:, :, 1:] - targets[:, :, :1]
    predicted_scale = torch.sqrt(
        (predicted_palm.square().sum(-1).mean(-1, keepdim=True)).clamp_min(1e-8)
    )
    target_scale = torch.sqrt(
        (target_palm.square().sum(-1).mean(-1, keepdim=True)).clamp_min(1e-8)
    )
    predicted_normalized = predicted_palm / predicted_scale[..., None]
    target_normalized = target_palm / target_scale[..., None]
    palm_error = (
        (predicted_normalized - target_normalized).square().mean(dim=(-1, -2))
    )
    palm_loss = weighted_mean(palm_error, weights)
    return wrist_loss, palm_loss


def regularization_loss(
    delta: torch.Tensor, args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prior_weights = torch.tensor(
        PRIOR_JOINT_WEIGHTS, device=delta.device, dtype=delta.dtype
    )
    prior = (delta.square() * prior_weights[None, :, None]).mean()
    velocity = (
        (delta[1:] - delta[:-1]).square().mean()
        if len(delta) > 1
        else delta.new_zeros(())
    )
    acceleration = (
        (delta[2:] - 2.0 * delta[1:-1] + delta[:-2]).square().mean()
        if len(delta) > 2
        else delta.new_zeros(())
    )
    total = (
        args.pose_prior_weight * prior
        + args.velocity_weight * velocity
        + args.acceleration_weight * acceleration
    )
    return total, {
        "prior": prior,
        "velocity": velocity,
        "acceleration": acceleration,
    }


def clamp_delta(delta: torch.Tensor) -> None:
    limits = torch.tensor(
        MAX_DELTA_RADIANS, device=delta.device, dtype=delta.dtype
    )[None, :, None]
    norms = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    scale = torch.minimum(torch.ones_like(norms), limits / norms.clamp_min(1e-8))
    delta.mul_(scale)


def predicted_targets_for_delta(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    delta: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    outputs = []
    joint_indices = torch.tensor(
        REFINED_BODY_JOINT_INDICES, device=delta.device, dtype=torch.long
    )
    for start in range(0, len(delta), args.batch_size):
        stop = min(start + args.batch_size, len(delta))
        body_pose = values["body"][start:stop].clone()
        body_pose[:, joint_indices] = body_pose[:, joint_indices] + delta[start:stop]
        joints = smplx_palm_joints(
            model,
            values,
            body_pose,
            start,
            stop,
            return_verts=not args.no_verts,
        )
        outputs.append(
            project_smplx_joints(
                joints,
                values["camera"][start:stop],
                values["image_shape"][start:stop],
                args.aios_focal_length,
            )
        )
    return torch.cat(outputs, dim=0)


def optimize_arm_pose(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    targets: np.ndarray,
    side_weights: np.ndarray,
    metric_image_size: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    device = values["body"].device
    target_tensor = tensor(targets, device)
    weight_tensor = tensor(side_weights, device)
    num_frames = len(target_tensor)
    delta = torch.nn.Parameter(
        torch.zeros(
            (num_frames, len(REFINED_BODY_JOINT_INDICES), 3),
            device=device,
        )
    )
    optimizer = torch.optim.Adam([delta], lr=args.learning_rate)

    with torch.no_grad():
        before = predicted_targets_for_delta(model, values, delta, args)

    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        regularization, regularization_parts = regularization_loss(delta, args)
        regularization.backward()
        wrist_value = 0.0
        palm_value = 0.0
        weight_total = float(weight_tensor.sum().item())
        for start in range(0, num_frames, args.batch_size):
            stop = min(start + args.batch_size, num_frames)
            predicted = predicted_targets_for_delta(
                model,
                {key: value[start:stop] for key, value in values.items()},
                delta[start:stop],
                argparse.Namespace(**{**vars(args), "batch_size": stop - start}),
            )
            chunk_weights = weight_tensor[start:stop]
            wrist_loss, palm_loss = data_losses(
                predicted, target_tensor[start:stop], chunk_weights
            )
            chunk_fraction = float(chunk_weights.sum().item()) / max(
                weight_total, 1e-8
            )
            data_loss = chunk_fraction * (
                args.wrist_weight * wrist_loss + args.palm_weight * palm_loss
            )
            data_loss.backward()
            wrist_value += chunk_fraction * float(wrist_loss.detach().item())
            palm_value += chunk_fraction * float(palm_loss.detach().item())
        optimizer.step()
        with torch.no_grad():
            clamp_delta(delta)

        if iteration == 1 or iteration % args.log_every == 0:
            total = (
                args.wrist_weight * wrist_value
                + args.palm_weight * palm_value
                + float(regularization.detach().item())
            )
            tqdm.write(
                f"  iter={iteration:04d} loss={total:.6f} "
                f"wrist={wrist_value:.6f} palm={palm_value:.6f} "
                f"prior={float(regularization_parts['prior']):.6f} "
                f"vel={float(regularization_parts['velocity']):.6f} "
                f"acc={float(regularization_parts['acceleration']):.6f}"
            )

    with torch.no_grad():
        after = predicted_targets_for_delta(model, values, delta, args)
    refined_body = values["body"].clone()
    refined_body[:, REFINED_BODY_JOINT_INDICES] += delta

    metrics = wrist_metrics(
        before,
        after,
        target_tensor,
        weight_tensor,
        tensor(side_weights > 0, device).bool(),
        tensor(metric_image_size, device),
        delta,
    )
    return (
        refined_body.detach()
        .reshape(num_frames, 63)
        .cpu()
        .numpy()
        .astype(np.float32),
        delta.detach().cpu().numpy().astype(np.float32),
        metrics,
    )


def wrist_metrics(
    before: torch.Tensor,
    after: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    image_size: torch.Tensor,
    delta: torch.Tensor,
) -> dict[str, float]:
    del weights
    before_error = torch.linalg.vector_norm(
        (before[:, :, 0] - targets[:, :, 0]) * image_size, dim=-1
    )[valid]
    after_error = torch.linalg.vector_norm(
        (after[:, :, 0] - targets[:, :, 0]) * image_size, dim=-1
    )[valid]
    delta_degrees = torch.linalg.vector_norm(delta, dim=-1) * (180.0 / np.pi)
    return {
        "wrist_error_before_median_px": float(before_error.median()),
        "wrist_error_before_p95_px": float(torch.quantile(before_error, 0.95)),
        "wrist_error_after_median_px": float(after_error.median()),
        "wrist_error_after_p95_px": float(torch.quantile(after_error, 0.95)),
        "arm_delta_rms_degrees": float(torch.sqrt(delta_degrees.square().mean())),
        "arm_delta_max_degrees": float(delta_degrees.max()),
    }


def write_passthrough(
    fused: dict[str, np.ndarray],
    output_path: Path,
    num_frames: int,
    args: argparse.Namespace,
) -> None:
    """Write a v4 file for a clip that had no WiLoR hand targets.

    Body pose is copied unchanged and delta is zero, so the clip is identical to
    its stage-1 input. The field set matches a refined clip so downstream code
    needs no special case; ``arm_refined`` is False and the wrist metrics are
    NaN to mark that no fit was performed.
    """
    payload = dict(fused)
    payload.update(
        {
            "fusion_schema_version": np.int32(FUSION_SCHEMA_VERSION),
            "input_fusion_schema_version": np.int32(
                int(fused["fusion_schema_version"])
            ),
            "arm_refined": np.bool_(False),
            "arm_refinement_skipped_reason": np.asarray(
                "no_valid_wilor_hand_targets"
            ),
            "arm_refinement_method": np.asarray("passthrough"),
            "arm_refined_body_joint_indices": np.asarray(
                REFINED_BODY_JOINT_INDICES, dtype=np.int32
            ),
            "arm_refined_body_joint_names": np.asarray(REFINED_BODY_JOINT_NAMES),
            "arm_refinement_pose_delta": np.zeros(
                (num_frames, len(REFINED_BODY_JOINT_INDICES), 3), dtype=np.float32
            ),
            "arm_refinement_iterations": np.int32(0),
            "arm_refinement_learning_rate": np.float32(args.learning_rate),
            "arm_target_cutoff_hz": np.float32(args.target_cutoff_hz),
            "arm_wrist_weight": np.float32(args.wrist_weight),
            "arm_palm_weight": np.float32(args.palm_weight),
            "arm_pose_prior_weight": np.float32(args.pose_prior_weight),
            "arm_velocity_weight": np.float32(args.velocity_weight),
            "arm_acceleration_weight": np.float32(args.acceleration_weight),
            "arm_interpolated_target_weight": np.float32(args.interpolated_weight),
        }
    )
    for key in (
        "wrist_error_before_median_px",
        "wrist_error_before_p95_px",
        "wrist_error_after_median_px",
        "wrist_error_after_p95_px",
        "arm_delta_rms_degrees",
        "arm_delta_max_degrees",
    ):
        payload[key] = np.float32(np.nan)
    save_npz_atomic(output_path, payload)


def refine_clip(
    fused_path: Path,
    wilor_path: Path,
    output_path: Path,
    smplx_model: torch.nn.Module,
    mano_model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float] | None:
    fused = load_npz(fused_path)
    wilor = load_npz(wilor_path)
    clip_id, num_frames, fps = validate_inputs(
        fused, wilor, fused_path, wilor_path
    )
    targets, observed_confidence = build_wilor_targets(
        wilor, mano_model, device, fps, args.target_cutoff_hz
    )
    side_weights = make_side_weights(
        wilor, observed_confidence, args.interpolated_weight
    )
    if not np.any(side_weights > 0):
        # WiLoR never saw either hand in this clip, so there is nothing to fit
        # against. Pass the fused clip through unrefined (delta = 0) so the
        # output set stays complete, and flag it for downstream filtering.
        tqdm.write(
            f"[passthrough] {clip_id}: no valid WiLoR hand targets, "
            f"writing unrefined copy (delta = 0)"
        )
        write_passthrough(fused, output_path, num_frames, args)
        return None

    tqdm.write(
        f"[refine] {clip_id}: frames={num_frames}, "
        f"observed_targets={int(wilor['observed_mask'].sum())}, "
        f"interpolated_targets={int(wilor['interpolated_mask'].sum())}"
    )
    values = fused_tensors(fused, device)
    refined_body, delta, metrics = optimize_arm_pose(
        smplx_model,
        values,
        targets,
        side_weights,
        np.asarray(wilor["image_size"], dtype=np.float32),
        args,
    )

    payload = dict(fused)
    payload.update(
        {
            "fusion_schema_version": np.int32(FUSION_SCHEMA_VERSION),
            "input_fusion_schema_version": np.int32(
                int(fused["fusion_schema_version"])
            ),
            "smplx_body_pose": refined_body,
            "arm_refined": np.bool_(True),
            "arm_refinement_method": np.asarray(
                "wilor_2d_wrist_palm_clip_ik"
            ),
            "arm_refined_body_joint_indices": np.asarray(
                REFINED_BODY_JOINT_INDICES, dtype=np.int32
            ),
            "arm_refined_body_joint_names": np.asarray(
                REFINED_BODY_JOINT_NAMES
            ),
            "arm_refinement_pose_delta": delta,
            "arm_refinement_iterations": np.int32(args.iterations),
            "arm_refinement_learning_rate": np.float32(args.learning_rate),
            "arm_target_cutoff_hz": np.float32(args.target_cutoff_hz),
            "arm_wrist_weight": np.float32(args.wrist_weight),
            "arm_palm_weight": np.float32(args.palm_weight),
            "arm_pose_prior_weight": np.float32(args.pose_prior_weight),
            "arm_velocity_weight": np.float32(args.velocity_weight),
            "arm_acceleration_weight": np.float32(
                args.acceleration_weight
            ),
            "arm_interpolated_target_weight": np.float32(
                args.interpolated_weight
            ),
        }
    )
    for key, value in metrics.items():
        payload[key] = np.float32(value)

    for key in (
        "smplx_lhand_pose",
        "smplx_rhand_pose",
        "cam_trans",
        "smplx_shape",
        "smplx_root_pose",
        "smplx_jaw_pose",
        "smplx_expr",
    ):
        if not np.array_equal(payload[key], fused[key]):
            raise AssertionError(f"{clip_id}: protected field changed: {key}")
    save_npz_atomic(output_path, payload)
    return metrics


def main() -> int:
    args = parse_args()
    try:
        clip_paths = discover_clip_paths(args.fused_root, args.clip_ids)
        if args.max_clips is not None:
            clip_paths = clip_paths[: args.max_clips]
        if not clip_paths:
            raise RuntimeError(f"No NPZ files found under {args.fused_root}")
        if not args.wilor_root.is_dir():
            raise FileNotFoundError(f"WiLoR root does not exist: {args.wilor_root}")
        if args.output_root.resolve() in {
            args.fused_root.resolve(),
            args.wilor_root.resolve(),
        }:
            raise ValueError("Output root must differ from both input roots")

        validated = []
        validate_bar = tqdm(
            clip_paths,
            desc="validate",
            unit="clip",
            dynamic_ncols=True,
        )
        for fused_path in validate_bar:
            wilor_path = args.wilor_root / fused_path.name
            fused = load_npz(fused_path)
            wilor = load_npz(wilor_path)
            clip_id, num_frames, fps = validate_inputs(
                fused, wilor, fused_path, wilor_path
            )
            validated.append((fused_path, wilor_path))
            validate_bar.set_postfix_str(f"{num_frames}f @ {fps:g}fps")
        validate_bar.close()
        if args.dry_run:
            print(f"[dry-run] PASS: validated {len(validated)} clip(s)")
            return 0

        device = resolve_device(args.device)
        print(f"[device] {device}", flush=True)
        smplx_model = make_smplx_model(args.smplx_model_root, device)
        mano_model = make_mano_model(args.mano_model_path, device)
        processed = 0
        skipped = 0
        passthrough = 0
        passthrough_clips: list[str] = []
        clip_bar = tqdm(
            validated,
            desc="refine",
            unit="clip",
            dynamic_ncols=True,
        )
        for fused_path, wilor_path in clip_bar:
            clip_bar.set_postfix(
                done=processed, pass_=passthrough, skip=skipped, refresh=False
            )
            output_path = args.output_root / fused_path.name
            if output_path.is_file() and not args.overwrite:
                skipped += 1
                continue
            metrics = refine_clip(
                fused_path,
                wilor_path,
                output_path,
                smplx_model,
                mano_model,
                device,
                args,
            )
            if metrics is None:
                passthrough += 1
                passthrough_clips.append(fused_path.name)
                continue
            processed += 1
            tqdm.write(
                f"[done] {fused_path.name}  "
                f"wrist median {metrics['wrist_error_before_median_px']:.2f} -> "
                f"{metrics['wrist_error_after_median_px']:.2f} px  |  "
                f"p95 {metrics['wrist_error_before_p95_px']:.2f} -> "
                f"{metrics['wrist_error_after_p95_px']:.2f} px  |  "
                f"delta RMS/max "
                f"{metrics['arm_delta_rms_degrees']:.2f}/"
                f"{metrics['arm_delta_max_degrees']:.2f} deg"
            )
        clip_bar.set_postfix(done=processed, pass_=passthrough, skip=skipped)
        clip_bar.close()
        print(
            f"Arm refinement complete: processed={processed}, "
            f"passthrough={passthrough}, skipped={skipped}"
        )
        if passthrough_clips:
            print("Clips written unrefined (no valid WiLoR hand targets):")
            for name in passthrough_clips:
                print(f"  {name}")
        return 0
    except (
        FileNotFoundError,
        KeyError,
        PermissionError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
