#!/usr/bin/env python3
"""Refine fused SMPL-X arms with OpenASL RTMPose supervision.

This is intentionally separate from ``refine_aios_arms_with_wilor.py``.
WiLoR wrist/palm positions and their interpolated masks are never used as
optimization targets.  The input WiLoR finger rotations are copied unchanged,
while RTMPose shoulder, elbow, and wrist keypoints guide the SMPL-X upper body.

RTMPose and fused clips can have different frame rates.  Both streams are put
on the clip time line encoded in the OpenASL clip id, followed by an optional
small global-offset search.  Low-confidence observations and sequence edges
are masked rather than nearest-filled.  A differentiable 3-D torso-volume
penalty discourages upper-arm and forearm samples from passing through the
body while still allowing valid signs in front of the chest.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import smplx
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from fuse_shared_aios_wilor import save_npz_atomic


DEFAULT_FUSED_ROOT = Path(
    "shared_samples/aios_smoothed_wilor_hands_fused_final"
)
DEFAULT_RTMPOSE_ROOT = Path(
    "/home/student/hwu/Workplace/Uni-Sign/data/OpenASL/pose-rtmpose-192"
)
DEFAULT_OUTPUT_ROOT = Path("shared_samples/aios_rtmpose_arm_refined")
DEFAULT_SMPLX_MODEL_ROOT = Path(
    "/home/student/hwu/Workplace/SOKE/prepare/deps/smpl_models"
)

REFINEMENT_SCHEMA_VERSION = 1
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
PRIOR_JOINT_WEIGHTS = (8.0, 4.0, 4.0, 1.5, 1.5, 1.0, 1.0, 0.75, 0.75)
MAX_DELTA_RADIANS = (0.20, 0.30, 0.30, 0.60, 0.60, 0.75, 0.75, 0.60, 0.60)

# [side, shoulder/elbow/wrist].  These are COCO-WholeBody / RTMPose indices.
RTMPOSE_ARM_JOINT_INDICES = np.asarray(((5, 7, 9), (6, 8, 10)), dtype=np.int64)
RTMPOSE_ARM_JOINT_NAMES = (
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)

# SMPL-X output-joint indices in the same [side, shoulder/elbow/wrist] layout.
SMPLX_ARM_JOINT_INDICES = ((16, 18, 20), (17, 19, 21))
SMPLX_HIP_JOINT_INDICES = (1, 2)

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
    "aios_person_id",
    "camera_fixed_clipwise",
}

PROTECTED_FIELDS = (
    "smplx_lhand_pose",
    "smplx_rhand_pose",
    "cam_trans",
    "smplx_shape",
    "smplx_root_pose",
    "smplx_jaw_pose",
    "smplx_expr",
)

CLIP_TIME_RE = re.compile(
    r"-(?P<start>\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    r"-(?P<end>\d{2}:\d{2}:\d{2}(?:\.\d+)?)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-root", type=Path, default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--rtmpose-root", type=Path, default=DEFAULT_RTMPOSE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--smplx-model-root", type=Path, default=DEFAULT_SMPLX_MODEL_ROOT
    )
    parser.add_argument(
        "--clip-id",
        dest="clip_ids",
        action="append",
        help="Process only this fused clip; may be repeated.",
    )
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--aios-focal-length", type=float, default=5000.0)

    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument(
        "--single-neighbor-weight",
        type=float,
        default=0.5,
        help=(
            "Weight for a target supported by only one adjacent RTMPose frame; "
            "this never fills sequence edges or longer missing intervals."
        ),
    )
    parser.add_argument("--shoulder-weight", type=float, default=1.0)
    parser.add_argument("--elbow-weight", type=float, default=2.0)
    parser.add_argument("--wrist-weight", type=float, default=1.0)
    parser.add_argument(
        "--huber-delta-px",
        type=float,
        default=8.0,
        help="Pixel transition point for robust RTMPose reprojection loss.",
    )

    parser.add_argument("--pose-prior-weight", type=float, default=0.08)
    parser.add_argument("--velocity-weight", type=float, default=0.20)
    parser.add_argument("--acceleration-weight", type=float, default=1.0)
    parser.add_argument("--collision-weight", type=float, default=0.05)
    parser.add_argument(
        "--collision-margin",
        type=float,
        default=0.02,
        help="Dimensionless margin outside the torso ellipsoid.",
    )
    parser.add_argument("--torso-width-scale", type=float, default=0.42)
    parser.add_argument("--torso-height-scale", type=float, default=0.52)
    parser.add_argument("--torso-depth-scale", type=float, default=0.22)

    parser.add_argument(
        "--max-time-offset-frames",
        type=int,
        default=4,
        help="Maximum integer 24-fps-frame offset considered during alignment.",
    )
    parser.add_argument(
        "--time-offset-penalty-px",
        type=float,
        default=0.25,
        help="Per-frame penalty used to prefer a zero temporal offset.",
    )
    parser.add_argument(
        "--time-offset-min-improvement-px",
        type=float,
        default=0.5,
        help="Keep offset zero unless another offset improves by this many pixels.",
    )
    parser.add_argument(
        "--max-duration-error-seconds",
        type=float,
        default=0.25,
        help="Reject clips whose fused duration disagrees with the clip id.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all contracts and time axes without loading SMPL-X.",
    )
    args = parser.parse_args()

    positive = (
        "iterations",
        "learning_rate",
        "batch_size",
        "aios_focal_length",
        "huber_delta_px",
        "torso_width_scale",
        "torso_height_scale",
        "torso_depth_scale",
        "max_duration_error_seconds",
        "log_every",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    nonnegative = (
        "shoulder_weight",
        "elbow_weight",
        "wrist_weight",
        "pose_prior_weight",
        "velocity_weight",
        "acceleration_weight",
        "collision_weight",
        "collision_margin",
        "max_time_offset_frames",
        "time_offset_penalty_px",
        "time_offset_min_improvement_px",
    )
    for name in nonnegative:
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if not 0 <= args.confidence_threshold < 1:
        parser.error("--confidence-threshold must be in [0, 1)")
    if not 0 <= args.single_neighbor_weight <= 1:
        parser.error("--single-neighbor-weight must be in [0, 1]")
    if args.max_clips is not None and args.max_clips <= 0:
        parser.error("--max-clips must be positive")
    if args.shoulder_weight + args.elbow_weight + args.wrist_weight <= 0:
        parser.error("at least one RTMPose joint weight must be positive")
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


def discover_clip_paths(
    root: Path, requested: Sequence[str] | None
) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Fused root is not a directory: {root}")
    if requested:
        if len(requested) != len(set(requested)):
            raise ValueError("--clip-id values must not contain duplicates")
        paths = [
            root / f"{value[:-4] if value.endswith('.npz') else value}.npz"
            for value in requested
        ]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Requested fused clips are missing: {missing}")
        return paths
    return sorted(root.glob("*.npz"))


def validate_fused(
    fused: dict[str, np.ndarray], path: Path
) -> tuple[str, int, float]:
    missing = sorted(REQUIRED_FUSED_FIELDS - fused.keys())
    if missing:
        raise KeyError(f"{path}: missing fields {missing}")
    clip_id = str(fused["clip_id"])
    if clip_id != path.stem:
        raise ValueError(f"{path}: clip_id {clip_id!r} disagrees with filename")
    if int(fused["fusion_schema_version"]) < 3:
        raise ValueError(f"{clip_id}: requires fusion schema v3+")
    if int(fused["aios_person_id"]) != 0:
        raise ValueError(f"{clip_id}: fused input does not use AIOS person 0")
    if not bool(fused["camera_fixed_clipwise"]):
        raise ValueError(f"{clip_id}: camera must be fixed clipwise")

    num_frames = int(fused["num_frames"])
    if num_frames <= 0:
        raise ValueError(f"{clip_id}: num_frames must be positive")
    expected_shapes = {
        "frame_names": (num_frames,),
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
    for key, expected in expected_shapes.items():
        if fused[key].shape != expected:
            raise ValueError(
                f"{path}: {key} shape {fused[key].shape}, expected {expected}"
            )
    if np.any(np.asarray(fused["img_shape"]) <= 0):
        raise ValueError(f"{clip_id}: img_shape must be positive")
    for key, value in fused.items():
        array = np.asarray(value)
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"{path}: {key} contains NaN or Inf")

    fps = float(fused["source_fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{clip_id}: invalid source_fps={fps}")
    return clip_id, num_frames, fps


def load_rtmpose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"RTMPose pickle does not exist: {path}")
    with path.open("rb") as handle:
        pose = pickle.load(handle)
    if not isinstance(pose, dict):
        raise ValueError(f"{path}: expected a dict, got {type(pose).__name__}")
    missing = sorted({"keypoints", "scores"} - pose.keys())
    if missing:
        raise KeyError(f"{path}: missing fields {missing}")
    keypoints = np.asarray(pose["keypoints"], dtype=np.float32)
    scores = np.asarray(pose["scores"], dtype=np.float32)
    if keypoints.ndim != 4 or keypoints.shape[1:] != (1, 133, 2):
        raise ValueError(
            f"{path}: keypoints shape {keypoints.shape}, expected [T,1,133,2]"
        )
    if scores.shape != (len(keypoints), 1, 133):
        raise ValueError(
            f"{path}: scores shape {scores.shape}, "
            f"expected {(len(keypoints), 1, 133)}"
        )
    if len(keypoints) == 0:
        raise ValueError(f"{path}: RTMPose sequence is empty")
    if not np.isfinite(keypoints).all() or not np.isfinite(scores).all():
        raise ValueError(f"{path}: keypoints or scores contain NaN/Inf")
    return keypoints[:, 0], scores[:, 0]


def timestamp_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)


def clip_duration_seconds(clip_id: str) -> float:
    match = CLIP_TIME_RE.search(clip_id)
    if match is None:
        raise ValueError(f"{clip_id}: clip id has no OpenASL start/end timestamps")
    duration = timestamp_seconds(match.group("end")) - timestamp_seconds(
        match.group("start")
    )
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"{clip_id}: invalid timestamp duration {duration}")
    return duration


def validate_time_axis(
    clip_id: str,
    num_frames: int,
    fps: float,
    rtmpose_frames: int,
    max_duration_error: float,
) -> tuple[float, float, float]:
    duration = clip_duration_seconds(clip_id)
    fused_duration = num_frames / fps
    duration_error = abs(fused_duration - duration)
    if duration_error > max_duration_error:
        raise ValueError(
            f"{clip_id}: fused duration {fused_duration:.4f}s disagrees with "
            f"clip-id duration {duration:.4f}s by {duration_error:.4f}s"
        )
    inferred_rtmpose_fps = rtmpose_frames / duration
    return duration, inferred_rtmpose_fps, duration_error


def align_rtmpose_targets(
    keypoints: np.ndarray,
    scores: np.ndarray,
    num_frames: int,
    fused_fps: float,
    duration: float,
    offset_frames: int,
    confidence_threshold: float,
    single_neighbor_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample RTMPose without filling invalid gaps or sequence edges."""
    source_frames = len(keypoints)
    source_step = duration / source_frames
    query_time = (
        np.arange(num_frames, dtype=np.float64) + float(offset_frames)
    ) / fused_fps
    source_position = query_time / source_step
    in_range = (source_position >= 0.0) & (
        source_position <= source_frames - 1 + 1e-7
    )
    clipped = np.clip(source_position, 0.0, source_frames - 1)
    lower = np.floor(clipped).astype(np.int64)
    upper = np.ceil(clipped).astype(np.int64)
    alpha = (clipped - lower).astype(np.float32)
    joint_indices = RTMPOSE_ARM_JOINT_INDICES.reshape(-1)
    lower_xy = keypoints[lower][:, joint_indices]
    upper_xy = keypoints[upper][:, joint_indices]
    lower_score = scores[lower][:, joint_indices]
    upper_score = scores[upper][:, joint_indices]

    interpolated_targets = (
        lower_xy * (1.0 - alpha[:, None, None])
        + upper_xy * alpha[:, None, None]
    )
    interpolated_scores = (
        lower_score * (1.0 - alpha[:, None]) + upper_score * alpha[:, None]
    )
    lower_valid = lower_score > confidence_threshold
    upper_valid = upper_score > confidence_threshold
    both_valid = lower_valid & upper_valid
    only_lower = lower_valid & ~upper_valid
    only_upper = upper_valid & ~lower_valid
    targets = np.where(
        only_lower[..., None],
        lower_xy,
        np.where(only_upper[..., None], upper_xy, interpolated_targets),
    )
    target_scores = np.where(
        only_lower,
        lower_score,
        np.where(only_upper, upper_score, interpolated_scores),
    )
    valid = (both_valid | only_lower | only_upper) & in_range[:, None]
    support_weight = np.where(both_valid, 1.0, single_neighbor_weight)
    weights = np.where(
        valid,
        np.clip(target_scores, 0.0, 1.0) * support_weight,
        0.0,
    )
    targets = np.where(valid[..., None], targets, 0.0)
    return (
        targets.reshape(num_frames, 2, 3, 2).astype(np.float32),
        weights.reshape(num_frames, 2, 3).astype(np.float32),
    )


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


def tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array)).to(device=device, dtype=torch.float32)


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


def slice_values(
    values: dict[str, torch.Tensor], start: int, stop: int
) -> dict[str, torch.Tensor]:
    return {key: value[start:stop] for key, value in values.items()}


def smplx_arm_and_hip_joints(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    body_pose: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = len(body_pose)
    zeros = torch.zeros((count, 3), device=body_pose.device, dtype=body_pose.dtype)
    output = model(
        betas=values["shape"],
        global_orient=values["root"],
        body_pose=body_pose.reshape(count, -1),
        left_hand_pose=values["left_hand"],
        right_hand_pose=values["right_hand"],
        jaw_pose=values["jaw"],
        leye_pose=zeros,
        reye_pose=zeros,
        expression=values["expression"],
        return_verts=False,
    )
    arm_indices = torch.tensor(
        SMPLX_ARM_JOINT_INDICES, device=body_pose.device, dtype=torch.long
    )
    hip_indices = torch.tensor(
        SMPLX_HIP_JOINT_INDICES, device=body_pose.device, dtype=torch.long
    )
    return output.joints[:, arm_indices], output.joints[:, hip_indices]


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


def normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(
        vector, dim=-1, keepdim=True
    ).clamp_min(1e-6)


def torso_penetration(
    arm_joints: torch.Tensor,
    hip_joints: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dimensionless collision loss and penetration depth in metres."""
    shoulder = arm_joints[:, :, 0]
    elbow = arm_joints[:, :, 1]
    wrist = arm_joints[:, :, 2]
    shoulder_mid = shoulder.mean(dim=1)
    hip_mid = hip_joints.mean(dim=1)

    x_axis = normalize(shoulder[:, 1] - shoulder[:, 0])
    y_hint = normalize(shoulder_mid - hip_mid)
    z_axis = normalize(torch.cross(x_axis, y_hint, dim=-1))
    y_axis = normalize(torch.cross(z_axis, x_axis, dim=-1))
    center = 0.5 * (shoulder_mid + hip_mid)

    shoulder_width = torch.linalg.vector_norm(
        shoulder[:, 1] - shoulder[:, 0], dim=-1
    ).clamp_min(1e-4)
    torso_height = torch.linalg.vector_norm(
        shoulder_mid - hip_mid, dim=-1
    ).clamp_min(1e-4)
    radii = torch.stack(
        (
            args.torso_width_scale * shoulder_width,
            args.torso_height_scale * torso_height,
            args.torso_depth_scale * shoulder_width,
        ),
        dim=-1,
    ).clamp_min(1e-4)

    upper_alpha = torch.tensor(
        (0.35, 0.55, 0.75, 1.0), device=arm_joints.device, dtype=arm_joints.dtype
    )
    forearm_alpha = torch.tensor(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        device=arm_joints.device,
        dtype=arm_joints.dtype,
    )
    upper = shoulder[:, :, None] * (1.0 - upper_alpha[None, None, :, None])
    upper = upper + elbow[:, :, None] * upper_alpha[None, None, :, None]
    forearm = elbow[:, :, None] * (1.0 - forearm_alpha[None, None, :, None])
    forearm = forearm + wrist[:, :, None] * forearm_alpha[None, None, :, None]
    samples = torch.cat((upper, forearm), dim=2)

    relative = samples - center[:, None, None]
    local = torch.stack(
        (
            (relative * x_axis[:, None, None]).sum(dim=-1),
            (relative * y_axis[:, None, None]).sum(dim=-1),
            (relative * z_axis[:, None, None]).sum(dim=-1),
        ),
        dim=-1,
    )
    ellipsoid_radius = torch.sqrt(
        ((local / radii[:, None, None]) ** 2).sum(dim=-1).clamp_min(1e-8)
    )
    margin_penetration = F.relu(1.0 + args.collision_margin - ellipsoid_radius)
    collision_loss = margin_penetration.square().mean()
    penetration_metres = F.relu(1.0 - ellipsoid_radius) * radii.min(
        dim=-1
    ).values[:, None, None]
    return collision_loss, penetration_metres


def body_with_delta(
    base_body: torch.Tensor, delta: torch.Tensor
) -> torch.Tensor:
    result = base_body.clone()
    indices = torch.tensor(
        REFINED_BODY_JOINT_INDICES, device=base_body.device, dtype=torch.long
    )
    result[:, indices] = result[:, indices] + delta
    return result


def predict_chunk(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    delta: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    arm_joints, hip_joints = smplx_arm_and_hip_joints(
        model, values, body_with_delta(values["body"], delta)
    )
    projected = project_smplx_joints(
        arm_joints,
        values["camera"],
        values["image_shape"],
        args.aios_focal_length,
    )
    collision_loss, penetration = torso_penetration(arm_joints, hip_joints, args)
    return projected, collision_loss, penetration


def predict_sequence(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    delta: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected_parts = []
    penetration_parts = []
    for start in range(0, len(delta), args.batch_size):
        stop = min(start + args.batch_size, len(delta))
        projected, _, penetration = predict_chunk(
            model,
            slice_values(values, start, stop),
            delta[start:stop],
            args,
        )
        projected_parts.append(projected)
        penetration_parts.append(penetration)
    return torch.cat(projected_parts), torch.cat(penetration_parts)


def joint_loss_weights(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        (args.shoulder_weight, args.elbow_weight, args.wrist_weight),
        device=device,
        dtype=torch.float32,
    )[None, None]


def reprojection_loss(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    image_shape: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    pixel_scale = torch.stack((image_shape[:, 1], image_shape[:, 0]), dim=-1)
    absolute_error_px = torch.abs(predicted - targets) * pixel_scale[:, None, None]
    delta = args.huber_delta_px
    smooth_l1_px = torch.where(
        absolute_error_px < delta,
        0.5 * absolute_error_px.square() / delta,
        absolute_error_px - 0.5 * delta,
    )
    image_diagonal = torch.linalg.vector_norm(pixel_scale, dim=-1).clamp_min(1.0)
    per_joint = smooth_l1_px.mean(dim=-1) / image_diagonal[:, None, None]
    combined_weights = weights * joint_loss_weights(args, predicted.device)
    return (per_joint * combined_weights).sum() / combined_weights.sum().clamp_min(
        1e-8
    )


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


def offset_score(
    predicted: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    image_shape: np.ndarray,
    offset: int,
    args: argparse.Namespace,
) -> float:
    if not np.any(weights > 0):
        return float("inf")
    pixel_scale = np.stack((image_shape[:, 1], image_shape[:, 0]), axis=-1)
    error = np.linalg.norm(
        (predicted - targets) * pixel_scale[:, None, None], axis=-1
    )
    robust_error = np.minimum(error, 200.0)
    joint_weights = np.asarray(
        (args.shoulder_weight, args.elbow_weight, args.wrist_weight),
        dtype=np.float32,
    )[None, None]
    combined = weights * joint_weights
    score = float((robust_error * combined).sum() / max(combined.sum(), 1e-8))
    return score + abs(offset) * args.time_offset_penalty_px


def select_time_offset(
    predicted: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    num_frames: int,
    fused_fps: float,
    duration: float,
    image_shape: np.ndarray,
    args: argparse.Namespace,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    candidates: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}
    for offset in range(
        -args.max_time_offset_frames, args.max_time_offset_frames + 1
    ):
        targets, weights = align_rtmpose_targets(
            keypoints,
            scores,
            num_frames,
            fused_fps,
            duration,
            offset,
            args.confidence_threshold,
            args.single_neighbor_weight,
        )
        candidates[offset] = (
            offset_score(predicted, targets, weights, image_shape, offset, args),
            targets,
            weights,
        )
    if not any(np.isfinite(value[0]) for value in candidates.values()):
        score, targets, weights = candidates[0]
        return 0, score, targets, weights
    best_offset = min(candidates, key=lambda value: candidates[value][0])
    zero_score = candidates[0][0]
    best_score = candidates[best_offset][0]
    if (
        best_offset != 0
        and np.isfinite(zero_score)
        and zero_score - best_score < args.time_offset_min_improvement_px
    ):
        best_offset = 0
    score, targets, weights = candidates[best_offset]
    return best_offset, score, targets, weights


def optimize_arm_pose(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    targets: np.ndarray,
    weights: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, dict[str, float]]:
    device = values["body"].device
    target_tensor = tensor(targets, device)
    weight_tensor = tensor(weights, device)
    num_frames = len(target_tensor)
    delta = torch.nn.Parameter(
        torch.zeros(
            (num_frames, len(REFINED_BODY_JOINT_INDICES), 3), device=device
        )
    )
    optimizer = torch.optim.Adam([delta], lr=args.learning_rate)

    with torch.no_grad():
        before, penetration_before = predict_sequence(model, values, delta, args)

    joint_weights = joint_loss_weights(args, device)
    total_target_weight = float((weight_tensor * joint_weights).sum().item())
    for iteration in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        regularization, regularization_parts = regularization_loss(delta, args)
        regularization.backward()
        data_value = 0.0
        collision_value = 0.0

        for start in range(0, num_frames, args.batch_size):
            stop = min(start + args.batch_size, num_frames)
            chunk_values = slice_values(values, start, stop)
            predicted, collision_loss, _ = predict_chunk(
                model, chunk_values, delta[start:stop], args
            )
            chunk_weights = weight_tensor[start:stop]
            chunk_target_weight = float(
                (chunk_weights * joint_weights).sum().detach().item()
            )
            data_fraction = chunk_target_weight / max(total_target_weight, 1e-8)
            data_loss = reprojection_loss(
                predicted,
                target_tensor[start:stop],
                chunk_weights,
                chunk_values["image_shape"],
                args,
            )
            collision_fraction = (stop - start) / num_frames
            chunk_loss = (
                data_fraction * data_loss
                + collision_fraction * args.collision_weight * collision_loss
            )
            chunk_loss.backward()
            data_value += data_fraction * float(data_loss.detach())
            collision_value += collision_fraction * float(collision_loss.detach())

        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError("arm optimization produced non-finite gradients")
        optimizer.step()
        with torch.no_grad():
            clamp_delta(delta)

        if iteration == 1 or iteration % args.log_every == 0:
            total = (
                data_value
                + args.collision_weight * collision_value
                + float(regularization.detach())
            )
            tqdm.write(
                f"  iter={iteration:04d} loss={total:.6f} "
                f"rtmpose={data_value:.6f} collision={collision_value:.6f} "
                f"prior={float(regularization_parts['prior']):.6f} "
                f"vel={float(regularization_parts['velocity']):.6f} "
                f"acc={float(regularization_parts['acceleration']):.6f}"
            )

    with torch.no_grad():
        after, penetration_after = predict_sequence(model, values, delta, args)
    refined_body = body_with_delta(values["body"], delta)
    metrics = refinement_metrics(
        before,
        after,
        target_tensor,
        weight_tensor,
        values["image_shape"],
        penetration_before,
        penetration_after,
        delta,
    )
    return (
        refined_body.reshape(num_frames, 63).detach().cpu().numpy().astype(np.float32),
        delta.detach().cpu().numpy().astype(np.float32),
        before,
        after,
        metrics,
    )


def error_summary(values: torch.Tensor) -> tuple[float, float]:
    if values.numel() == 0:
        return float("nan"), float("nan")
    return float(values.median()), float(torch.quantile(values, 0.95))


def metric_field_names() -> list[str]:
    names = [
        "rtmpose_arm_error_before_median_px",
        "rtmpose_arm_error_before_p95_px",
        "rtmpose_arm_error_after_median_px",
        "rtmpose_arm_error_after_p95_px",
    ]
    for side in ("left", "right"):
        for joint in ("shoulder", "elbow", "wrist"):
            prefix = f"rtmpose_{side}_{joint}_error"
            names.extend(
                (
                    f"{prefix}_before_median_px",
                    f"{prefix}_before_p95_px",
                    f"{prefix}_after_median_px",
                    f"{prefix}_after_p95_px",
                )
            )
    names.extend(
        (
            "torso_collision_frames_before",
            "torso_collision_frames_after",
            "torso_max_penetration_before_mm",
            "torso_max_penetration_after_mm",
            "arm_delta_rms_degrees",
            "arm_delta_max_degrees",
        )
    )
    return names


def refinement_metrics(
    before: torch.Tensor,
    after: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    image_shape: torch.Tensor,
    penetration_before: torch.Tensor,
    penetration_after: torch.Tensor,
    delta: torch.Tensor,
) -> dict[str, float]:
    pixel_scale = torch.stack((image_shape[:, 1], image_shape[:, 0]), dim=-1)
    before_error = torch.linalg.vector_norm(
        (before - targets) * pixel_scale[:, None, None], dim=-1
    )
    after_error = torch.linalg.vector_norm(
        (after - targets) * pixel_scale[:, None, None], dim=-1
    )
    valid = weights > 0
    metrics: dict[str, float] = {}
    overall_before = error_summary(before_error[valid])
    overall_after = error_summary(after_error[valid])
    metrics.update(
        {
            "rtmpose_arm_error_before_median_px": overall_before[0],
            "rtmpose_arm_error_before_p95_px": overall_before[1],
            "rtmpose_arm_error_after_median_px": overall_after[0],
            "rtmpose_arm_error_after_p95_px": overall_after[1],
        }
    )
    sides = ("left", "right")
    joints = ("shoulder", "elbow", "wrist")
    for side_index, side in enumerate(sides):
        for joint_index, joint in enumerate(joints):
            joint_valid = valid[:, side_index, joint_index]
            before_values = before_error[:, side_index, joint_index][joint_valid]
            after_values = after_error[:, side_index, joint_index][joint_valid]
            before_summary = error_summary(before_values)
            after_summary = error_summary(after_values)
            prefix = f"rtmpose_{side}_{joint}_error"
            metrics[f"{prefix}_before_median_px"] = before_summary[0]
            metrics[f"{prefix}_before_p95_px"] = before_summary[1]
            metrics[f"{prefix}_after_median_px"] = after_summary[0]
            metrics[f"{prefix}_after_p95_px"] = after_summary[1]

    # Older PyTorch does not accept a tuple of dims in .any(); flatten instead.
    before_frame_collision = (penetration_before > 0).flatten(1).any(dim=1)
    after_frame_collision = (penetration_after > 0).flatten(1).any(dim=1)
    metrics.update(
        {
            "torso_collision_frames_before": float(before_frame_collision.sum()),
            "torso_collision_frames_after": float(after_frame_collision.sum()),
            "torso_max_penetration_before_mm": float(
                penetration_before.max() * 1000.0
            ),
            "torso_max_penetration_after_mm": float(
                penetration_after.max() * 1000.0
            ),
        }
    )
    delta_degrees = torch.linalg.vector_norm(delta, dim=-1) * (180.0 / np.pi)
    metrics["arm_delta_rms_degrees"] = float(
        torch.sqrt(delta_degrees.square().mean())
    )
    metrics["arm_delta_max_degrees"] = float(delta_degrees.max())
    return metrics


def assert_protected_fields(
    input_fused: dict[str, np.ndarray], payload: dict[str, np.ndarray]
) -> None:
    clip_id = str(input_fused["clip_id"])
    for key in PROTECTED_FIELDS:
        if not np.array_equal(payload[key], input_fused[key]):
            raise AssertionError(f"{clip_id}: protected field changed: {key}")
    before = input_fused["smplx_body_pose"].reshape(-1, BODY_JOINT_COUNT, 3)
    after = payload["smplx_body_pose"].reshape(-1, BODY_JOINT_COUNT, 3)
    protected_indices = sorted(
        set(range(BODY_JOINT_COUNT)) - set(REFINED_BODY_JOINT_INDICES)
    )
    if not np.array_equal(before[:, protected_indices], after[:, protected_indices]):
        raise AssertionError(f"{clip_id}: a non-refined body joint changed")


def base_output_metadata(
    fused: dict[str, np.ndarray],
    raw_pose_frames: int,
    duration: float,
    inferred_pose_fps: float,
    offset: int,
    alignment_score: float,
    target_weights: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    return {
        # The existing renderer treats fusion schema v4 as the common
        # arm-refined envelope. Keep the input version separately below.
        "fusion_schema_version": np.int32(4),
        "rtmpose_arm_refinement_schema_version": np.int32(
            REFINEMENT_SCHEMA_VERSION
        ),
        "input_fusion_schema_version": np.int32(
            int(fused["fusion_schema_version"])
        ),
        "arm_refinement_method": np.asarray(
            "rtmpose_2d_arm_torso_collision_clip_ik"
        ),
        "arm_refined_body_joint_indices": np.asarray(
            REFINED_BODY_JOINT_INDICES, dtype=np.int32
        ),
        "arm_refined_body_joint_names": np.asarray(REFINED_BODY_JOINT_NAMES),
        "rtmpose_arm_joint_indices": RTMPOSE_ARM_JOINT_INDICES.astype(np.int32),
        "rtmpose_arm_joint_names": np.asarray(RTMPOSE_ARM_JOINT_NAMES),
        "rtmpose_coordinates": np.asarray("normalized_image_xy"),
        "rtmpose_time_alignment": np.asarray(
            "clip_duration_linear_small_global_offset"
        ),
        "rtmpose_edge_fill": np.bool_(False),
        "rtmpose_raw_num_frames": np.int32(raw_pose_frames),
        "rtmpose_aligned_num_frames": np.int32(int(fused["num_frames"])),
        "rtmpose_clip_duration_seconds": np.float32(duration),
        "rtmpose_inferred_fps": np.float32(inferred_pose_fps),
        "rtmpose_selected_time_offset_frames": np.int32(offset),
        "rtmpose_alignment_score_px": np.float32(alignment_score),
        "rtmpose_valid_target_counts": (target_weights > 0).sum(axis=0).astype(
            np.int32
        ),
        "rtmpose_confidence_threshold": np.float32(args.confidence_threshold),
        "rtmpose_single_neighbor_weight": np.float32(
            args.single_neighbor_weight
        ),
        "rtmpose_shoulder_weight": np.float32(args.shoulder_weight),
        "rtmpose_elbow_weight": np.float32(args.elbow_weight),
        "rtmpose_wrist_weight": np.float32(args.wrist_weight),
        "rtmpose_huber_delta_px": np.float32(args.huber_delta_px),
        "arm_refinement_iterations": np.int32(args.iterations),
        "arm_refinement_learning_rate": np.float32(args.learning_rate),
        "arm_pose_prior_weight": np.float32(args.pose_prior_weight),
        "arm_velocity_weight": np.float32(args.velocity_weight),
        "arm_acceleration_weight": np.float32(args.acceleration_weight),
        "arm_torso_collision_weight": np.float32(args.collision_weight),
        "arm_torso_collision_margin": np.float32(args.collision_margin),
        "arm_torso_collision_method": np.asarray(
            "smplx_shoulder_hip_local_ellipsoid"
        ),
    }


def write_passthrough(
    fused: dict[str, np.ndarray],
    output_path: Path,
    raw_pose_frames: int,
    duration: float,
    inferred_pose_fps: float,
    offset: int,
    alignment_score: float,
    target_weights: np.ndarray,
    args: argparse.Namespace,
) -> None:
    payload = dict(fused)
    payload.update(
        base_output_metadata(
            fused,
            raw_pose_frames,
            duration,
            inferred_pose_fps,
            offset,
            alignment_score,
            target_weights,
            args,
        )
    )
    payload.update(
        {
            "arm_refined": np.bool_(False),
            "arm_refinement_skipped_reason": np.asarray(
                "no_valid_rtmpose_arm_targets"
            ),
            "arm_refinement_pose_delta": np.zeros(
                (
                    int(fused["num_frames"]),
                    len(REFINED_BODY_JOINT_INDICES),
                    3,
                ),
                dtype=np.float32,
            ),
            "arm_refinement_iterations": np.int32(0),
        }
    )
    for name in metric_field_names():
        payload[name] = np.float32(np.nan)
    assert_protected_fields(fused, payload)
    save_npz_atomic(output_path, payload)


def refine_clip(
    fused_path: Path,
    pose_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float] | None:
    fused = load_npz(fused_path)
    clip_id, num_frames, fps = validate_fused(fused, fused_path)
    keypoints, scores = load_rtmpose(pose_path)
    duration, pose_fps, _ = validate_time_axis(
        clip_id,
        num_frames,
        fps,
        len(keypoints),
        args.max_duration_error_seconds,
    )
    values = fused_tensors(fused, device)
    zero_delta = torch.zeros(
        (num_frames, len(REFINED_BODY_JOINT_INDICES), 3), device=device
    )
    with torch.no_grad():
        initial_prediction, _ = predict_sequence(model, values, zero_delta, args)
    offset, alignment_score, targets, weights = select_time_offset(
        initial_prediction.cpu().numpy(),
        keypoints,
        scores,
        num_frames,
        fps,
        duration,
        np.asarray(fused["img_shape"], dtype=np.float32),
        args,
    )

    valid_counts = (weights > 0).sum(axis=0)
    tqdm.write(
        f"[refine] {clip_id}: fused={num_frames}f pose={len(keypoints)}f "
        f"pose_fps~{pose_fps:.2f} offset={offset:+d} "
        f"valid={int(valid_counts.sum())}/{num_frames * 6}"
    )
    if not np.any(weights > 0):
        tqdm.write(f"[passthrough] {clip_id}: no valid RTMPose arm targets")
        write_passthrough(
            fused,
            output_path,
            len(keypoints),
            duration,
            pose_fps,
            offset,
            alignment_score,
            weights,
            args,
        )
        return None

    refined_body, delta, _, _, metrics = optimize_arm_pose(
        model, values, targets, weights, args
    )
    payload = dict(fused)
    payload.update(
        base_output_metadata(
            fused,
            len(keypoints),
            duration,
            pose_fps,
            offset,
            alignment_score,
            weights,
            args,
        )
    )
    payload.update(
        {
            "smplx_body_pose": refined_body,
            "arm_refined": np.bool_(True),
            "arm_refinement_pose_delta": delta,
        }
    )
    for key, value in metrics.items():
        payload[key] = np.float32(value)
    assert_protected_fields(fused, payload)
    save_npz_atomic(output_path, payload)
    return metrics


def validate_all(
    clip_paths: list[Path], args: argparse.Namespace
) -> list[tuple[Path, Path]]:
    validated: list[tuple[Path, Path]] = []
    fused_counts = []
    pose_counts = []
    duration_errors = []
    valid_counts = np.zeros((2, 3), dtype=np.int64)
    bar = tqdm(clip_paths, desc="validate", unit="clip", dynamic_ncols=True)
    for fused_path in bar:
        fused = load_npz(fused_path)
        clip_id, num_frames, fps = validate_fused(fused, fused_path)
        pose_path = args.rtmpose_root / f"{clip_id}.pkl"
        keypoints, scores = load_rtmpose(pose_path)
        duration, pose_fps, duration_error = validate_time_axis(
            clip_id,
            num_frames,
            fps,
            len(keypoints),
            args.max_duration_error_seconds,
        )
        del duration
        fused_counts.append(num_frames)
        pose_counts.append(len(keypoints))
        duration_errors.append(duration_error)
        source_arm_scores = scores[:, RTMPOSE_ARM_JOINT_INDICES.reshape(-1)]
        valid_counts += (source_arm_scores > args.confidence_threshold).sum(
            axis=0
        ).reshape(2, 3)
        validated.append((fused_path, pose_path))
        bar.set_postfix_str(
            f"fused={num_frames} pose={len(keypoints)} pose_fps~{pose_fps:.2f}"
        )
    bar.close()
    ratios = np.asarray(pose_counts, dtype=np.float64) / np.asarray(
        fused_counts, dtype=np.float64
    )
    print(
        f"[validate] clips={len(validated)} fused_frames={sum(fused_counts):,} "
        f"rtmpose_frames={sum(pose_counts):,}"
    )
    print(
        "[validate] pose/fused frame ratio "
        f"min/median/max={ratios.min():.3f}/{np.median(ratios):.3f}/{ratios.max():.3f}"
    )
    print(
        f"[validate] max clip-duration error={max(duration_errors):.4f}s; "
        f"raw valid target counts={valid_counts.tolist()}"
    )
    return validated


def main() -> int:
    args = parse_args()
    try:
        clip_paths = discover_clip_paths(args.fused_root, args.clip_ids)
        if args.max_clips is not None:
            clip_paths = clip_paths[: args.max_clips]
        if not clip_paths:
            raise RuntimeError(f"No NPZ files found under {args.fused_root}")
        if not args.rtmpose_root.is_dir():
            raise FileNotFoundError(
                f"RTMPose root is not a directory: {args.rtmpose_root}"
            )
        if args.output_root.resolve() in {
            args.fused_root.resolve(),
            args.rtmpose_root.resolve(),
        }:
            raise ValueError("Output root must differ from both input roots")

        validated = validate_all(clip_paths, args)
        if args.dry_run:
            print(f"[dry-run] PASS: validated {len(validated)} clip(s)")
            return 0

        device = resolve_device(args.device)
        print(f"[device] {device}", flush=True)
        model = make_smplx_model(args.smplx_model_root, device)
        processed = 0
        passthrough = 0
        skipped = 0
        bar = tqdm(validated, desc="refine", unit="clip", dynamic_ncols=True)
        for fused_path, pose_path in bar:
            output_path = args.output_root / fused_path.name
            if output_path.is_file() and not args.overwrite:
                skipped += 1
                continue
            metrics = refine_clip(
                fused_path, pose_path, output_path, model, device, args
            )
            if metrics is None:
                passthrough += 1
            else:
                processed += 1
                tqdm.write(
                    f"[done] {fused_path.name} RTMPose median "
                    f"{metrics['rtmpose_arm_error_before_median_px']:.2f} -> "
                    f"{metrics['rtmpose_arm_error_after_median_px']:.2f}px; "
                    f"collision frames "
                    f"{int(metrics['torso_collision_frames_before'])} -> "
                    f"{int(metrics['torso_collision_frames_after'])}; "
                    f"penetration max "
                    f"{metrics['torso_max_penetration_before_mm']:.2f} -> "
                    f"{metrics['torso_max_penetration_after_mm']:.2f}mm"
                )
            bar.set_postfix(
                done=processed, pass_=passthrough, skip=skipped, refresh=False
            )
        bar.close()
        print(
            f"RTMPose arm refinement complete: processed={processed}, "
            f"passthrough={passthrough}, skipped={skipped}"
        )
        return 0
    except (
        EOFError,
        FileNotFoundError,
        KeyError,
        OSError,
        PermissionError,
        pickle.UnpicklingError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
