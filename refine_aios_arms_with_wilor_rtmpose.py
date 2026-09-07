#!/usr/bin/env python3
"""Jointly refine SMPL-X arms with WiLoR hands and optional RTMPose arms.

This is the single entry point that merges ``refine_aios_arms_with_wilor.py``
(WiLoR-only wrist/palm fitting) with the RTMPose arm refinement. Supervision:

* The WiLoR MANO wrist, projected into the image, anchors the absolute 2-D hand
  position with a robust pixel (Huber) loss on the same scale as RTMPose.
* The WiLoR MANO ``global_orient`` constrains the absolute 3-D rotation of the
  SMPL-X wrist. The projected palm shape alone cannot tell a palm facing the
  camera from one facing away, so it is kept only as a weak auxiliary term.
* RTMPose shoulders and elbows constrain the rest of the arm chain. The
  RTMPose wrist is a different anatomical point (COCO keypoint) than the MANO
  wrist, so it is switched off wherever WiLoR supervises that hand and only
  acts as a fallback on frames without WiLoR. Pass ``--no-rtmpose`` for
  WiLoR-only refinement.

The optimization runs in two stages: first only positional terms place the arm
and hand, then the palm orientation term is enabled while every positional term
stays active, so turning the palm cannot drag the hand away.

WiLoR targets are used at real observations and, optionally, inside short gaps
bounded by observations on both sides; ``--wilor-unbounded-weight`` can also
supervise long gaps and nearest-filled prefix/suffix frames. A 3-D torso-volume
loss discourages arm penetration. Every clip is optimized independently. Finger
pose, camera, shape, root, face, and all non-listed body joints are copied
unchanged from the fused input.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from smplx.lbs import batch_rodrigues
from tqdm.auto import tqdm

import refine_aios_arms_with_rtmpose as rtmpose
from fuse_shared_aios_wilor import save_npz_atomic
from refine_aios_arms_with_wilor import (
    build_wilor_targets,
    data_losses as wilor_data_losses,
    make_mano_model,
    validate_inputs as validate_fused_wilor,
    weighted_mean,
)


DEFAULT_FUSED_ROOT = rtmpose.DEFAULT_FUSED_ROOT
DEFAULT_RTMPOSE_ROOT = rtmpose.DEFAULT_RTMPOSE_ROOT
DEFAULT_WILOR_ROOT = Path("shared_samples/wilor_params_interpolated")
DEFAULT_OUTPUT_ROOT = Path("shared_samples/aios_wilor_rtmpose_arm_refined")
DEFAULT_SMPLX_MODEL_ROOT = rtmpose.DEFAULT_SMPLX_MODEL_ROOT
DEFAULT_MANO_MODEL_PATH = Path("mano_data/MANO_RIGHT.pkl")

COMBINED_SCHEMA_VERSION = 3
SMPLX_PALM_JOINT_INDICES = (
    (20, 25, 28, 31, 34, 37),
    (21, 40, 43, 46, 49, 52),
)
# SMPL-X left/right wrist joints. Their global rotation is the hand-root
# rotation, i.e. the SMPL-X counterpart of MANO's global_orient.
SMPLX_WRIST_JOINT_INDICES = (20, 21)
# Kinematic-tree joints needed to reach both wrists: pelvis + 21 body joints.
SMPLX_WRIST_CHAIN_JOINTS = 22
MIRROR_X = np.diag((-1.0, 1.0, 1.0))
# Index of the wrist inside the RTMPose [shoulder, elbow, wrist] layout.
RTMPOSE_WRIST_SLOT = 2
# Positions in rtmpose.REFINED_BODY_JOINT_NAMES. Proximal joints shape the
# torso and shoulder line; distal joints (elbows, wrists) orient the hand
# without moving the shoulders.
DISTAL_DELTA_INDICES = tuple(
    index
    for index, name in enumerate(rtmpose.REFINED_BODY_JOINT_NAMES)
    if name.endswith("elbow") or name.endswith("wrist")
)
PROXIMAL_DELTA_INDICES = tuple(
    index
    for index in range(len(rtmpose.REFINED_BODY_JOINT_NAMES))
    if index not in DISTAL_DELTA_INDICES
)
# Frames whose palm orientation error exceeds this are counted as "bad" in the
# metrics; the count is reported per clip so unconverged frames are visible.
ORIENT_BAD_FRAME_DEGREES = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused-root", type=Path, default=DEFAULT_FUSED_ROOT)
    parser.add_argument("--rtmpose-root", type=Path, default=DEFAULT_RTMPOSE_ROOT)
    parser.add_argument(
        "--no-rtmpose",
        action="store_true",
        help=(
            "WiLoR-only refinement: do not load RTMPose at all. Equivalent to "
            "the standalone refine_aios_arms_with_wilor.py, plus palm "
            "orientation, torso collision, and the two-stage schedule."
        ),
    )
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
        help="Process only this fused clip; may be repeated.",
    )
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N")
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument(
        "--orient-warmup-iterations",
        type=int,
        default=None,
        help=(
            "Stage-1 iterations with positional terms only, before the palm "
            "orientation loss is enabled (default: a third of --iterations; "
            "position converges in about 40 iterations, orientation needs "
            "the rest)."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--aios-focal-length", type=float, default=5000.0)

    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--single-neighbor-weight", type=float, default=0.5)
    parser.add_argument("--shoulder-weight", type=float, default=1.0)
    parser.add_argument("--elbow-weight", type=float, default=2.0)
    parser.add_argument(
        "--wrist-weight",
        type=float,
        default=0.5,
        help=(
            "RTMPose wrist weight on frames where WiLoR does NOT supervise that "
            "hand (fallback only). WiLoR has its own wrist weight."
        ),
    )
    parser.add_argument(
        "--rtmpose-wrist-with-wilor",
        type=float,
        default=0.0,
        help=(
            "Multiplier on the RTMPose wrist weight for frames where WiLoR "
            "supervises the same hand. The COCO wrist and the MANO wrist are "
            "different points, so the default 0 hands the wrist to WiLoR."
        ),
    )
    parser.add_argument(
        "--huber-delta-px",
        type=float,
        default=8.0,
        help="Pixel transition point of the robust RTMPose and WiLoR wrist losses.",
    )

    parser.add_argument(
        "--wilor-wrist-weight",
        type=float,
        default=4.0,
        help=(
            "Weight of the WiLoR wrist reprojection loss (pixels over the "
            "image diagonal, the same scale as the RTMPose loss)."
        ),
    )
    parser.add_argument(
        "--wilor-wrist-loss",
        choices=("quadratic", "huber"),
        default="quadratic",
        help=(
            "quadratic: squared pixel error normalized like the Huber "
            "quadratic branch (0.5*e^2/delta/diagonal) but never switching to "
            "the linear branch, so no frame may drift far from the WiLoR "
            "wrist. huber: the same robust loss as RTMPose, which tolerates "
            "outlier frames."
        ),
    )
    parser.add_argument(
        "--max-wrist-delta-degrees",
        type=float,
        default=80.0,
        help=(
            "Clamp on each wrist joint correction. The shared 34-degree limit "
            "cannot absorb typical 60-degree palm orientation errors, which "
            "then leak into the elbow and shoulder and move the hand."
        ),
    )
    parser.add_argument(
        "--max-elbow-delta-degrees",
        type=float,
        default=60.0,
        help=(
            "Clamp on each elbow joint correction (shared default: 43). With "
            "the shoulders frozen in stage 2, forearm twist for the palm goes "
            "through the elbow and competes with the stage-1 bend correction "
            "for this budget. Elbow bending is still held by the quadratic "
            "WiLoR wrist loss and the RTMPose elbow."
        ),
    )
    parser.add_argument(
        "--no-stage2-freeze-proximal",
        action="store_true",
        help=(
            "By default stage 2 (orientation) updates only elbows and wrists "
            "and keeps spine3, collars, and shoulders at their stage-1 values, "
            "so the palm cannot drag the torso or the hand position. Pass this "
            "to let all nine joints move in stage 2."
        ),
    )
    parser.add_argument(
        "--wilor-palm-weight",
        type=float,
        default=0.1,
        help=(
            "Weight of the projected 2-D palm-shape loss. Kept weak because it "
            "is redundant with wrist position plus orientation and has a "
            "front/back ambiguity that can fight the orientation loss."
        ),
    )
    parser.add_argument(
        "--wilor-orient-weight",
        type=float,
        default=1.0,
        help=(
            "Weight of the absolute 3-D palm orientation loss: SMPL-X wrist "
            "global rotation versus WiLoR MANO global_orient."
        ),
    )
    parser.add_argument(
        "--wilor-orient-loss",
        choices=("angle", "cosine"),
        default="angle",
        help=(
            "angle: chordal distance 2*sin(angle/2), about the angle in radians "
            "(90 degrees costs 1.41); its gradient stays near 1 at small "
            "angles, so residual 10-30 degree errors keep being pulled in. "
            "cosine: 1 - cos(angle) (90 degrees costs 1.0); smooth, but its "
            "gradient fades like sin(angle) and leaves small errors behind."
        ),
    )
    parser.add_argument(
        "--wilor-target-cutoff-hz",
        type=float,
        default=5.0,
        help="Low-pass cutoff for the 2-D WiLoR wrist/palm position targets.",
    )
    parser.add_argument(
        "--wilor-orient-cutoff-hz",
        type=float,
        default=8.0,
        help=(
            "Low-pass cutoff for the WiLoR palm orientation targets. Kept "
            "higher than the position cutoff and applied without Hampel "
            "outlier rejection, because fast two-to-three-frame wrist flips "
            "are real signing motion, not outliers."
        ),
    )
    parser.add_argument(
        "--wilor-interpolated-weight",
        type=float,
        default=0.15,
        help="Weight for short, observation-bounded WiLoR gaps.",
    )
    parser.add_argument(
        "--max-wilor-interpolation-gap",
        type=int,
        default=12,
        help=(
            "Maximum missing frames between two observed WiLoR frames. "
            "Use 0 for observed-only supervision."
        ),
    )
    parser.add_argument(
        "--wilor-unbounded-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for valid WiLoR frames that are neither observed nor inside "
            "a short bounded gap: long gaps and nearest-filled prefix/suffix. "
            "0 (default) rejects them; 0.25 reproduces the standalone WiLoR "
            "script's coverage."
        ),
    )

    parser.add_argument("--pose-prior-weight", type=float, default=0.08)
    parser.add_argument("--velocity-weight", type=float, default=0.20)
    parser.add_argument("--acceleration-weight", type=float, default=1.0)
    parser.add_argument("--collision-weight", type=float, default=0.05)
    parser.add_argument("--collision-margin", type=float, default=0.02)
    parser.add_argument("--torso-width-scale", type=float, default=0.42)
    parser.add_argument("--torso-height-scale", type=float, default=0.52)
    parser.add_argument("--torso-depth-scale", type=float, default=0.22)

    parser.add_argument("--max-time-offset-frames", type=int, default=4)
    parser.add_argument("--time-offset-penalty-px", type=float, default=0.25)
    parser.add_argument(
        "--time-offset-min-improvement-px", type=float, default=0.5
    )
    parser.add_argument(
        "--max-duration-error-seconds", type=float, default=0.25
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all inputs without loading SMPL-X or MANO.",
    )
    args = parser.parse_args()

    positive = (
        "iterations",
        "learning_rate",
        "batch_size",
        "aios_focal_length",
        "huber_delta_px",
        "max_wrist_delta_degrees",
        "max_elbow_delta_degrees",
        "wilor_target_cutoff_hz",
        "wilor_orient_cutoff_hz",
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
        "rtmpose_wrist_with_wilor",
        "wilor_wrist_weight",
        "wilor_palm_weight",
        "wilor_orient_weight",
        "wilor_interpolated_weight",
        "wilor_unbounded_weight",
        "max_wilor_interpolation_gap",
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
    unit_interval = (
        "confidence_threshold",
        "single_neighbor_weight",
        "rtmpose_wrist_with_wilor",
        "wilor_interpolated_weight",
        "wilor_unbounded_weight",
    )
    for name in unit_interval:
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.confidence_threshold == 1:
        parser.error("--confidence-threshold must be less than 1")
    if args.max_clips is not None and args.max_clips <= 0:
        parser.error("--max-clips must be positive")
    if args.orient_warmup_iterations is None:
        args.orient_warmup_iterations = args.iterations // 3
    if not 0 <= args.orient_warmup_iterations <= args.iterations:
        parser.error("--orient-warmup-iterations must be in [0, --iterations]")
    if (
        not args.no_rtmpose
        and args.shoulder_weight + args.elbow_weight + args.wrist_weight <= 0
    ):
        parser.error("at least one RTMPose joint weight must be positive")
    if (
        args.wilor_wrist_weight + args.wilor_palm_weight + args.wilor_orient_weight
        <= 0
    ):
        parser.error("at least one WiLoR loss weight must be positive")
    return args


def discover_clip_paths(
    root: Path, requested: Sequence[str] | None
) -> list[Path]:
    return rtmpose.discover_clip_paths(root, requested)


def bounded_wilor_weights(
    wilor: dict[str, np.ndarray],
    interpolated_weight: float,
    max_gap: int,
    unbounded_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame, per-hand WiLoR loss weights.

    Observed frames get their detection confidence (clipped to [0.25, 1]).
    Interpolated frames inside a gap of at most ``max_gap`` frames bounded by
    observations on both sides get ``interpolated_weight``. Every other valid
    frame (long gaps, nearest-filled prefix/suffix) gets ``unbounded_weight``,
    which is 0 by default. Returns weights, the bounded-interpolation mask, and
    the unbounded mask.
    """
    valid = np.asarray(wilor["valid_mask"], dtype=bool)
    observed = np.asarray(wilor["observed_mask"], dtype=bool) & valid
    interpolated = np.asarray(wilor["interpolated_mask"], dtype=bool) & valid
    confidence = np.asarray(wilor["detection_confidence"], dtype=np.float32)
    weights = np.zeros(valid.shape, dtype=np.float32)
    weights[observed] = np.clip(confidence[observed], 0.25, 1.0)
    allowed_interpolation = np.zeros(valid.shape, dtype=bool)

    if max_gap > 0 and interpolated_weight > 0:
        for side in range(2):
            observed_indices = np.flatnonzero(observed[:, side])
            for left, right in zip(observed_indices[:-1], observed_indices[1:]):
                gap = int(right - left - 1)
                if 0 < gap <= max_gap:
                    allowed_interpolation[left + 1 : right, side] = True
        allowed_interpolation &= interpolated
        weights[allowed_interpolation] = interpolated_weight

    unbounded = valid & ~observed & ~allowed_interpolation
    if unbounded_weight > 0:
        weights[unbounded] = unbounded_weight
    return weights, allowed_interpolation, unbounded


def gate_rtmpose_wrist(
    rtmpose_weights: np.ndarray,
    wilor_weights: np.ndarray,
    multiplier: float,
) -> np.ndarray:
    """Hand the wrist to WiLoR wherever WiLoR supervises that hand.

    The RTMPose wrist (COCO keypoint) and the MANO wrist are different
    anatomical points, so fitting both pulls the hand to a compromise. Scale the
    RTMPose wrist weight by ``multiplier`` (0 = off) on frames with WiLoR weight;
    frames without WiLoR keep the full RTMPose wrist weight as a fallback.
    """
    gated = np.array(rtmpose_weights, dtype=np.float32, copy=True)
    scale = np.where(wilor_weights > 0, multiplier, 1.0).astype(np.float32)
    gated[:, :, RTMPOSE_WRIST_SLOT] *= scale
    return gated


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt the first two (filtered) rotation columns back onto SO(3)."""
    first = rotation_6d[..., :3]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = rotation_6d[..., 3:]
    second = second - (first * second).sum(-1, keepdims=True) * first
    second = second / np.maximum(
        np.linalg.norm(second, axis=-1, keepdims=True), 1e-8
    )
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def nearest_valid_indices(valid: np.ndarray) -> np.ndarray:
    """For every frame, the index of the nearest frame where ``valid`` holds."""
    frames = np.arange(len(valid))
    valid_indices = np.flatnonzero(valid)
    position = np.searchsorted(valid_indices, frames)
    upper = valid_indices[np.clip(position, 0, len(valid_indices) - 1)]
    lower = valid_indices[np.clip(position - 1, 0, len(valid_indices) - 1)]
    return np.where(np.abs(lower - frames) <= np.abs(upper - frames), lower, upper)


def zero_phase_lowpass(
    values: np.ndarray, fps: float, cutoff_hz: float, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth low-pass along time, without Hampel rejection.

    Same filter as ``fuse_shared_aios_wilor.zero_phase_filter`` minus the
    outlier step, which would replace brief real wrist flips by the median.
    """
    original_shape = values.shape
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    if len(values) < 4:
        return flat.reshape(original_shape)
    effective_cutoff = min(cutoff_hz, fps * 0.475)
    sos = butter(order, effective_cutoff, btype="lowpass", fs=fps, output="sos")
    padlen = min(len(values) - 1, 3 * (2 * len(sos) + 1))
    return sosfiltfilt(sos, flat, axis=0, padlen=padlen).reshape(original_shape)


def build_wilor_orientation_targets(
    wilor: dict[str, np.ndarray],
    fps: float,
    cutoff_hz: float,
) -> np.ndarray:
    """Absolute palm (hand-root) rotation targets, shape [T, 2, 3, 3].

    WiLoR stores the MANO right-hand ``global_orient`` in camera coordinates for
    both slots; the left hand was predicted on a horizontally flipped crop, so
    its rotation is mirrored back with ``MIRROR_X @ R @ MIRROR_X`` (the same
    convention the fusion stage uses for the finger rotations). Fused AIOS
    SMPL-X joints already live in camera-axis orientation (see
    ``project_smplx_joints``), so the SMPL-X wrist global rotation is directly
    comparable. The rotations are low-pass filtered in a continuous 6-D
    representation with their own, higher cutoff and WITHOUT the Hampel
    outlier step used for the 2-D targets: a wrist flip that lasts two or three
    frames is real signing motion, and Hampel would replace it by the
    neighbourhood median, leaving those frames with a wrong target. Frames
    outside ``valid_mask`` never receive loss weight; they are nearest-filled
    here only so the zero-phase filter never sees zero matrices.
    """
    valid = np.asarray(wilor["valid_mask"], dtype=bool)
    rotations = np.asarray(wilor["global_orient_rotmat"], dtype=np.float64)[:, :, 0]
    num_frames = len(valid)
    targets = np.broadcast_to(np.eye(3), (num_frames, 2, 3, 3)).copy()
    for side in range(2):
        if not valid[:, side].any():
            continue
        side_rotations = rotations[:, side]
        if side == 0:
            side_rotations = MIRROR_X @ side_rotations @ MIRROR_X
        side_rotations = side_rotations[nearest_valid_indices(valid[:, side])]
        rotation_6d = np.concatenate(
            (side_rotations[..., :, 0], side_rotations[..., :, 1]), axis=-1
        )
        filtered = zero_phase_lowpass(rotation_6d, fps, cutoff_hz)
        targets[:, side] = rotation_6d_to_matrix(filtered)
    return targets.astype(np.float32)


def smplx_arm_palm_and_hip_joints(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    body_pose: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        rtmpose.SMPLX_ARM_JOINT_INDICES,
        device=body_pose.device,
        dtype=torch.long,
    )
    palm_indices = torch.tensor(
        SMPLX_PALM_JOINT_INDICES, device=body_pose.device, dtype=torch.long
    )
    hip_indices = torch.tensor(
        rtmpose.SMPLX_HIP_JOINT_INDICES,
        device=body_pose.device,
        dtype=torch.long,
    )
    return (
        output.joints[:, arm_indices],
        output.joints[:, palm_indices],
        output.joints[:, hip_indices],
    )


def smplx_wrist_rotations(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    body_pose: torch.Tensor,
) -> torch.Tensor:
    """Global (camera-frame) rotation of both SMPL-X wrists, shape [N, 2, 3, 3].

    SMPL-X rest joint frames are all axis-aligned with the template, so the
    accumulated rotation root @ spine ... @ wrist is exactly the rotation applied
    to the (MANO-compatible) hand template, i.e. MANO's global_orient. It is
    computed by forward kinematics on the axis-angle chain, so it stays
    differentiable with respect to every refined arm joint.
    """
    count = len(body_pose)
    rotation_vectors = torch.cat((values["root"][:, None, :], body_pose), dim=1)
    local = batch_rodrigues(rotation_vectors.reshape(-1, 3)).reshape(
        count, SMPLX_WRIST_CHAIN_JOINTS, 3, 3
    )
    parents = model.parents[:SMPLX_WRIST_CHAIN_JOINTS].tolist()
    global_rotations = [local[:, 0]]
    for joint in range(1, SMPLX_WRIST_CHAIN_JOINTS):
        global_rotations.append(global_rotations[parents[joint]] @ local[:, joint])
    stacked = torch.stack(global_rotations, dim=1)
    return stacked[:, list(SMPLX_WRIST_JOINT_INDICES)]


def orientation_cosine(
    predicted: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """cos(geodesic angle) between rotation matrices, shape [N, 2]."""
    trace = (predicted * targets).sum(dim=(-1, -2))
    return ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)


def orientation_loss(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Weighted mean orientation distance per frame and hand.

    ``cosine``: 1 - cos(angle). ``angle``: chordal distance
    sqrt(2 - 2 cos(angle)) = 2 sin(angle / 2), which is the angle in radians
    to first order and keeps a unit gradient down to the epsilon, so frames
    with a 10-30 degree residual are still pulled in.
    """
    cosine = orientation_cosine(predicted, targets)
    if args.wilor_orient_loss == "angle":
        per_frame = torch.sqrt((2.0 - 2.0 * cosine).clamp_min(0.0) + 1e-6)
    else:
        per_frame = 1.0 - cosine
    return weighted_mean(per_frame, weights)


def orientation_degrees(
    predicted: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return torch.rad2deg(torch.acos(orientation_cosine(predicted, targets)))


def wilor_wrist_loss(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    image_shape: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """WiLoR wrist reprojection loss on the RTMPose scale.

    Pixel error averaged over x/y and divided by the image diagonal, exactly
    like ``rtmpose.reprojection_loss``. The standalone WiLoR script used a
    squared error in normalized coordinates, two orders of magnitude smaller,
    which lost every tug-of-war against the RTMPose and orientation terms.

    ``quadratic`` (default) keeps the Huber quadratic branch for every error
    magnitude. The WiLoR wrist is trusted, so a frame that drifts 40 px must
    cost 25x a frame at 8 px, not 4.5x as with the linear Huber branch.
    """
    pixel_scale = torch.stack((image_shape[:, 1], image_shape[:, 0]), dim=-1)
    absolute_px = (
        torch.abs(predicted[:, :, 0] - targets[:, :, 0]) * pixel_scale[:, None]
    )
    delta = args.huber_delta_px
    quadratic_px = 0.5 * absolute_px.square() / delta
    if args.wilor_wrist_loss == "quadratic":
        loss_px = quadratic_px
    else:
        loss_px = torch.where(
            absolute_px < delta, quadratic_px, absolute_px - 0.5 * delta
        )
    image_diagonal = torch.linalg.vector_norm(pixel_scale, dim=-1).clamp_min(1.0)
    per_hand = loss_px.mean(dim=-1) / image_diagonal[:, None]
    return weighted_mean(per_hand, weights)


def delta_limit_values(args: argparse.Namespace) -> np.ndarray:
    """Per-joint clamp on the correction norm; elbows and wrists get their own."""
    limits = np.asarray(rtmpose.MAX_DELTA_RADIANS, dtype=np.float32).copy()
    for index, name in enumerate(rtmpose.REFINED_BODY_JOINT_NAMES):
        if name.endswith("wrist"):
            limits[index] = np.deg2rad(args.max_wrist_delta_degrees)
        elif name.endswith("elbow"):
            limits[index] = np.deg2rad(args.max_elbow_delta_degrees)
    return limits


def clamp_delta(delta: torch.Tensor, limits: torch.Tensor) -> None:
    norms = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    scale = torch.minimum(
        torch.ones_like(norms), limits[None, :, None] / norms.clamp_min(1e-8)
    )
    delta.mul_(scale)


def predict_chunk(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    delta: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    body_pose = rtmpose.body_with_delta(values["body"], delta)
    arm_joints, palm_joints, hip_joints = smplx_arm_palm_and_hip_joints(
        model, values, body_pose
    )
    wrist_rotations = smplx_wrist_rotations(model, values, body_pose)
    arm_projected = rtmpose.project_smplx_joints(
        arm_joints,
        values["camera"],
        values["image_shape"],
        args.aios_focal_length,
    )
    palm_projected = rtmpose.project_smplx_joints(
        palm_joints,
        values["camera"],
        values["image_shape"],
        args.aios_focal_length,
    )
    collision_loss, penetration = rtmpose.torso_penetration(
        arm_joints, hip_joints, args
    )
    return (
        arm_projected,
        palm_projected,
        wrist_rotations,
        collision_loss,
        penetration,
    )


def predict_sequence(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    delta: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    arm_parts = []
    palm_parts = []
    rotation_parts = []
    penetration_parts = []
    for start in range(0, len(delta), args.batch_size):
        stop = min(start + args.batch_size, len(delta))
        arm, palm, rotations, _, penetration = predict_chunk(
            model,
            rtmpose.slice_values(values, start, stop),
            delta[start:stop],
            args,
        )
        arm_parts.append(arm)
        palm_parts.append(palm)
        rotation_parts.append(rotations)
        penetration_parts.append(penetration)
    return (
        torch.cat(arm_parts),
        torch.cat(palm_parts),
        torch.cat(rotation_parts),
        torch.cat(penetration_parts),
    )


def wilor_metrics(
    before: torch.Tensor,
    after: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    image_shape: torch.Tensor,
) -> dict[str, float]:
    valid = weights > 0
    pixel_scale = torch.stack((image_shape[:, 1], image_shape[:, 0]), dim=-1)
    before_wrist = torch.linalg.vector_norm(
        (before[:, :, 0] - targets[:, :, 0]) * pixel_scale[:, None], dim=-1
    )
    after_wrist = torch.linalg.vector_norm(
        (after[:, :, 0] - targets[:, :, 0]) * pixel_scale[:, None], dim=-1
    )

    def palm_error(predicted: torch.Tensor) -> torch.Tensor:
        predicted_palm = predicted[:, :, 1:] - predicted[:, :, :1]
        target_palm = targets[:, :, 1:] - targets[:, :, :1]
        predicted_scale = torch.sqrt(
            predicted_palm.square().sum(-1).mean(-1).clamp_min(1e-8)
        )
        target_scale = torch.sqrt(
            target_palm.square().sum(-1).mean(-1).clamp_min(1e-8)
        )
        predicted_palm = predicted_palm / predicted_scale[:, :, None, None]
        target_palm = target_palm / target_scale[:, :, None, None]
        return (predicted_palm - target_palm).square().mean(dim=(-1, -2))

    before_palm = palm_error(before)
    after_palm = palm_error(after)
    wrist_before_summary = rtmpose.error_summary(before_wrist[valid])
    wrist_after_summary = rtmpose.error_summary(after_wrist[valid])
    palm_before_summary = rtmpose.error_summary(before_palm[valid])
    palm_after_summary = rtmpose.error_summary(after_palm[valid])
    return {
        "wilor_wrist_error_before_median_px": wrist_before_summary[0],
        "wilor_wrist_error_before_p95_px": wrist_before_summary[1],
        "wilor_wrist_error_after_median_px": wrist_after_summary[0],
        "wilor_wrist_error_after_p95_px": wrist_after_summary[1],
        "wilor_palm_error_before_median": palm_before_summary[0],
        "wilor_palm_error_before_p95": palm_before_summary[1],
        "wilor_palm_error_after_median": palm_after_summary[0],
        "wilor_palm_error_after_p95": palm_after_summary[1],
    }


def orientation_metrics(
    before: torch.Tensor,
    after: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, float]:
    valid = weights > 0
    before_degrees = orientation_degrees(before, targets)[valid]
    after_degrees = orientation_degrees(after, targets)[valid]
    before_summary = rtmpose.error_summary(before_degrees)
    after_summary = rtmpose.error_summary(after_degrees)
    return {
        "wilor_palm_orient_error_before_median_deg": before_summary[0],
        "wilor_palm_orient_error_before_p95_deg": before_summary[1],
        "wilor_palm_orient_error_after_median_deg": after_summary[0],
        "wilor_palm_orient_error_after_p95_deg": after_summary[1],
        "wilor_palm_orient_bad_hand_frames_before": float(
            (before_degrees > ORIENT_BAD_FRAME_DEGREES).sum()
        ),
        "wilor_palm_orient_bad_hand_frames_after": float(
            (after_degrees > ORIENT_BAD_FRAME_DEGREES).sum()
        ),
    }


def optimize_arm_pose(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    rtmpose_targets: np.ndarray,
    rtmpose_weights: np.ndarray,
    wilor_targets: np.ndarray,
    wilor_orient_targets: np.ndarray,
    wilor_weights: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    device = values["body"].device
    rtmpose_target_tensor = rtmpose.tensor(rtmpose_targets, device)
    rtmpose_weight_tensor = rtmpose.tensor(rtmpose_weights, device)
    wilor_target_tensor = rtmpose.tensor(wilor_targets, device)
    wilor_orient_tensor = rtmpose.tensor(wilor_orient_targets, device)
    wilor_weight_tensor = rtmpose.tensor(wilor_weights, device)
    num_frames = len(rtmpose_target_tensor)
    delta = torch.nn.Parameter(
        torch.zeros(
            (num_frames, len(rtmpose.REFINED_BODY_JOINT_INDICES), 3),
            device=device,
        )
    )
    optimizer = torch.optim.Adam([delta], lr=args.learning_rate)

    with torch.no_grad():
        arm_before, palm_before, orient_before, penetration_before = (
            predict_sequence(model, values, delta, args)
        )

    rtmpose_joint_weights = rtmpose.joint_loss_weights(args, device)
    total_rtmpose_weight = float(
        (rtmpose_weight_tensor * rtmpose_joint_weights).sum()
    )
    total_wilor_weight = float(wilor_weight_tensor.sum())
    limits = torch.from_numpy(delta_limit_values(args)).to(device)
    proximal = torch.tensor(
        PROXIMAL_DELTA_INDICES, device=device, dtype=torch.long
    )
    frozen_proximal: torch.Tensor | None = None
    warmup = args.orient_warmup_iterations
    for iteration in range(1, args.iterations + 1):
        # Stage 1: positional terms only, all nine joints. Stage 2: add palm
        # orientation while keeping every positional term; by default only
        # elbows and wrists may still move, so the palm is turned through
        # wrist rotation and forearm twist, neither of which moves the hand.
        orient_weight = args.wilor_orient_weight if iteration > warmup else 0.0
        if iteration == warmup + 1 and warmup > 0 and orient_weight > 0:
            if args.no_stage2_freeze_proximal:
                tqdm.write("  [stage 2] enabling WiLoR palm orientation loss")
            else:
                frozen_proximal = delta.detach()[:, proximal].clone()
                tqdm.write(
                    "  [stage 2] enabling WiLoR palm orientation loss; "
                    "spine3/collars/shoulders frozen at stage-1 values"
                )

        optimizer.zero_grad(set_to_none=True)
        regularization, regularization_parts = rtmpose.regularization_loss(
            delta, args
        )
        regularization.backward()
        rtmpose_value = 0.0
        wilor_wrist_value = 0.0
        wilor_palm_value = 0.0
        wilor_orient_value = 0.0
        collision_value = 0.0

        for start in range(0, num_frames, args.batch_size):
            stop = min(start + args.batch_size, num_frames)
            chunk_values = rtmpose.slice_values(values, start, stop)
            arm, palm, wrist_rotations, collision_loss, _ = predict_chunk(
                model, chunk_values, delta[start:stop], args
            )
            rt_weights = rtmpose_weight_tensor[start:stop]
            rt_chunk_weight = float(
                (rt_weights * rtmpose_joint_weights).sum().detach()
            )
            rt_fraction = rt_chunk_weight / max(total_rtmpose_weight, 1e-8)
            rt_loss = rtmpose.reprojection_loss(
                arm,
                rtmpose_target_tensor[start:stop],
                rt_weights,
                chunk_values["image_shape"],
                args,
            )

            wi_weights = wilor_weight_tensor[start:stop]
            wi_chunk_weight = float(wi_weights.sum().detach())
            wi_fraction = wi_chunk_weight / max(total_wilor_weight, 1e-8)
            wi_targets = wilor_target_tensor[start:stop]
            wrist_loss = wilor_wrist_loss(
                palm, wi_targets, wi_weights, chunk_values["image_shape"], args
            )
            _, palm_loss = wilor_data_losses(palm, wi_targets, wi_weights)
            orient_loss = orientation_loss(
                wrist_rotations, wilor_orient_tensor[start:stop], wi_weights, args
            )
            wilor_terms = (
                args.wilor_wrist_weight * wrist_loss
                + args.wilor_palm_weight * palm_loss
            )
            if orient_weight > 0:
                wilor_terms = wilor_terms + orient_weight * orient_loss
            collision_fraction = (stop - start) / num_frames
            chunk_loss = (
                rt_fraction * rt_loss
                + wi_fraction * wilor_terms
                + collision_fraction * args.collision_weight * collision_loss
            )
            chunk_loss.backward()
            rtmpose_value += rt_fraction * float(rt_loss.detach())
            wilor_wrist_value += wi_fraction * float(wrist_loss.detach())
            wilor_palm_value += wi_fraction * float(palm_loss.detach())
            wilor_orient_value += wi_fraction * float(orient_loss.detach())
            collision_value += collision_fraction * float(collision_loss.detach())

        if delta.grad is None or not torch.isfinite(delta.grad).all():
            raise RuntimeError("combined arm optimization produced non-finite gradients")
        optimizer.step()
        with torch.no_grad():
            clamp_delta(delta, limits)
            if frozen_proximal is not None:
                delta[:, proximal] = frozen_proximal

        if iteration == 1 or iteration % args.log_every == 0:
            total = (
                rtmpose_value
                + args.wilor_wrist_weight * wilor_wrist_value
                + args.wilor_palm_weight * wilor_palm_value
                + orient_weight * wilor_orient_value
                + args.collision_weight * collision_value
                + float(regularization.detach())
            )
            stage = 1 if iteration <= warmup else 2
            tqdm.write(
                f"  iter={iteration:04d} stage={stage} loss={total:.6f} "
                f"rtmpose={rtmpose_value:.6f} "
                f"wilor_wrist={wilor_wrist_value:.6f} "
                f"wilor_palm={wilor_palm_value:.6f} "
                f"wilor_orient={wilor_orient_value:.6f} "
                f"collision={collision_value:.6f} "
                f"prior={float(regularization_parts['prior']):.6f} "
                f"vel={float(regularization_parts['velocity']):.6f} "
                f"acc={float(regularization_parts['acceleration']):.6f}"
            )

    with torch.no_grad():
        arm_after, palm_after, orient_after, penetration_after = (
            predict_sequence(model, values, delta, args)
        )
    refined_body = rtmpose.body_with_delta(values["body"], delta)
    metrics = rtmpose.refinement_metrics(
        arm_before,
        arm_after,
        rtmpose_target_tensor,
        rtmpose_weight_tensor,
        values["image_shape"],
        penetration_before,
        penetration_after,
        delta,
    )
    metrics.update(
        wilor_metrics(
            palm_before,
            palm_after,
            wilor_target_tensor,
            wilor_weight_tensor,
            values["image_shape"],
        )
    )
    metrics.update(
        orientation_metrics(
            orient_before, orient_after, wilor_orient_tensor, wilor_weight_tensor
        )
    )
    return (
        refined_body.reshape(num_frames, 63).detach().cpu().numpy().astype(np.float32),
        delta.detach().cpu().numpy().astype(np.float32),
        metrics,
    )


def rtmpose_supervision(
    fused: dict[str, np.ndarray],
    clip_id: str,
    num_frames: int,
    fps: float,
    pose_path: Path,
    initial_arm: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    """RTMPose targets, weights, and alignment metadata; all-zero if disabled."""
    if args.no_rtmpose:
        try:
            duration = rtmpose.clip_duration_seconds(clip_id)
        except ValueError:
            duration = num_frames / fps
        return {
            "targets": np.zeros((num_frames, 2, 3, 2), dtype=np.float32),
            "weights": np.zeros((num_frames, 2, 3), dtype=np.float32),
            "offset": 0,
            "alignment_score": float("nan"),
            "raw_frames": 0,
            "duration": duration,
            "pose_fps": float("nan"),
        }
    keypoints, scores = rtmpose.load_rtmpose(pose_path)
    duration, pose_fps, _ = rtmpose.validate_time_axis(
        clip_id,
        num_frames,
        fps,
        len(keypoints),
        args.max_duration_error_seconds,
    )
    offset, alignment_score, targets, weights = rtmpose.select_time_offset(
        initial_arm,
        keypoints,
        scores,
        num_frames,
        fps,
        duration,
        np.asarray(fused["img_shape"], dtype=np.float32),
        args,
    )
    return {
        "targets": targets,
        "weights": weights,
        "offset": offset,
        "alignment_score": alignment_score,
        "raw_frames": len(keypoints),
        "duration": duration,
        "pose_fps": pose_fps,
    }


def combined_metadata(
    fused: dict[str, np.ndarray],
    supervision: dict[str, object],
    rtmpose_weights: np.ndarray,
    wilor: dict[str, np.ndarray],
    wilor_weights: np.ndarray,
    allowed_interpolation: np.ndarray,
    unbounded: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    metadata = rtmpose.base_output_metadata(
        fused,
        int(supervision["raw_frames"]),
        float(supervision["duration"]),
        float(supervision["pose_fps"]),
        int(supervision["offset"]),
        float(supervision["alignment_score"]),
        rtmpose_weights,
        args,
    )
    method = "wilor_wrist_palm_orient"
    if not args.no_rtmpose:
        method += "_rtmpose_arm"
    method += "_torso_collision_two_stage_clip_ik"
    target_policy = "observed_plus_short_bounded_interpolation"
    if args.wilor_unbounded_weight > 0:
        target_policy += "_plus_unbounded"
    metadata.update(
        {
            "wilor_rtmpose_arm_refinement_schema_version": np.int32(
                COMBINED_SCHEMA_VERSION
            ),
            "arm_refinement_method": np.asarray(method),
            "rtmpose_used": np.bool_(not args.no_rtmpose),
            "rtmpose_wrist_with_wilor_weight": np.float32(
                args.rtmpose_wrist_with_wilor
            ),
            "rtmpose_wrist_gated_by_wilor": np.bool_(
                args.rtmpose_wrist_with_wilor < 1
            ),
            "wilor_target_policy": np.asarray(target_policy),
            "wilor_edge_fill_used_for_loss": np.bool_(
                args.wilor_unbounded_weight > 0
            ),
            "wilor_observed_target_counts": np.asarray(
                wilor["observed_mask"], dtype=bool
            ).sum(axis=0).astype(np.int32),
            "wilor_bounded_interpolated_target_counts": allowed_interpolation.sum(
                axis=0
            ).astype(np.int32),
            "wilor_unbounded_target_counts": (
                unbounded & (wilor_weights > 0)
            ).sum(axis=0).astype(np.int32),
            "wilor_weighted_target_counts": (wilor_weights > 0).sum(axis=0).astype(
                np.int32
            ),
            "wilor_interpolated_target_weight": np.float32(
                args.wilor_interpolated_weight
            ),
            "wilor_unbounded_target_weight": np.float32(
                args.wilor_unbounded_weight
            ),
            "wilor_max_interpolation_gap_frames": np.int32(
                args.max_wilor_interpolation_gap
            ),
            "wilor_target_cutoff_hz": np.float32(args.wilor_target_cutoff_hz),
            "wilor_orient_cutoff_hz": np.float32(args.wilor_orient_cutoff_hz),
            "wilor_orient_hampel": np.bool_(False),
            "wilor_orient_loss": np.asarray(args.wilor_orient_loss),
            "wilor_orient_bad_frame_threshold_deg": np.float32(
                ORIENT_BAD_FRAME_DEGREES
            ),
            "wilor_wrist_loss": np.asarray(
                f"{args.wilor_wrist_loss}_px_over_image_diagonal"
            ),
            "wilor_wrist_weight": np.float32(args.wilor_wrist_weight),
            "arm_delta_limits_radians": delta_limit_values(args),
            "arm_max_wrist_delta_degrees": np.float32(
                args.max_wrist_delta_degrees
            ),
            "arm_max_elbow_delta_degrees": np.float32(
                args.max_elbow_delta_degrees
            ),
            "stage2_freeze_proximal_joints": np.bool_(
                not args.no_stage2_freeze_proximal
            ),
            "stage2_frozen_joint_names": np.asarray(
                [
                    rtmpose.REFINED_BODY_JOINT_NAMES[index]
                    for index in PROXIMAL_DELTA_INDICES
                ]
            ),
            "wilor_palm_weight": np.float32(args.wilor_palm_weight),
            "wilor_orient_weight": np.float32(args.wilor_orient_weight),
            "wilor_orientation_target": np.asarray(
                "mano_global_orient_rotmat_camera_frame_left_mirror_x"
            ),
            "orient_warmup_iterations": np.int32(args.orient_warmup_iterations),
        }
    )
    return metadata


def refine_clip(
    fused_path: Path,
    pose_path: Path,
    wilor_path: Path,
    output_path: Path,
    smplx_model: torch.nn.Module,
    mano_model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    fused = rtmpose.load_npz(fused_path)
    wilor = rtmpose.load_npz(wilor_path)
    clip_id, num_frames, fps = rtmpose.validate_fused(fused, fused_path)
    validate_fused_wilor(fused, wilor, fused_path, wilor_path)
    values = rtmpose.fused_tensors(fused, device)
    zero_delta = torch.zeros(
        (num_frames, len(rtmpose.REFINED_BODY_JOINT_INDICES), 3), device=device
    )
    with torch.no_grad():
        initial_arm, _, _, _ = predict_sequence(
            smplx_model, values, zero_delta, args
        )
    supervision = rtmpose_supervision(
        fused,
        clip_id,
        num_frames,
        fps,
        pose_path,
        initial_arm.cpu().numpy(),
        args,
    )
    wi_targets, _ = build_wilor_targets(
        wilor,
        mano_model,
        device,
        fps,
        args.wilor_target_cutoff_hz,
    )
    wi_orient_targets = build_wilor_orientation_targets(
        wilor, fps, args.wilor_orient_cutoff_hz
    )
    wi_weights, allowed_interpolation, unbounded = bounded_wilor_weights(
        wilor,
        args.wilor_interpolated_weight,
        args.max_wilor_interpolation_gap,
        args.wilor_unbounded_weight,
    )
    rt_weights = gate_rtmpose_wrist(
        supervision["weights"], wi_weights, args.rtmpose_wrist_with_wilor
    )
    if not np.any(rt_weights > 0) and not np.any(wi_weights > 0):
        raise RuntimeError(f"{clip_id}: neither RTMPose nor WiLoR has valid targets")

    gated_wrists = int(
        (
            (np.asarray(supervision["weights"])[:, :, RTMPOSE_WRIST_SLOT] > 0)
            & (rt_weights[:, :, RTMPOSE_WRIST_SLOT] == 0)
        ).sum()
    )
    tqdm.write(
        f"[refine] {clip_id}: fused={num_frames} "
        f"pose={int(supervision['raw_frames'])} "
        f"offset={int(supervision['offset']):+d} "
        f"rt_valid={int((rt_weights > 0).sum())} rt_wrist_gated={gated_wrists} "
        f"wilor_observed={int(np.asarray(wilor['observed_mask']).sum())} "
        f"wilor_bounded_interp={int(allowed_interpolation.sum())} "
        f"wilor_unbounded={int((unbounded & (wi_weights > 0)).sum())}"
    )
    refined_body, delta, metrics = optimize_arm_pose(
        smplx_model,
        values,
        supervision["targets"],
        rt_weights,
        wi_targets,
        wi_orient_targets,
        wi_weights,
        args,
    )
    payload = dict(fused)
    payload.update(
        combined_metadata(
            fused,
            supervision,
            rt_weights,
            wilor,
            wi_weights,
            allowed_interpolation,
            unbounded,
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
    rtmpose.assert_protected_fields(fused, payload)
    save_npz_atomic(output_path, payload)
    return metrics


def validate_all(
    clip_paths: list[Path], args: argparse.Namespace
) -> list[tuple[Path, Path, Path]]:
    validated = []
    pose_counts = []
    fused_counts = []
    observed_counts = np.zeros(2, dtype=np.int64)
    bounded_counts = np.zeros(2, dtype=np.int64)
    unbounded_counts = np.zeros(2, dtype=np.int64)
    edge_fill_counts = np.zeros(2, dtype=np.int64)
    bar = tqdm(clip_paths, desc="validate", unit="clip", dynamic_ncols=True)
    for fused_path in bar:
        fused = rtmpose.load_npz(fused_path)
        clip_id, num_frames, fps = rtmpose.validate_fused(fused, fused_path)
        pose_path = args.rtmpose_root / f"{clip_id}.pkl"
        wilor_path = args.wilor_root / fused_path.name
        wilor = rtmpose.load_npz(wilor_path)
        validate_fused_wilor(fused, wilor, fused_path, wilor_path)
        pose_frames = 0
        pose_fps = float("nan")
        if not args.no_rtmpose:
            keypoints, _ = rtmpose.load_rtmpose(pose_path)
            pose_frames = len(keypoints)
            _, pose_fps, _ = rtmpose.validate_time_axis(
                clip_id,
                num_frames,
                fps,
                pose_frames,
                args.max_duration_error_seconds,
            )
        weights, allowed, unbounded = bounded_wilor_weights(
            wilor,
            args.wilor_interpolated_weight,
            args.max_wilor_interpolation_gap,
            args.wilor_unbounded_weight,
        )
        observed = np.asarray(wilor["observed_mask"], dtype=bool)
        valid = np.asarray(wilor["valid_mask"], dtype=bool)
        bounded_span = np.zeros_like(valid)
        for side in range(2):
            indices = np.flatnonzero(observed[:, side])
            if len(indices):
                bounded_span[indices[0] : indices[-1] + 1, side] = True
        observed_counts += observed.sum(axis=0)
        bounded_counts += allowed.sum(axis=0)
        unbounded_counts += (unbounded & (weights > 0)).sum(axis=0)
        edge_fill_counts += (valid & ~bounded_span).sum(axis=0)
        if args.wilor_unbounded_weight == 0:
            assert not np.any((weights > 0) & (valid & ~bounded_span))
        fused_counts.append(num_frames)
        pose_counts.append(pose_frames)
        validated.append((fused_path, pose_path, wilor_path))
        bar.set_postfix_str(
            f"fused={num_frames} pose={pose_frames} pose_fps~{pose_fps:.2f}"
        )
    bar.close()
    print(
        f"[validate] clips={len(validated)} fused_frames={sum(fused_counts):,} "
        f"rtmpose_frames={sum(pose_counts):,}"
        + (" (RTMPose disabled)" if args.no_rtmpose else "")
    )
    if not args.no_rtmpose:
        ratios = np.asarray(pose_counts) / np.asarray(fused_counts)
        print(
            "[validate] pose/fused ratio min/median/max="
            f"{ratios.min():.3f}/{np.median(ratios):.3f}/{ratios.max():.3f}"
        )
    print(
        f"[validate] WiLoR observed={observed_counts.tolist()} "
        f"bounded_interp_used={bounded_counts.tolist()} "
        f"unbounded_used={unbounded_counts.tolist()} "
        f"edge_fill_frames={edge_fill_counts.tolist()}"
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
        input_roots = [("WiLoR", args.wilor_root)]
        if not args.no_rtmpose:
            input_roots.append(("RTMPose", args.rtmpose_root))
        for label, path in input_roots:
            if not path.is_dir():
                raise FileNotFoundError(f"{label} root is not a directory: {path}")
        if args.output_root.resolve() in {
            args.fused_root.resolve(),
            *(path.resolve() for _, path in input_roots),
        }:
            raise ValueError("Output root must differ from all input roots")

        validated = validate_all(clip_paths, args)
        if args.dry_run:
            print(f"[dry-run] PASS: validated {len(validated)} clip(s)")
            return 0

        device = rtmpose.resolve_device(args.device)
        print(f"[device] {device}", flush=True)
        print(
            f"[schedule] stage 1 (position, all joints) iterations 1-"
            f"{args.orient_warmup_iterations}, stage 2 (+orientation, "
            f"{'all joints' if args.no_stage2_freeze_proximal else 'elbows+wrists only'}) "
            f"{args.orient_warmup_iterations + 1}-{args.iterations}; "
            f"WiLoR wrist loss {args.wilor_wrist_loss}, orient loss "
            f"{args.wilor_orient_loss} @ {args.wilor_orient_cutoff_hz:g}Hz, "
            f"clamps wrist {args.max_wrist_delta_degrees:g}deg / elbow "
            f"{args.max_elbow_delta_degrees:g}deg; "
            f"RTMPose wrist with WiLoR x{args.rtmpose_wrist_with_wilor:g}"
            + ("; RTMPose disabled" if args.no_rtmpose else ""),
            flush=True,
        )
        smplx_model = rtmpose.make_smplx_model(args.smplx_model_root, device)
        mano_model = make_mano_model(args.mano_model_path, device)
        processed = 0
        skipped = 0
        bar = tqdm(validated, desc="refine", unit="clip", dynamic_ncols=True)
        for fused_path, pose_path, wilor_path in bar:
            output_path = args.output_root / fused_path.name
            if output_path.is_file() and not args.overwrite:
                skipped += 1
                continue
            metrics = refine_clip(
                fused_path,
                pose_path,
                wilor_path,
                output_path,
                smplx_model,
                mano_model,
                device,
                args,
            )
            processed += 1
            rtmpose_part = (
                ""
                if args.no_rtmpose
                else (
                    f"RTMPose median "
                    f"{metrics['rtmpose_arm_error_before_median_px']:.2f} -> "
                    f"{metrics['rtmpose_arm_error_after_median_px']:.2f}px; "
                )
            )
            tqdm.write(
                f"[done] {fused_path.name} {rtmpose_part}"
                f"WiLoR wrist "
                f"{metrics['wilor_wrist_error_before_median_px']:.2f} -> "
                f"{metrics['wilor_wrist_error_after_median_px']:.2f}px "
                f"(p95 {metrics['wilor_wrist_error_before_p95_px']:.1f} -> "
                f"{metrics['wilor_wrist_error_after_p95_px']:.1f}); "
                f"palm {metrics['wilor_palm_error_before_median']:.4f} -> "
                f"{metrics['wilor_palm_error_after_median']:.4f}; "
                f"palm orient "
                f"{metrics['wilor_palm_orient_error_before_median_deg']:.1f} -> "
                f"{metrics['wilor_palm_orient_error_after_median_deg']:.1f}deg "
                f"(p95 {metrics['wilor_palm_orient_error_before_p95_deg']:.1f} -> "
                f"{metrics['wilor_palm_orient_error_after_p95_deg']:.1f}, "
                f">{ORIENT_BAD_FRAME_DEGREES:g}deg hand-frames "
                f"{int(metrics['wilor_palm_orient_bad_hand_frames_before'])} -> "
                f"{int(metrics['wilor_palm_orient_bad_hand_frames_after'])}); "
                f"collision frames "
                f"{int(metrics['torso_collision_frames_before'])} -> "
                f"{int(metrics['torso_collision_frames_after'])}"
            )
            bar.set_postfix(done=processed, skip=skipped, refresh=False)
        bar.close()
        print(
            f"Combined WiLoR+RTMPose refinement complete: "
            f"processed={processed}, skipped={skipped}"
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
