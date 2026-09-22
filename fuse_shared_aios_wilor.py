#!/usr/bin/env python3
"""Smooth AIOS person 0 and replace its hands with interpolated WiLoR.

This prototype operates on the packed clip-level NPZ files in ``shared_samples``.
By default, only clip IDs present in both input directories are processed;
unpaired files are reported and skipped.
AIOS arrays are packed over all detected people, so person 0 for frame ``t`` is
stored at ``person_offsets[t]`` rather than necessarily at row ``t``.

AIOS and WiLoR rotations are smoothed in a continuous 6D rotation
representation with a robust Hampel prefilter, zero-phase Butterworth
filtering, and projection back onto SO(3). This avoids directly averaging
wrapped axis-angle values and does not introduce temporal lag. Missing AIOS
person-0 frames are filled with SLERP for rotations and linear interpolation
for Euclidean parameters.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp


DEFAULT_WILOR_ROOT = Path("shared_samples/wilor_params_interpolated")
DEFAULT_AIOS_ROOT = Path("shared_samples/aios_smplx_params")
DEFAULT_OUTPUT_ROOT = Path("shared_samples/aios_smoothed_wilor_hands_fused")
DEFAULT_SMPLX_MODEL_PATH = Path(
    "/home/student/hwu/Workplace/SOKE/"
    "prepare/deps/smpl_models/smplx/SMPLX_NEUTRAL.npz"
)

AIOS_PERSON_ID = 0
AIOS_PARAMETER_SHAPES = {
    "cam_trans": (3,),
    "smplx_root_pose": (3,),
    "smplx_body_pose": (63,),
    "smplx_lhand_pose": (45,),
    "smplx_rhand_pose": (45,),
    "smplx_jaw_pose": (3,),
    "smplx_expr": (10,),
    "smplx_shape": (10,),
    "score": (),
}
MIRROR_X = np.diag([-1.0, 1.0, 1.0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create temporally smoothed AIOS person-0 clips whose hand poses "
            "come from interpolated WiLoR. Process matching clip IDs only; "
            "skip unpaired files."
        )
    )
    parser.add_argument("--wilor-root", type=Path, default=DEFAULT_WILOR_ROOT)
    parser.add_argument("--aios-root", type=Path, default=DEFAULT_AIOS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--smplx-model-path",
        type=Path,
        default=DEFAULT_SMPLX_MODEL_PATH,
        help=(
            "SMPL-X NPZ providing hands_meanl/hands_meanr "
            f"(default: {DEFAULT_SMPLX_MODEL_PATH})"
        ),
    )
    parser.add_argument(
        "--clip-id",
        dest="clip_ids",
        action="append",
        help="Process only this clip; may be repeated.",
    )
    parser.add_argument(
        "--body-cutoff-hz",
        type=float,
        default=6.0,
        help="Cutoff for root/body rotations (default: 6.0 Hz)",
    )
    parser.add_argument(
        "--hand-cutoff-hz",
        type=float,
        default=8.0,
        help="Cutoff for fallback AIOS hand rotations (default: 8.0 Hz)",
    )
    parser.add_argument(
        "--wilor-hand-cutoff-hz",
        type=float,
        default=4.0,
        help=(
            "Cutoff used only with --smooth-wilor-hands "
            "(default: 4.0 Hz)"
        ),
    )
    parser.add_argument(
        "--smooth-wilor-hands",
        action="store_true",
        help=(
            "Low-pass-filter WiLoR local finger rotations. Disabled by "
            "default to preserve WiLoR finger detail."
        ),
    )
    parser.add_argument(
        "--camera-cutoff-hz",
        type=float,
        default=3.0,
        help="Cutoff for --camera-mode smooth (default: 3.0 Hz)",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("fixed-median", "smooth"),
        default="fixed-median",
        help=(
            "Use one robust camera translation for the whole clip, or only "
            "low-pass-filter it (default: fixed-median)."
        ),
    )
    parser.add_argument(
        "--expression-cutoff-hz",
        type=float,
        default=5.0,
        help="Cutoff for jaw and expression (default: 5.0 Hz)",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=4,
        help="Butterworth filter order (default: 4)",
    )
    parser.add_argument(
        "--hampel-window",
        type=int,
        default=7,
        help="Odd temporal window for robust spike rejection (default: 7)",
    )
    parser.add_argument(
        "--hampel-sigmas",
        type=float,
        default=3.0,
        help="Hampel outlier threshold in robust sigmas (default: 3.0)",
    )
    parser.add_argument(
        "--invalid-wilor-policy",
        choices=("fallback-aios", "error"),
        default="fallback-aios",
        help=(
            "What to do when a hand side was never observed by WiLoR "
            "(default: fallback-aios)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing fused NPZ files.",
    )
    args = parser.parse_args()

    for name in (
        "body_cutoff_hz",
        "hand_cutoff_hz",
        "wilor_hand_cutoff_hz",
        "camera_cutoff_hz",
        "expression_cutoff_hz",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.filter_order < 1:
        parser.error("--filter-order must be at least 1")
    if args.hampel_window < 3 or args.hampel_window % 2 == 0:
        parser.error("--hampel-window must be an odd integer of at least 3")
    if args.hampel_sigmas <= 0:
        parser.error("--hampel-sigmas must be positive")
    return args


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {path}")


def load_smplx_hand_means(model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not model_path.is_file():
        raise FileNotFoundError(f"SMPL-X model does not exist: {model_path}")
    with np.load(model_path, allow_pickle=True) as model:
        for key in ("hands_meanl", "hands_meanr"):
            if key not in model:
                raise KeyError(f"{model_path}: missing '{key}'")
        left = np.asarray(model["hands_meanl"], dtype=np.float64).reshape(-1)
        right = np.asarray(model["hands_meanr"], dtype=np.float64).reshape(-1)
    if left.shape != (45,) or right.shape != (45,):
        raise ValueError(
            f"{model_path}: hand means must both have shape (45,), "
            f"got {left.shape} and {right.shape}"
        )
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"{model_path}: hand means contain NaN or Inf")
    return left, right


def validate_clip_sets(
    wilor_root: Path,
    aios_root: Path,
    requested_clip_ids: list[str] | None,
) -> list[str]:
    wilor_ids = {path.stem for path in wilor_root.glob("*.npz")}
    aios_ids = {path.stem for path in aios_root.glob("*.npz")}
    if requested_clip_ids is None:
        only_wilor = sorted(wilor_ids - aios_ids)
        only_aios = sorted(aios_ids - wilor_ids)
        matched_ids = sorted(wilor_ids & aios_ids)
        print(
            f"Clip pairing: matched={len(matched_ids):,}, "
            f"only WiLoR={len(only_wilor):,}, only AIOS={len(only_aios):,}",
            flush=True,
        )
        if only_wilor or only_aios:
            print(
                "Skipping unpaired clips (up to 10 examples per source): "
                f"only WiLoR={only_wilor[:10]}, only AIOS={only_aios[:10]}"
            )
        if not matched_ids:
            raise ValueError("No matching clip IDs found in the input directories")
        return matched_ids

    if len(requested_clip_ids) != len(set(requested_clip_ids)):
        raise ValueError("--clip-id values must not contain duplicates")
    missing_wilor = sorted(set(requested_clip_ids) - wilor_ids)
    missing_aios = sorted(set(requested_clip_ids) - aios_ids)
    if missing_wilor or missing_aios:
        raise FileNotFoundError(
            "Requested clips are missing: "
            f"WiLoR={missing_wilor}, AIOS={missing_aios}"
        )
    return requested_clip_ids


def validate_aios_packing(data: np.lib.npyio.NpzFile, clip_id: str) -> tuple:
    num_frames = int(data["num_frames"])
    counts = np.asarray(data["num_person_per_frame"], dtype=np.int64)
    offsets = np.asarray(data["person_offsets"], dtype=np.int64)
    if counts.shape != (num_frames,):
        raise ValueError(
            f"{clip_id}: num_person_per_frame has shape {counts.shape}, "
            f"expected {(num_frames,)}"
        )
    if offsets.shape != (num_frames + 1,):
        raise ValueError(
            f"{clip_id}: person_offsets has shape {offsets.shape}, "
            f"expected {(num_frames + 1,)}"
        )
    if offsets[0] != 0 or not np.array_equal(np.diff(offsets), counts):
        raise ValueError(f"{clip_id}: person_offsets do not match person counts")
    if np.any(counts < 0):
        raise ValueError(f"{clip_id}: negative AIOS person count")

    total_people = int(offsets[-1])
    for key, trailing_shape in AIOS_PARAMETER_SHAPES.items():
        expected = (total_people, *trailing_shape)
        if data[key].shape != expected:
            raise ValueError(
                f"{clip_id}: {key} has shape {data[key].shape}, "
                f"expected {expected}"
            )
    observed = counts > AIOS_PERSON_ID
    person_indices = offsets[:-1][observed] + AIOS_PERSON_ID
    return num_frames, counts, offsets, observed, person_indices


def select_person_zero(
    packed: np.ndarray,
    num_frames: int,
    observed: np.ndarray,
    person_indices: np.ndarray,
) -> np.ndarray:
    result = np.full(
        (num_frames, *packed.shape[1:]),
        np.nan,
        dtype=np.float64,
    )
    result[observed] = packed[person_indices]
    return result


def fill_missing_linear(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if not observed.any():
        raise ValueError("AIOS person 0 is absent from every frame")
    result = np.asarray(values, dtype=np.float64).copy()
    flat = result.reshape(result.shape[0], -1)
    frame_indices = np.arange(result.shape[0])
    observed_indices = frame_indices[observed]
    for column in range(flat.shape[1]):
        flat[:, column] = np.interp(
            frame_indices,
            observed_indices,
            flat[observed, column],
        )
    return flat.reshape(result.shape)


def fill_missing_rotations(
    rotvecs: np.ndarray,
    observed: np.ndarray,
    num_joints: int,
) -> np.ndarray:
    if not observed.any():
        raise ValueError("AIOS person 0 is absent from every frame")
    values = np.asarray(rotvecs, dtype=np.float64).reshape(-1, num_joints, 3)
    result = values.copy()
    all_indices = np.arange(len(values))
    observed_indices = all_indices[observed]
    for joint_index in range(num_joints):
        if len(observed_indices) == 1:
            result[:, joint_index] = values[observed_indices[0], joint_index]
            continue
        first = int(observed_indices[0])
        last = int(observed_indices[-1])
        result[:first, joint_index] = values[first, joint_index]
        result[last + 1 :, joint_index] = values[last, joint_index]
        slerp = Slerp(
            observed_indices,
            Rotation.from_rotvec(values[observed, joint_index]),
        )
        result[first : last + 1, joint_index] = slerp(
            all_indices[first : last + 1]
        ).as_rotvec()
    return result.reshape(rotvecs.shape)


def hampel_prefilter(
    values: np.ndarray,
    window: int,
    num_sigmas: float,
) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    medians = median_filter(values, size=(window, 1), mode="nearest")
    deviations = np.abs(values - medians)
    mad = median_filter(deviations, size=(window, 1), mode="nearest")
    thresholds = num_sigmas * 1.4826 * mad + 1e-8
    return np.where(deviations > thresholds, medians, values)


def zero_phase_filter(
    values: np.ndarray,
    fps: float,
    cutoff_hz: float,
    order: int,
    hampel_window: int,
    hampel_sigmas: float,
) -> np.ndarray:
    original_shape = values.shape
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    robust = hampel_prefilter(flat, hampel_window, hampel_sigmas)
    if len(values) < 4:
        return robust.reshape(original_shape)
    effective_cutoff = min(cutoff_hz, fps * 0.475)
    sos = butter(order, effective_cutoff, btype="lowpass", fs=fps, output="sos")
    default_padlen = 3 * (2 * len(sos) + 1)
    padlen = min(len(values) - 1, default_padlen)
    filtered = sosfiltfilt(sos, robust, axis=0, padlen=padlen)
    return filtered.reshape(original_shape)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def smooth_rotations(
    rotvecs: np.ndarray,
    observed: np.ndarray,
    num_joints: int,
    fps: float,
    cutoff_hz: float,
    order: int,
    hampel_window: int,
    hampel_sigmas: float,
) -> np.ndarray:
    filled = fill_missing_rotations(rotvecs, observed, num_joints)
    matrices = Rotation.from_rotvec(filled.reshape(-1, 3)).as_matrix()
    matrices = matrices.reshape(len(filled), num_joints, 3, 3)
    rotation_6d = np.concatenate(
        [matrices[..., :, 0], matrices[..., :, 1]],
        axis=-1,
    )
    filtered_6d = zero_phase_filter(
        rotation_6d,
        fps,
        cutoff_hz,
        order,
        hampel_window,
        hampel_sigmas,
    )
    first = normalize(filtered_6d[..., :3])
    second_raw = filtered_6d[..., 3:]
    second = normalize(second_raw - (first * second_raw).sum(-1, keepdims=True) * first)
    third = np.cross(first, second)
    projected = np.stack([first, second, third], axis=-1)
    return (
        Rotation.from_matrix(projected.reshape(-1, 3, 3))
        .as_rotvec()
        .reshape(rotvecs.shape)
        .astype(np.float32)
    )


def smooth_linear(
    values: np.ndarray,
    observed: np.ndarray,
    fps: float,
    cutoff_hz: float,
    order: int,
    hampel_window: int,
    hampel_sigmas: float,
) -> np.ndarray:
    filled = fill_missing_linear(values, observed)
    return zero_phase_filter(
        filled,
        fps,
        cutoff_hz,
        order,
        hampel_window,
        hampel_sigmas,
    ).astype(np.float32)


def wilor_hand_axis_angles(
    wilor: np.lib.npyio.NpzFile,
    clip_id: str,
    left_hand_mean: np.ndarray,
    right_hand_mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrices = np.asarray(wilor["hand_pose_rotmat"], dtype=np.float64)
    num_frames = int(wilor["num_frames"])
    if matrices.shape != (num_frames, 2, 15, 3, 3):
        raise ValueError(
            f"{clip_id}: WiLoR hand_pose_rotmat has shape {matrices.shape}, "
            f"expected {(num_frames, 2, 15, 3, 3)}"
        )
    left_mirrored = MIRROR_X @ matrices[:, 0] @ MIRROR_X
    right = matrices[:, 1]
    left_full_axis_angle = Rotation.from_matrix(
        left_mirrored.reshape(-1, 3, 3)
    ).as_rotvec()
    right_full_axis_angle = Rotation.from_matrix(
        right.reshape(-1, 3, 3)
    ).as_rotvec()
    # SMPL-X is reconstructed with flat_hand_mean=False, so its forward pass
    # adds hands_meanl/hands_meanr. Store residuals here to avoid adding the
    # model hand mean a second time to WiLoR's already-complete rotations.
    left_axis_angle = left_full_axis_angle.reshape(num_frames, 45) - left_hand_mean
    right_axis_angle = (
        right_full_axis_angle.reshape(num_frames, 45) - right_hand_mean
    )
    return (
        left_axis_angle.astype(np.float32),
        right_axis_angle.astype(np.float32),
    )


def smooth_wilor_hand(
    residual_axis_angles: np.ndarray,
    hand_mean: np.ndarray,
    valid: np.ndarray,
    fps: float,
    args: argparse.Namespace,
) -> np.ndarray:
    """Smooth complete WiLoR rotations, then restore SMPL-X residuals."""
    if not valid.any():
        return residual_axis_angles.astype(np.float32)
    full_axis_angles = residual_axis_angles + hand_mean
    smoothed_full = smooth_rotations(
        full_axis_angles,
        valid,
        15,
        fps,
        args.wilor_hand_cutoff_hz,
        args.filter_order,
        args.hampel_window,
        args.hampel_sigmas,
    )
    return smoothed_full - hand_mean.astype(np.float32)


def save_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.",
        suffix=".tmp.npz",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, **payload)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def process_clip(
    wilor_path: Path,
    aios_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    left_hand_mean: np.ndarray,
    right_hand_mean: np.ndarray,
) -> dict[str, int]:
    clip_id = wilor_path.stem
    with np.load(wilor_path, allow_pickle=False) as wilor, np.load(
        aios_path, allow_pickle=False
    ) as aios:
        if str(wilor["clip_id"]) != clip_id or str(aios["clip_id"]) != clip_id:
            raise ValueError(f"{clip_id}: clip_id metadata do not match filename")
        num_frames, counts, _, observed, person_indices = validate_aios_packing(
            aios, clip_id
        )
        if int(wilor["num_frames"]) != num_frames:
            raise ValueError(
                f"{clip_id}: frame mismatch: WiLoR={int(wilor['num_frames'])}, "
                f"AIOS={num_frames}"
            )
        if wilor["frame_names"].shape != (num_frames,):
            raise ValueError(f"{clip_id}: invalid WiLoR frame_names shape")
        if tuple(wilor["hand_slot_order"].tolist()) != ("left", "right"):
            raise ValueError(f"{clip_id}: unexpected WiLoR hand slot order")
        if not observed.any():
            raise ValueError(f"{clip_id}: AIOS person 0 is absent from all frames")

        fps = float(aios["source_fps"])
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{clip_id}: invalid source_fps={fps}")
        selected = {
            key: select_person_zero(aios[key], num_frames, observed, person_indices)
            for key in AIOS_PARAMETER_SHAPES
        }

        smoothed_root = smooth_rotations(
            selected["smplx_root_pose"], observed, 1, fps,
            args.body_cutoff_hz, args.filter_order,
            args.hampel_window, args.hampel_sigmas,
        )
        smoothed_body = smooth_rotations(
            selected["smplx_body_pose"], observed, 21, fps,
            args.body_cutoff_hz, args.filter_order,
            args.hampel_window, args.hampel_sigmas,
        )
        # AIOS stores SMPL-X hand residuals because flat_hand_mean=False.
        # Smooth the actual rotations after adding the model mean, then return
        # to the residual convention used by the output SMPL-X parameters.
        left_full = selected["smplx_lhand_pose"] + left_hand_mean
        right_full = selected["smplx_rhand_pose"] + right_hand_mean
        smoothed_lhand = smooth_rotations(
            left_full, observed, 15, fps,
            args.hand_cutoff_hz, args.filter_order,
            args.hampel_window, args.hampel_sigmas,
        ) - left_hand_mean.astype(np.float32)
        smoothed_rhand = smooth_rotations(
            right_full, observed, 15, fps,
            args.hand_cutoff_hz, args.filter_order,
            args.hampel_window, args.hampel_sigmas,
        ) - right_hand_mean.astype(np.float32)
        smoothed_jaw = smooth_rotations(
            selected["smplx_jaw_pose"], observed, 1, fps,
            args.expression_cutoff_hz, args.filter_order,
            args.hampel_window, args.hampel_sigmas,
        )
        if args.camera_mode == "fixed-median":
            camera_reference = np.median(
                selected["cam_trans"][observed], axis=0
            ).astype(np.float32)
            stabilized_camera = np.broadcast_to(
                camera_reference, (num_frames, 3)
            ).copy()
        else:
            stabilized_camera = smooth_linear(
                selected["cam_trans"], observed, fps,
                args.camera_cutoff_hz, args.filter_order,
                args.hampel_window, args.hampel_sigmas,
            )
            camera_reference = np.median(
                stabilized_camera, axis=0
            ).astype(np.float32)
        smoothed_expression = smooth_linear(
            selected["smplx_expr"], observed, fps,
            args.expression_cutoff_hz, args.filter_order,
            args.hampel_window, args.hampel_sigmas,
        )
        stable_shape = np.median(selected["smplx_shape"][observed], axis=0)
        stable_shape = np.broadcast_to(stable_shape, (num_frames, 10)).astype(
            np.float32
        ).copy()

        wilor_left, wilor_right = wilor_hand_axis_angles(
            wilor,
            clip_id,
            left_hand_mean,
            right_hand_mean,
        )
        valid_mask = np.asarray(wilor["valid_mask"], dtype=bool)
        if valid_mask.shape != (num_frames, 2):
            raise ValueError(f"{clip_id}: invalid WiLoR valid_mask shape")
        if args.invalid_wilor_policy == "error" and not valid_mask.all():
            invalid_sides = np.flatnonzero(~valid_mask.any(axis=0)).tolist()
            raise ValueError(
                f"{clip_id}: WiLoR has invalid hand sides {invalid_sides}"
            )
        if args.smooth_wilor_hands:
            fused_wilor_left = smooth_wilor_hand(
                wilor_left,
                left_hand_mean,
                valid_mask[:, 0],
                fps,
                args,
            )
            fused_wilor_right = smooth_wilor_hand(
                wilor_right,
                right_hand_mean,
                valid_mask[:, 1],
                fps,
                args,
            )
        else:
            fused_wilor_left = wilor_left
            fused_wilor_right = wilor_right
        fused_left = np.where(
            valid_mask[:, 0:1], fused_wilor_left, smoothed_lhand
        )
        fused_right = np.where(
            valid_mask[:, 1:2], fused_wilor_right, smoothed_rhand
        )

        score = np.zeros(num_frames, dtype=np.float32)
        score[observed] = selected["score"][observed].astype(np.float32)
        payload = {
            "schema_version": np.int32(1),
            "fusion_schema_version": np.int32(3),
            "clip_id": np.asarray(clip_id),
            "split": np.asarray(wilor["split"]),
            "source_fps": np.float32(fps),
            "num_frames": np.int32(num_frames),
            "frame_names": np.asarray(wilor["frame_names"]),
            "aios_person_id": np.int32(AIOS_PERSON_ID),
            "num_person_per_frame": np.ones(num_frames, dtype=np.int32),
            "person_offsets": np.arange(num_frames + 1, dtype=np.int32),
            "source_num_person_per_frame": counts.astype(np.int32),
            "aios_person_observed_mask": observed,
            "aios_person_interpolated_mask": ~observed,
            "additional_aios_people_discarded": np.int32(
                np.maximum(counts - 1, 0).sum()
            ),
            "img_shape": np.asarray(aios["img_shape"], dtype=np.int32),
            "score_threshold": np.float32(aios["score_threshold"]),
            "cam_trans": stabilized_camera,
            "smplx_root_pose": smoothed_root,
            "smplx_body_pose": smoothed_body,
            "smplx_lhand_pose": fused_left.astype(np.float32),
            "smplx_rhand_pose": fused_right.astype(np.float32),
            "smplx_jaw_pose": smoothed_jaw,
            "smplx_expr": smoothed_expression,
            "smplx_shape": stable_shape,
            "score": score,
            "wilor_hand_used_mask": valid_mask,
            "wilor_observed_mask": np.asarray(wilor["observed_mask"], dtype=bool),
            "wilor_interpolated_mask": np.asarray(
                wilor["interpolated_mask"], dtype=bool
            ),
            "wilor_detection_confidence": np.asarray(
                wilor["detection_confidence"], dtype=np.float32
            ),
            "hand_source": np.asarray(
                "smoothed_wilor_valid_else_smoothed_aios"
                if args.smooth_wilor_hands
                else "wilor_valid_else_smoothed_aios"
            ),
            "left_hand_conversion": np.asarray("mirror_x_RMR"),
            "smplx_flat_hand_mean": np.bool_(False),
            "smplx_hand_mean_subtracted": np.bool_(True),
            "wilor_hands_smoothed": np.bool_(args.smooth_wilor_hands),
            "camera_mode": np.asarray(args.camera_mode),
            "camera_fixed_clipwise": np.bool_(
                args.camera_mode == "fixed-median"
            ),
            "camera_reference": camera_reference,
            "smoothing_method": np.asarray(
                "hampel_rotation6d_zero_phase_butterworth_aios_and_wilor"
                if args.smooth_wilor_hands
                else "hampel_rotation6d_zero_phase_butterworth_aios"
            ),
            "body_cutoff_hz": np.float32(args.body_cutoff_hz),
            "hand_fallback_cutoff_hz": np.float32(args.hand_cutoff_hz),
            "wilor_hand_cutoff_hz": np.float32(
                args.wilor_hand_cutoff_hz
            ),
            "camera_cutoff_hz": np.float32(args.camera_cutoff_hz),
            "expression_cutoff_hz": np.float32(args.expression_cutoff_hz),
            "filter_order": np.int32(args.filter_order),
            "hampel_window": np.int32(args.hampel_window),
            "hampel_sigmas": np.float32(args.hampel_sigmas),
        }
        for key, value in payload.items():
            array = np.asarray(value)
            if array.dtype.kind in "fc" and not np.isfinite(array).all():
                raise ValueError(f"{clip_id}: non-finite output in {key}")
        save_npz_atomic(output_path, payload)
        return {
            "frames": num_frames,
            "missing_aios_frames": int((~observed).sum()),
            "additional_aios_people": int(np.maximum(counts - 1, 0).sum()),
            "wilor_fallback_slots": int((~valid_mask).sum()),
        }


def main() -> int:
    args = parse_args()
    try:
        require_directory(args.wilor_root, "WiLoR root")
        require_directory(args.aios_root, "AIOS root")
        if args.output_root.resolve() in {
            args.wilor_root.resolve(),
            args.aios_root.resolve(),
        }:
            raise ValueError("Output root must differ from both input roots")
        clip_ids = validate_clip_sets(
            args.wilor_root, args.aios_root, args.clip_ids
        )
        left_hand_mean, right_hand_mean = load_smplx_hand_means(
            args.smplx_model_path
        )

        totals = {
            "processed": 0,
            "skipped": 0,
            "frames": 0,
            "missing_aios_frames": 0,
            "additional_aios_people": 0,
            "wilor_fallback_slots": 0,
        }
        for index, clip_id in enumerate(clip_ids, start=1):
            output_path = args.output_root / f"{clip_id}.npz"
            if output_path.is_file() and not args.overwrite:
                totals["skipped"] += 1
                continue
            result = process_clip(
                args.wilor_root / f"{clip_id}.npz",
                args.aios_root / f"{clip_id}.npz",
                output_path,
                args,
                left_hand_mean,
                right_hand_mean,
            )
            totals["processed"] += 1
            for key, value in result.items():
                totals[key] += value
            if index % 100 == 0 or index == len(clip_ids):
                print(f"[{index}/{len(clip_ids)}] processed {clip_id}", flush=True)

        print("\nFusion complete")
        for key, value in totals.items():
            print(f"  {key}: {value:,}")
        print(f"  output: {args.output_root}")
        return 0
    except (FileNotFoundError, PermissionError, ValueError, KeyError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
