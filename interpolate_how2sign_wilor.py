#!/usr/bin/env python3
"""Convert raw WiLoR frame PKLs into fixed left/right, temporally filled PKLs.

The raw extractor intentionally stores every detector result, so the first
dimension can be 0, 1, 2, or larger.  This postprocessor creates a separate,
non-destructive output tree where every frame has two stable hand slots:

    index 0: left hand
    index 1: right hand

For a missing side, rotations are interpolated with quaternion SLERP and other
continuous parameters are linearly interpolated between the nearest observed
frames.  Missing values before the first or after the last observation use the
nearest observed frame.  If one side is never observed in a clip, that slot is
filled with neutral finite values and marked invalid.
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
from scipy.spatial.transform import Rotation


DEFAULT_DATASET_ROOT = Path("/data/hwu/how2sign")
DEFAULT_SPLITS = ("test", "val", "train")
SIDE_NAMES = ("left", "right")

ROTATION_FIELDS = {
    "global_orient_rotmat": (1, 3, 3),
    "hand_pose_rotmat": (15, 3, 3),
}
LINEAR_FIELDS = {
    "betas": (10,),
    "pred_cam": (3,),
    "cam_t_full": (3,),
    "bbox_xyxy": (4,),
    "box_center": (2,),
    "box_size": (),
    "image_size": (2,),
    "focal_length": (),
}
REPLACED_FIELDS = {
    "num_hands",
    "hand_pose",
    "global_orient",
    "mano_pose",
    "global_orient_rotmat",
    "hand_pose_rotmat",
    "global_orient_axis_angle",
    "hand_pose_axis_angle",
    "mano_pose_axis_angle",
    "betas",
    "pred_cam",
    "cam_t_full",
    "bbox_xyxy",
    "detection_confidence",
    "is_right",
    "box_center",
    "box_size",
    "image_size",
    "focal_length",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create fixed [left, right] WiLoR PKLs with temporal interpolation."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Directory containing how2sign_images_<split>_wilor_out.",
    )
    parser.add_argument(
        "--split-output-root",
        type=Path,
        default=None,
        help=(
            "Process one output directory directly, for example "
            "/data/hwu/how2sign/how2sign_images_test_wilor_out. "
            "When set, --dataset-root and --splits are not used."
        ),
    )
    parser.add_argument(
        "--split-name",
        default=None,
        help=(
            "Metadata/sharding name for a generic --split-output-root, "
            "for example 'youtube'. If omitted, How2Sign names are inferred."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=list(DEFAULT_SPLITS),
        help="Splits to process in order.",
    )
    parser.add_argument(
        "--input-subdir",
        default="wilor_params",
        help="Raw parameter directory inside each split output root.",
    )
    parser.add_argument(
        "--output-subdir",
        default="wilor_params_interpolated",
        help="Separate destination directory inside each split output root.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of independent CPU workers sharing the clip set.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="This worker's zero-based shard index.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Process at most this many new clips (smoke testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute frame PKLs even when the destination clip is complete.",
    )
    args = parser.parse_args()

    if args.input_subdir == args.output_subdir:
        parser.error("--input-subdir and --output-subdir must be different")
    if args.num_shards < 1:
        parser.error("--num-shards must be at least 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, num-shards)")
    if args.max_clips is not None and args.max_clips < 1:
        parser.error("--max-clips must be at least 1")
    return args


def infer_split_from_output_root(split_output_root: Path) -> str:
    for split in ("test", "val", "train"):
        if split_output_root.name == f"how2sign_images_{split}_wilor_out":
            return split
    raise ValueError(
        "Could not infer split from --split-output-root name "
        f"{split_output_root.name!r}; expected "
        "how2sign_images_test_wilor_out, "
        "how2sign_images_val_wilor_out, or "
        "how2sign_images_train_wilor_out"
    )


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


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(payload).__name__}")
    return payload


def save_pickle_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp.pkl",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def validate_first_dimension(
    frame: dict,
    field: str,
    expected_hands: int,
    frame_path: Path,
) -> np.ndarray:
    if field not in frame:
        raise KeyError(f"{frame_path}: missing field {field!r}")
    value = np.asarray(frame[field], dtype=np.float32)
    if value.shape[0] != expected_hands:
        raise ValueError(
            f"{frame_path}: {field} has first dimension {value.shape[0]}, "
            f"expected {expected_hands}"
        )
    return value


def select_detections(
    frames: list[dict],
    frame_paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select the highest-confidence detection independently for each side."""
    num_frames = len(frames)
    selected = np.full((num_frames, 2), -1, dtype=np.int32)
    observed = np.zeros((num_frames, 2), dtype=bool)
    selected_confidence = np.zeros((num_frames, 2), dtype=np.float32)
    discarded_duplicates = np.zeros((num_frames, 2), dtype=np.int32)

    for frame_index, (frame, frame_path) in enumerate(zip(frames, frame_paths)):
        num_hands = int(frame.get("num_hands", 0))
        sides = validate_first_dimension(
            frame,
            "is_right",
            num_hands,
            frame_path,
        ).reshape(-1)
        confidence = validate_first_dimension(
            frame,
            "detection_confidence",
            num_hands,
            frame_path,
        ).reshape(-1)

        for side_index in range(2):
            candidates = np.flatnonzero(
                (sides >= 0.5) if side_index == 1 else (sides < 0.5)
            )
            if candidates.size == 0:
                continue
            best = int(candidates[np.argmax(confidence[candidates])])
            selected[frame_index, side_index] = best
            observed[frame_index, side_index] = True
            selected_confidence[frame_index, side_index] = confidence[best]
            discarded_duplicates[frame_index, side_index] = candidates.size - 1

    return selected, observed, selected_confidence, discarded_duplicates


def build_observed_values(
    frames: list[dict],
    frame_paths: list[Path],
    selected: np.ndarray,
    observed: np.ndarray,
    field: str,
    item_shape: tuple[int, ...],
) -> np.ndarray:
    values = np.zeros(
        (len(frames), 2, *item_shape),
        dtype=np.float32,
    )
    for frame_index, (frame, frame_path) in enumerate(zip(frames, frame_paths)):
        num_hands = int(frame.get("num_hands", 0))
        source = validate_first_dimension(
            frame,
            field,
            num_hands,
            frame_path,
        )
        expected_shape = (num_hands, *item_shape)
        if source.shape != expected_shape:
            raise ValueError(
                f"{frame_path}: {field} has shape {source.shape}, "
                f"expected {expected_shape}"
            )
        for side_index in range(2):
            if observed[frame_index, side_index]:
                values[frame_index, side_index] = source[
                    selected[frame_index, side_index]
                ]
    return values


def interpolation_plan(
    observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return previous/next observed frame, alpha, and clip-level side validity."""
    num_frames = observed.shape[0]
    previous = np.full((num_frames, 2), -1, dtype=np.int32)
    following = np.full((num_frames, 2), -1, dtype=np.int32)
    alpha = np.zeros((num_frames, 2), dtype=np.float32)
    valid_side = np.zeros(2, dtype=bool)

    frame_indices = np.arange(num_frames, dtype=np.int32)
    for side_index in range(2):
        known = np.flatnonzero(observed[:, side_index]).astype(np.int32)
        if known.size == 0:
            continue
        valid_side[side_index] = True

        insertion = np.searchsorted(known, frame_indices, side="left")
        previous_pos = np.clip(insertion - 1, 0, known.size - 1)
        following_pos = np.clip(insertion, 0, known.size - 1)
        previous[:, side_index] = known[previous_pos]
        following[:, side_index] = known[following_pos]

        exact = observed[:, side_index]
        previous[exact, side_index] = frame_indices[exact]
        following[exact, side_index] = frame_indices[exact]

        denominator = following[:, side_index] - previous[:, side_index]
        between = denominator > 0
        alpha[between, side_index] = (
            (frame_indices[between] - previous[between, side_index])
            / denominator[between]
        )

    return previous, following, alpha, valid_side


def fill_linear(
    observed_values: np.ndarray,
    previous: np.ndarray,
    following: np.ndarray,
    alpha: np.ndarray,
    valid_side: np.ndarray,
) -> np.ndarray:
    filled = np.zeros_like(observed_values, dtype=np.float32)
    expand_dims = (1,) * (observed_values.ndim - 2)

    for side_index in range(2):
        if not valid_side[side_index]:
            continue
        prev_values = observed_values[previous[:, side_index], side_index]
        next_values = observed_values[following[:, side_index], side_index]
        side_alpha = alpha[:, side_index].reshape(-1, *expand_dims)
        filled[:, side_index] = (
            (1.0 - side_alpha) * prev_values + side_alpha * next_values
        )
    return filled.astype(np.float32, copy=False)


def quaternion_slerp(
    start: np.ndarray,
    end: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Vectorized SLERP for J quaternions over K interpolation positions."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64).copy()
    alpha = np.asarray(alpha, dtype=np.float64)

    dot = np.sum(start * end, axis=-1)
    negative = dot < 0.0
    end[negative] *= -1.0
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)

    result = np.empty((alpha.size, start.shape[0], 4), dtype=np.float64)
    nearly_parallel = dot > 0.9995

    if np.any(nearly_parallel):
        linear = (
            (1.0 - alpha[:, None, None]) * start[None, :, :]
            + alpha[:, None, None] * end[None, :, :]
        )
        norm = np.linalg.norm(linear, axis=-1, keepdims=True)
        result[:, nearly_parallel, :] = (
            linear[:, nearly_parallel, :]
            / np.maximum(norm[:, nearly_parallel, :], 1e-12)
        )

    spherical = ~nearly_parallel
    if np.any(spherical):
        theta = np.arccos(dot[spherical])
        sin_theta = np.sin(theta)
        weight_start = np.sin(
            (1.0 - alpha[:, None]) * theta[None, :]
        ) / sin_theta[None, :]
        weight_end = np.sin(
            alpha[:, None] * theta[None, :]
        ) / sin_theta[None, :]
        result[:, spherical, :] = (
            weight_start[:, :, None] * start[None, spherical, :]
            + weight_end[:, :, None] * end[None, spherical, :]
        )

    return result


def fill_rotations(
    observed_values: np.ndarray,
    observed: np.ndarray,
    valid_side: np.ndarray,
) -> np.ndarray:
    """Fill [F,2,J,3,3] rotation matrices with piecewise quaternion SLERP."""
    num_frames, _, num_joints = observed_values.shape[:3]
    identity = np.eye(3, dtype=np.float32)
    filled = np.broadcast_to(
        identity,
        (num_frames, 2, num_joints, 3, 3),
    ).copy()

    for side_index in range(2):
        if not valid_side[side_index]:
            continue
        known = np.flatnonzero(observed[:, side_index])
        filled[known, side_index] = observed_values[known, side_index]

        first = int(known[0])
        last = int(known[-1])
        filled[:first, side_index] = observed_values[first, side_index]
        filled[last + 1 :, side_index] = observed_values[last, side_index]

        for start_index, end_index in zip(known[:-1], known[1:]):
            start_index = int(start_index)
            end_index = int(end_index)
            if end_index == start_index + 1:
                continue

            frame_indices = np.arange(start_index + 1, end_index)
            interval_alpha = (
                (frame_indices - start_index) / (end_index - start_index)
            )
            start_quat = Rotation.from_matrix(
                observed_values[start_index, side_index]
            ).as_quat()
            end_quat = Rotation.from_matrix(
                observed_values[end_index, side_index]
            ).as_quat()
            interpolated_quat = quaternion_slerp(
                start_quat,
                end_quat,
                interval_alpha,
            )
            interpolated_matrix = Rotation.from_quat(
                interpolated_quat.reshape(-1, 4)
            ).as_matrix()
            filled[frame_indices, side_index] = interpolated_matrix.reshape(
                len(frame_indices),
                num_joints,
                3,
                3,
            )

    return filled.astype(np.float32, copy=False)


def rotation_matrices_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    flat = rotmat.reshape(-1, 3, 3).astype(np.float64, copy=False)
    rotvec = Rotation.from_matrix(flat).as_rotvec().astype(np.float32)
    return rotvec.reshape(*rotmat.shape[:-2], 3)


def process_clip(
    *,
    split: str,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, int]:
    frame_paths = sorted(input_dir.glob("*.pkl"))
    if not frame_paths:
        raise ValueError(f"No PKL frames in completed clip: {input_dir}")
    frames = [load_pickle(path) for path in frame_paths]

    selected, observed, confidence, discarded = select_detections(
        frames,
        frame_paths,
    )
    previous, following, alpha, valid_side = interpolation_plan(observed)

    linear_outputs = {}
    for field, item_shape in LINEAR_FIELDS.items():
        observed_values = build_observed_values(
            frames,
            frame_paths,
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
            frames,
            frame_paths,
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

    global_axis_angle = rotation_matrices_to_axis_angle(
        rotation_outputs["global_orient_rotmat"]
    ).reshape(len(frames), 2, 3)
    hand_axis_angle = rotation_matrices_to_axis_angle(
        rotation_outputs["hand_pose_rotmat"]
    ).reshape(len(frames), 2, 45)
    mano_axis_angle = np.concatenate(
        [global_axis_angle, hand_axis_angle],
        axis=2,
    )

    valid_mask = np.broadcast_to(valid_side, observed.shape).copy()
    interpolated_mask = valid_mask & ~observed
    output_confidence = confidence.copy()
    output_confidence[interpolated_mask] = 0.0
    source_num_hands = np.asarray(
        [int(frame.get("num_hands", 0)) for frame in frames],
        dtype=np.int32,
    )

    written = 0
    for frame_index, (frame_path, raw_frame) in enumerate(
        zip(frame_paths, frames)
    ):
        output_path = output_dir / frame_path.name
        if output_path.is_file() and not overwrite:
            continue

        payload = {
            key: value
            for key, value in raw_frame.items()
            if key not in REPLACED_FIELDS
        }
        payload.update(
            {
                "schema_version": 2,
                "interpolation_schema_version": 1,
                "split": split,
                "num_hands": 2,
                "num_observed_hands": int(observed[frame_index].sum()),
                "num_valid_hands": int(valid_mask[frame_index].sum()),
                "source_num_hands": int(source_num_hands[frame_index]),
                "hand_slot_order": SIDE_NAMES,
                "is_right": np.asarray([0.0, 1.0], dtype=np.float32),
                "observed_mask": observed[frame_index].copy(),
                "interpolated_mask": interpolated_mask[frame_index].copy(),
                "valid_mask": valid_mask[frame_index].copy(),
                "selected_detection_index": selected[frame_index].copy(),
                "discarded_duplicate_count": discarded[frame_index].copy(),
                "interpolation_prev_frame": previous[frame_index].copy(),
                "interpolation_next_frame": following[frame_index].copy(),
                "interpolation_alpha": alpha[frame_index].copy(),
                "interpolation_method": "rotation_slerp_linear_values",
                "edge_fill_mode": "nearest",
                "global_orient_rotmat": rotation_outputs[
                    "global_orient_rotmat"
                ][frame_index],
                "hand_pose_rotmat": rotation_outputs["hand_pose_rotmat"][
                    frame_index
                ],
                "global_orient": global_axis_angle[frame_index],
                "hand_pose": hand_axis_angle[frame_index],
                "mano_pose": mano_axis_angle[frame_index],
                "global_orient_axis_angle": global_axis_angle[frame_index],
                "hand_pose_axis_angle": hand_axis_angle[frame_index],
                "mano_pose_axis_angle": mano_axis_angle[frame_index],
                "detection_confidence": output_confidence[frame_index],
            }
        )
        for field, values in linear_outputs.items():
            payload[field] = values[frame_index]

        save_pickle_atomic(output_path, payload)
        written += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".complete").touch()
    return {
        "frames": len(frames),
        "written": written,
        "observed": int(observed.sum()),
        "interpolated": int(interpolated_mask.sum()),
        "missing_frames": int(interpolated_mask.any(axis=1).sum()),
        "invalid": int((~valid_mask).sum()),
        "duplicates": int(discarded.sum()),
    }


def main() -> int:
    args = parse_args()
    processed_clips = 0
    skipped_complete = 0
    skipped_incomplete = 0
    failed_clips = 0
    total_frames = 0
    total_observed = 0
    total_interpolated = 0
    total_missing_frames = 0
    total_invalid = 0
    started_at = time.monotonic()

    log(
        f"interpolation shard={args.shard_index}/{args.num_shards}, "
        f"input={args.input_subdir}, output={args.output_subdir}"
    )

    if args.split_output_root is not None:
        split_roots = [
            (
                args.split_name
                or infer_split_from_output_root(args.split_output_root),
                args.split_output_root,
            )
        ]
    else:
        split_roots = [
            (
                split,
                args.dataset_root / f"how2sign_images_{split}_wilor_out",
            )
            for split in args.splits
        ]

    for split, split_root in split_roots:
        input_root = split_root / args.input_subdir
        output_root = split_root / args.output_subdir
        if not input_root.is_dir():
            raise FileNotFoundError(
                f"Input parameter root does not exist: {input_root}"
            )

        clip_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
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
            f"split={split}: {len(assigned)}/{len(clip_dirs)} raw clips assigned; "
            f"output={output_root}"
        )

        for input_dir in assigned:
            if args.max_clips is not None and processed_clips >= args.max_clips:
                log("Reached --max-clips; stopping cleanly")
                return 1 if failed_clips else 0

            if not (input_dir / ".complete").is_file():
                skipped_incomplete += 1
                continue

            output_dir = output_root / input_dir.name
            if (output_dir / ".complete").is_file() and not args.overwrite:
                skipped_complete += 1
                continue

            clip_started_at = time.monotonic()
            try:
                stats = process_clip(
                    split=split,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    overwrite=args.overwrite,
                )
            except Exception:
                failed_clips += 1
                log(f"FAIL {split}/{input_dir.name}\n{traceback.format_exc()}")
                continue

            processed_clips += 1
            total_frames += stats["frames"]
            total_observed += stats["observed"]
            total_interpolated += stats["interpolated"]
            total_missing_frames += stats["missing_frames"]
            total_invalid += stats["invalid"]
            elapsed = time.monotonic() - clip_started_at
            log(
                f"DONE {split}/{input_dir.name}: frames={stats['frames']}, "
                f"missing_frames={stats['missing_frames']}, "
                f"interpolated_slots={stats['interpolated']}, "
                f"invalid_slots={stats['invalid']}, "
                f"duplicates={stats['duplicates']}, "
                f"processing_fps="
                f"{stats['frames'] / max(elapsed, 1e-6):.2f}"
            )

    elapsed = time.monotonic() - started_at
    log(
        f"COMPLETE clips={processed_clips}, frames={total_frames}, "
        f"missing_frames={total_missing_frames}, "
        f"observed_slots={total_observed}, "
        f"interpolated_slots={total_interpolated}, "
        f"invalid_slots={total_invalid}, failed={failed_clips}, "
        f"skipped_complete={skipped_complete}, "
        f"skipped_incomplete={skipped_incomplete}, seconds={elapsed:.1f}"
    )
    return 1 if failed_clips else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted")
        raise SystemExit(130)
