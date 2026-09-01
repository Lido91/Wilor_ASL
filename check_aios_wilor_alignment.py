#!/usr/bin/env python3
"""Check frame-level alignment between source images, WiLoR, and AIOS.

The default paths target the How2Sign test outputs used in this repository.
AIOS files are matched by frame ID for person 0 only, so additional people do
not incorrectly appear as additional video frames.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_FRAMES_ROOT = Path("/data/hwu/how2sign/how2sign_images_test")
DEFAULT_WILOR_ROOT = Path(
    "/data/hwu/how2sign/"
    "how2sign_images_test_wilor_out/wilor_params_interpolated"
)
DEFAULT_AIOS_ROOT = Path(
    "/data/hwu/how2sign/how2sign_images_test_out/smplx_params"
)

FRAME_RE = re.compile(r"^frame_(\d+)$")
WILOR_RE = re.compile(r"^(frame_(\d+))\.pkl$")
AIOS_RE = re.compile(r"^(frame_(\d+))_person_(\d+)\.pkl$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
AIOS_PERSON_ID = 0


@dataclass
class Inventory:
    frames: set[str] = field(default_factory=set)
    invalid_names: list[str] = field(default_factory=list)
    additional_person_files: int = 0
    additional_person_ids: set[int] = field(default_factory=set)
    complete: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that every source frame has one interpolated WiLoR PKL "
            "and one AIOS PKL for the selected person."
        )
    )
    parser.add_argument(
        "--frames-root",
        type=Path,
        default=DEFAULT_FRAMES_ROOT,
        help=f"Source frame directory (default: {DEFAULT_FRAMES_ROOT})",
    )
    parser.add_argument(
        "--wilor-root",
        type=Path,
        default=DEFAULT_WILOR_ROOT,
        help=f"Interpolated WiLoR root (default: {DEFAULT_WILOR_ROOT})",
    )
    parser.add_argument(
        "--aios-root",
        type=Path,
        default=DEFAULT_AIOS_ROOT,
        help=f"AIOS SMPL-X root (default: {DEFAULT_AIOS_ROOT})",
    )
    parser.add_argument(
        "--no-source-check",
        action="store_true",
        help=(
            "Compare only WiLoR and AIOS. This cannot detect a frame missing "
            "from both outputs."
        ),
    )
    parser.add_argument(
        "--allow-missing-complete",
        action="store_true",
        help="Do not fail when a WiLoR clip lacks its .complete marker.",
    )
    parser.add_argument(
        "--max-details",
        type=int,
        default=20,
        help="Maximum clip/frame names shown per issue category (default: 20)",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        help="Optionally write the complete machine-readable report as JSON.",
    )
    args = parser.parse_args()
    if args.max_details < 1:
        parser.error("--max-details must be at least 1")
    return args


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")


def clip_names(root: Path) -> set[str]:
    return {path.name for path in root.iterdir() if path.is_dir()}


def scan_source_clip(clip_dir: Path) -> Inventory:
    result = Inventory()
    for path in clip_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if FRAME_RE.fullmatch(path.stem):
            result.frames.add(path.stem)
        else:
            result.invalid_names.append(path.name)
    return result


def scan_wilor_clip(clip_dir: Path) -> Inventory:
    result = Inventory(complete=(clip_dir / ".complete").is_file())
    for path in clip_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".pkl":
            continue
        match = WILOR_RE.fullmatch(path.name)
        if match is None:
            result.invalid_names.append(path.name)
        else:
            result.frames.add(match.group(1))
    return result


def scan_aios_clip(clip_dir: Path, person_id: int) -> Inventory:
    result = Inventory()
    for path in clip_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".pkl":
            continue
        match = AIOS_RE.fullmatch(path.name)
        if match is None:
            result.invalid_names.append(path.name)
            continue
        file_person_id = int(match.group(3))
        if file_person_id == person_id:
            result.frames.add(match.group(1))
        else:
            result.additional_person_files += 1
            result.additional_person_ids.add(file_person_id)
    return result


def frame_sort_key(frame: str) -> tuple[int, str]:
    match = FRAME_RE.fullmatch(frame)
    return (int(match.group(1)) if match else sys.maxsize, frame)


def add_issue(
    report: dict,
    category: str,
    clip_id: str,
    values: set[str] | list[str],
) -> None:
    if not values:
        return
    report["issues"].setdefault(category, {})[clip_id] = sorted(
        values, key=frame_sort_key
    )


def compare(args: argparse.Namespace) -> dict:
    require_directory(args.wilor_root, "WiLoR root")
    require_directory(args.aios_root, "AIOS root")
    if not args.no_source_check:
        require_directory(args.frames_root, "Source frame root")

    source_clips = set() if args.no_source_check else clip_names(args.frames_root)
    wilor_clips = clip_names(args.wilor_root)
    aios_clips = clip_names(args.aios_root)
    expected_clips = (
        wilor_clips | aios_clips if args.no_source_check else source_clips
    )

    report = {
        "paths": {
            "frames_root": None if args.no_source_check else str(args.frames_root),
            "wilor_root": str(args.wilor_root),
            "aios_root": str(args.aios_root),
        },
        "person_id": AIOS_PERSON_ID,
        "clip_counts": {
            "source": None if args.no_source_check else len(source_clips),
            "wilor": len(wilor_clips),
            "aios": len(aios_clips),
        },
        "frame_counts": {"source": 0, "wilor": 0, "aios": 0},
        "additional_aios_person_files": 0,
        "additional_aios_person_ids": [],
        "issues": {},
        "passed": False,
    }

    if not args.no_source_check:
        add_issue(
            report,
            "clips_missing_from_wilor",
            "<root>",
            source_clips - wilor_clips,
        )
        add_issue(
            report,
            "clips_missing_from_aios",
            "<root>",
            source_clips - aios_clips,
        )
        add_issue(report, "extra_wilor_clips", "<root>", wilor_clips - source_clips)
        add_issue(report, "extra_aios_clips", "<root>", aios_clips - source_clips)
    else:
        add_issue(report, "clips_missing_from_wilor", "<root>", aios_clips - wilor_clips)
        add_issue(report, "clips_missing_from_aios", "<root>", wilor_clips - aios_clips)

    all_additional_person_ids: set[int] = set()
    for clip_id in sorted(expected_clips):
        source = (
            Inventory()
            if args.no_source_check or clip_id not in source_clips
            else scan_source_clip(args.frames_root / clip_id)
        )
        wilor = (
            Inventory()
            if clip_id not in wilor_clips
            else scan_wilor_clip(args.wilor_root / clip_id)
        )
        aios = (
            Inventory()
            if clip_id not in aios_clips
            else scan_aios_clip(args.aios_root / clip_id, AIOS_PERSON_ID)
        )

        report["frame_counts"]["source"] += len(source.frames)
        report["frame_counts"]["wilor"] += len(wilor.frames)
        report["frame_counts"]["aios"] += len(aios.frames)
        report["additional_aios_person_files"] += aios.additional_person_files
        all_additional_person_ids.update(aios.additional_person_ids)

        expected_frames = (
            wilor.frames | aios.frames if args.no_source_check else source.frames
        )
        add_issue(
            report,
            "frames_missing_from_wilor",
            clip_id,
            expected_frames - wilor.frames,
        )
        add_issue(
            report,
            "frames_missing_from_aios",
            clip_id,
            expected_frames - aios.frames,
        )
        add_issue(
            report,
            "extra_wilor_frames",
            clip_id,
            wilor.frames - expected_frames,
        )
        add_issue(
            report,
            "extra_aios_frames",
            clip_id,
            aios.frames - expected_frames,
        )
        add_issue(report, "invalid_source_names", clip_id, source.invalid_names)
        add_issue(report, "invalid_wilor_names", clip_id, wilor.invalid_names)
        add_issue(report, "invalid_aios_names", clip_id, aios.invalid_names)

        if (
            clip_id in wilor_clips
            and not wilor.complete
            and not args.allow_missing_complete
        ):
            report["issues"].setdefault("wilor_clips_without_complete", {})[
                clip_id
            ] = [".complete"]

    report["additional_aios_person_ids"] = sorted(all_additional_person_ids)
    report["passed"] = not report["issues"]
    return report


def print_limited(values: list[str], limit: int) -> str:
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f" ... (+{len(values) - limit} more)"
    return ", ".join(shown) + suffix


def print_report(report: dict, max_details: int) -> None:
    print("Alignment inputs")
    for label, path in report["paths"].items():
        if path is not None:
            print(f"  {label}: {path}")
    print(f"  AIOS person ID: {report['person_id']}")

    print("\nCounts")
    for key in ("source", "wilor", "aios"):
        clip_count = report["clip_counts"][key]
        frame_count = report["frame_counts"][key]
        if clip_count is not None:
            print(f"  {key:6s}: {clip_count:,} clips, {frame_count:,} frames")
    print(
        "  ignored AIOS files for other people: "
        f"{report['additional_aios_person_files']:,}"
    )
    if report["additional_aios_person_ids"]:
        ids = ", ".join(map(str, report["additional_aios_person_ids"]))
        print(f"  other AIOS person IDs: {ids}")

    if report["passed"]:
        print("\nPASS: every expected frame has both WiLoR and AIOS person output.")
        return

    print("\nFAIL: alignment problems were found.")
    for category, clips in report["issues"].items():
        total = sum(len(values) for values in clips.values())
        print(f"\n  {category}: {total:,} item(s) across {len(clips):,} clip(s)")
        for clip_id, values in list(clips.items())[:max_details]:
            print(f"    {clip_id}: {print_limited(values, max_details)}")
        if len(clips) > max_details:
            print(f"    ... (+{len(clips) - max_details} more clips)")


def main() -> int:
    args = parse_args()
    try:
        report = compare(args)
    except (FileNotFoundError, PermissionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print_report(report, args.max_details)
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        with args.json_report.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"\nJSON report: {args.json_report}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
