#!/usr/bin/env python3
"""Dump what is inside one or more .npz files: keys, shapes, dtypes, values.

For every array it reports shape and dtype, plus a summary suited to the kind
of data: scalars print their value, floats print NaN/Inf counts and range,
booleans print how many are True, integers print their range and (when few)
the distinct values, strings print the first few entries.

Usage:
    python inspect_npz.py <file.npz> [<file.npz> ...]
    python inspect_npz.py --rows 3 <file.npz>     # also print the first 3 rows
    python inspect_npz.py --sort <file.npz>       # keys alphabetically

Filenames containing ':' need quoting in the shell:
    python inspect_npz.py '/data/hwu/.../clip-00:00:07.440-00:00:09.199.npz'
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{num:.0f} B"
        num /= 1024
    return f"{num:.1f} GB"


def truncate(text: str, limit: int = 110) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def describe(array: np.ndarray) -> str:
    """One-line summary of the values in an array."""
    if array.dtype == object:
        flat = array.reshape(-1)
        if array.ndim == 0:
            return f"object -> {truncate(repr(array.item()))}"
        return f"object, first={truncate(repr(flat[0]))}" if flat.size else "object, empty"

    kind = array.dtype.kind

    if array.ndim == 0:
        return f"= {truncate(repr(array.item()))}"

    if array.size == 0:
        return "empty"

    if kind in "US":
        flat = array.reshape(-1)
        shown = ", ".join(truncate(repr(v), 40) for v in flat[:3])
        more = f", ... (+{flat.size - 3})" if flat.size > 3 else ""
        return f"{shown}{more}"

    if kind == "b":
        true_count = int(array.sum())
        pct = 100.0 * true_count / array.size
        return f"True {true_count:,}/{array.size:,} ({pct:.1f}%)"

    if kind in "iu":
        uniques = np.unique(array)
        if uniques.size <= 8:
            return f"min={array.min()} max={array.max()} unique={uniques.tolist()}"
        return f"min={array.min()} max={array.max()} unique={uniques.size:,}"

    if kind in "fc":
        finite = np.isfinite(array)
        n_nan = int(np.isnan(array).sum())
        n_inf = int(np.isinf(array).sum())
        flags = []
        if n_nan:
            flags.append(f"NaN={n_nan:,}")
        if n_inf:
            flags.append(f"Inf={n_inf:,}")
        if not finite.any():
            return "all non-finite " + " ".join(flags)
        good = array[finite]
        summary = (
            f"min={good.min():.4g} max={good.max():.4g} mean={good.mean():.4g}"
        )
        return f"{summary} " + " ".join(flags) if flags else summary

    return f"dtype kind '{kind}' not summarized"


def inspect(path: str, rows: int, sort_keys: bool) -> int:
    if not os.path.isfile(path):
        print(f"ERROR: not a file: {path}", file=sys.stderr)
        return 1

    print("=" * 100)
    print(f"file : {path}")
    print(f"size : {human_bytes(os.path.getsize(path))}")

    try:
        handle = np.load(path, allow_pickle=True)
    except Exception as error:  # noqa: BLE001 - report whatever numpy raises
        print(f"ERROR: cannot load: {error}", file=sys.stderr)
        return 1

    with handle:
        keys = sorted(handle.files) if sort_keys else list(handle.files)
        print(f"keys : {len(keys)}")
        print("-" * 100)
        width = max((len(k) for k in keys), default=0)

        for key in keys:
            try:
                array = np.asanyarray(handle[key])
            except Exception as error:  # noqa: BLE001
                print(f"  {key:<{width}}  <unreadable: {error}>")
                continue

            shape = "()" if array.ndim == 0 else str(array.shape)
            print(
                f"  {key:<{width}}  {shape:<18} {str(array.dtype):<10} "
                f"{describe(array)}"
            )

            if rows > 0 and array.ndim >= 1 and array.dtype != object:
                with np.printoptions(
                    precision=4, suppress=True, threshold=200, linewidth=90
                ):
                    preview = np.array2string(array[:rows])
                for line in preview.splitlines():
                    print(f"  {'':<{width}}    {line}")

    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump the keys, shapes, dtypes and value ranges of .npz files.",
    )
    parser.add_argument("paths", nargs="+", help="one or more .npz files")
    parser.add_argument(
        "--rows",
        type=int,
        default=0,
        help="also print the first N rows of every array (default: 0)",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="list keys alphabetically instead of in archive order",
    )
    args = parser.parse_args()

    if args.rows < 0:
        parser.error("--rows must not be negative")

    failures = sum(inspect(p, args.rows, args.sort) for p in args.paths)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
