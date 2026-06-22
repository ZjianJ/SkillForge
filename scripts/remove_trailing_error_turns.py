#!/usr/bin/env python3
"""Remove trailing error turns from conversation.json rollout files.

A turn is treated as an error turn when its ``reasoning`` field equals ``"error"``
(the placeholder used when model inference fails). Only contiguous error turns at
the end of each conversation are removed; earlier error turns are kept.

Usage:
    python scripts/remove_trailing_error_turns.py rollouts/teacher_gpt4o_alfworld_rollouts
    python scripts/remove_trailing_error_turns.py rollouts/teacher_gpt4o_alfworld_rollouts --in-place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def is_error_turn(turn: dict[str, Any]) -> bool:
    return turn.get("reasoning") == "error"


def trim_trailing_errors(turns: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    trimmed = list(turns)
    removed = 0
    while trimmed and is_error_turn(trimmed[-1]):
        trimmed.pop()
        removed += 1
    return trimmed, removed


def process_file(path: Path, *, in_place: bool) -> int:
    with path.open() as f:
        turns = json.load(f)
    if not isinstance(turns, list):
        raise ValueError(f"expected a JSON array: {path}")

    trimmed, removed = trim_trailing_errors(turns)
    if removed == 0:
        return 0

    if in_place:
        with path.open("w") as f:
            json.dump(trimmed, f, indent=2)
            f.write("\n")

    print(f"{path}: removed {removed} trailing error turn(s) ({len(turns)} -> {len(trimmed)})")
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove trailing error turns from conversation.json files."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Root directory to search for conversation.json files",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite files on disk (default: dry run, print only)",
    )
    args = parser.parse_args()

    root = args.directory
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1

    files = sorted(root.rglob("conversation.json"))
    if not files:
        print(f"error: no conversation.json files found under {root}", file=sys.stderr)
        return 1

    total_removed = 0
    changed_files = 0
    for path in files:
        try:
            removed = process_file(path, in_place=args.in_place)
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            return 1
        if removed:
            changed_files += 1
            total_removed += removed

    mode = "updated" if args.in_place else "would update"
    print(
        f"\n{changed_files} file(s) {mode}; "
        f"{total_removed} trailing error turn(s) removed in total"
    )
    if not args.in_place and changed_files:
        print("Re-run with --in-place to apply changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
