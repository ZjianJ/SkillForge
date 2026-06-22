#!/usr/bin/env python3
"""Compute average number of turns across conversation.json files in a directory.

Each conversation.json is a JSON array; one element = one turn (model action + env feedback).
Success is inferred when any step has reward > 0 (ALFWorld win reward is 10.0).

Usage:
    python scripts/avg_conversation_turns.py outputs/plain_baseline_alfworld
    python scripts/avg_conversation_turns.py outputs/plain_baseline_alfworld --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_conversation(path: Path) -> tuple[int, bool]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array: {path}")
    success = any(float(step.get("reward", 0)) > 0 for step in data if isinstance(step, dict))
    return len(data), success


def summarize_turns(turn_counts: list[int]) -> dict[str, Any]:
    if not turn_counts:
        return {
            "count": 0,
            "avg_turns": None,
            "median_turns": None,
            "min_turns": None,
            "max_turns": None,
        }
    return {
        "count": len(turn_counts),
        "avg_turns": statistics.mean(turn_counts),
        "median_turns": statistics.median(turn_counts),
        "min_turns": min(turn_counts),
        "max_turns": max(turn_counts),
    }


def format_group(label: str, stats: dict[str, Any]) -> None:
    if stats["count"] == 0:
        print(f"{label}: (none)")
        return
    print(f"{label} ({stats['count']} conversations):")
    print(f"  Average turns:   {stats['avg_turns']:.2f}")
    print(f"  Median turns:    {stats['median_turns']:.1f}")
    print(f"  Min / max turns: {stats['min_turns']} / {stats['max_turns']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Average turn count across conversation.json files in a directory."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Root directory to search (e.g. outputs/plain_baseline_alfworld)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON instead of a human-readable summary",
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

    all_turns: list[int] = []
    success_turns: list[int] = []
    failure_turns: list[int] = []
    for path in files:
        n_turns, success = load_conversation(path)
        all_turns.append(n_turns)
        (success_turns if success else failure_turns).append(n_turns)

    overall = summarize_turns(all_turns)
    by_outcome = {
        "success": summarize_turns(success_turns),
        "failure": summarize_turns(failure_turns),
    }
    result = {
        "directory": str(root),
        "overall": overall,
        "by_outcome": by_outcome,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Directory:     {root}")
        print()
        format_group("Overall", overall)
        print()
        format_group("Success", by_outcome["success"])
        print()
        format_group("Failure", by_outcome["failure"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
