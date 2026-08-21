#!/usr/bin/env python3
"""Validate and compare matched SpreadsheetBench result files.

Each positional input has the form ``NAME=PATH``.  The script refuses to
compare incomplete runs, duplicated task IDs, or result files with different
task sets.  It emits machine-readable JSON followed by compact Markdown tables.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any


def parse_method(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def load_results(name: str, path: Path, expected_tasks: int) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(rows) != expected_tasks:
        raise ValueError(f"{name}: expected {expected_tasks} rows, found {len(rows)}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}: duplicate task IDs")
    if any(not task_id for task_id in ids):
        raise ValueError(f"{name}: empty task ID")
    for row in rows:
        ok = bool(row["ok"])
        if "hard" in row and bool(row["hard"]) != ok:
            raise ValueError(f"{name}/{row['id']}: hard and ok scores disagree")
        if "soft" in row and bool(row["soft"]) != ok:
            raise ValueError(f"{name}/{row['id']}: soft and ok scores disagree")
    return dict(zip(ids, rows, strict=True))


def exact_paired_p(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def wilson_interval(success: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = success / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return [center - radius, center + radius]


def summarize(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(rows.values())
    success = sum(bool(row["ok"]) for row in values)
    by_type: dict[str, dict[str, int]] = {}
    for row in values:
        task_type = str(row.get("task_type", "unknown"))
        counts = by_type.setdefault(task_type, {"success": 0, "total": 0})
        counts["total"] += 1
        counts["success"] += int(bool(row["ok"]))
    execution_failure = sum(not bool(row["ok"]) and int(row.get("n_exec_pass", 0)) == 0 for row in values)
    semantic_failure = len(values) - success - execution_failure
    return {
        "success": success,
        "total": len(values),
        "rate": success / len(values),
        "wilson_95": wilson_interval(success, len(values)),
        "by_type": by_type,
        "execution_failure": execution_failure,
        "semantic_failure": semantic_failure,
    }


def compare(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> dict[str, int | float]:
    both = sum(bool(a[key]["ok"]) and bool(b[key]["ok"]) for key in a)
    a_only = sum(bool(a[key]["ok"]) and not bool(b[key]["ok"]) for key in a)
    b_only = sum(not bool(a[key]["ok"]) and bool(b[key]["ok"]) for key in a)
    neither = len(a) - both - a_only - b_only
    return {
        "both": both,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
        "delta_tasks": a_only - b_only,
        "exact_two_sided_p": exact_paired_p(a_only, b_only),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("methods", nargs="+", type=parse_method)
    parser.add_argument("--expected-tasks", type=int, default=280)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    loaded = {name: load_results(name, path, args.expected_tasks) for name, path in args.methods}
    if len(loaded) != len(args.methods):
        raise ValueError("method names must be unique")
    reference_ids = set(next(iter(loaded.values())))
    for name, rows in loaded.items():
        if set(rows) != reference_ids:
            missing = sorted(reference_ids - set(rows))
            extra = sorted(set(rows) - reference_ids)
            raise ValueError(f"{name}: mismatched IDs; missing={missing[:5]}, extra={extra[:5]}")

    summaries = {name: summarize(rows) for name, rows in loaded.items()}
    pairwise = {f"{a}__vs__{b}": compare(loaded[a], loaded[b]) for a, b in combinations(loaded, 2)}
    report = {"expected_tasks": args.expected_tasks, "methods": summaries, "pairwise": pairwise}
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)

    print("\n| Method | Success | Wilson 95% CI | Cell | Sheet | Exec fail | Semantic fail |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, item in summaries.items():
        cell = item["by_type"].get("cell_level", {"success": 0, "total": 0})
        sheet = item["by_type"].get("sheet_level", {"success": 0, "total": 0})
        interval = item["wilson_95"]
        print(
            f"| {name} | {item['success']}/{item['total']} ({item['rate']:.2%}) "
            f"| [{interval[0]:.2%}, {interval[1]:.2%}] "
            f"| {cell['success']}/{cell['total']} | {sheet['success']}/{sheet['total']} "
            f"| {item['execution_failure']} | {item['semantic_failure']} |"
        )

    print("\n| Pair | First only | Second only | Delta | Exact paired p |")
    print("|---|---:|---:|---:|---:|")
    for key, item in pairwise.items():
        a, b = key.split("__vs__")
        print(
            f"| {a} vs {b} | {item['a_only']} | {item['b_only']} "
            f"| {item['delta_tasks']:+d} | {item['exact_two_sided_p']:.6g} |"
        )


if __name__ == "__main__":
    main()
