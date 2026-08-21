#!/usr/bin/env python3
"""Summarize the frozen Val40 runs for the four-signal locator study."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = {
    "A0 G+JS": ROOT
    / "outputs/SpreadsheetBench_meta_locator_a0_fixed_g_js_full128_seed1_val40/valid_seen/results.jsonl",
    "A1 fixed four-additive": ROOT
    / "outputs/SpreadsheetBench_meta_locator_a1_fixed_four_additive_full128_seed1_val40/valid_seen/results.jsonl",
    "A2 adaptive additive": ROOT
    / "outputs/SpreadsheetBench_meta_locator_a2_adaptive_additive_full128_seed1_val40/valid_seen/results.jsonl",
    "A3 adaptive multiplicative": ROOT
    / "outputs/SpreadsheetBench_meta_locator_a3_adaptive_multiplicative_full128_seed1_val40/valid_seen/results.jsonl",
    "Combined10 full-vocab KL": ROOT
    / "outputs/SpreadsheetBench_combined_core10_full_vocab_skillkl_shared_preserve_len8_seed1/eval/final/valid_seen/results.jsonl",
    "SE-KD": ROOT
    / "outputs/SpreadsheetBench_sekd_prefix_official_top20_len8_seed1/eval/final/valid_seen/results.jsonl",
    "OPCD": ROOT
    / "outputs/SpreadsheetBench_opcd_prefix_official_top256_len8_seed1/eval/final/valid_seen/results.jsonl",
    "Hard Skill": ROOT
    / "outputs/SpreadsheetBench_qwen36_hard_skill_val40_dynamic_additive_matched/eval/hard_skill/valid_seen/results.jsonl",
}


def read_results(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {str(row["id"]): row for row in rows}


def succeeded(row: dict) -> bool:
    return bool(row.get("hard", row.get("ok", False)))


def pairwise(left: dict[str, dict], right: dict[str, dict]) -> dict:
    common = sorted(set(left) & set(right))
    right_only = sum(succeeded(right[key]) and not succeeded(left[key]) for key in common)
    left_only = sum(succeeded(left[key]) and not succeeded(right[key]) for key in common)
    discordant = right_only + left_only
    p_value = (
        float(binomtest(min(right_only, left_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "common_tasks": len(common),
        "right_wins": right_only,
        "right_losses": left_only,
        "exact_p_value": p_value,
    }


def summarize(name: str, rows_by_id: dict[str, dict]) -> dict:
    rows = list(rows_by_id.values())
    by_type = {}
    for task_type in ("cell_level", "sheet_level", "other"):
        subset = [row for row in rows if row.get("task_type") == task_type]
        by_type[task_type] = {
            "tasks": len(subset),
            "successes": sum(succeeded(row) for row in subset),
        }
    failures = {"execution": 0, "semantic": 0, "no_cases": 0}
    for row in rows:
        if succeeded(row):
            continue
        if int(row.get("n_cases", 0)) == 0:
            failures["no_cases"] += 1
        elif int(row.get("n_exec_pass", 0)) < int(row.get("n_cases", 0)):
            failures["execution"] += 1
        else:
            failures["semantic"] += 1
    return {
        "name": name,
        "tasks": len(rows),
        "successes": sum(succeeded(row) for row in rows),
        "rate": sum(succeeded(row) for row in rows) / len(rows),
        "by_type": by_type,
        "failures": failures,
        "mean_response_chars": sum(len(str(row.get("response", ""))) for row in rows)
        / len(rows),
    }


def main() -> None:
    data = {name: read_results(path) for name, path in RESULTS.items()}
    dynamic_names = list(RESULTS)[:4]
    report = {
        "summaries": {name: summarize(name, rows) for name, rows in data.items()},
        "dynamic_pairwise": {},
        "comparisons_vs_a0": {},
    }
    for left_name, right_name in itertools.combinations(dynamic_names, 2):
        key = f"{left_name} -> {right_name}"
        report["dynamic_pairwise"][key] = pairwise(data[left_name], data[right_name])
    for name in list(RESULTS)[1:]:
        report["comparisons_vs_a0"][name] = pairwise(data["A0 G+JS"], data[name])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
