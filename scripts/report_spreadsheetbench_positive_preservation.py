#!/usr/bin/env python3
"""Report the focused positive-gain + no-Skill preservation experiment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sign_p(gains: int, losses: int) -> float:
    total = gains + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**total))


def _best(run_dir: Path) -> tuple[dict, list[dict]]:
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    best = max(history, key=lambda row: (row["valid_seen_hard"], -row["epoch"]))
    rows = _read_jsonl(
        run_dir / "eval" / f"epoch_{best['epoch']:02d}" / "valid_seen/results.jsonl"
    )
    return best, rows


def _paired(candidate: list[dict], baseline: list[dict]) -> dict:
    first = {str(row["id"]): bool(row.get("ok")) for row in candidate}
    second = {str(row["id"]): bool(row.get("ok")) for row in baseline}
    shared = sorted(set(first) & set(second))
    gains = sum(first[key] and not second[key] for key in shared)
    losses = sum(second[key] and not first[key] for key in shared)
    return {
        "shared_tasks": len(shared),
        "candidate_only_successes": gains,
        "baseline_only_successes": losses,
        "exact_two_sided_sign_p": _sign_p(gains, losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", default="outputs/SpreadsheetBench_selective_stage2_len8_seed1_safe"
    )
    parser.add_argument("--variant", default="positive_gain_top0.05_core_preserve")
    args = parser.parse_args()
    root = _resolve(args.run_root)
    candidate_best, candidate_rows = _best(root / args.variant)
    is_window = "L2_R8" in args.variant
    is_random = args.variant.startswith("random_")
    selector = (
        "positive_gain_top0.05_L2_R8"
        if is_window
        else ("random_top0.05_core" if is_random else "positive_gain_top0.05_core")
    )
    candidate_history = json.loads((root / args.variant / "history.json").read_text(encoding="utf-8"))
    result = {
        "protocol": {
            "selector": selector,
            "preservation_reference": "no-Skill Qwen Top-64 plus residual bucket",
            "preservation_sampling": (
                "one deterministic unselected token per original core token"
                if is_window
                else "one deterministic unselected token per selected token"
            ),
            "preservation_loss_weight": 1.0,
            "soft_prefix_length": 8,
            "epochs": len(candidate_history),
            "validation_tasks": 40,
            "test_set_accessed": False,
        },
        "candidate": {
            "best_epoch": candidate_best["epoch"],
            "hard_success": candidate_best["valid_seen_hard"],
            "successes": sum(bool(row.get("ok")) for row in candidate_rows),
            "history": candidate_history,
        },
        "comparisons": {},
    }
    baselines = ["clean_full", "random_top0.05_core"]
    if is_window:
        baselines.insert(0, "positive_gain_top0.05_core_preserve")
    elif is_random:
        baselines.insert(0, "positive_gain_top0.05_core_preserve")
    for baseline in baselines:
        best, rows = _best(root / baseline)
        result["comparisons"][baseline] = {
            "baseline_best_epoch": best["epoch"],
            "baseline_hard_success": best["valid_seen_hard"],
            "absolute_delta": candidate_best["valid_seen_hard"] - best["valid_seen_hard"],
            "paired": _paired(candidate_rows, rows),
        }

    out_dir = root / args.variant
    (out_dir / "preservation_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Positive-gain selective distillation with unselected-position preservation",
        "",
        "- Frozen Qwen3.6-35B-A3B; soft-prefix length 8; seed 1; three epochs.",
        (
            "- Selected objective: CE on merged L2/R8 windows around positive-gain Top-5% core positions plus EOS."
            if is_window
            else "- Selected objective: CE on positive-gain Top-5% core positions plus EOS."
        ),
        "- Preservation objective: KL to the no-Skill Top-64 distribution plus residual bucket.",
        "- Preservation locations: deterministic 1:1 sample from unselected non-EOS positions; weight 1.0.",
        "- Validation: free greedy generation on the same 40-task validation split; test split untouched.",
        "",
        "| Method | Best epoch | Successes / 40 | Hard success | Delta vs candidate |",
        "|---|---:|---:|---:|---:|",
        f"| {selector} + preservation | {candidate_best['epoch']} | "
        f"{result['candidate']['successes']} | {candidate_best['valid_seen_hard']:.2%} | — |",
    ]
    for baseline in baselines:
        row = result["comparisons"][baseline]
        lines.append(
            f"| {baseline} | {row['baseline_best_epoch']} | "
            f"{round(row['baseline_hard_success'] * 40)} | {row['baseline_hard_success']:.2%} | "
            f"{-row['absolute_delta']:+.2%} |"
        )
    lines.extend(
        [
            "",
            (
                "这是一轮窗口扩展诊断：`positive_gain_top0.05_core_preserve` 是直接单变量对照。若窗口候选有效，"
                "下一步仍需运行 token 数匹配的 `random_core5_L2_R8 + preservation`。"
                if is_window
                else "这是一轮诊断性比较：已有 Full/Random 基线未使用保持损失。若候选有效，下一步仍需运行 "
                "`random_top0.05_core + preservation`，才能把选点收益与保持损失收益分离。"
            ),
        ]
    )
    (out_dir / "PRESERVATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
