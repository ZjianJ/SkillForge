#!/usr/bin/env python3
"""Aggregate stage-2 validation runs into a machine-readable and Markdown report."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORDER = [
    "clean_full",
    "random_top0.05_core",
    "positive_gain_top0.05_core",
    "combined_top0.05_core",
    "positive_gain_top0.1_core",
    "combined_top0.05_L1_R2",
]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _nearest_rank(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _failure_profile(rows: list[dict]) -> dict:
    lengths = [len(str(row.get("response", ""))) for row in rows]
    failures = [row for row in rows if not row.get("ok", False)]
    return {
        "response_chars_median": median(lengths) if lengths else 0,
        "response_chars_p95": _nearest_rank(lengths, 0.95),
        "response_chars_max": max(lengths, default=0),
        "responses_at_least_12000_chars": sum(length >= 12_000 for length in lengths),
        "exec_failures": sum(str(row.get("fail_reason", "")).startswith("exec-error") for row in failures),
        "evaluation_mismatches": sum(
            str(row.get("fail_reason", "")).startswith("eval-mismatch") for row in failures
        ),
        "other_failures": sum(
            not str(row.get("fail_reason", "")).startswith(("exec-error", "eval-mismatch"))
            for row in failures
        ),
    }


def _exact_sign_p_value(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired_comparison(candidate: list[dict], baseline: list[dict]) -> dict:
    candidate_by_id = {str(row["id"]): bool(row.get("ok", False)) for row in candidate}
    baseline_by_id = {str(row["id"]): bool(row.get("ok", False)) for row in baseline}
    shared = sorted(set(candidate_by_id) & set(baseline_by_id))
    gains = sum(candidate_by_id[key] and not baseline_by_id[key] for key in shared)
    losses = sum(not candidate_by_id[key] and baseline_by_id[key] for key in shared)
    both_success = sum(candidate_by_id[key] and baseline_by_id[key] for key in shared)
    return {
        "shared_tasks": len(shared),
        "candidate_only_successes": gains,
        "baseline_only_successes": losses,
        "both_success": both_success,
        "both_fail": len(shared) - gains - losses - both_success,
        "exact_two_sided_sign_p": _exact_sign_p_value(gains, losses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", default="outputs/SpreadsheetBench_selective_stage2_len8_seed1_safe"
    )
    parser.add_argument(
        "--manifest-summary",
        default="outputs/SpreadsheetBench_selective_stage2_manifests/summary.json",
    )
    args = parser.parse_args()
    run_root = _resolve(args.run_root)
    manifest = json.loads(_resolve(args.manifest_summary).read_text(encoding="utf-8"))
    stage1_path = PROJECT_ROOT / "outputs/SpreadsheetBench_selective_stage1_qwen36_gpt55/summary.json"
    stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))

    results: dict[str, dict] = {}
    missing: list[str] = []
    for name in ORDER:
        history_path = run_root / name / "history.json"
        summary_path = run_root / name / "summary.json"
        if not history_path.exists() or not summary_path.exists():
            missing.append(name)
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best = max(history, key=lambda row: (row["valid_seen_hard"], row["valid_seen_soft"], -row["epoch"]))
        best_rows = _read_jsonl(
            run_root / name / "eval" / f"epoch_{best['epoch']:02d}" / "valid_seen" / "results.jsonl"
        )
        successes = sum(bool(row.get("ok", False)) for row in best_rows)
        stats = manifest["variants"][name]
        results[name] = {
            "best_epoch": best["epoch"],
            "best_valid_hard": best["valid_seen_hard"],
            "best_valid_soft": best["valid_seen_soft"],
            "best_valid_count_of_40": round(best["valid_seen_hard"] * 40),
            "best_valid_wilson_95": _wilson_interval(successes, len(best_rows)),
            "best_train_loss": best["loss"],
            "supervised_tokens_per_epoch": best.get("supervised_tokens"),
            "mean_supervised_tokens_per_trajectory": stats["mean_supervised_tokens"],
            "non_eos_token_coverage": stats["non_eos_coverage"],
            "wall_time_s_total": sum(float(row.get("wall_time_s", 0)) for row in history),
            "best_prefix_path": summary.get("best_prefix_path", ""),
            "history": history,
            "failure_profile_at_best": _failure_profile(best_rows),
        }

    comparisons: dict[str, object] = {}
    if "random_top0.05_core" in results:
        random_score = results["random_top0.05_core"]["best_valid_hard"]
        selector_names = [
            name
            for name in (
                "positive_gain_top0.05_core",
                "combined_top0.05_core",
                "combined_top0.05_L1_R2",
            )
            if name in results
        ]
        comparisons["selectors_vs_matched_random"] = {
            name: results[name]["best_valid_hard"] - random_score for name in selector_names
        }
        comparisons["any_selector_beats_matched_random"] = any(
            results[name]["best_valid_hard"] > random_score for name in selector_names
        )
    if "clean_full" in results:
        full_score = results["clean_full"]["best_valid_hard"]
        comparisons["variants_vs_clean_full"] = {
            name: row["best_valid_hard"] - full_score
            for name, row in results.items()
            if name != "clean_full"
        }
    best_rows_by_name = {
        name: _read_jsonl(
            run_root
            / name
            / "eval"
            / f"epoch_{row['best_epoch']:02d}"
            / "valid_seen"
            / "results.jsonl"
        )
        for name, row in results.items()
    }
    if "random_top0.05_core" in best_rows_by_name:
        comparisons["paired_vs_matched_random"] = {
            name: _paired_comparison(rows, best_rows_by_name["random_top0.05_core"])
            for name, rows in best_rows_by_name.items()
            if name not in ("random_top0.05_core", "clean_full")
        }
    if "clean_full" in best_rows_by_name:
        comparisons["paired_vs_clean_full"] = {
            name: _paired_comparison(rows, best_rows_by_name["clean_full"])
            for name, rows in best_rows_by_name.items()
            if name != "clean_full"
        }

    payload = {
        "protocol": {
            "model": "Qwen3.6-35B-A3B@995ad96e",
            "frozen_base_model": True,
            "prefix_length": 8,
            "seed": 1,
            "epochs": 3,
            "learning_rate": 0.001,
            "validation_max_new_tokens": 4096,
            "validation_generation_batch_size": 8,
            "left_padding_prefix_insertion_fixed": True,
            "canonical_inputs_copied_before_generated_code_execution": True,
            "input_integrity_checked_before_and_after": True,
            "successful_training_trajectories": 61,
            "validation_tasks": 40,
            "test_set_accessed": False,
            "teacher_skill_in_student_prompt": False,
            "eos_always_supervised": True,
            "token_weighted_gradient_accumulation": True,
        },
        "stage1_supports_training": stage1["supports_selective_training"],
        "results": results,
        "comparisons": comparisons,
        "missing_runs": missing,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# SpreadsheetBench Selective Soft Prompt Distillation — Stage 2",
        "",
        "## Protocol",
        "",
        "- Frozen Qwen3.6-35B-A3B, soft-prefix length 8, seed 1, 3 epochs, LR 1e-3.",
        "- Validation generation is capped at 4,096 tokens; the longest successful teacher target is 3,647 tokens.",
        "- Validation generation batch size is 8 on one GH200; decoding remains greedy and deterministic.",
        "- Batched left padding is handled as PAD → soft prefix → prompt, making prefix placement independent of batch peers.",
        "- Generated code receives a temporary copy of each input workbook; canonical benchmark files are integrity-checked against the release archive before and after the run.",
        "- 61 GPT-5.5 successful trajectories; student prompts contain the task but no text Skill.",
        "- All variants use the same initialization, shuffle seed, optimizer, 40-task validation set, and hard-score checkpoint gate.",
        "- EOS is always supervised. Gradients are normalized by the actual selected-token count in each accumulation group.",
        "- The 280-task test split was not accessed during screening.",
        "",
        "## Validation results",
        "",
        "| Variant | Non-EOS coverage | Supervised tokens / trajectory | Best epoch | Hard success | Successes / 40 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ORDER:
        if name not in results:
            lines.append(f"| {name} | — | — | — | incomplete | — |")
            continue
        row = results[name]
        ci_low, ci_high = row["best_valid_wilson_95"]
        lines.append(
            f"| {name} | {row['non_eos_token_coverage']:.2%} | "
            f"{row['mean_supervised_tokens_per_trajectory']:.1f} | {row['best_epoch']} | "
            f"{row['best_valid_hard']:.2%} | {row['best_valid_count_of_40']} |"
        )

    lines.extend(["", "Wilson 95% intervals and all epoch histories are recorded in `results.json`."])

    lines.extend(["", "## Interpretation", ""])
    if missing:
        lines.append("- Report is partial; incomplete runs: " + ", ".join(missing) + ".")
    elif comparisons.get("any_selector_beats_matched_random"):
        lines.append(
            "- At least one Skill-derived selector outperforms the token-count-matched random control, "
            "so the stage-1 importance signal has predictive training value at seed 1."
        )
    else:
        lines.append(
            "- No Skill-derived selector beats the matched-random control at seed 1; the concentration "
            "result alone therefore does not establish useful selective training."
        )
    if not missing and "clean_full" in results:
        best_selective_name = max(
            (name for name in results if name != "clean_full"),
            key=lambda name: results[name]["best_valid_hard"],
        )
        delta = results[best_selective_name]["best_valid_hard"] - results["clean_full"]["best_valid_hard"]
        lines.append(
            f"- Best non-full variant is `{best_selective_name}`; its hard-success delta versus clean Full CE is {delta:+.2%}."
        )
        if "random_top0.05_core" in results and best_selective_name != "random_top0.05_core":
            paired = comparisons["paired_vs_matched_random"][best_selective_name]
            lines.append(
                f"- Against matched random on the same 40 tasks, it flips {paired['candidate_only_successes']} failures to successes "
                f"and {paired['baseline_only_successes']} successes to failures "
                f"(exact paired sign p={paired['exact_two_sided_sign_p']:.4f})."
            )
        lines.append(
            "- This is a one-seed selection experiment. A final claim requires locked configurations, "
            "three seeds, shorter prefixes, and one test evaluation after validation selection."
        )
    lines.extend(
        [
            "",
            "## Generation and failure audit",
            "",
            "`results.json` reports response-length proxies plus execution, evaluation-mismatch, and other failure counts "
            "at every variant's selected checkpoint. Character length is an audit proxy, not a tokenizer-exact token count.",
            "The earlier pre-fix batch-size-dependent diagnostics are excluded from every table and comparison.",
            "The earlier source-mutating diagnostics under the `_fixed` run root are also excluded; only the protected `_safe` root is reported.",
            "",
            "Machine-readable details, epoch histories, paths, and pairwise deltas are in `results.json`.",
        ]
    )
    (run_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {run_root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
