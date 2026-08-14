#!/usr/bin/env python3
"""Compare target-token gain and full-distribution-aware token localization.

This analysis is cache-only: it does not load the language model, train a
prefix, generate trajectories, or access the SpreadsheetBench test split.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


def _top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    return np.argsort(-scores, kind="stable")[:count]


def _capture(values: np.ndarray, indices: np.ndarray) -> float:
    denominator = float(values.sum())
    return float(values[indices].sum() / denominator) if denominator > 0 else 0.0


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    points = np.quantile(array, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "mean": float(array.mean()),
        "p01": float(points[0]),
        "p05": float(points[1]),
        "median": float(points[2]),
        "p95": float(points[3]),
        "p99": float(points[4]),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    n_rows = values.shape[0]
    draws = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        indices = rng.integers(0, n_rows, size=n_rows)
        draws[sample_index] = float(values[indices].mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return [float(low), float(high)]


def _selection_stats(
    gain: np.ndarray,
    js: np.ndarray,
    combined: np.ndarray,
    skill_top1: np.ndarray,
    clean_top1: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    return {
        "positive_gain_capture": _capture(gain, indices),
        "exact_js_capture": _capture(js, indices),
        "combined_mass_capture": _capture(combined, indices),
        "mean_positive_gain": float(gain[indices].mean()),
        "mean_exact_js": float(js[indices].mean()),
        "mean_combined_score": float(combined[indices].mean()),
        "top1_flip_fraction": float((skill_top1[indices] != clean_top1[indices]).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default="outputs/SpreadsheetBench_selective_stage1_qwen36_gpt55/token_scores",
    )
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-json",
        default=(
            "outputs/SpreadsheetBench_selective_stage1_qwen36_gpt55/"
            "full_distribution_statistics.json"
        ),
    )
    parser.add_argument(
        "--output-report",
        default=(
            "outputs/SpreadsheetBench_selective_stage1_qwen36_gpt55/"
            "FULL_DISTRIBUTION_REPORT.md"
        ),
    )
    args = parser.parse_args()
    if not 0 < args.ratio < 1:
        raise ValueError("--ratio must be between zero and one")

    cache_paths = sorted(_resolve(args.cache_dir).glob("*.npz"))
    if not cache_paths:
        raise FileNotFoundError(f"No score caches found under {_resolve(args.cache_dir)}")

    required = {
        "target_ids",
        "positive_gain",
        "js",
        "combined",
        "skill_topk_ids",
        "skill_topk_logp",
        "skill_residual_log_mass",
        "clean_topk_ids",
        "clean_topk_logp",
        "clean_residual_log_mass",
    }
    per_trajectory: list[dict[str, float | int | str]] = []
    positive_unique: list[list[float]] = []
    combined_unique: list[list[float]] = []
    skill_top64_mass: list[float] = []
    clean_top64_mass: list[float] = []
    top64_id_jaccard: list[float] = []
    all_gain: list[float] = []
    all_js: list[float] = []
    total_gain = total_js = total_combined = 0.0
    selected_totals = {
        "positive_gain": np.zeros(3, dtype=np.float64),
        "combined": np.zeros(3, dtype=np.float64),
    }
    target_in_skill_top64 = target_in_clean_top64 = 0
    total_tokens = 0

    for cache_path in cache_paths:
        with np.load(cache_path) as cached:
            missing = required - set(cached.files)
            if missing:
                raise ValueError(f"{cache_path} is missing fields: {sorted(missing)}")
            # The final target id is EOS and is always supervised separately.
            selectable = len(cached["target_ids"]) - 1
            count = max(1, math.ceil(selectable * args.ratio))
            gain = cached["positive_gain"][:selectable].astype(np.float64)
            js = cached["js"][:selectable].astype(np.float64)
            combined = cached["combined"][:selectable].astype(np.float64)
            skill_ids = cached["skill_topk_ids"][:selectable]
            clean_ids = cached["clean_topk_ids"][:selectable]
            skill_top1 = skill_ids[:, 0]
            clean_top1 = clean_ids[:, 0]
            target_ids = cached["target_ids"][:selectable]

            positive_indices = _top_indices(gain, count)
            combined_indices = _top_indices(combined, count)
            positive_set = set(positive_indices.tolist())
            combined_set = set(combined_indices.tolist())
            intersection = positive_set & combined_set
            union = positive_set | combined_set

            positive_stats = _selection_stats(
                gain, js, combined, skill_top1, clean_top1, positive_indices
            )
            combined_stats = _selection_stats(
                gain, js, combined, skill_top1, clean_top1, combined_indices
            )
            row: dict[str, float | int | str] = {
                "id": cache_path.stem.split("_", 1)[-1],
                "tokens": selectable,
                "selected_tokens": count,
                "jaccard": len(intersection) / len(union),
            }
            for prefix, stats in (
                ("positive", positive_stats),
                ("combined", combined_stats),
            ):
                row.update({f"{prefix}_{key}": value for key, value in stats.items()})
            per_trajectory.append(row)

            for index in sorted(positive_set - combined_set):
                positive_unique.append([gain[index], js[index], combined[index]])
            for index in sorted(combined_set - positive_set):
                combined_unique.append([gain[index], js[index], combined[index]])

            skill_mass = 1.0 - np.exp(
                cached["skill_residual_log_mass"][:selectable].astype(np.float64)
            )
            clean_mass = 1.0 - np.exp(
                cached["clean_residual_log_mass"][:selectable].astype(np.float64)
            )
            skill_top64_mass.extend(skill_mass.tolist())
            clean_top64_mass.extend(clean_mass.tolist())
            for skill_row, clean_row in zip(skill_ids, clean_ids, strict=True):
                first, second = set(skill_row.tolist()), set(clean_row.tolist())
                top64_id_jaccard.append(len(first & second) / len(first | second))

            target_in_skill_top64 += int(
                (skill_ids == target_ids[:, None]).any(axis=1).sum()
            )
            target_in_clean_top64 += int(
                (clean_ids == target_ids[:, None]).any(axis=1).sum()
            )
            total_tokens += selectable
            total_gain += float(gain.sum())
            total_js += float(js.sum())
            total_combined += float(combined.sum())
            selected_totals["positive_gain"] += [
                gain[positive_indices].sum(),
                js[positive_indices].sum(),
                combined[positive_indices].sum(),
            ]
            selected_totals["combined"] += [
                gain[combined_indices].sum(),
                js[combined_indices].sum(),
                combined[combined_indices].sum(),
            ]
            all_gain.extend(gain.tolist())
            all_js.extend(js.tolist())

    paired_names = [
        "positive_gain_capture",
        "exact_js_capture",
        "combined_mass_capture",
        "mean_positive_gain",
        "mean_exact_js",
        "mean_combined_score",
        "top1_flip_fraction",
    ]
    paired_differences: dict[str, dict[str, float | list[float]]] = {}
    for metric_index, metric in enumerate(paired_names):
        differences = np.asarray(
            [
                float(row[f"combined_{metric}"]) - float(row[f"positive_{metric}"])
                for row in per_trajectory
            ],
            dtype=np.float64,
        )
        paired_differences[metric] = {
            "combined_minus_positive_mean": float(differences.mean()),
            "bootstrap_95_ci": _bootstrap_mean_ci(
                differences,
                samples=args.bootstrap_samples,
                seed=args.seed + metric_index,
            ),
            "combined_better_trajectory_fraction": float((differences > 0).mean()),
        }

    positive_total = selected_totals["positive_gain"]
    combined_total = selected_totals["combined"]
    global_capture = {
        "positive_gain_selector": {
            "positive_gain": float(positive_total[0] / total_gain),
            "exact_js": float(positive_total[1] / total_js),
            "combined_mass": float(positive_total[2] / total_combined),
        },
        "combined_selector": {
            "positive_gain": float(combined_total[0] / total_gain),
            "exact_js": float(combined_total[1] / total_js),
            "combined_mass": float(combined_total[2] / total_combined),
        },
    }
    gain_retention = (
        global_capture["combined_selector"]["positive_gain"]
        / global_capture["positive_gain_selector"]["positive_gain"]
    )
    mean_jaccard = float(np.mean([float(row["jaccard"]) for row in per_trajectory]))
    js_difference_ci = paired_differences["exact_js_capture"]["bootstrap_95_ci"]
    combined_difference_ci = paired_differences["combined_mass_capture"]["bootstrap_95_ci"]
    criteria = {
        "non_redundant_mean_jaccard_le_0.80": mean_jaccard <= 0.80,
        "paired_js_capture_improvement_ci_above_zero": float(js_difference_ci[0]) > 0,
        "paired_combined_capture_improvement_ci_above_zero": float(combined_difference_ci[0]) > 0,
        "retain_at_least_90pct_positive_gain": gain_retention >= 0.90,
    }
    supports_validation = all(criteria.values())

    def unique_summary(values: list[list[float]]) -> dict[str, object]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "tokens": int(len(array)),
            "mean_positive_gain": float(array[:, 0].mean()),
            "median_positive_gain": float(np.median(array[:, 0])),
            "mean_exact_js": float(array[:, 1].mean()),
            "median_exact_js": float(np.median(array[:, 1])),
            "mean_combined_score": float(array[:, 2].mean()),
            "median_combined_score": float(np.median(array[:, 2])),
        }

    result = {
        "analysis_scope": {
            "cache_only": True,
            "language_model_forward_passes": 0,
            "soft_prompt_training_runs": 0,
            "test_split_accesses": 0,
            "trajectories": len(cache_paths),
            "selectable_tokens": total_tokens,
            "selection_ratio": args.ratio,
        },
        "top64_quality": {
            "skill_probability_mass": _quantiles(skill_top64_mass),
            "clean_probability_mass": _quantiles(clean_top64_mass),
            "target_token_in_skill_top64_fraction": target_in_skill_top64 / total_tokens,
            "target_token_in_clean_top64_fraction": target_in_clean_top64 / total_tokens,
            "skill_clean_top64_id_jaccard": _quantiles(top64_id_jaccard),
        },
        "signal_relationship": {
            "positive_gain_exact_js_pearson": float(
                np.corrcoef(np.asarray(all_gain), np.asarray(all_js))[0, 1]
            ),
            "mean_selection_jaccard": mean_jaccard,
            "median_selection_jaccard": float(
                np.median([float(row["jaccard"]) for row in per_trajectory])
            ),
        },
        "global_capture": global_capture,
        "combined_positive_gain_retention_vs_positive_selector": gain_retention,
        "paired_trajectory_differences": paired_differences,
        "selector_unique_tokens": {
            "positive_only": unique_summary(positive_unique),
            "combined_only": unique_summary(combined_unique),
        },
        "decision_rule": {
            "criteria": criteria,
            "supports_full_distribution_localization_validation": supports_validation,
            "interpretation": (
                "The cache statistics support a controlled combined-selector training validation."
                if supports_validation
                else "The cache statistics do not yet justify a combined-selector training validation."
            ),
            "limitation": (
                "This is a localization proxy, not downstream evidence; only a matched validation "
                "training run can establish whether execution success improves."
            ),
        },
        "per_trajectory": per_trajectory,
    }

    output_json = _resolve(args.output_json)
    output_report = _resolve(args.output_report)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    positive_capture = global_capture["positive_gain_selector"]
    combined_capture = global_capture["combined_selector"]
    verdict = "支持进入匹配验证实验" if supports_validation else "暂不支持进入训练验证"
    report = [
        "# SpreadsheetBench 全分布定位统计",
        "",
        f"- 轨迹：{len(cache_paths)}",
        f"- 可选择 token：{total_tokens}",
        f"- 选择预算：{args.ratio:.1%}",
        "- 本次模型前向/训练/测试集访问：0 / 0 / 0",
        "",
        "## Top-64 缓存质量",
        "",
        f"- Skill Top-64 平均概率质量：{np.mean(skill_top64_mass):.4%}",
        f"- No-Skill Top-64 平均概率质量：{np.mean(clean_top64_mass):.4%}",
        f"- 成功目标 token 位于 Skill Top-64：{target_in_skill_top64 / total_tokens:.4%}",
        f"- 成功目标 token 位于 No-Skill Top-64：{target_in_clean_top64 / total_tokens:.4%}",
        "",
        "## Top 5% 定位对比",
        "",
        "| 选择器 | 捕获正增益 | 捕获精确 JS | 捕获 combined 质量 |",
        "|---|---:|---:|---:|",
        (
            f"| positive_gain | {positive_capture['positive_gain']:.2%} | "
            f"{positive_capture['exact_js']:.2%} | {positive_capture['combined_mass']:.2%} |"
        ),
        (
            f"| positive_gain × JS | {combined_capture['positive_gain']:.2%} | "
            f"{combined_capture['exact_js']:.2%} | {combined_capture['combined_mass']:.2%} |"
        ),
        "",
        f"- 平均选择 Jaccard：{mean_jaccard:.4f}",
        f"- Combined 保留 Positive 方案正增益的比例：{gain_retention:.2%}",
        (
            "- 逐轨迹 JS 捕获差异（Combined−Positive）："
            f"{paired_differences['exact_js_capture']['combined_minus_positive_mean']:+.2%}，"
            f"95% bootstrap CI [{js_difference_ci[0]:+.2%}, {js_difference_ci[1]:+.2%}]"
        ),
        (
            "- 逐轨迹 combined 质量捕获差异："
            f"{paired_differences['combined_mass_capture']['combined_minus_positive_mean']:+.2%}，"
            f"95% bootstrap CI [{combined_difference_ci[0]:+.2%}, {combined_difference_ci[1]:+.2%}]"
        ),
        "",
        "## 判断",
        "",
        f"**{verdict}。**",
        "",
        "该判断只说明全分布信号具备非冗余且更集中的定位信息；它不等同于任务完成率提升。下一步需要在相同 prefix、epoch、KL、token 预算和验证集下，对 combined 与 positive 做单变量训练比较。",
    ]
    output_report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_report": str(output_report), "decision": result["decision_rule"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
