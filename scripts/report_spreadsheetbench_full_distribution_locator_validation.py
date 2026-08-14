#!/usr/bin/env python3
"""Report the paired Positive-vs-Combined shared-preservation validation."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSITIVE = "positive_gain_top0.05_core_shared_preserve"
COMBINED = "combined_top0.05_core_shared_preserve"


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _category(row: dict) -> str:
    if row.get("hard"):
        return "success"
    stages = {case.get("stage") for case in row.get("cases", []) if not case.get("ok")}
    if "exec" in stages:
        return "exec_error"
    if "eval" in stages:
        return "eval_mismatch"
    return "other"


def _wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (successes / total + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        successes * (total - successes) / total**3 + z * z / (4 * total * total)
    ) / denominator
    return [center - radius, center + radius]


def _exact_sign_p(first: int, second: int) -> float:
    discordant = first + second
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(first, second) + 1))
    return min(1.0, 2.0 * tail / 2**discordant)


def _summarize(history: dict, rows: list[dict]) -> dict:
    lengths = [len(str(row.get("response", ""))) for row in rows]
    successes = sum(int(bool(row.get("hard"))) for row in rows)
    categories = Counter(_category(row) for row in rows)
    executable = categories["success"] + categories["eval_mismatch"]
    return {
        "tasks": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "wilson_95_ci": _wilson(successes, len(rows)),
        "categories": dict(categories),
        "executable_tasks": executable,
        "conditional_success_given_executable": successes / executable,
        "median_response_chars": statistics.median(lengths),
        "mean_response_chars": statistics.mean(lengths),
        "p95_response_chars": sorted(lengths)[math.ceil(0.95 * len(lengths)) - 1],
        "max_response_chars": max(lengths),
        "unclosed_responses": sum(
            not str(row.get("response", "")).rstrip().endswith("```") for row in rows
        ),
        "loss": history["loss"],
        "selected_ce_loss": history["selected_ce_loss"],
        "preservation_kl_loss": history["preservation_kl_loss"],
        "supervised_tokens": history["supervised_tokens"],
        "preservation_tokens": history["preservation_tokens"],
        "wall_time_s": history["wall_time_s"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default="outputs/SpreadsheetBench_full_distribution_locator_len8_seed1_shared",
    )
    parser.add_argument(
        "--manifest-root",
        default="outputs/SpreadsheetBench_selective_stage2_manifests",
    )
    args = parser.parse_args()
    run_root = _resolve(args.run_root)
    manifest_root = _resolve(args.manifest_root)

    histories: dict[str, dict] = {}
    results: dict[str, list[dict]] = {}
    manifests: dict[str, list[dict]] = {}
    for name in (POSITIVE, COMBINED):
        history = json.loads((run_root / name / "history.json").read_text(encoding="utf-8"))
        if len(history) != 1 or int(history[0]["epoch"]) != 1:
            raise ValueError(f"{name} is not an epoch-1-only run")
        histories[name] = history[0]
        results[name] = _read_jsonl(
            run_root / name / "eval" / "epoch_01" / "valid_seen" / "results.jsonl"
        )
        manifests[name] = _read_jsonl(manifest_root / f"{name}.jsonl")

    for positive, combined in zip(manifests[POSITIVE], manifests[COMBINED], strict=True):
        if positive["id"] != combined["id"]:
            raise ValueError("Paired manifests have different task order")
        if positive["preserve_indices"] != combined["preserve_indices"]:
            raise ValueError(f"Preservation mismatch for {positive['id']}")
        if set(positive["preserve_indices"]) & set(positive["selected_indices"]):
            raise ValueError(f"Positive selected/preserve overlap for {positive['id']}")
        if set(combined["preserve_indices"]) & set(combined["selected_indices"]):
            raise ValueError(f"Combined selected/preserve overlap for {positive['id']}")

    positive_by_id = {str(row["id"]): row for row in results[POSITIVE]}
    combined_by_id = {str(row["id"]): row for row in results[COMBINED]}
    if set(positive_by_id) != set(combined_by_id) or len(positive_by_id) != 40:
        raise ValueError("Validation result IDs are not the same complete 40-task set")

    combined_only = sorted(
        identifier
        for identifier in positive_by_id
        if combined_by_id[identifier].get("hard") and not positive_by_id[identifier].get("hard")
    )
    positive_only = sorted(
        identifier
        for identifier in positive_by_id
        if positive_by_id[identifier].get("hard") and not combined_by_id[identifier].get("hard")
    )
    both = sorted(
        identifier
        for identifier in positive_by_id
        if positive_by_id[identifier].get("hard") and combined_by_id[identifier].get("hard")
    )
    neither = sorted(
        identifier
        for identifier in positive_by_id
        if not positive_by_id[identifier].get("hard") and not combined_by_id[identifier].get("hard")
    )
    transitions = Counter(
        (_category(positive_by_id[identifier]), _category(combined_by_id[identifier]))
        for identifier in positive_by_id
    )
    summaries = {
        "positive": _summarize(histories[POSITIVE], results[POSITIVE]),
        "combined": _summarize(histories[COMBINED], results[COMBINED]),
    }
    paired_p = _exact_sign_p(len(combined_only), len(positive_only))
    report_json = {
        "protocol": {
            "trajectories": len(manifests[POSITIVE]),
            "validation_tasks": len(positive_by_id),
            "prefix_length": 8,
            "epochs": 1,
            "selected_tokens_including_eos": histories[POSITIVE]["supervised_tokens"],
            "shared_preservation_tokens": histories[POSITIVE]["preservation_tokens"],
            "shared_preservation_exactly_matched": True,
            "test_split_accessed": False,
        },
        "summaries": summaries,
        "effect": {
            "absolute_success_delta": summaries["combined"]["successes"] - summaries["positive"]["successes"],
            "percentage_point_delta": 100.0 * (
                summaries["combined"]["success_rate"] - summaries["positive"]["success_rate"]
            ),
            "combined_only_success_ids": combined_only,
            "positive_only_success_ids": positive_only,
            "both_success_ids": both,
            "neither_success_count": len(neither),
            "exact_two_sided_sign_p": paired_p,
            "statistically_significant_at_0.05": paired_p < 0.05,
        },
        "category_transitions": {
            f"{source}->{target}": count for (source, target), count in sorted(transitions.items())
        },
        "interpretation": (
            "Combined localization is directionally better under the strictly matched preservation control, "
            "but the 40-task paired result is not statistically significant."
        ),
    }
    json_path = run_root / "FULL_DISTRIBUTION_LOCATOR_VALIDATION.json"
    report_path = run_root / "FULL_DISTRIBUTION_LOCATOR_VALIDATION_REPORT.md"
    json_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    p = summaries["positive"]
    c = summaries["combined"]
    report = [
        "# SpreadsheetBench 全分布定位配对验证",
        "",
        "## 协议",
        "",
        "- Positive 与 Combined 各训练 1 epoch，prefix length 8，seed 1。",
        f"- 每组监督 token：{p['supervised_tokens']}（含 EOS）。",
        f"- 每组 KL token：{p['preservation_tokens']}，逐轨迹位置完全相同。",
        "- 只评估 40 题验证集；未访问测试集。",
        "",
        "## 结果",
        "",
        "| Selector | 成功 | 执行错误 | 结果不匹配 | 条件成功率 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Positive | {p['successes']}/40 ({p['success_rate']:.1%}) | "
            f"{p['categories'].get('exec_error', 0)} | {p['categories'].get('eval_mismatch', 0)} | "
            f"{p['conditional_success_given_executable']:.1%} |"
        ),
        (
            f"| Combined | **{c['successes']}/40 ({c['success_rate']:.1%})** | "
            f"{c['categories'].get('exec_error', 0)} | {c['categories'].get('eval_mismatch', 0)} | "
            f"**{c['conditional_success_given_executable']:.1%}** |"
        ),
        "",
        f"- Combined 绝对提升：{c['successes'] - p['successes']} 题，{100 * (c['success_rate'] - p['success_rate']):+.1f} 个百分点。",
        f"- Combined 独占成功：{len(combined_only)} 题；Positive 独占成功：{len(positive_only)} 题。",
        f"- 精确双侧符号检验：p={paired_p:.6f}。",
        "",
        "## 训练与生成诊断",
        "",
        "| Selector | Selected CE | Preserve KL | 响应中位字符 | 未闭合响应 |",
        "|---|---:|---:|---:|---:|",
        f"| Positive | {p['selected_ce_loss']:.6f} | {p['preservation_kl_loss']:.6f} | {p['median_response_chars']:.1f} | {p['unclosed_responses']} |",
        f"| Combined | {c['selected_ce_loss']:.6f} | {c['preservation_kl_loss']:.6f} | {c['median_response_chars']:.1f} | {c['unclosed_responses']} |",
        "",
        "## 结论",
        "",
        "全分布 Combined 定位在严格共享 preservation 控制下呈现明显正向结果，并同时减少执行错误；但 40 题上的配对差异尚未达到统计显著。该结果支持继续做多 preservation seed 复验，不能单独证明稳定泛化。",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "report": str(report_path), "effect": report_json["effect"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
