#!/usr/bin/env python3
"""Stage 1 of selective soft-prompt distillation.

Run one batched frozen-model forward pass per successful trajectory with two
conditions (text Skill and no Skill), score target-relative token positions,
circle trainable windows, and export clean-prompt manifests for stage 2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skillopt.envs.spreadsheetbench.codegen_agent import _build_system
from skillopt.softprefix.data import _apply_text_chat_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", default="", help="Local snapshot; avoids Hub access")
    parser.add_argument("--out-root", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Recompute existing per-example caches")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: Any, index: int) -> str:
    raw = str(value or f"row-{index:04d}")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw)
    return f"{index:04d}_{safe[:80]}"


def _read_examples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if "messages" not in row or "target" not in row:
                    raise ValueError(f"{path}:{line_no}: messages/target are required")
                rows.append(row)
    return rows


def _clean_messages(messages: list[dict[str, Any]], skill_text: str) -> list[dict[str, Any]]:
    clean = [dict(message) for message in messages]
    if not clean or clean[0].get("role") != "system":
        raise ValueError("Each trajectory must start with a system message")
    skill_system = str(clean[0].get("content", ""))
    if skill_text.strip() not in skill_system:
        raise ValueError("The original system message does not contain the configured full text Skill")
    clean[0]["content"] = _build_system("")
    return clean


def _encode_example(
    tokenizer: Any,
    row: dict[str, Any],
    skill_text: str,
    max_prompt_tokens: int,
    max_target_tokens: int,
) -> dict[str, Any]:
    skill_messages = [dict(message) for message in row["messages"]]
    clean_messages = _clean_messages(skill_messages, skill_text)
    skill_prompt = _apply_text_chat_template(
        tokenizer, skill_messages, enable_thinking=False, add_generation_prompt=True
    )
    clean_prompt = _apply_text_chat_template(
        tokenizer, clean_messages, enable_thinking=False, add_generation_prompt=True
    )
    target = str(row["target"]).strip()
    eos = getattr(tokenizer, "eos_token", None)
    if eos and not target.endswith(eos):
        target += eos

    def encode(text: str, max_length: int) -> list[int]:
        return list(
            tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
        )

    target_ids = encode(target, max_target_tokens)
    if not target_ids:
        raise ValueError("Empty target after tokenization")
    return {
        "skill_prompt_ids": encode(skill_prompt, max_prompt_tokens),
        "clean_prompt_ids": encode(clean_prompt, max_prompt_tokens),
        "target_ids": target_ids,
        "clean_messages": clean_messages,
        "target_text": str(row["target"]).strip(),
    }


def _forward_pair(model: Any, encoded: dict[str, Any], pad_id: int, device: str) -> Any:
    import torch

    target = encoded["target_ids"]
    sequences = [
        encoded["skill_prompt_ids"] + target,
        encoded["clean_prompt_ids"] + target,
    ]
    max_len = max(map(len, sequences))
    input_ids = []
    masks = []
    for sequence in sequences:
        padding = max_len - len(sequence)
        input_ids.append([pad_id] * padding + sequence)
        masks.append([0] * padding + [1] * len(sequence))
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    mask_tensor = torch.tensor(masks, dtype=torch.long, device=device)
    # Both rows are left-padded and share the target suffix, so these absolute
    # indices select the same target-relative next-token predictions.
    prediction_positions = torch.arange(
        max_len - len(target) - 1,
        max_len - 1,
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode():
        output = model(
            input_ids=input_tensor,
            attention_mask=mask_tensor,
            use_cache=False,
            output_router_logits=False,
            logits_to_keep=prediction_positions,
            return_dict=True,
        )
    return output.logits


def _score_logits(
    logits: Any,
    target_ids: list[int],
    *,
    exact_js: bool,
    top_k: int,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    import torch

    n_tokens = len(target_ids)
    target = torch.tensor(target_ids, dtype=torch.long, device=logits.device)
    gain = np.empty(n_tokens, dtype=np.float32)
    skill_logp = np.empty(n_tokens, dtype=np.float32)
    clean_logp = np.empty(n_tokens, dtype=np.float32)
    js = np.zeros(n_tokens, dtype=np.float32)
    top_ids = np.empty((n_tokens, top_k), dtype=np.int32)
    top_logp = np.empty((n_tokens, top_k), dtype=np.float16)
    residual_log_mass = np.empty(n_tokens, dtype=np.float16)
    clean_top_ids = np.empty((n_tokens, top_k), dtype=np.int32)
    clean_top_logp = np.empty((n_tokens, top_k), dtype=np.float16)
    clean_residual_log_mass = np.empty(n_tokens, dtype=np.float16)

    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        skill_lp = torch.log_softmax(logits[0, start:end].float(), dim=-1)
        clean_lp = torch.log_softmax(logits[1, start:end].float(), dim=-1)
        ids = target[start:end, None]
        s_target = skill_lp.gather(1, ids).squeeze(1)
        c_target = clean_lp.gather(1, ids).squeeze(1)
        skill_logp[start:end] = s_target.cpu().numpy()
        clean_logp[start:end] = c_target.cpu().numpy()
        gain[start:end] = (s_target - c_target).cpu().numpy()

        values, indices = torch.topk(skill_lp, k=top_k, dim=-1)
        top_ids[start:end] = indices.cpu().numpy().astype(np.int32)
        top_logp[start:end] = values.cpu().numpy().astype(np.float16)
        top_mass = values.exp().sum(dim=-1).clamp(max=1.0 - 1e-7)
        residual_log_mass[start:end] = torch.log1p(-top_mass).cpu().numpy().astype(np.float16)

        clean_values, clean_indices = torch.topk(clean_lp, k=top_k, dim=-1)
        clean_top_ids[start:end] = clean_indices.cpu().numpy().astype(np.int32)
        clean_top_logp[start:end] = clean_values.cpu().numpy().astype(np.float16)
        clean_top_mass = clean_values.exp().sum(dim=-1).clamp(max=1.0 - 1e-7)
        clean_residual_log_mass[start:end] = (
            torch.log1p(-clean_top_mass).cpu().numpy().astype(np.float16)
        )

        if exact_js:
            log_mix = torch.logaddexp(skill_lp, clean_lp) - math.log(2.0)
            divergence = 0.5 * (
                (skill_lp.exp() * (skill_lp - log_mix)).sum(dim=-1)
                + (clean_lp.exp() * (clean_lp - log_mix)).sum(dim=-1)
            )
            js[start:end] = divergence.cpu().numpy()

    positive = np.maximum(gain, 0.0).astype(np.float32)
    combined = (positive * js).astype(np.float32)
    return {
        "target_ids": np.asarray(target_ids, dtype=np.int32),
        "skill_target_logp": skill_logp,
        "clean_target_logp": clean_logp,
        "gain": gain,
        "positive_gain": positive,
        "js": js,
        "combined": combined,
        "skill_topk_ids": top_ids,
        "skill_topk_logp": top_logp,
        "skill_residual_log_mass": residual_log_mass,
        "clean_topk_ids": clean_top_ids,
        "clean_topk_logp": clean_top_logp,
        "clean_residual_log_mass": clean_residual_log_mass,
    }


def _select_core(scores: np.ndarray, selectable_count: int, ratio: float) -> list[int]:
    count = max(1, math.ceil(selectable_count * ratio))
    candidates = np.arange(selectable_count)
    order = candidates[np.argsort(-scores[:selectable_count], kind="stable")]
    return sorted(int(index) for index in order[:count] if scores[index] > 0)


def _expand_windows(core: list[int], selectable_count: int, left: int, right: int) -> tuple[list[int], list[list[int]]]:
    if not core:
        return [], []
    intervals = sorted((max(0, index - left), min(selectable_count - 1, index + right)) for index in core)
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    indices = [index for start, end in merged for index in range(start, end + 1)]
    return indices, merged


def _curve(values: np.ndarray, selectable_count: int, ratios: list[float]) -> dict[str, float]:
    values = np.asarray(values[:selectable_count], dtype=np.float64)
    total = float(values.sum())
    ordered = np.sort(values)[::-1]
    result: dict[str, float] = {}
    for ratio in ratios:
        count = max(1, math.ceil(selectable_count * ratio))
        result[f"{ratio:.4f}"] = float(ordered[:count].sum() / total) if total > 0 else 0.0
    return result


def _bootstrap(curves: list[dict[str, float]], ratios: list[float], samples: int, seed: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    matrix = np.asarray([[curve[f"{ratio:.4f}"] for ratio in ratios] for curve in curves])
    output: dict[str, list[float]] = {}
    for column, ratio in enumerate(ratios):
        means = np.empty(samples, dtype=np.float64)
        for sample in range(samples):
            means[sample] = matrix[rng.integers(0, len(matrix), len(matrix)), column].mean()
        output[f"{ratio:.4f}"] = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return output


def _random_baseline(arrays: list[np.ndarray], ratios: list[float], trials: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.concatenate(arrays).astype(np.float64)
    total = float(values.sum())
    result: dict[str, float] = {}
    for ratio in ratios:
        count = max(1, math.ceil(len(values) * ratio))
        captures = [float(values[rng.choice(len(values), count, replace=False)].sum() / total) for _ in range(trials)]
        result[f"{ratio:.4f}"] = float(np.mean(captures)) if total > 0 else 0.0
    return result


def _decode_fragments(tokenizer: Any, token_ids: np.ndarray, windows: list[list[int]], scores: np.ndarray) -> list[dict[str, Any]]:
    fragments = []
    for start, end in windows:
        fragments.append(
            {
                "start": start,
                "end_inclusive": end,
                "max_score": float(scores[start : end + 1].max()),
                "text": tokenizer.decode(token_ids[start : end + 1].tolist(), skip_special_tokens=False),
            }
        )
    return fragments


def _write_svg(path: Path, ratios: list[float], global_curve: dict[str, float], mean_curve: dict[str, float], random_curve: dict[str, float]) -> None:
    width, height, margin = 760, 440, 60
    x = lambda ratio: margin + ratio / max(ratios) * (width - 2 * margin)
    y = lambda value: height - margin - value * (height - 2 * margin)
    def poly(curve: dict[str, float], color: str) -> str:
        points = " ".join(f"{x(r):.1f},{y(curve[f'{r:.4f}']):.1f}" for r in ratios)
        return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>',
        '<text x="380" y="425" text-anchor="middle">selected fraction of all target tokens</text>',
        '<text x="18" y="220" transform="rotate(-90 18 220)" text-anchor="middle">positive Skill gain captured</text>',
        poly(global_curve, "#1261a0"), poly(mean_curve, "#d1495b"), poly(random_curve, "#888888"),
        '<text x="500" y="30" fill="#1261a0">global</text>',
        '<text x="570" y="30" fill="#d1495b">trajectory mean</text>',
        '<text x="700" y="30" fill="#888888">random</text>',
    ]
    for ratio in ratios:
        parts.append(f'<text x="{x(ratio):.1f}" y="{height-margin+20}" text-anchor="middle">{ratio:g}</text>')
    for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
        parts.append(f'<text x="{margin-8}" y="{y(value)+4:.1f}" text-anchor="end">{value:g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _load_frozen_model(model_source: str, model_cfg: dict[str, Any], device: str, local_only: bool) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    revision = model_cfg.get("revision") or None
    dtype = getattr(torch, str(model_cfg.get("dtype", "bfloat16")))
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        revision=revision,
        dtype=dtype,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=local_only,
        low_cpu_mem_usage=True,
        device_map={"": device},
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _aggregate(
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    cache_records: list[dict[str, Any]],
    tokenizer: Any,
    out_root: Path,
) -> None:
    selection = cfg["selection"]
    ratios = [float(value) for value in selection["ratios"]]
    arrays: list[np.ndarray] = []
    per_curves: list[dict[str, float]] = []
    per_rows: list[dict[str, Any]] = []
    total_tokens = total_positive = 0
    total_gain = 0.0

    for record in cache_records:
        with np.load(record["cache_path"]) as data:
            gain = data["positive_gain"].astype(np.float64)
            selectable = int(record["selectable_count"])
            values = gain[:selectable]
            curve = _curve(values, selectable, ratios)
            arrays.append(values)
            per_curves.append(curve)
            total_tokens += selectable
            total_positive += int(np.count_nonzero(values > 0))
            total_gain += float(values.sum())
            per_rows.append({"id": record["id"], "tokens": selectable, "positive_tokens": int(np.count_nonzero(values > 0)), **curve})

    joined = np.concatenate(arrays)
    global_curve = _curve(joined, len(joined), ratios)
    mean_curve = {f"{ratio:.4f}": float(np.mean([curve[f"{ratio:.4f}"] for curve in per_curves])) for ratio in ratios}
    ci = _bootstrap(per_curves, ratios, int(selection["bootstrap_samples"]), int(selection["seed"]))
    random_curve = _random_baseline(arrays, ratios, int(selection["random_trials"]), int(selection["seed"]))

    support_by_trajectory = {
        "C10_ge_0.60_fraction": float(np.mean([curve["0.1000"] >= 0.60 for curve in per_curves])),
        "C20_ge_0.80_fraction": float(np.mean([curve["0.2000"] >= 0.80 for curve in per_curves])),
        "both_fraction": float(
            np.mean([curve["0.1000"] >= 0.60 and curve["0.2000"] >= 0.80 for curve in per_curves])
        ),
    }

    window_statistics: dict[str, Any] = {}
    core_overlap: dict[str, Any] = {}
    fragment_payloads = [json.loads(Path(record["fragment_path"]).read_text(encoding="utf-8")) for record in cache_records]
    for score_name in selection["score_names"]:
        window_statistics[score_name] = {}
        for ratio in [float(value) for value in selection["training_ratios"]]:
            key = f"{ratio:.4f}"
            core_fractions, window_fractions, window_counts = [], [], []
            for payload in fragment_payloads:
                selected = payload["selections"][score_name][key]
                denominator = payload["selectable_count"]
                core_fractions.append(len(selected["core_indices"]) / denominator)
                window_fractions.append(len(selected["window_indices"]) / denominator)
                window_counts.append(len(selected["windows"]))
            window_statistics[score_name][key] = {
                "mean_core_fraction": float(np.mean(core_fractions)),
                "mean_window_fraction": float(np.mean(window_fractions)),
                "median_window_fraction": float(np.median(window_fractions)),
                "mean_merged_windows": float(np.mean(window_counts)),
            }
    if {"positive_gain", "combined"}.issubset(set(selection["score_names"])):
        for ratio in [float(value) for value in selection["training_ratios"]]:
            key = f"{ratio:.4f}"
            similarities = []
            for payload in fragment_payloads:
                first = set(payload["selections"]["positive_gain"][key]["core_indices"])
                second = set(payload["selections"]["combined"][key]["core_indices"])
                similarities.append(len(first & second) / len(first | second))
            core_overlap[key] = {
                "mean_jaccard": float(np.mean(similarities)),
                "median_jaccard": float(np.median(similarities)),
            }

    top_token_examples: list[dict[str, Any]] = []
    for record in cache_records:
        with np.load(record["cache_path"]) as data:
            selectable = int(record["selectable_count"])
            order = np.argsort(-data["positive_gain"][:selectable], kind="stable")[:3]
            for position in order:
                start, end = max(0, int(position) - 8), min(selectable, int(position) + 9)
                top_token_examples.append(
                    {
                        "id": record["id"],
                        "position": int(position),
                        "positive_gain": float(data["positive_gain"][position]),
                        "js": float(data["js"][position]),
                        "token": tokenizer.decode([int(data["target_ids"][position])], skip_special_tokens=False),
                        "context": tokenizer.decode(data["target_ids"][start:end].tolist(), skip_special_tokens=False),
                    }
                )
    top_token_examples.sort(key=lambda item: item["positive_gain"], reverse=True)
    _atomic_json(out_root / "top_token_examples.json", top_token_examples)

    with (out_root / "concentration.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["id", "tokens", "positive_tokens"] + [f"{ratio:.4f}" for ratio in ratios]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_rows)

    _write_svg(out_root / "concentration.svg", ratios, global_curve, mean_curve, random_curve)
    summary = {
        "num_trajectories": len(cache_records),
        "total_selectable_target_tokens": total_tokens,
        "positive_gain_tokens": total_positive,
        "positive_gain_token_fraction": total_positive / total_tokens,
        "total_positive_gain": total_gain,
        "global_concentration": global_curve,
        "mean_trajectory_concentration": mean_curve,
        "mean_trajectory_95pct_bootstrap_ci": ci,
        "random_baseline": random_curve,
        "support_by_trajectory": support_by_trajectory,
        "window_statistics": window_statistics,
        "positive_gain_vs_combined_core_overlap": core_overlap,
        "preregistered_support_rule": "mean C(10%) >= 0.60 and mean C(20%) >= 0.80",
        "supports_selective_training": mean_curve["0.1000"] >= 0.60 and mean_curve["0.2000"] >= 0.80,
    }
    _atomic_json(out_root / "summary.json", summary)

    manifests = out_root / "training_manifests"
    manifests.mkdir(exist_ok=True)
    left, right = int(selection["window_left"]), int(selection["window_right"])
    for score_name in selection["score_names"]:
        for ratio in [float(value) for value in selection["training_ratios"]]:
            path = manifests / f"{score_name}_top{ratio:g}_L{left}_R{right}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row, record in zip(rows, cache_records, strict=True):
                    fragment_path = Path(record["fragment_path"])
                    fragments = json.loads(fragment_path.read_text(encoding="utf-8"))
                    key = f"{ratio:.4f}"
                    selection_data = fragments["selections"][score_name][key]
                    manifest = {
                        "id": record["id"],
                        "messages": record["clean_messages"],
                        "target": str(row["target"]).strip(),
                        "score_cache": str(Path(record["cache_path"]).resolve()),
                        "score_name": score_name,
                        "selection_ratio": ratio,
                        "selected_core_indices": selection_data["core_indices"],
                        "selected_token_indices": selection_data["window_indices"],
                        "windows": selection_data["windows"],
                    }
                    handle.write(json.dumps(manifest, ensure_ascii=False) + "\n")

    q10, q20 = mean_curve["0.1000"], mean_curve["0.2000"]
    verdict = "支持进入第二阶段" if summary["supports_selective_training"] else "未达到预注册阈值，暂不支持直接进入第二阶段"
    report = [
        "# SpreadsheetBench Selective Distillation — Stage 1",
        "",
        f"- 成功轨迹：{len(cache_records)}",
        f"- 可选择目标 token：{total_tokens}",
        f"- 正 Skill 增益 token：{total_positive} ({total_positive / total_tokens:.2%})",
        f"- 平均轨迹 C(10%)：{q10:.4f}，95% bootstrap CI [{ci['0.1000'][0]:.4f}, {ci['0.1000'][1]:.4f}]",
        f"- 平均轨迹 C(20%)：{q20:.4f}，95% bootstrap CI [{ci['0.2000'][0]:.4f}, {ci['0.2000'][1]:.4f}]",
        f"- 预注册判据：C(10%) ≥ 0.60 且 C(20%) ≥ 0.80",
        f"- 单轨迹同时达标比例：{support_by_trajectory['both_fraction']:.2%}",
        f"- Top 5% positive-gain core 经 L2/R8 扩窗后平均覆盖：{window_statistics['positive_gain']['0.0500']['mean_window_fraction']:.2%}",
        f"- Top 10% positive-gain core 经 L2/R8 扩窗后平均覆盖：{window_statistics['positive_gain']['0.1000']['mean_window_fraction']:.2%}",
        f"- Top 20% positive-gain core 经 L2/R8 扩窗后平均覆盖：{window_statistics['positive_gain']['0.2000']['mean_window_fraction']:.2%}",
        f"- 结论：**{verdict}**",
        "",
        "完整统计见 `summary.json`，逐轨迹曲线见 `concentration.csv`，代表 token 见 `top_token_examples.json`，可训练掩码见 `training_manifests/`。",
    ]
    (out_root / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg, model_cfg, scoring_cfg = cfg["data"], cfg["model"], cfg["scoring"]
    examples_path = _resolve(data_cfg["examples_path"])
    skill_path = _resolve(data_cfg["skill_path"])
    out_root = _resolve(args.out_root or cfg["output"]["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    rows = _read_examples(examples_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    skill_text = skill_path.read_text(encoding="utf-8").strip()
    if not skill_text:
        raise ValueError("A non-empty full text Skill is required")

    import torch
    from transformers import AutoTokenizer

    model_source = args.model_path or model_cfg["name"]
    revision = model_cfg.get("revision") or None
    print(f"Loading tokenizer and inspecting caches for {model_source}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        revision=revision,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        local_files_only=bool(args.model_path),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = str(model_cfg.get("device", "cuda"))
    model = None

    run_metadata = {
        "config": cfg,
        "config_path": str(config_path),
        "model_source": str(Path(model_source).resolve()) if args.model_path else model_source,
        "model_revision": revision,
        "examples_sha256": _sha256(examples_path),
        "skill_sha256": _sha256(skill_path),
        "skill_chars": len(skill_text),
        "num_examples": len(rows),
        "created_unix": time.time(),
    }
    _atomic_json(out_root / "run_metadata.json", run_metadata)
    cache_dir, fragment_dir = out_root / "token_scores", out_root / "fragments"
    cache_records: list[dict[str, Any]] = []
    selection_cfg = cfg["selection"]
    ratios = [float(value) for value in selection_cfg["ratios"]]

    for index, row in enumerate(rows):
        identifier = str(row.get("id") or f"row-{index}")
        slug = _slug(identifier, index)
        cache_path = cache_dir / f"{slug}.npz"
        fragment_path = fragment_dir / f"{slug}.json"
        encoded = _encode_example(
            tokenizer,
            row,
            skill_text,
            int(data_cfg["max_prompt_tokens"]),
            int(data_cfg["max_target_tokens"]),
        )
        target_ids = encoded["target_ids"]
        eos_ids = set(tokenizer.encode(tokenizer.eos_token or "", add_special_tokens=False))
        selectable = len(target_ids) - 1 if target_ids[-1] in eos_ids else len(target_ids)
        if args.force or not cache_path.exists():
            print(
                f"[{index + 1}/{len(rows)}] {identifier}: skill_prompt={len(encoded['skill_prompt_ids'])}, "
                f"clean_prompt={len(encoded['clean_prompt_ids'])}, target={len(target_ids)}",
                flush=True,
            )
            if model is None:
                model = _load_frozen_model(model_source, model_cfg, device, bool(args.model_path))
            logits = _forward_pair(model, encoded, int(tokenizer.pad_token_id), device)
            arrays = _score_logits(
                logits,
                target_ids,
                exact_js=bool(scoring_cfg.get("exact_js", True)),
                top_k=int(scoring_cfg.get("top_k", 64)),
                chunk_size=int(scoring_cfg.get("score_chunk_size", 16)),
            )
            _atomic_npz(cache_path, **arrays)
            del logits
            torch.cuda.empty_cache()
        with np.load(cache_path) as cached:
            selections: dict[str, Any] = {}
            for score_name in selection_cfg["score_names"]:
                score = cached[score_name]
                score_selections: dict[str, Any] = {}
                for ratio in ratios:
                    core = _select_core(score, selectable, ratio)
                    window_indices, windows = _expand_windows(
                        core,
                        selectable,
                        int(selection_cfg["window_left"]),
                        int(selection_cfg["window_right"]),
                    )
                    score_selections[f"{ratio:.4f}"] = {
                        "core_indices": core,
                        "window_indices": window_indices,
                        "windows": windows,
                        "core_fraction": len(core) / selectable,
                        "window_fraction": len(window_indices) / selectable,
                        "fragments": _decode_fragments(tokenizer, cached["target_ids"], windows, score),
                    }
                selections[score_name] = score_selections
        fragment = {
            "id": identifier,
            "cache_path": str(cache_path.resolve()),
            "target_token_count": len(target_ids),
            "selectable_count": selectable,
            "selections": selections,
        }
        _atomic_json(fragment_path, fragment)
        cache_records.append(
            {
                "id": identifier,
                "cache_path": str(cache_path.resolve()),
                "fragment_path": str(fragment_path.resolve()),
                "selectable_count": selectable,
                "clean_messages": encoded["clean_messages"],
            }
        )

    _aggregate(cfg, rows, cache_records, tokenizer, out_root)
    print(f"Stage 1 complete: {out_root}", flush=True)


if __name__ == "__main__":
    main()
