#!/usr/bin/env python3
"""Train the PRCB family of staged soft-prefix optimizers.

Both methods warm-start the frozen Combined length-8 prefix. At every round they:
1. recompute residual Combined scores against the current prefix;
2. select dynamic harmful No-Skill anchor positions;
3. choose rows by a gradient probe or a fixed causal sliding schedule;
4. optimize only those rows on gold teacher-forced trajectories; and
5. commit a fixed or monitor-selected fraction of the staged update.

PRCB-v5 additionally retains the stage-start Top-64 distribution on replay
positions and allows alpha=0 to reject a harmful stage.

The test split is never read.  Final evaluation is valid_seen only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skillopt.envs.spreadsheetbench.dataloader import SpreadsheetBenchDataLoader
from skillopt.softprefix.data import (
    PrefixBatchCollator,
    TextTrajectoryPrefixDataset,
    _apply_text_chat_template,
)
from skillopt.softprefix.model import SoftPrefixCausalLM
from skillopt.softprefix.prcb import (
    causal_prefix_pair,
    choose_prefix_pair,
    mask_prefix_gradient_,
    margin_decision_locator,
    reference_decision_margin,
    residual_combined_scores,
    select_harmful_anchor_positions,
    select_margin_decision_top_fraction,
    select_positive_top_fraction,
    shrink_pair_update_,
    sliding_prefix_pair,
    topk_residual_forward_kl,
    topk_residual_js,
)
from skillopt.softprefix.trainer import (
    _batch_to_tensors,
    _normalized_adapter_accumulation_loss,
    evaluate_spreadsheet_prefix,
)


def parse_args(
    *,
    default_pair_policy: str = "gradient",
    default_out_root: str = "outputs/SpreadsheetBench_prcb_v1_len8_seed1",
    default_locator_policy: str = "combined_residual",
    default_method_version: str = "auto",
    default_teacher_kl_weight: float = 0.0,
    default_teacher_margin_weight: float = 0.0,
    default_rounds: int = 4,
    default_round_step_pattern: str = "",
    default_early_stop: bool = False,
    default_sliding_window_size: int = 2,
    default_retention_weight: float = 0.0,
    default_stage_alpha_grid: str = "",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("SPREADSHEETBENCH_MODEL", "Qwen/Qwen3.6-35B-A3B"),
    )
    parser.add_argument(
        "--initial-checkpoint",
        default=(
            "outputs/SpreadsheetBench_full_distribution_locator_len8_seed1_shared/"
            "combined_top0.05_core_shared_preserve/best_prefix.pt"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=(
            "outputs/SpreadsheetBench_selective_stage2_manifests/"
            "combined_top0.05_core_shared_preserve.jsonl"
        ),
    )
    parser.add_argument(
        "--out-root",
        default=default_out_root,
    )
    parser.add_argument(
        "--pair-policy",
        choices=("gradient", "tail_to_head", "head_to_tail", "sliding_head_to_tail"),
        default=default_pair_policy,
        help="How to choose the two trainable prefix rows in each round.",
    )
    parser.add_argument(
        "--locator-policy",
        choices=("combined_residual", "margin_decision"),
        default=default_locator_policy,
    )
    parser.add_argument(
        "--method-version",
        choices=("auto", "v1", "v2", "v3", "v4", "v4_es", "v5"),
        default=default_method_version,
    )
    parser.add_argument("--split-dir", default="data/spreadsheetbench_split")
    parser.add_argument("--data-root", default="data/spreadsheetbench_verified_400")
    parser.add_argument("--rounds", type=int, default=default_rounds)
    parser.add_argument("--steps-per-round", type=int, default=8)
    parser.add_argument(
        "--round-step-pattern",
        default=default_round_step_pattern,
        help="Optional comma-separated optimizer-step count for each round.",
    )
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--probe-examples", type=int, default=4)
    parser.add_argument("--selection-ratio", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--shrinkage", type=float, default=0.25)
    parser.add_argument("--preservation-weight", type=float, default=1.0)
    parser.add_argument(
        "--retention-weight",
        type=float,
        default=default_retention_weight,
        help="KL weight for retaining the stage-start distribution on replay tokens.",
    )
    parser.add_argument(
        "--retention-kl-limit",
        type=float,
        default=0.02,
        help="Maximum monitor KL from the stage-start prefix on replay tokens.",
    )
    parser.add_argument(
        "--stage-alpha-grid",
        default=default_stage_alpha_grid,
        help="Optional comma-separated stage commit scales; include 0 to allow rejection.",
    )
    parser.add_argument("--teacher-kl-weight", type=float, default=default_teacher_kl_weight)
    parser.add_argument(
        "--teacher-margin-weight", type=float, default=default_teacher_margin_weight
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--score-chunk-size", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=16384)
    parser.add_argument("--max-target-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--early-stop",
        action=argparse.BooleanOptionalAction,
        default=default_early_stop,
    )
    parser.add_argument("--monitor-trajectories", type=int, default=12)
    parser.add_argument("--monitor-eval-interval", type=int, default=2)
    parser.add_argument("--monitor-min-steps", type=int, default=4)
    parser.add_argument("--monitor-patience", type=int, default=3)
    parser.add_argument("--monitor-min-relative-improvement", type=float, default=0.002)
    parser.add_argument("--monitor-preservation-ratio-limit", type=float, default=1.10)
    parser.add_argument("--max-steps-per-stage", type=int, default=32)
    parser.add_argument(
        "--sliding-window-size",
        type=int,
        default=default_sliding_window_size,
        help="Number of contiguous prefix rows updated by sliding_head_to_tail.",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_model_reference(value: str) -> str:
    """Keep Hub model IDs intact while resolving local filesystem paths."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    local_path = PROJECT_ROOT / path
    return str(local_path) if local_path.exists() else value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    """Hash tensor values without depending on torch.save metadata."""
    raw = tensor.detach().cpu().contiguous().float().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def slug(identifier: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier)[:100]


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_stage_alpha_grid(raw: str) -> list[float]:
    """Parse and validate deterministic stage-commit interpolation factors."""
    value = str(raw or "").strip()
    if not value:
        return []
    alphas = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not alphas:
        raise ValueError("stage-alpha-grid must contain at least one number")
    if any(not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("Every stage alpha must be finite and between zero and one")
    if 0.0 not in alphas:
        raise ValueError("stage-alpha-grid must include alpha=0 for stage rejection")
    return alphas


def encode_trajectory(
    prefix_model: SoftPrefixCausalLM,
    row: dict[str, Any],
    *,
    max_prompt_tokens: int,
    max_target_tokens: int,
) -> tuple[list[int], list[int]]:
    prompt = _apply_text_chat_template(
        prefix_model.tokenizer,
        list(row["messages"]),
        enable_thinking=False,
        add_generation_prompt=True,
    )
    target = str(row["target"]).strip()
    eos = prefix_model.tokenizer.eos_token
    if eos and not target.endswith(eos):
        target += eos
    prompt_ids = prefix_model.tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_tokens,
    )["input_ids"]
    target_ids = prefix_model.tokenizer(
        target,
        add_special_tokens=False,
        truncation=True,
        max_length=max_target_tokens,
    )["input_ids"]
    return list(prompt_ids), list(target_ids)


def score_current_prefix(
    prefix_model: SoftPrefixCausalLM,
    row: dict[str, Any],
    *,
    max_prompt_tokens: int,
    max_target_tokens: int,
    chunk_size: int,
    locator_policy: str = "combined_residual",
    cache_current_topk: bool = False,
) -> dict[str, np.ndarray]:
    import torch

    prompt_ids, target_ids = encode_trajectory(
        prefix_model,
        row,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
    )
    cache_path = resolve(str(row["score_cache"]))
    with np.load(cache_path) as cached:
        arrays = {name: cached[name] for name in cached.files}
    if arrays["target_ids"].astype(np.int64).tolist() != target_ids:
        raise ValueError(f"Tokenizer/cache mismatch for trajectory {row['id']!r}")

    input_ids = torch.tensor(
        [prompt_ids + target_ids],
        dtype=torch.long,
        device=prefix_model.device,
    )
    attention_mask = torch.ones_like(input_ids)
    inputs_embeds, full_attention, _ = prefix_model._with_prefix(input_ids, attention_mask)
    first_logit = prefix_model.prefix_length + len(prompt_ids) - 1
    positions = torch.arange(
        first_logit,
        first_logit + len(target_ids),
        dtype=torch.long,
        device=prefix_model.device,
    )
    with torch.inference_mode():
        output = prefix_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            use_cache=False,
            output_router_logits=False,
            logits_to_keep=positions,
            return_dict=True,
        )
    logits = output.logits[0]
    if int(logits.shape[0]) != len(target_ids):
        raise ValueError(
            f"Unexpected selected-logit shape for {row['id']}: {tuple(logits.shape)}"
        )

    current_target_logp = np.empty(len(target_ids), dtype=np.float32)
    teacher_student_js = np.empty(len(target_ids), dtype=np.float32)
    base_student_kl = np.empty(len(target_ids), dtype=np.float32)
    current_margin = np.empty(len(target_ids), dtype=np.float32)
    current_top1_gold = np.empty(len(target_ids), dtype=np.bool_)
    retention_top_k = min(64, int(logits.shape[-1])) if cache_current_topk else 2
    current_topk_ids = (
        np.empty((len(target_ids), retention_top_k), dtype=np.int32)
        if cache_current_topk
        else None
    )
    current_topk_logp = (
        np.empty((len(target_ids), retention_top_k), dtype=np.float32)
        if cache_current_topk
        else None
    )
    current_residual_log_mass = (
        np.empty(len(target_ids), dtype=np.float32) if cache_current_topk else None
    )
    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=prefix_model.device)
    for start in range(0, len(target_ids), chunk_size):
        end = min(start + chunk_size, len(target_ids))
        chunk = logits[start:end].float()
        current_lp = torch.log_softmax(chunk, dim=-1)
        retention_values, retention_ids = torch.topk(
            current_lp, k=retention_top_k, dim=-1
        )
        if cache_current_topk:
            if current_topk_ids is None or current_topk_logp is None:
                raise AssertionError("Current Top-k cache was not allocated")
            if current_residual_log_mass is None:
                raise AssertionError("Current residual cache was not allocated")
            retention_mass = retention_values.exp().sum(dim=-1).clamp(max=1.0 - 1e-7)
            current_topk_ids[start:end] = retention_ids.cpu().numpy().astype(np.int32)
            current_topk_logp[start:end] = retention_values.cpu().numpy().astype(np.float32)
            current_residual_log_mass[start:end] = (
                torch.log1p(-retention_mass).cpu().numpy().astype(np.float32)
            )
        current_target_logp[start:end] = (
            current_lp.gather(1, target_tensor[start:end, None]).squeeze(1).cpu().numpy()
        )
        top_values, top_ids = retention_values[:, :2], retention_ids[:, :2]
        local_targets = target_tensor[start:end]
        competitor_logp = torch.where(
            top_ids[:, 0] == local_targets,
            top_values[:, 1],
            top_values[:, 0],
        )
        current_margin[start:end] = (
            current_lp.gather(1, local_targets[:, None]).squeeze(1) - competitor_logp
        ).cpu().numpy()
        current_top1_gold[start:end] = (top_ids[:, 0] == local_targets).cpu().numpy()
        teacher_student_js[start:end] = topk_residual_js(
            torch,
            chunk,
            reference_topk_ids=torch.from_numpy(arrays["skill_topk_ids"][start:end]),
            reference_topk_logp=torch.from_numpy(arrays["skill_topk_logp"][start:end]),
            reference_residual_log_mass=torch.from_numpy(
                arrays["skill_residual_log_mass"][start:end]
            ),
        ).cpu().numpy()
        base_student_kl[start:end] = topk_residual_forward_kl(
            torch,
            chunk,
            reference_topk_ids=torch.from_numpy(arrays["clean_topk_ids"][start:end]),
            reference_topk_logp=torch.from_numpy(arrays["clean_topk_logp"][start:end]),
            reference_residual_log_mass=torch.from_numpy(
                arrays["clean_residual_log_mass"][start:end]
            ),
        ).cpu().numpy()

    skill_margin, skill_top1_gold = reference_decision_margin(
        target_ids=np.asarray(target_ids),
        target_logp=arrays["skill_target_logp"],
        topk_ids=arrays["skill_topk_ids"],
        topk_logp=arrays["skill_topk_logp"],
    )
    clean_margin, clean_top1_gold = reference_decision_margin(
        target_ids=np.asarray(target_ids),
        target_logp=arrays["clean_target_logp"],
        topk_ids=arrays["clean_topk_ids"],
        topk_logp=arrays["clean_topk_logp"],
    )
    margin_values = margin_decision_locator(
        skill_margin=skill_margin,
        clean_margin=clean_margin,
        current_margin=current_margin,
        skill_top1_gold=skill_top1_gold,
        clean_top1_gold=clean_top1_gold,
        current_top1_gold=current_top1_gold,
    )
    if locator_policy == "margin_decision":
        dynamic = margin_values["margin_residual_mass"]
        teacher_beneficial = margin_values["original_margin_gain"] > 0
    elif locator_policy == "combined_residual":
        dynamic = residual_combined_scores(
            teacher_target_logp=arrays["skill_target_logp"],
            student_target_logp=current_target_logp,
            teacher_beneficial=arrays["positive_gain"] > 0,
            teacher_student_js=teacher_student_js,
        )
        teacher_beneficial = arrays["positive_gain"] > 0
    else:
        raise ValueError(f"Unknown locator policy: {locator_policy}")
    del output, logits, inputs_embeds, full_attention, input_ids, attention_mask
    gc.collect()
    torch.cuda.empty_cache()
    result = {
        "target_ids": np.asarray(target_ids, dtype=np.int32),
        "dynamic_score": dynamic,
        "current_target_logp": current_target_logp,
        "teacher_student_js": teacher_student_js,
        "base_student_kl": base_student_kl,
        "teacher_beneficial": np.asarray(teacher_beneficial, dtype=np.bool_),
        "skill_margin": skill_margin,
        "clean_margin": clean_margin,
        "current_margin": current_margin,
        "skill_top1_gold": skill_top1_gold.astype(np.bool_),
        "clean_top1_gold": clean_top1_gold.astype(np.bool_),
        "current_top1_gold": current_top1_gold.astype(np.bool_),
        **margin_values,
    }
    if cache_current_topk:
        result.update(
            current_topk_ids=current_topk_ids,
            current_topk_logp=current_topk_logp,
            current_residual_log_mass=current_residual_log_mass,
        )
    return result


def round_schedule(
    total: int,
    *,
    round_index: int,
    examples: int,
    seed: int,
    start_offset: int | None = None,
) -> list[int]:
    order = list(range(total))
    random.Random(seed).shuffle(order)
    start = (
        int(start_offset) % total
        if start_offset is not None
        else ((round_index - 1) * examples) % total
    )
    return [order[(start + offset) % total] for offset in range(examples)]


def optimizer_steps_by_round(
    *,
    rounds: int,
    steps_per_round: int,
    pattern: str,
) -> list[int]:
    """Resolve a fixed per-round optimizer budget, validating it up front."""
    raw = str(pattern or "").strip()
    if raw:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
        if len(values) != rounds:
            raise ValueError(
                f"round-step-pattern must contain {rounds} values, got {len(values)}"
            )
    else:
        values = [int(steps_per_round)] * rounds
    if any(value <= 0 for value in values):
        raise ValueError("Every round must contain at least one optimizer step")
    return values


def fixed_monitor_split(
    rows: list[dict[str, Any]],
    *,
    monitor_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Return deterministic train/monitor IDs without consulting outcomes."""
    if not 0 < monitor_count < len(rows):
        raise ValueError("monitor-trajectories must be between 1 and N-1")
    identifiers = [str(row["id"]) for row in rows]
    shuffled = list(identifiers)
    random.Random(seed).shuffle(shuffled)
    monitor = set(shuffled[:monitor_count])
    train_ids = [identifier for identifier in identifiers if identifier not in monitor]
    monitor_ids = [identifier for identifier in identifiers if identifier in monitor]
    return train_ids, monitor_ids


def build_round_rows(
    rows: list[dict[str, Any]],
    scores: dict[str, dict[str, np.ndarray]],
    previous_core: dict[str, list[int]],
    *,
    ratio: float,
    locator_policy: str = "combined_residual",
) -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    current_core: dict[str, list[int]] = {}
    total_score = selected_score = 0.0
    selected_count = anchor_count = replay_count = overlap_count = union_count = 0
    eligible_count = decisive_count = selected_decisive_count = 0
    per_trajectory: list[dict[str, Any]] = []
    for row in rows:
        identifier = str(row["id"])
        values = scores[identifier]
        selectable = len(values["target_ids"]) - 1
        dynamic = values["dynamic_score"][:selectable]
        if locator_policy == "margin_decision":
            core = select_margin_decision_top_fraction(
                priority=values["margin_priority"][:selectable],
                residual=values["margin_residual"][:selectable],
                original_gain=values["original_margin_gain"][:selectable],
                teacher_student_js=values["teacher_student_js"][:selectable],
                ratio=ratio,
            )
        elif locator_policy == "combined_residual":
            core = select_positive_top_fraction(dynamic, ratio)
        else:
            raise ValueError(f"Unknown locator policy: {locator_policy}")
        if not core:
            raise ValueError(f"No eligible {locator_policy} positions for {identifier}")
        replay = [
            int(index)
            for index in previous_core.get(identifier, [])
            if 0 <= int(index) < selectable
        ]
        train_indices = sorted(set(core) | set(replay))
        anchor = select_harmful_anchor_positions(
            values["base_student_kl"][:selectable],
            count=len(core),
            excluded=set(train_indices),
            teacher_beneficial=values["teacher_beneficial"][:selectable],
        )
        if len(anchor) != len(core):
            raise ValueError(f"Could not match anchor budget for {identifier}")
        result = {
            "id": identifier,
            "messages": row["messages"],
            "target": row["target"],
            "score_cache": row["score_cache"],
            "selected_indices": train_indices,
            "selected_core_indices": core,
            "replay_indices": replay,
            "preserve_indices": anchor,
            "stage2_selector": f"prcb_{locator_policy}",
        }
        output.append(result)
        current_core[identifier] = core
        total_score += float(dynamic.sum())
        selected_score += float(dynamic[core].sum())
        selected_count += len(core)
        anchor_count += len(anchor)
        replay_count += len(replay)
        if locator_policy == "margin_decision":
            priorities = values["margin_priority"][:selectable]
            eligible_count += int((priorities > 0).sum())
            decisive_count += int((priorities == 2).sum())
            selected_decisive_count += sum(int(priorities[index]) == 2 for index in core)
        old, new = set(replay), set(core)
        overlap_count += len(old & new)
        union_count += len(old | new)
        per_trajectory.append(
            {
                "id": identifier,
                "selectable": selectable,
                "core": len(core),
                "replay": len(replay),
                "anchor": len(anchor),
                "residual_mass": float(dynamic.sum()),
                "selected_residual_mass": float(dynamic[core].sum()),
                "jaccard_vs_previous": (
                    len(old & new) / len(old | new) if old | new else None
                ),
                "eligible_margin_tokens": (
                    int((values["margin_priority"][:selectable] > 0).sum())
                    if locator_policy == "margin_decision"
                    else None
                ),
                "decisive_margin_tokens": (
                    int((values["margin_priority"][:selectable] == 2).sum())
                    if locator_policy == "margin_decision"
                    else None
                ),
                "selected_decisive_tokens": (
                    sum(
                        int(values["margin_priority"][index]) == 2
                        for index in core
                    )
                    if locator_policy == "margin_decision"
                    else None
                ),
            }
        )
    statistics = {
        "trajectories": len(rows),
        "selected_core_tokens": selected_count,
        "replay_tokens": replay_count,
        "anchor_tokens": anchor_count,
        "total_residual_combined_mass": total_score,
        "selected_residual_combined_mass": selected_score,
        "total_locator_mass": total_score,
        "selected_locator_mass": selected_score,
        "residual_mass_capture": selected_score / total_score if total_score > 0 else 0.0,
        "global_jaccard_vs_previous": (
            overlap_count / union_count if union_count else None
        ),
        "locator_policy": locator_policy,
        "eligible_margin_tokens": eligible_count,
        "decisive_margin_tokens": decisive_count,
        "selected_decisive_tokens": selected_decisive_count,
        "per_trajectory": per_trajectory,
    }
    return output, current_core, statistics


def tensorize_example(
    torch: Any,
    dataset: TextTrajectoryPrefixDataset,
    collator: PrefixBatchCollator,
    index: int,
    device: Any,
) -> dict[str, Any]:
    return _batch_to_tensors(torch, collator([dataset[index]]), device)


def gradient_probe(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    dataset: TextTrajectoryPrefixDataset,
    collator: PrefixBatchCollator,
    indices: list[int],
    *,
    preservation_weight: float,
) -> tuple[list[int], list[float]]:
    prefix_model.prefix_embeddings.grad = None
    for index in indices:
        batch = tensorize_example(torch, dataset, collator, index, prefix_model.device)
        batch["preservation_loss_weight"] = preservation_weight
        output = prefix_model.forward(batch)
        (output.loss / len(indices)).backward()
        del batch, output
        gc.collect()
        torch.cuda.empty_cache()
    if prefix_model.prefix_embeddings.grad is None:
        raise RuntimeError("Gradient probe produced no prefix gradient")
    pair, scores = choose_prefix_pair(
        prefix_model.prefix_embeddings.grad,
        prefix_model.prefix_embeddings,
    )
    prefix_model.prefix_embeddings.grad = None
    return pair, scores


def train_round(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    dataset: TextTrajectoryPrefixDataset,
    collator: PrefixBatchCollator,
    schedule: list[int],
    *,
    pair: list[int],
    steps: int,
    accumulation: int,
    learning_rate: float,
    shrinkage: float,
    preservation_weight: float,
    teacher_kl_weight: float = 0.0,
    teacher_margin_weight: float = 0.0,
) -> dict[str, Any]:
    if len(schedule) != steps * accumulation:
        raise ValueError("Training schedule size must equal steps * accumulation")
    round_start = prefix_model.prefix_embeddings.detach().clone()
    frozen_rows = [index for index in range(prefix_model.prefix_length) if index not in pair]
    optimizer = torch.optim.AdamW(
        [prefix_model.prefix_embeddings],
        lr=learning_rate,
        weight_decay=0.0,
    )
    selected_sum = preservation_sum = teacher_kl_sum = teacher_margin_sum = 0.0
    selected_tokens_seen = preservation_tokens_seen = teacher_tokens_seen = 0
    progress = tqdm(range(steps), desc="  Pair train", unit="step", leave=False)
    for step in progress:
        group_indices = schedule[step * accumulation : (step + 1) * accumulation]
        tensor_group = [
            tensorize_example(torch, dataset, collator, index, prefix_model.device)
            for index in group_indices
        ]
        selected_counts = [int((batch["labels"] != -100).sum().item()) for batch in tensor_group]
        preservation_counts = [
            int(batch["preservation_mask"].sum().item()) for batch in tensor_group
        ]
        teacher_counts = [
            int(batch["teacher_mask"].sum().item()) if "teacher_mask" in batch else 0
            for batch in tensor_group
        ]
        group_selected = sum(selected_counts)
        group_preservation = sum(preservation_counts)
        group_teacher = sum(teacher_counts)
        optimizer.zero_grad(set_to_none=True)
        last_loss = 0.0
        for batch, selected_count, preservation_count, teacher_count in zip(
            tensor_group,
            selected_counts,
            preservation_counts,
            teacher_counts,
            strict=True,
        ):
            batch["preservation_loss_weight"] = preservation_weight
            batch["teacher_kl_loss_weight"] = teacher_kl_weight
            batch["teacher_margin_loss_weight"] = teacher_margin_weight
            output = prefix_model.forward(batch)
            loss = output.selected_loss * (selected_count / group_selected)
            if preservation_count:
                loss = loss + (
                    preservation_weight
                    * output.preservation_loss
                    * (preservation_count / group_preservation)
                )
            if teacher_count:
                if group_teacher <= 0:
                    raise ValueError("Teacher tokens require a positive group total")
                teacher_scale = teacher_count / group_teacher
                loss = (
                    loss
                    + teacher_kl_weight * output.teacher_kl_loss * teacher_scale
                    + teacher_margin_weight * output.teacher_margin_loss * teacher_scale
                )
            loss.backward()
            selected_value = float(output.selected_loss.detach().cpu())
            preservation_value = float(output.preservation_loss.detach().cpu())
            teacher_kl_value = float(output.teacher_kl_loss.detach().cpu())
            teacher_margin_value = float(output.teacher_margin_loss.detach().cpu())
            selected_sum += selected_value * selected_count
            preservation_sum += preservation_value * preservation_count
            teacher_kl_sum += teacher_kl_value * teacher_count
            teacher_margin_sum += teacher_margin_value * teacher_count
            selected_tokens_seen += selected_count
            preservation_tokens_seen += preservation_count
            teacher_tokens_seen += teacher_count
            last_loss = (
                selected_value
                + preservation_weight * preservation_value
                + teacher_kl_weight * teacher_kl_value
                + teacher_margin_weight * teacher_margin_value
            )
            del batch, output, loss
            gc.collect()
            torch.cuda.empty_cache()
        gradient = prefix_model.prefix_embeddings.grad
        if gradient is None:
            raise RuntimeError("Pair training produced no gradient")
        mask_prefix_gradient_(gradient, pair)
        torch.nn.utils.clip_grad_norm_([prefix_model.prefix_embeddings], max_norm=1.0)
        optimizer.step()
        if not torch.equal(
            prefix_model.prefix_embeddings.detach()[frozen_rows],
            round_start[frozen_rows],
        ):
            raise AssertionError("A frozen prefix row changed during pair training")
        progress.set_postfix(loss=f"{last_loss:.4f}", pair=str(pair))
    progress.close()
    pre_shrink_delta = (
        prefix_model.prefix_embeddings.detach() - round_start
    ).float().norm(dim=1).cpu().tolist()
    shrink_pair_update_(
        torch,
        prefix_model.prefix_embeddings,
        round_start=round_start,
        active_rows=pair,
        shrinkage=shrinkage,
    )
    if not torch.equal(
        prefix_model.prefix_embeddings.detach()[frozen_rows],
        round_start[frozen_rows],
    ):
        raise AssertionError("A frozen prefix row changed during shrinkage")
    post_shrink_delta = (
        prefix_model.prefix_embeddings.detach() - round_start
    ).float().norm(dim=1).cpu().tolist()
    return {
        "selected_ce_loss": selected_sum / max(selected_tokens_seen, 1),
        "preservation_kl_loss": preservation_sum / max(preservation_tokens_seen, 1),
        "teacher_top64_kl_loss": teacher_kl_sum / max(teacher_tokens_seen, 1),
        "teacher_margin_hinge_loss": teacher_margin_sum / max(teacher_tokens_seen, 1),
        "selected_tokens_seen": selected_tokens_seen,
        "preservation_tokens_seen": preservation_tokens_seen,
        "teacher_tokens_seen": teacher_tokens_seen,
        "pre_shrink_row_delta_norms": pre_shrink_delta,
        "post_shrink_row_delta_norms": post_shrink_delta,
    }


def evaluate_monitor_objective(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    dataset: TextTrajectoryPrefixDataset,
    collator: PrefixBatchCollator,
    *,
    preservation_weight: float,
    teacher_kl_weight: float,
    teacher_margin_weight: float,
    retention_weight: float = 0.0,
) -> dict[str, float | int]:
    """Evaluate one fixed teacher-forced monitor set without gradients."""
    was_training = bool(prefix_model.model.training)
    prefix_model.model.eval()
    selected_sum = preservation_sum = teacher_kl_sum = teacher_margin_sum = 0.0
    retention_sum = 0.0
    selected_tokens = preservation_tokens = teacher_tokens = retention_tokens = 0
    with torch.inference_mode():
        for index in range(len(dataset)):
            batch = tensorize_example(
                torch, dataset, collator, index, prefix_model.device
            )
            batch["preservation_loss_weight"] = preservation_weight
            batch["teacher_kl_loss_weight"] = teacher_kl_weight
            batch["teacher_margin_loss_weight"] = teacher_margin_weight
            batch["retention_loss_weight"] = retention_weight
            output = prefix_model.forward(batch)
            selected_count = int((batch["labels"] != -100).sum().item())
            preservation_count = int(batch["preservation_mask"].sum().item())
            teacher_count = int(batch["teacher_mask"].sum().item())
            retention_count = (
                int(batch["retention_mask"].sum().item())
                if "retention_mask" in batch
                else 0
            )
            selected_sum += float(output.selected_loss.cpu()) * selected_count
            preservation_sum += float(output.preservation_loss.cpu()) * preservation_count
            teacher_kl_sum += float(output.teacher_kl_loss.cpu()) * teacher_count
            teacher_margin_sum += float(output.teacher_margin_loss.cpu()) * teacher_count
            retention_sum += float(output.retention_loss.cpu()) * retention_count
            selected_tokens += selected_count
            preservation_tokens += preservation_count
            teacher_tokens += teacher_count
            retention_tokens += retention_count
            del batch, output
            gc.collect()
            torch.cuda.empty_cache()
    if was_training:
        prefix_model.model.train()
    selected = selected_sum / max(selected_tokens, 1)
    preservation = preservation_sum / max(preservation_tokens, 1)
    teacher_kl = teacher_kl_sum / max(teacher_tokens, 1)
    teacher_margin = teacher_margin_sum / max(teacher_tokens, 1)
    retention = retention_sum / max(retention_tokens, 1)
    composite = (
        selected
        + preservation_weight * preservation
        + teacher_kl_weight * teacher_kl
        + teacher_margin_weight * teacher_margin
        + retention_weight * retention
    )
    return {
        "composite_loss": composite,
        "selected_ce_loss": selected,
        "preservation_kl_loss": preservation,
        "teacher_top64_kl_loss": teacher_kl,
        "teacher_margin_hinge_loss": teacher_margin,
        "retention_top64_kl_loss": retention,
        "selected_tokens": selected_tokens,
        "preservation_tokens": preservation_tokens,
        "teacher_tokens": teacher_tokens,
        "retention_tokens": retention_tokens,
    }


def train_round_early_stopping(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    train_dataset: TextTrajectoryPrefixDataset,
    monitor_dataset: TextTrajectoryPrefixDataset,
    collator: PrefixBatchCollator,
    schedule: list[int],
    *,
    pair: list[int],
    max_steps: int,
    accumulation: int,
    learning_rate: float,
    shrinkage: float,
    preservation_weight: float,
    teacher_kl_weight: float,
    teacher_margin_weight: float,
    min_steps: int,
    eval_interval: int,
    patience: int,
    min_relative_improvement: float,
    preservation_ratio_limit: float,
    retention_weight: float = 0.0,
    retention_kl_limit: float = math.inf,
    stage_alpha_grid: list[float] | None = None,
) -> dict[str, Any]:
    """Train one pair and commit the best safe shrink-aware monitor candidate."""
    if len(schedule) != max_steps * accumulation:
        raise ValueError("Early-stop schedule size must equal max_steps * accumulation")
    if min_steps <= 0 or eval_interval <= 0 or patience <= 0:
        raise ValueError("Early-stop cadence values must be positive")
    round_start = prefix_model.prefix_embeddings.detach().clone()
    frozen_rows = [index for index in range(prefix_model.prefix_length) if index not in pair]
    optimizer = torch.optim.AdamW(
        [prefix_model.prefix_embeddings], lr=learning_rate, weight_decay=0.0
    )
    baseline = evaluate_monitor_objective(
        torch,
        prefix_model,
        monitor_dataset,
        collator,
        preservation_weight=preservation_weight,
        teacher_kl_weight=teacher_kl_weight,
        teacher_margin_weight=teacher_margin_weight,
        retention_weight=retention_weight,
    )
    baseline_preservation = float(baseline["preservation_kl_loss"])
    preservation_limit = baseline_preservation * preservation_ratio_limit
    retention_tokens = int(baseline["retention_tokens"])
    retention_limit = float(retention_kl_limit) if retention_tokens else math.inf
    best_loss = float(baseline["composite_loss"])
    patience_reference = best_loss
    best_step = 0
    best_pair = round_start[pair].clone()
    best_raw_pair = round_start[pair].clone()
    best_monitor = dict(baseline)
    history: list[dict[str, Any]] = [
        {
            "step": 0,
            **baseline,
            "preservation_safe": True,
            "significant_improvement": False,
        }
    ]
    stale_evaluations = 0
    stop_reason = "max_steps"
    selected_sum = preservation_sum = teacher_kl_sum = teacher_margin_sum = 0.0
    retention_sum = 0.0
    selected_tokens_seen = preservation_tokens_seen = teacher_tokens_seen = 0
    retention_tokens_seen = 0
    steps_executed = 0
    last_raw_delta = [0.0] * prefix_model.prefix_length
    progress = tqdm(range(max_steps), desc="  Pair train ES", unit="step", leave=False)
    for step_zero in progress:
        group_indices = schedule[
            step_zero * accumulation : (step_zero + 1) * accumulation
        ]
        tensor_group = [
            tensorize_example(
                torch, train_dataset, collator, index, prefix_model.device
            )
            for index in group_indices
        ]
        selected_counts = [
            int((batch["labels"] != -100).sum().item()) for batch in tensor_group
        ]
        preservation_counts = [
            int(batch["preservation_mask"].sum().item()) for batch in tensor_group
        ]
        teacher_counts = [
            int(batch["teacher_mask"].sum().item()) for batch in tensor_group
        ]
        retention_counts = [
            int(batch["retention_mask"].sum().item())
            if "retention_mask" in batch
            else 0
            for batch in tensor_group
        ]
        group_selected = sum(selected_counts)
        group_preservation = sum(preservation_counts)
        group_teacher = sum(teacher_counts)
        group_retention = sum(retention_counts)
        optimizer.zero_grad(set_to_none=True)
        last_loss = 0.0
        for batch, selected_count, preservation_count, teacher_count, retention_count in zip(
            tensor_group,
            selected_counts,
            preservation_counts,
            teacher_counts,
            retention_counts,
            strict=True,
        ):
            batch["preservation_loss_weight"] = preservation_weight
            batch["teacher_kl_loss_weight"] = teacher_kl_weight
            batch["teacher_margin_loss_weight"] = teacher_margin_weight
            batch["retention_loss_weight"] = retention_weight
            output = prefix_model.forward(batch)
            loss = output.selected_loss * (selected_count / group_selected)
            loss = loss + (
                preservation_weight
                * output.preservation_loss
                * (preservation_count / group_preservation)
            )
            teacher_scale = teacher_count / group_teacher
            loss = (
                loss
                + teacher_kl_weight * output.teacher_kl_loss * teacher_scale
                + teacher_margin_weight * output.teacher_margin_loss * teacher_scale
            )
            if retention_count:
                if group_retention <= 0:
                    raise ValueError("Retention tokens require a positive group total")
                loss = (
                    loss
                    + retention_weight
                    * output.retention_loss
                    * (retention_count / group_retention)
                )
            loss.backward()
            selected_value = float(output.selected_loss.detach().cpu())
            preservation_value = float(output.preservation_loss.detach().cpu())
            teacher_kl_value = float(output.teacher_kl_loss.detach().cpu())
            teacher_margin_value = float(output.teacher_margin_loss.detach().cpu())
            retention_value = float(output.retention_loss.detach().cpu())
            selected_sum += selected_value * selected_count
            preservation_sum += preservation_value * preservation_count
            teacher_kl_sum += teacher_kl_value * teacher_count
            teacher_margin_sum += teacher_margin_value * teacher_count
            retention_sum += retention_value * retention_count
            selected_tokens_seen += selected_count
            preservation_tokens_seen += preservation_count
            teacher_tokens_seen += teacher_count
            retention_tokens_seen += retention_count
            last_loss = (
                selected_value
                + preservation_weight * preservation_value
                + teacher_kl_weight * teacher_kl_value
                + teacher_margin_weight * teacher_margin_value
                + retention_weight * retention_value
            )
            del batch, output, loss
            gc.collect()
            torch.cuda.empty_cache()
        gradient = prefix_model.prefix_embeddings.grad
        if gradient is None:
            raise RuntimeError("Early-stop pair training produced no gradient")
        mask_prefix_gradient_(gradient, pair)
        torch.nn.utils.clip_grad_norm_([prefix_model.prefix_embeddings], max_norm=1.0)
        optimizer.step()
        if not torch.equal(
            prefix_model.prefix_embeddings.detach()[frozen_rows], round_start[frozen_rows]
        ):
            raise AssertionError("A frozen prefix row changed during early-stop training")
        steps_executed = step_zero + 1
        progress.set_postfix(loss=f"{last_loss:.4f}", pair=str(pair))
        if steps_executed % eval_interval != 0:
            continue

        with torch.no_grad():
            raw_pair = prefix_model.prefix_embeddings.detach()[pair].clone()
            candidate_pair = round_start[pair] + shrinkage * (
                raw_pair - round_start[pair]
            )
            pair_tensor = torch.tensor(
                pair, dtype=torch.long, device=prefix_model.prefix_embeddings.device
            )
            prefix_model.prefix_embeddings.index_copy_(0, pair_tensor, candidate_pair)
        monitor = evaluate_monitor_objective(
            torch,
            prefix_model,
            monitor_dataset,
            collator,
            preservation_weight=preservation_weight,
            teacher_kl_weight=teacher_kl_weight,
            teacher_margin_weight=teacher_margin_weight,
            retention_weight=retention_weight,
        )
        with torch.no_grad():
            prefix_model.prefix_embeddings.index_copy_(0, pair_tensor, raw_pair)
        current_loss = float(monitor["composite_loss"])
        current_preservation = float(monitor["preservation_kl_loss"])
        current_retention = float(monitor["retention_top64_kl_loss"])
        preservation_safe = current_preservation <= preservation_limit + 1e-12
        retention_safe = current_retention <= retention_limit + 1e-12
        candidate_safe = preservation_safe and retention_safe
        significant = (
            candidate_safe
            and current_loss
            <= patience_reference * (1.0 - min_relative_improvement)
        )
        if candidate_safe and current_loss < best_loss:
            best_loss = current_loss
            best_step = steps_executed
            best_pair = candidate_pair.clone()
            best_raw_pair = raw_pair.clone()
            best_monitor = dict(monitor)
        if significant:
            patience_reference = current_loss
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        history.append(
            {
                "step": steps_executed,
                **monitor,
                "preservation_safe": preservation_safe,
                "retention_safe": retention_safe,
                "significant_improvement": significant,
                "stale_evaluations": stale_evaluations,
            }
        )
        if not preservation_safe:
            stop_reason = "preservation_guard"
            break
        if not retention_safe:
            stop_reason = "retention_guard"
            break
        if steps_executed >= min_steps and stale_evaluations >= patience:
            stop_reason = "patience"
            break
    progress.close()
    last_raw_delta = (
        prefix_model.prefix_embeddings.detach() - round_start
    ).float().norm(dim=1).cpu().tolist()
    alpha_line_search: list[dict[str, Any]] = []
    selected_alpha = float(shrinkage if best_step > 0 else 0.0)
    alpha_grid = list(stage_alpha_grid or [])
    if alpha_grid:
        # Only the best raw optimizer state is searched. Alpha=0 reuses the
        # baseline, and alpha=shrinkage reuses the already-computed monitor,
        # keeping v5 substantially cheaper than a per-monitor line search.
        selected_alpha = 0.0
        selected_pair = round_start[pair].clone()
        selected_monitor = dict(baseline)
        selected_loss = float(baseline["composite_loss"])
        pair_tensor = torch.tensor(
            pair, dtype=torch.long, device=prefix_model.prefix_embeddings.device
        )
        for alpha in alpha_grid:
            candidate_pair = round_start[pair] + float(alpha) * (
                best_raw_pair - round_start[pair]
            )
            if alpha == 0.0 or best_step == 0:
                candidate_monitor = dict(baseline)
            elif math.isclose(alpha, shrinkage, rel_tol=0.0, abs_tol=1e-12):
                candidate_monitor = dict(best_monitor)
            else:
                with torch.no_grad():
                    prefix_model.prefix_embeddings.copy_(round_start)
                    prefix_model.prefix_embeddings.index_copy_(
                        0, pair_tensor, candidate_pair
                    )
                candidate_monitor = evaluate_monitor_objective(
                    torch,
                    prefix_model,
                    monitor_dataset,
                    collator,
                    preservation_weight=preservation_weight,
                    teacher_kl_weight=teacher_kl_weight,
                    teacher_margin_weight=teacher_margin_weight,
                    retention_weight=retention_weight,
                )
            candidate_preservation_safe = (
                float(candidate_monitor["preservation_kl_loss"])
                <= preservation_limit + 1e-12
            )
            candidate_retention_safe = (
                float(candidate_monitor["retention_top64_kl_loss"])
                <= retention_limit + 1e-12
            )
            candidate_safe = candidate_preservation_safe and candidate_retention_safe
            candidate_loss = float(candidate_monitor["composite_loss"])
            alpha_line_search.append(
                {
                    "alpha": float(alpha),
                    **candidate_monitor,
                    "preservation_safe": candidate_preservation_safe,
                    "retention_safe": candidate_retention_safe,
                    "safe": candidate_safe,
                }
            )
            if candidate_safe and candidate_loss < selected_loss - 1e-12:
                selected_loss = candidate_loss
                selected_alpha = float(alpha)
                selected_pair = candidate_pair.clone()
                selected_monitor = dict(candidate_monitor)
        best_pair = selected_pair
        best_monitor = selected_monitor
        best_loss = selected_loss
    with torch.no_grad():
        prefix_model.prefix_embeddings.copy_(round_start)
        pair_tensor = torch.tensor(
            pair, dtype=torch.long, device=prefix_model.prefix_embeddings.device
        )
        prefix_model.prefix_embeddings.index_copy_(0, pair_tensor, best_pair)
    committed_delta = (
        prefix_model.prefix_embeddings.detach() - round_start
    ).float().norm(dim=1).cpu().tolist()
    if not torch.equal(
        prefix_model.prefix_embeddings.detach()[frozen_rows], round_start[frozen_rows]
    ):
        raise AssertionError("A frozen prefix row changed during best-pair rollback")
    return {
        "selected_ce_loss": selected_sum / max(selected_tokens_seen, 1),
        "preservation_kl_loss": preservation_sum / max(preservation_tokens_seen, 1),
        "teacher_top64_kl_loss": teacher_kl_sum / max(teacher_tokens_seen, 1),
        "teacher_margin_hinge_loss": teacher_margin_sum / max(teacher_tokens_seen, 1),
        "retention_top64_kl_loss": retention_sum / max(retention_tokens_seen, 1),
        "selected_tokens_seen": selected_tokens_seen,
        "preservation_tokens_seen": preservation_tokens_seen,
        "teacher_tokens_seen": teacher_tokens_seen,
        "retention_tokens_seen": retention_tokens_seen,
        "steps_executed": steps_executed,
        "best_step": best_step,
        "stop_reason": stop_reason,
        "baseline_monitor": baseline,
        "best_monitor": best_monitor,
        "preservation_limit": preservation_limit,
        "retention_limit": retention_limit if math.isfinite(retention_limit) else None,
        "selected_alpha": selected_alpha,
        "stage_rejected": selected_alpha == 0.0,
        "alpha_line_search": alpha_line_search,
        "monitor_history": history,
        "last_raw_row_delta_norms": last_raw_delta,
        "post_shrink_row_delta_norms": committed_delta,
    }


def paired_counts(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> dict[str, Any]:
    first_by_id = {str(row["id"]): row for row in first}
    second_by_id = {str(row["id"]): row for row in second}
    if set(first_by_id) != set(second_by_id):
        raise ValueError("Paired result IDs differ")
    first_only = sum(
        bool(first_by_id[key].get("hard")) and not bool(second_by_id[key].get("hard"))
        for key in first_by_id
    )
    second_only = sum(
        bool(second_by_id[key].get("hard")) and not bool(first_by_id[key].get("hard"))
        for key in first_by_id
    )
    discordant = first_only + second_only
    lower = min(first_only, second_only)
    p_value = (
        min(
            1.0,
            2.0
            * sum(math.comb(discordant, index) for index in range(lower + 1))
            / 2**discordant,
        )
        if discordant
        else 1.0
    )
    return {
        "first_only": first_only,
        "second_only": second_only,
        "exact_two_sided_p": p_value,
    }


def main(
    *,
    default_pair_policy: str = "gradient",
    default_out_root: str = "outputs/SpreadsheetBench_prcb_v1_len8_seed1",
    default_locator_policy: str = "combined_residual",
    default_method_version: str = "auto",
    default_teacher_kl_weight: float = 0.0,
    default_teacher_margin_weight: float = 0.0,
    default_rounds: int = 4,
    default_round_step_pattern: str = "",
    default_early_stop: bool = False,
    default_sliding_window_size: int = 2,
    default_retention_weight: float = 0.0,
    default_stage_alpha_grid: str = "",
) -> None:
    args = parse_args(
        default_pair_policy=default_pair_policy,
        default_out_root=default_out_root,
        default_locator_policy=default_locator_policy,
        default_method_version=default_method_version,
        default_teacher_kl_weight=default_teacher_kl_weight,
        default_teacher_margin_weight=default_teacher_margin_weight,
        default_rounds=default_rounds,
        default_round_step_pattern=default_round_step_pattern,
        default_early_stop=default_early_stop,
        default_sliding_window_size=default_sliding_window_size,
        default_retention_weight=default_retention_weight,
        default_stage_alpha_grid=default_stage_alpha_grid,
    )
    if args.rounds <= 0 or args.steps_per_round <= 0 or args.accumulation <= 0:
        raise ValueError("rounds, steps-per-round, and accumulation must be positive")
    if args.pair_policy in {"tail_to_head", "head_to_tail"} and args.rounds > 4:
        raise ValueError("A length-8 fixed causal schedule supports at most four rounds")
    if args.pair_policy == "sliding_head_to_tail":
        if not 2 <= args.sliding_window_size <= 8:
            raise ValueError("sliding-window-size must be between 2 and 8")
        maximum_sliding_rounds = 8 - args.sliding_window_size + 1
        if not 1 <= args.rounds <= maximum_sliding_rounds:
            raise ValueError(
                "A length-8 sliding schedule with window size "
                f"{args.sliding_window_size} supports 1 to "
                f"{maximum_sliding_rounds} rounds"
            )
    planned_steps = optimizer_steps_by_round(
        rounds=args.rounds,
        steps_per_round=args.steps_per_round,
        pattern=args.round_step_pattern,
    )
    if (
        args.teacher_kl_weight < 0
        or args.teacher_margin_weight < 0
        or args.retention_weight < 0
    ):
        raise ValueError("Teacher and retention loss weights must be non-negative")
    if args.retention_kl_limit < 0:
        raise ValueError("retention-kl-limit must be non-negative")
    stage_alpha_grid = parse_stage_alpha_grid(args.stage_alpha_grid)
    if args.method_version == "auto":
        method_version = "v1" if args.pair_policy == "gradient" else "v2"
    else:
        method_version = args.method_version
    if method_version in {"v3", "v4", "v4_es", "v5"} and args.locator_policy != "margin_decision":
        raise ValueError(f"PRCB-{method_version} requires locator-policy=margin_decision")
    if method_version == "v5":
        if not args.early_stop:
            raise ValueError("PRCB-v5 requires teacher-forced early stopping")
        if args.retention_weight <= 0:
            raise ValueError("PRCB-v5 requires retention-weight > 0")
        if not stage_alpha_grid:
            raise ValueError("PRCB-v5 requires a non-empty stage-alpha-grid")
    set_seed(args.seed)
    import torch

    model_path = resolve_model_reference(args.model_path)
    checkpoint_path = resolve(args.initial_checkpoint)
    manifest_path = resolve(args.manifest)
    out_root = resolve(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No trajectories loaded")
    if args.early_stop:
        if args.pair_policy != "sliding_head_to_tail":
            raise ValueError("PRCB early stopping currently requires sliding_head_to_tail")
        if any(value != args.max_steps_per_stage for value in planned_steps):
            raise ValueError(
                "Early-stop round-step-pattern must equal max-steps-per-stage in every round"
            )
        if not 0 < args.monitor_min_relative_improvement < 1:
            raise ValueError("monitor-min-relative-improvement must be between zero and one")
        if args.monitor_preservation_ratio_limit <= 1:
            raise ValueError("monitor-preservation-ratio-limit must exceed one")
        if args.monitor_min_steps > args.max_steps_per_stage:
            raise ValueError("monitor-min-steps cannot exceed max-steps-per-stage")
        train_ids, monitor_ids = fixed_monitor_split(
            rows,
            monitor_count=args.monitor_trajectories,
            seed=args.seed,
        )
    else:
        train_ids = [str(row["id"]) for row in rows]
        monitor_ids = []
    run_config = {
        **vars(args),
        "model_path": str(model_path),
        "initial_checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "out_root": str(out_root),
        "initial_checkpoint_sha256": sha256(checkpoint_path),
        "trajectories": len(rows),
        "test_split_accessed": False,
        "dynamic_js": "teacher_top64_plus_residual",
        "dynamic_anchor": "top_no_skill_kl_on_teacher_nonbeneficial_positions",
        "replay": "previous_round_core",
        "method_version": method_version,
        "pair_policy": args.pair_policy,
        "locator_formula": (
            "tier2(skill_top1_gold & !clean_top1_gold & !current_top1_gold), "
            "tier1(positive_skill_margin_gain & positive_current_margin_residual), "
            "lexicographic residual/gain/JS tie-break"
            if args.locator_policy == "margin_decision"
            else "positive_target_gain * current_target_residual * teacher_current_js"
        ),
        "teacher_forcing_only_locator": True,
        "optimizer_steps_by_round": planned_steps,
        "total_optimizer_steps": sum(planned_steps),
        "total_trajectory_presentations": sum(planned_steps) * args.accumulation,
        "train_trajectory_ids": train_ids,
        "monitor_trajectory_ids": monitor_ids,
        "gradient_trajectories": len(train_ids),
        "monitor_trajectories_actual": len(monitor_ids),
        "stage_alpha_values": stage_alpha_grid,
        "retention_reference": "stage_start_prefix_top64_plus_residual_on_replay_gold",
    }
    config_path = out_root / "prcb_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        immutable = [
            "model_path",
            "initial_checkpoint_sha256",
            "manifest",
            "seed",
            "selection_ratio",
            "learning_rate",
            "shrinkage",
            "steps_per_round",
            "accumulation",
            "trajectories",
            "rounds",
            "round_step_pattern",
            "preservation_weight",
            "pair_policy",
            "sliding_window_size",
            "locator_policy",
            "teacher_kl_weight",
            "teacher_margin_weight",
            "retention_weight",
            "retention_kl_limit",
            "stage_alpha_values",
            "method_version",
            "early_stop",
            "monitor_trajectories",
            "monitor_eval_interval",
            "monitor_min_steps",
            "monitor_patience",
            "monitor_min_relative_improvement",
            "monitor_preservation_ratio_limit",
            "max_steps_per_stage",
            "train_trajectory_ids",
            "monitor_trajectory_ids",
        ]
        mismatched = [
            key
            for key in immutable
            if existing.get(key) != run_config.get(key)
            and not (
                key == "pair_policy"
                and key not in existing
                and run_config.get(key) == "gradient"
            )
            and not (
                key == "round_step_pattern"
                and key not in existing
                and run_config.get(key) == ""
            )
        ]
        if mismatched:
            raise ValueError(f"Existing PRCB run config mismatch: {mismatched}")
    else:
        atomic_json(config_path, run_config)
    split_path = out_root / "trajectory_split.json"
    split_record = {
        "seed": args.seed,
        "train_count": len(train_ids),
        "monitor_count": len(monitor_ids),
        "train_ids": train_ids,
        "monitor_ids": monitor_ids,
        "monitor_backpropagation": False,
        "test_split_accessed": False,
    }
    if split_path.exists():
        if json.loads(split_path.read_text(encoding="utf-8")) != split_record:
            raise ValueError("Existing trajectory split does not match this run")
    else:
        atomic_json(split_path, split_record)

    print(f"Loading frozen model from {model_path}", flush=True)
    prefix_model = SoftPrefixCausalLM(
        str(model_path),
        prefix_length=8,
        init_strategy="random",
        torch_dtype="bfloat16",
        device="cuda",
    )
    initial_state = torch.load(checkpoint_path, map_location="cpu")
    prefix_model.load_state_dict(initial_state)
    prefix_model.prefix_embeddings.requires_grad_(True)

    previous_core: dict[str, list[int]] = {}
    completed_rounds = 0
    for round_index in range(1, args.rounds + 1):
        round_dir = out_root / f"round_{round_index:02d}"
        checkpoint = round_dir / "prefix.pt"
        manifest = round_dir / "manifest.jsonl"
        summary_path = round_dir / "summary.json"
        if checkpoint.exists() and manifest.exists() and summary_path.exists():
            prefix_model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            previous_core = {
                str(row["id"]): [int(index) for index in row["selected_core_indices"]]
                for row in read_jsonl(manifest)
            }
            completed_rounds = round_index
            print(f"[round {round_index}] complete cache found; resumed prefix", flush=True)
            continue

        round_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        round_start_sha = tensor_sha256(prefix_model.prefix_embeddings)
        locator_meta_path = round_dir / "locator_metadata.json"
        locator_meta = {
            "round": round_index,
            "prefix_sha256": round_start_sha,
            "trajectories": len(rows),
            "locator_policy": args.locator_policy,
        }
        if locator_meta_path.exists():
            existing_meta = json.loads(locator_meta_path.read_text(encoding="utf-8"))
            if existing_meta != locator_meta:
                raise ValueError(f"Round {round_index} locator cache does not match prefix")
        else:
            atomic_json(locator_meta_path, locator_meta)

        print(f"[round {round_index}/{args.rounds}] dynamic localization", flush=True)
        # The previous round leaves the backbone in checkpointed training mode.
        # Localization is inference-only and must not construct a graph.
        prefix_model.model.gradient_checkpointing_disable()
        prefix_model.model.config.use_cache = False
        prefix_model.model.eval()
        score_rows: dict[str, dict[str, np.ndarray]] = {}
        score_dir = round_dir / "scores"
        for row in tqdm(rows, desc=f"  Locate {round_index}", unit="traj"):
            identifier = str(row["id"])
            cache_file = score_dir / f"{slug(identifier)}.npz"
            if not cache_file.exists():
                values = score_current_prefix(
                    prefix_model,
                    row,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_target_tokens=args.max_target_tokens,
                    chunk_size=args.score_chunk_size,
                    locator_policy=args.locator_policy,
                    cache_current_topk=method_version == "v5",
                )
                atomic_npz(cache_file, **values)
            with np.load(cache_file) as cached:
                score_rows[identifier] = {name: cached[name] for name in cached.files}

        round_rows, current_core, locator_stats = build_round_rows(
            rows,
            score_rows,
            previous_core,
            ratio=args.selection_ratio,
            locator_policy=args.locator_policy,
        )
        for round_row in round_rows:
            round_row["stage2_selector"] = f"prcb_{method_version}_{args.pair_policy}"
            if method_version == "v5":
                round_row["retention_cache"] = str(
                    score_dir / f"{slug(str(round_row['id']))}.npz"
                )
        atomic_jsonl(manifest, round_rows)
        atomic_json(round_dir / "locator_statistics.json", locator_stats)

        train_id_set = set(train_ids)
        monitor_id_set = set(monitor_ids)
        gradient_rows = [
            row for row in round_rows if str(row["id"]) in train_id_set
        ]
        monitor_rows = [
            row for row in round_rows if str(row["id"]) in monitor_id_set
        ]
        dataset = TextTrajectoryPrefixDataset(
            gradient_rows,
            prefix_model.tokenizer,
            max_prompt_tokens=args.max_prompt_tokens,
            max_target_tokens=args.max_target_tokens,
            selective_label_field="selected_indices",
            always_supervise_eos=True,
            preservation_loss_weight=args.preservation_weight,
            preservation_label_field="preserve_indices",
            teacher_distillation_loss_weight=(
                args.teacher_kl_weight + args.teacher_margin_weight
            ),
            teacher_label_field="selected_indices",
            retention_loss_weight=args.retention_weight,
            retention_label_field="replay_indices",
            retention_cache_field="retention_cache",
        )
        monitor_dataset = (
            TextTrajectoryPrefixDataset(
                monitor_rows,
                prefix_model.tokenizer,
                max_prompt_tokens=args.max_prompt_tokens,
                max_target_tokens=args.max_target_tokens,
                selective_label_field="selected_indices",
                always_supervise_eos=True,
                preservation_loss_weight=args.preservation_weight,
                preservation_label_field="preserve_indices",
                teacher_distillation_loss_weight=(
                    args.teacher_kl_weight + args.teacher_margin_weight
                ),
                teacher_label_field="selected_indices",
                retention_loss_weight=args.retention_weight,
                retention_label_field="replay_indices",
                retention_cache_field="retention_cache",
            )
            if args.early_stop
            else None
        )
        collator = PrefixBatchCollator(prefix_model.tokenizer.pad_token_id)
        current_round_steps = planned_steps[round_index - 1]
        examples_per_round = current_round_steps * args.accumulation
        schedule_start = sum(planned_steps[: round_index - 1]) * args.accumulation
        schedule = round_schedule(
            len(dataset),
            round_index=round_index,
            examples=examples_per_round,
            seed=args.seed,
            start_offset=schedule_start,
        )
        prefix_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        prefix_model.model.config.use_cache = False
        prefix_model.model.train()
        if args.pair_policy == "gradient":
            residual_mass = [
                float(score_rows[str(rows[index]["id"])]["dynamic_score"].sum())
                for index in schedule
            ]
            probe_order = [
                index
                for _, index in sorted(
                    zip(residual_mass, schedule, strict=True),
                    key=lambda item: (-item[0], item[1]),
                )
            ][: min(args.probe_examples, len(schedule))]
            print(f"[round {round_index}] gradient probe on {probe_order}", flush=True)
            pair, gradient_scores = gradient_probe(
                torch,
                prefix_model,
                dataset,
                collator,
                probe_order,
                preservation_weight=args.preservation_weight,
            )
        elif args.pair_policy == "sliding_head_to_tail":
            pair = sliding_prefix_pair(
                prefix_length=prefix_model.prefix_length,
                round_index=round_index,
                window_size=args.sliding_window_size,
            )
            gradient_scores = []
            probe_order = []
        else:
            pair = causal_prefix_pair(
                prefix_length=prefix_model.prefix_length,
                round_index=round_index,
                direction=args.pair_policy,
            )
            gradient_scores = []
            probe_order = []
        print(f"[round {round_index}] selected prefix rows {pair}", flush=True)
        if args.early_stop:
            if monitor_dataset is None:
                raise AssertionError("Early stopping requires a monitor dataset")
            training_stats = train_round_early_stopping(
                torch,
                prefix_model,
                dataset,
                monitor_dataset,
                collator,
                schedule,
                pair=pair,
                max_steps=current_round_steps,
                accumulation=args.accumulation,
                learning_rate=args.learning_rate,
                shrinkage=args.shrinkage,
                preservation_weight=args.preservation_weight,
                teacher_kl_weight=args.teacher_kl_weight,
                teacher_margin_weight=args.teacher_margin_weight,
                min_steps=args.monitor_min_steps,
                eval_interval=args.monitor_eval_interval,
                patience=args.monitor_patience,
                min_relative_improvement=args.monitor_min_relative_improvement,
                preservation_ratio_limit=args.monitor_preservation_ratio_limit,
                retention_weight=args.retention_weight,
                retention_kl_limit=args.retention_kl_limit,
                stage_alpha_grid=stage_alpha_grid,
            )
        else:
            training_stats = train_round(
                torch,
                prefix_model,
                dataset,
                collator,
                schedule,
                pair=pair,
                steps=current_round_steps,
                accumulation=args.accumulation,
                learning_rate=args.learning_rate,
                shrinkage=args.shrinkage,
                preservation_weight=args.preservation_weight,
                teacher_kl_weight=args.teacher_kl_weight,
                teacher_margin_weight=args.teacher_margin_weight,
            )
        torch.save(prefix_model.state_dict(), checkpoint)
        round_summary = {
            "round": round_index,
            "pair_policy": args.pair_policy,
            "prefix_rows": pair,
            "gradient_row_scores": gradient_scores,
            "probe_indices": probe_order,
            "training_schedule": schedule,
            "optimizer_steps": (
                training_stats.get("steps_executed", current_round_steps)
            ),
            "maximum_optimizer_steps": current_round_steps,
            "gradient_trajectories": len(dataset),
            "monitor_trajectories": len(monitor_dataset) if monitor_dataset else 0,
            "locator": {key: value for key, value in locator_stats.items() if key != "per_trajectory"},
            "training": training_stats,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "wall_time_s": round(time.time() - start_time, 1),
            "test_split_accessed": False,
        }
        atomic_json(summary_path, round_summary)
        previous_core = current_core
        completed_rounds = round_index
        print(json.dumps(round_summary, ensure_ascii=False, indent=2), flush=True)

    final_checkpoint = out_root / f"prcb_{method_version}_prefix.pt"
    torch.save(prefix_model.state_dict(), final_checkpoint)
    completed_summaries = [
        json.loads(
            (out_root / f"round_{index:02d}" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        for index in range(1, completed_rounds + 1)
    ]
    summary: dict[str, Any] = {
        "completed_rounds": completed_rounds,
        "final_checkpoint": str(final_checkpoint),
        "final_checkpoint_sha256": sha256(final_checkpoint),
        "test_split_accessed": False,
        "actual_total_optimizer_steps": sum(
            int(row.get("optimizer_steps", 0)) for row in completed_summaries
        ),
        "actual_total_trajectory_presentations": sum(
            int(row.get("optimizer_steps", 0)) * args.accumulation
            for row in completed_summaries
        ),
        "stage_best_steps": [
            row.get("training", {}).get("best_step") for row in completed_summaries
        ],
        "stage_stop_reasons": [
            row.get("training", {}).get("stop_reason") for row in completed_summaries
        ],
        "stage_selected_alphas": [
            row.get("training", {}).get("selected_alpha") for row in completed_summaries
        ],
        "stage_rejections": sum(
            bool(row.get("training", {}).get("stage_rejected"))
            for row in completed_summaries
        ),
    }
    if args.skip_eval:
        atomic_json(out_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    prefix_model.model.gradient_checkpointing_disable()
    prefix_model.model.config.use_cache = True
    prefix_model.model.eval()
    prefix_model.prefix_embeddings.requires_grad_(False)
    dataloader = SpreadsheetBenchDataLoader(
        split_dir=str(resolve(args.split_dir)),
        split_mode="split_dir",
        split_seed=42,
        data_root=str(resolve(args.data_root)),
        seed=args.seed,
    )
    # SplitDataLoader.setup() reads all three directories. Read only val here
    # so PRCB development never touches the held-out test split.
    validation_items = dataloader.load_split_items(
        str(resolve(args.split_dir) / "val")
    )[:40]
    hard, soft, results = evaluate_spreadsheet_prefix(
        prefix_model,
        validation_items,
        out_dir=str(out_root / "eval" / "final" / "valid_seen"),
        data_root=str(resolve(args.data_root)),
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        exec_timeout=600,
        desc=f"PRCB-{method_version.removeprefix('v')} Val",
        generator=None,
        injection_position="prompt_start",
        repair_turns=1,
        generation_batch_size=args.generation_batch_size,
    )
    summary["valid_seen_hard"] = hard
    summary["valid_seen_soft"] = soft
    summary["valid_seen_successes"] = sum(bool(row.get("hard")) for row in results)

    comparisons = {
        "combined": resolve(
            "outputs/SpreadsheetBench_full_distribution_locator_len8_seed1_shared/"
            "combined_top0.05_core_shared_preserve/eval/epoch_01/valid_seen/results.jsonl"
        ),
        "plain": resolve(
            "outputs/SpreadsheetBench_qwen36_noskill_noprefix_test280_softskill_matched/"
            "eval/plain/valid_seen/results.jsonl"
        ),
    }
    if method_version == "v2":
        comparisons["prcb_v1"] = resolve(
            "outputs/SpreadsheetBench_prcb_v1_len8_seed1/"
            "eval/final/valid_seen/results.jsonl"
        )
    if method_version == "v3":
        comparisons["prcb_v2_head_to_tail"] = resolve(
            "outputs/SpreadsheetBench_prcb_v2_head_to_tail_len8_seed1/"
            "eval/final/valid_seen/results.jsonl"
        )
    if method_version == "v4":
        comparisons["prcb_v3_overlap0"] = resolve(
            "outputs/SpreadsheetBench_prcb_v3_margin_head_to_tail_len8_seed1/"
            "eval/final/valid_seen/results.jsonl"
        )
    if method_version == "v4_es":
        comparisons["prcb_v4_fixed32"] = resolve(
            "outputs/SpreadsheetBench_prcb_v4_overlap1_len8_seed1/"
            "eval/final/valid_seen/results.jsonl"
        )
        comparisons["prcb_v3_overlap0"] = resolve(
            "outputs/SpreadsheetBench_prcb_v3_margin_head_to_tail_len8_seed1/"
            "eval/final/valid_seen/results.jsonl"
        )
        comparisons["prcb_v4_es_window2"] = resolve(
            "outputs/SpreadsheetBench_prcb_v4_es_overlap1_len8_seed1/"
            "eval/final/valid_seen/results.jsonl"
        )
    if method_version == "v5":
        comparisons["prcb_v4_fixed32"] = resolve(
            "outputs/SpreadsheetBench_prcb_v4_overlap1_len8_seed1/"
            "eval/final/valid_seen/results.jsonl"
        )
        comparisons["prcb_v4_es_window2"] = resolve(
            "outputs/SpreadsheetBench_prcb_v4_es_overlap1_len8_seed1/"
            "eval/final/valid_seen/results.jsonl"
        )
    if args.pair_policy == "head_to_tail":
        comparisons["prcb_v2_tail_to_head"] = resolve(
            "outputs/SpreadsheetBench_prcb_v2_tail_to_head_len8_seed1/"
            "eval/final/valid_seen/results.jsonl"
        )
    summary["paired_validation"] = {}
    for name, path in comparisons.items():
        if path.exists():
            baseline = read_jsonl(path)
            summary["paired_validation"][name] = paired_counts(results, baseline)
            summary["paired_validation"][name]["baseline_successes"] = sum(
                bool(row.get("hard")) for row in baseline
            )
            summary["paired_validation"][name]["protocol_matched"] = name in {
                "combined",
                "prcb_v1",
                "prcb_v2_tail_to_head",
                "prcb_v2_head_to_tail",
                "prcb_v3_overlap0",
                "prcb_v4_fixed32",
                "prcb_v4_es_window2",
            }
            if name == "plain":
                summary["paired_validation"][name]["protocol_note"] = (
                    "context only: stored plain run used max_new_tokens=8192 and "
                    f"generation_batch_size=2; PRCB-{method_version.removeprefix('v')} "
                    "used 4096 and 8"
                )
    atomic_json(out_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
