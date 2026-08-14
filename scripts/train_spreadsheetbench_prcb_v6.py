#!/usr/bin/env python3
"""PRCB-v6: independent length-8 soft prompts boosted in logit space.

Learner selection, early stopping, and alpha search use only a fixed 49/12
split of successful teacher-forced trajectories.  The SpreadsheetBench test
split is never loaded.  The final boosted teacher is distilled into one
length-8 prefix before the optional legacy Val40 evaluation.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
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
from skillopt.softprefix.model import SoftPrefixCausalLM
from skillopt.softprefix.prcb import (
    residual_combined_scores,
    select_harmful_anchor_positions,
    select_positive_top_fraction,
    topk_residual_forward_kl,
    topk_residual_js,
)
from skillopt.softprefix.prcb_v6 import (
    centered_logit_delta,
    choose_stage_alpha,
    combine_boosted_logits,
    topk_reference_from_logits,
    topk_residual_kl_from_logits,
)
from skillopt.softprefix.trainer import evaluate_spreadsheet_prefix
from scripts.train_spreadsheetbench_prcb_v1 import (
    atomic_json,
    atomic_jsonl,
    atomic_npz,
    encode_trajectory,
    fixed_monitor_split,
    read_jsonl,
    resolve,
    resolve_model_reference,
    round_schedule,
    set_seed,
    sha256,
    slug,
    tensor_sha256,
)


def parse_args() -> argparse.Namespace:
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
        default="outputs/SpreadsheetBench_prcb_v6_functional_len8_seed1",
    )
    parser.add_argument("--max-stages", type=int, default=4)
    parser.add_argument("--max-steps-per-stage", type=int, default=32)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--selection-ratio", type=float, default=0.05)
    parser.add_argument("--delta-weight", type=float, default=1e-4)
    parser.add_argument("--monitor-trajectories", type=int, default=12)
    parser.add_argument("--monitor-interval", type=int, default=2)
    parser.add_argument("--monitor-min-steps", type=int, default=4)
    parser.add_argument("--monitor-patience", type=int, default=3)
    parser.add_argument("--min-relative-improvement", type=float, default=0.002)
    parser.add_argument("--history-kl-limit", type=float, default=0.02)
    parser.add_argument("--alpha-grid", default="0,0.125,0.25,0.5,1.0")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--score-chunk-size", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=16384)
    parser.add_argument("--max-target-tokens", type=int, default=8192)
    parser.add_argument("--student-epochs", type=int, default=3)
    parser.add_argument("--student-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--learner-rank",
        type=int,
        default=0,
        help="0 trains a full prefix; a positive value trains P0 + A@B at this rank.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--split-dir", default="data/spreadsheetbench_split")
    parser.add_argument("--data-root", default="data/spreadsheetbench_verified_400")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--skip-distill", action="store_true")
    return parser.parse_args()


def parse_alphas(raw: str) -> list[float]:
    values = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or 0.0 not in values:
        raise ValueError("alpha-grid must contain alpha=0")
    if any(not 0 <= value <= 1 or not math.isfinite(value) for value in values):
        raise ValueError("alpha values must be finite and in [0, 1]")
    return values


def load_prefix(torch: Any, path: Path, device: Any, dtype: Any) -> Any:
    state = torch.load(path, map_location="cpu")
    value = state["prefix_embeddings"].to(device=device, dtype=dtype)
    if tuple(value.shape)[0] != 8:
        raise ValueError(f"PRCB-v6 requires length-8 prefixes, got {tuple(value.shape)}")
    return value.detach().clone()


def install_prefix(torch: Any, prefix_model: SoftPrefixCausalLM, value: Any) -> None:
    with torch.no_grad():
        prefix_model.prefix_embeddings.copy_(
            value.to(
                device=prefix_model.device,
                dtype=prefix_model.prefix_embeddings.dtype,
            )
        )


def logits_for_positions(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    row: dict[str, Any],
    target_positions: list[int],
    *,
    max_prompt_tokens: int,
    max_target_tokens: int,
    inference: bool,
    prefix_value: Any | None = None,
) -> Any:
    """Return logits predicting target-relative positions under the installed prefix."""
    prompt_ids, target_ids = encode_trajectory(
        prefix_model,
        row,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
    )
    if not target_positions:
        raise ValueError("At least one target position is required")
    if min(target_positions) < 0 or max(target_positions) >= len(target_ids):
        raise ValueError(f"Invalid target position for trajectory {row['id']}")
    input_ids = torch.tensor(
        [prompt_ids + target_ids], dtype=torch.long, device=prefix_model.device
    )
    attention = torch.ones_like(input_ids)
    if prefix_value is None:
        embeds, full_attention, _ = prefix_model._with_prefix(input_ids, attention)
    else:
        token_embeds = prefix_model.model.get_input_embeddings()(input_ids)
        prefix = prefix_value.to(
            device=prefix_model.device,
            dtype=token_embeds.dtype,
        ).unsqueeze(0)
        embeds = torch.cat([prefix, token_embeds], dim=1)
        full_attention = torch.cat(
            [
                torch.ones(
                    (1, prefix_model.prefix_length),
                    dtype=attention.dtype,
                    device=attention.device,
                ),
                attention,
            ],
            dim=1,
        )
    absolute = torch.tensor(
        [
            prefix_model.prefix_length + len(prompt_ids) + position - 1
            for position in target_positions
        ],
        dtype=torch.long,
        device=prefix_model.device,
    )

    def run():
        output = prefix_model.model(
            inputs_embeds=embeds,
            attention_mask=full_attention,
            use_cache=False,
            output_router_logits=False,
            logits_to_keep=absolute,
            return_dict=True,
        )
        result = output.logits[0]
        if int(result.shape[0]) != len(target_positions):
            raise ValueError(
                f"Unexpected logit shape for {row['id']}: {tuple(result.shape)}"
            )
        return result

    if inference:
        with torch.inference_mode():
            return run()
    return run()


def ensemble_logits_for_positions(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    row: dict[str, Any],
    target_positions: list[int],
    *,
    base_prefix: Any,
    learners: list[Any],
    alphas: list[float],
    max_prompt_tokens: int,
    max_target_tokens: int,
) -> tuple[Any, Any]:
    install_prefix(torch, prefix_model, base_prefix)
    base = logits_for_positions(
        torch,
        prefix_model,
        row,
        target_positions,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
        inference=True,
    )
    learner_logits = []
    for learner in learners:
        install_prefix(torch, prefix_model, learner)
        learner_logits.append(
            logits_for_positions(
                torch,
                prefix_model,
                row,
                target_positions,
                max_prompt_tokens=max_prompt_tokens,
                max_target_tokens=max_target_tokens,
                inference=True,
            )
        )
    combined = combine_boosted_logits(torch, base, learner_logits, alphas)
    return combined, base.float()


def score_ensemble(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    row: dict[str, Any],
    *,
    base_prefix: Any,
    learners: list[Any],
    alphas: list[float],
    max_prompt_tokens: int,
    max_target_tokens: int,
    chunk_size: int,
    top_k: int,
) -> dict[str, np.ndarray]:
    _, target_ids = encode_trajectory(
        prefix_model,
        row,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
    )
    positions = list(range(len(target_ids)))
    combined, _ = ensemble_logits_for_positions(
        torch,
        prefix_model,
        row,
        positions,
        base_prefix=base_prefix,
        learners=learners,
        alphas=alphas,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
    )
    with np.load(resolve(str(row["score_cache"]))) as cached:
        teacher = {key: cached[key] for key in cached.files}
    if teacher["target_ids"].astype(np.int64).tolist() != target_ids:
        raise ValueError(f"Tokenizer/cache mismatch for {row['id']}")
    count = len(target_ids)
    js = np.empty(count, dtype=np.float32)
    clean_kl = np.empty(count, dtype=np.float32)
    current_target_logp = np.empty(count, dtype=np.float32)
    current_topk_ids = np.empty((count, top_k), dtype=np.int32)
    current_topk_logp = np.empty((count, top_k), dtype=np.float16)
    current_residual = np.empty(count, dtype=np.float16)
    targets = torch.tensor(target_ids, dtype=torch.long, device=prefix_model.device)
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        logits = combined[start:end].to(prefix_model.device).float()
        logp = torch.log_softmax(logits, dim=-1)
        current_target_logp[start:end] = (
            logp.gather(1, targets[start:end, None]).squeeze(1).cpu().numpy()
        )
        ids, values, residual = topk_reference_from_logits(
            torch, logits, top_k=top_k
        )
        current_topk_ids[start:end] = ids.cpu().numpy().astype(np.int32)
        current_topk_logp[start:end] = values.cpu().numpy().astype(np.float16)
        current_residual[start:end] = residual.cpu().numpy().astype(np.float16)
        js[start:end] = topk_residual_js(
            torch,
            logits,
            reference_topk_ids=torch.from_numpy(teacher["skill_topk_ids"][start:end]),
            reference_topk_logp=torch.from_numpy(teacher["skill_topk_logp"][start:end]),
            reference_residual_log_mass=torch.from_numpy(
                teacher["skill_residual_log_mass"][start:end]
            ),
        ).cpu().numpy()
        clean_kl[start:end] = topk_residual_forward_kl(
            torch,
            logits,
            reference_topk_ids=torch.from_numpy(teacher["clean_topk_ids"][start:end]),
            reference_topk_logp=torch.from_numpy(teacher["clean_topk_logp"][start:end]),
            reference_residual_log_mass=torch.from_numpy(
                teacher["clean_residual_log_mass"][start:end]
            ),
        ).cpu().numpy()
    benefit = np.maximum(teacher["positive_gain"].astype(np.float32), 0.0)
    dynamic = residual_combined_scores(
        teacher_target_logp=teacher["skill_target_logp"],
        student_target_logp=current_target_logp,
        teacher_beneficial=teacher["positive_gain"] > 0,
        teacher_student_js=js,
    )
    del combined
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "target_ids": np.asarray(target_ids, dtype=np.int32),
        "skill_benefit": benefit,
        "teacher_student_js": js,
        "dynamic_score": dynamic,
        "current_target_logp": current_target_logp,
        "current_topk_ids": current_topk_ids,
        "current_topk_logp": current_topk_logp,
        "current_residual_log_mass": current_residual,
        "clean_kl": clean_kl,
    }


def build_stage_rows(
    rows: list[dict[str, Any]],
    scores: dict[str, dict[str, np.ndarray]],
    history: dict[str, set[int]],
    *,
    ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, set[int]], dict[str, Any]]:
    stage_rows = []
    updated_history = {key: set(value) for key, value in history.items()}
    total_mass = selected_mass = 0.0
    core_count = history_count = anchor_count = 0
    for row in rows:
        identifier = str(row["id"])
        values = scores[identifier]
        selectable = len(values["target_ids"]) - 1
        dynamic = values["dynamic_score"][:selectable]
        core = select_positive_top_fraction(dynamic, ratio)
        if not core:
            raise ValueError(f"No positive residual positions for {identifier}")
        current_core = set(core)
        prior = sorted(
            index
            for index in history.get(identifier, set())
            if 0 <= index < selectable and index not in current_core
        )
        excluded = current_core | set(prior)
        anchor = select_harmful_anchor_positions(
            values["clean_kl"][:selectable],
            count=len(core),
            excluded=excluded,
            teacher_beneficial=values["skill_benefit"][:selectable] > 0,
        )
        if len(anchor) != len(core):
            raise ValueError(f"Could not create matched anchors for {identifier}")
        stage_rows.append(
            {
                "id": identifier,
                "messages": row["messages"],
                "target": row["target"],
                "score_cache": row["score_cache"],
                "core_indices": core,
                "history_indices": prior,
                "anchor_indices": anchor,
            }
        )
        updated_history.setdefault(identifier, set()).update(core)
        total_mass += float(dynamic.sum())
        selected_mass += float(dynamic[core].sum())
        core_count += len(core)
        history_count += len(prior)
        anchor_count += len(anchor)
    return stage_rows, updated_history, {
        "trajectories": len(rows),
        "total_residual_mass": total_mass,
        "selected_residual_mass": selected_mass,
        "residual_capture": selected_mass / total_mass if total_mass > 0 else 0.0,
        "core_tokens": core_count,
        "cumulative_history_tokens": history_count,
        "anchor_tokens": anchor_count,
    }


def build_reference(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    row: dict[str, Any],
    *,
    base_prefix: Any,
    learners: list[Any],
    alphas: list[float],
    max_prompt_tokens: int,
    max_target_tokens: int,
    top_k: int,
) -> dict[str, Any]:
    positions = sorted(
        set(row["core_indices"])
        | set(row["history_indices"])
        | set(row["anchor_indices"])
    )
    fprev, base = ensemble_logits_for_positions(
        torch,
        prefix_model,
        row,
        positions,
        base_prefix=base_prefix,
        learners=learners,
        alphas=alphas,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
    )
    previous_ids, previous_logp, previous_residual = topk_reference_from_logits(
        torch, fprev.to(prefix_model.device), top_k=top_k
    )
    with np.load(resolve(str(row["score_cache"]))) as cached:
        teacher = {key: cached[key] for key in cached.files}
    index = {position: offset for offset, position in enumerate(positions)}
    return {
        "positions": positions,
        "core_offsets": [index[value] for value in row["core_indices"]],
        "history_offsets": [index[value] for value in row["history_indices"]],
        "anchor_offsets": [index[value] for value in row["anchor_indices"]],
        "fprev": fprev.to(dtype=torch.bfloat16, device="cpu"),
        "base": base.to(dtype=torch.bfloat16, device="cpu"),
        "previous_topk_ids": previous_ids.to(dtype=torch.int32, device="cpu"),
        "previous_topk_logp": previous_logp.to(dtype=torch.float16, device="cpu"),
        "previous_residual": previous_residual.to(dtype=torch.float16, device="cpu"),
        "skill_topk_ids": torch.from_numpy(
            teacher["skill_topk_ids"][positions].astype(np.int64)
        ),
        "skill_topk_logp": torch.from_numpy(
            teacher["skill_topk_logp"][positions].astype(np.float32)
        ),
        "skill_residual": torch.from_numpy(
            teacher["skill_residual_log_mass"][positions].astype(np.float32)
        ),
    }


def candidate_losses(
    torch: Any,
    candidate: Any,
    delta: Any,
    reference: dict[str, Any],
    *,
    delta_weight: float,
) -> dict[str, Any]:
    device = candidate.device

    def take(values: Any, offsets: list[int]):
        if not offsets:
            return values[:0]
        indices = torch.tensor(offsets, dtype=torch.long, device=device)
        return values.to(device)[indices]

    core = reference["core_offsets"]
    history = reference["history_offsets"]
    anchor = reference["anchor_offsets"]
    zero = candidate.sum() * 0.0
    core_loss = zero
    if core:
        core_loss = topk_residual_kl_from_logits(
            torch,
            take(candidate, core),
            reference_topk_ids=take(reference["skill_topk_ids"], core),
            reference_topk_logp=take(reference["skill_topk_logp"], core),
            reference_residual_log_mass=take(reference["skill_residual"], core),
        ).mean()

    def previous_loss(offsets: list[int]):
        if not offsets:
            return zero
        return topk_residual_kl_from_logits(
            torch,
            take(candidate, offsets),
            reference_topk_ids=take(reference["previous_topk_ids"], offsets),
            reference_topk_logp=take(reference["previous_topk_logp"], offsets),
            reference_residual_log_mass=take(reference["previous_residual"], offsets),
        ).mean()

    history_loss = previous_loss(history)
    anchor_loss = previous_loss(anchor)
    delta_loss = delta.float().square().mean()
    total = core_loss + history_loss + anchor_loss + float(delta_weight) * delta_loss
    return {
        "total": total,
        "core": core_loss,
        "history": history_loss,
        "anchor": anchor_loss,
        "delta": delta_loss,
        "core_count": len(core),
        "history_count": len(history),
        "anchor_count": len(anchor),
    }


def forward_candidate(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    row: dict[str, Any],
    reference: dict[str, Any],
    *,
    learner_prefix: Any,
    alpha: float,
    max_prompt_tokens: int,
    max_target_tokens: int,
    inference: bool,
) -> tuple[Any, Any]:
    learner_logits = logits_for_positions(
        torch,
        prefix_model,
        row,
        reference["positions"],
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
        inference=inference,
        prefix_value=learner_prefix,
    )
    base = reference["base"].to(prefix_model.device).float()
    fprev = reference["fprev"].to(prefix_model.device).float()
    delta = centered_logit_delta(torch, learner_logits, base)
    return fprev + float(alpha) * delta, delta


def evaluate_stage(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    rows: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    *,
    learner_prefix: Any,
    alpha: float,
    delta_weight: float,
    max_prompt_tokens: int,
    max_target_tokens: int,
) -> dict[str, float | int]:
    sums = {key: 0.0 for key in ("core", "history", "anchor", "delta")}
    counts = {key: 0 for key in ("core", "history", "anchor")}
    for row in rows:
        reference = references[str(row["id"])]
        candidate, delta = forward_candidate(
            torch,
            prefix_model,
            row,
            reference,
            learner_prefix=learner_prefix,
            alpha=alpha,
            max_prompt_tokens=max_prompt_tokens,
            max_target_tokens=max_target_tokens,
            inference=True,
        )
        losses = candidate_losses(
            torch, candidate, delta, reference, delta_weight=delta_weight
        )
        for key in ("core", "history", "anchor"):
            count = int(losses[f"{key}_count"])
            sums[key] += float(losses[key].cpu()) * count
            counts[key] += count
        sums["delta"] += float(losses["delta"].cpu())
        del candidate, delta, losses
        gc.collect()
        torch.cuda.empty_cache()
    means = {
        key: sums[key] / max(counts[key], 1)
        for key in ("core", "history", "anchor")
    }
    means["delta"] = sums["delta"] / max(len(rows), 1)
    means["global_loss"] = means["core"] + means["history"] + means["anchor"]
    return {
        **means,
        "core_tokens": counts["core"],
        "history_tokens": counts["history"],
        "anchor_tokens": counts["anchor"],
    }


def train_learner(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    train_rows: list[dict[str, Any]],
    monitor_rows: list[dict[str, Any]],
    train_references: dict[str, dict[str, Any]],
    monitor_references: dict[str, dict[str, Any]],
    *,
    base_prefix: Any,
    learning_rate: float,
    max_steps: int,
    accumulation: int,
    monitor_interval: int,
    min_steps: int,
    patience: int,
    min_relative_improvement: float,
    delta_weight: float,
    max_prompt_tokens: int,
    max_target_tokens: int,
    seed: int,
    schedule_offset: int,
) -> tuple[Any, dict[str, Any]]:
    install_prefix(torch, prefix_model, base_prefix)
    prefix_model.prefix_embeddings.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [prefix_model.prefix_embeddings], lr=learning_rate, weight_decay=0.0
    )
    schedule = round_schedule(
        len(train_rows),
        round_index=1,
        examples=max_steps * accumulation,
        seed=seed,
        start_offset=schedule_offset,
    )
    baseline = evaluate_stage(
        torch,
        prefix_model,
        monitor_rows,
        monitor_references,
        learner_prefix=base_prefix,
        alpha=1.0,
        delta_weight=delta_weight,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
    )
    # Fit the weak learner's *direction* on the residual objective.  The
    # global core+preservation objective is deliberately deferred to alpha
    # line search.  Selecting checkpoints at alpha=1 by global loss would
    # discard a useful direction before shrinkage can make it safe.
    best_loss = float(baseline["core"])
    best_prefix = base_prefix.detach().clone()
    best_step = 0
    best_monitor = dict(baseline)
    history = [{"step": 0, **baseline}]
    stale = 0
    steps_executed = 0
    stop_reason = "max_steps"
    progress = tqdm(range(max_steps), desc="  V6 learner", unit="step", leave=False)
    for step_zero in progress:
        group = schedule[step_zero * accumulation : (step_zero + 1) * accumulation]
        group_rows = [train_rows[index] for index in group]
        totals = {
            key: sum(
                len(train_references[str(row["id"])][f"{key}_offsets"])
                for row in group_rows
            )
            for key in ("core", "history", "anchor")
        }
        optimizer.zero_grad(set_to_none=True)
        last = 0.0
        for row in group_rows:
            reference = train_references[str(row["id"])]
            candidate, delta = forward_candidate(
                torch,
                prefix_model,
                row,
                reference,
                learner_prefix=prefix_model.prefix_embeddings,
                alpha=1.0,
                max_prompt_tokens=max_prompt_tokens,
                max_target_tokens=max_target_tokens,
                inference=False,
            )
            losses = candidate_losses(
                torch, candidate, delta, reference, delta_weight=delta_weight
            )
            loss = losses["delta"] * (delta_weight / len(group_rows))
            for key in ("core", "history", "anchor"):
                count = int(losses[f"{key}_count"])
                if count:
                    loss = loss + losses[key] * (count / totals[key])
            loss.backward()
            last = float(losses["total"].detach().cpu())
            del candidate, delta, losses, loss
            gc.collect()
            torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_([prefix_model.prefix_embeddings], max_norm=1.0)
        optimizer.step()
        steps_executed = step_zero + 1
        progress.set_postfix(loss=f"{last:.4f}")
        if steps_executed % monitor_interval:
            continue
        current_prefix = prefix_model.prefix_embeddings.detach().clone()
        monitor = evaluate_stage(
            torch,
            prefix_model,
            monitor_rows,
            monitor_references,
            learner_prefix=current_prefix,
            alpha=1.0,
            delta_weight=delta_weight,
            max_prompt_tokens=max_prompt_tokens,
            max_target_tokens=max_target_tokens,
        )
        current_loss = float(monitor["core"])
        relative = (best_loss - current_loss) / max(abs(best_loss), 1e-12)
        significant = relative >= min_relative_improvement
        if current_loss < best_loss:
            best_loss = current_loss
            best_prefix = current_prefix
            best_step = steps_executed
            best_monitor = dict(monitor)
        stale = 0 if significant else stale + 1
        history.append(
            {
                "step": steps_executed,
                **monitor,
                "relative_vs_best_before": relative,
                "significant_improvement": significant,
                "stale_evaluations": stale,
            }
        )
        install_prefix(torch, prefix_model, current_prefix)
        if steps_executed >= min_steps and stale >= patience:
            stop_reason = "patience"
            break
    progress.close()
    install_prefix(torch, prefix_model, best_prefix)
    return best_prefix, {
        "steps_executed": steps_executed,
        "best_step": best_step,
        "stop_reason": stop_reason,
        "baseline_monitor": baseline,
        "best_raw_monitor": best_monitor,
        "checkpoint_selection_metric": "monitor_core_skill_kl",
        "monitor_history": history,
        "training_schedule": schedule[: steps_executed * accumulation],
    }


def train_low_rank_learner(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    train_rows: list[dict[str, Any]],
    monitor_rows: list[dict[str, Any]],
    train_references: dict[str, dict[str, Any]],
    monitor_references: dict[str, dict[str, Any]],
    *,
    base_prefix: Any,
    rank: int,
    learning_rate: float,
    max_steps: int,
    accumulation: int,
    monitor_interval: int,
    min_steps: int,
    patience: int,
    min_relative_improvement: float,
    delta_weight: float,
    max_prompt_tokens: int,
    max_target_tokens: int,
    seed: int,
    schedule_offset: int,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Fit an independent learner constrained to P = P0 + A @ B."""
    if not 0 < int(rank) <= min(base_prefix.shape):
        raise ValueError(f"Invalid learner rank {rank} for {tuple(base_prefix.shape)}")
    prefix_model.prefix_embeddings.requires_grad_(False)
    hidden = int(base_prefix.shape[1])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 104729 * (int(schedule_offset) + 1))
    raw = torch.randn(hidden, int(rank), generator=generator, dtype=torch.float32)
    orthogonal, _ = torch.linalg.qr(raw, mode="reduced")
    left = torch.nn.Parameter(
        torch.zeros(
            int(base_prefix.shape[0]),
            int(rank),
            dtype=torch.float32,
            device=prefix_model.device,
        )
    )
    right = torch.nn.Parameter(
        orthogonal.T.contiguous().to(device=prefix_model.device)
    )
    parameters = [left, right]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)

    def realized() -> Any:
        return base_prefix.float() + left @ right

    schedule = round_schedule(
        len(train_rows),
        round_index=1,
        examples=max_steps * accumulation,
        seed=seed,
        start_offset=schedule_offset,
    )
    baseline = evaluate_stage(
        torch,
        prefix_model,
        monitor_rows,
        monitor_references,
        learner_prefix=base_prefix,
        alpha=1.0,
        delta_weight=delta_weight,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
    )
    best_loss = float(baseline["core"])
    best_prefix = base_prefix.detach().clone()
    best_left = left.detach().cpu().clone()
    best_right = right.detach().cpu().clone()
    best_step = 0
    best_monitor = dict(baseline)
    history = [{"step": 0, **baseline}]
    stale = 0
    steps_executed = 0
    stop_reason = "max_steps"
    progress = tqdm(
        range(max_steps),
        desc=f"  V6 low-rank r={rank}",
        unit="step",
        leave=False,
    )
    for step_zero in progress:
        group = schedule[step_zero * accumulation : (step_zero + 1) * accumulation]
        group_rows = [train_rows[index] for index in group]
        totals = {
            key: sum(
                len(train_references[str(row["id"])][f"{key}_offsets"])
                for row in group_rows
            )
            for key in ("core", "history", "anchor")
        }
        optimizer.zero_grad(set_to_none=True)
        last = 0.0
        for row in group_rows:
            reference = train_references[str(row["id"])]
            candidate, delta = forward_candidate(
                torch,
                prefix_model,
                row,
                reference,
                learner_prefix=realized(),
                alpha=1.0,
                max_prompt_tokens=max_prompt_tokens,
                max_target_tokens=max_target_tokens,
                inference=False,
            )
            losses = candidate_losses(
                torch, candidate, delta, reference, delta_weight=delta_weight
            )
            loss = losses["delta"] * (delta_weight / len(group_rows))
            for key in ("core", "history", "anchor"):
                count = int(losses[f"{key}_count"])
                if count:
                    loss = loss + losses[key] * (count / totals[key])
            loss.backward()
            last = float(losses["total"].detach().cpu())
            del candidate, delta, losses, loss
            gc.collect()
            torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        steps_executed = step_zero + 1
        progress.set_postfix(loss=f"{last:.4f}")
        if steps_executed % monitor_interval:
            continue
        current_prefix = realized().detach().to(dtype=base_prefix.dtype)
        monitor = evaluate_stage(
            torch,
            prefix_model,
            monitor_rows,
            monitor_references,
            learner_prefix=current_prefix,
            alpha=1.0,
            delta_weight=delta_weight,
            max_prompt_tokens=max_prompt_tokens,
            max_target_tokens=max_target_tokens,
        )
        current_loss = float(monitor["core"])
        relative = (best_loss - current_loss) / max(abs(best_loss), 1e-12)
        significant = relative >= min_relative_improvement
        if current_loss < best_loss:
            best_loss = current_loss
            best_prefix = current_prefix
            best_left = left.detach().cpu().clone()
            best_right = right.detach().cpu().clone()
            best_step = steps_executed
            best_monitor = dict(monitor)
        stale = 0 if significant else stale + 1
        history.append(
            {
                "step": steps_executed,
                **monitor,
                "relative_vs_best_before": relative,
                "significant_improvement": significant,
                "stale_evaluations": stale,
            }
        )
        if steps_executed >= min_steps and stale >= patience:
            stop_reason = "patience"
            break
    progress.close()
    trainable_parameters = int(left.numel() + right.numel())
    return best_prefix, {
        "steps_executed": steps_executed,
        "best_step": best_step,
        "stop_reason": stop_reason,
        "baseline_monitor": baseline,
        "best_raw_monitor": best_monitor,
        "checkpoint_selection_metric": "monitor_core_skill_kl",
        "monitor_history": history,
        "training_schedule": schedule[: steps_executed * accumulation],
        "learner_parameterization": "low_rank_additive",
        "learner_rank": int(rank),
        "trainable_parameters": trainable_parameters,
        "full_prefix_parameters": int(base_prefix.numel()),
        "parameter_fraction": trainable_parameters / int(base_prefix.numel()),
    }, {
        "rank": int(rank),
        "left": best_left,
        "right": best_right,
        "realized_prefix": best_prefix.detach().cpu(),
    }


def build_stage_references(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    rows: list[dict[str, Any]],
    *,
    base_prefix: Any,
    learners: list[Any],
    alphas: list[float],
    max_prompt_tokens: int,
    max_target_tokens: int,
    top_k: int,
    desc: str,
) -> dict[str, dict[str, Any]]:
    references = {}
    for row in tqdm(rows, desc=desc, unit="traj"):
        references[str(row["id"])] = build_reference(
            torch,
            prefix_model,
            row,
            base_prefix=base_prefix,
            learners=learners,
            alphas=alphas,
            max_prompt_tokens=max_prompt_tokens,
            max_target_tokens=max_target_tokens,
            top_k=top_k,
        )
        gc.collect()
        torch.cuda.empty_cache()
    return references


def distill_student(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    train_rows: list[dict[str, Any]],
    monitor_rows: list[dict[str, Any]],
    *,
    base_prefix: Any,
    learners: list[Any],
    alphas: list[float],
    epochs: int,
    learning_rate: float,
    accumulation: int,
    max_prompt_tokens: int,
    max_target_tokens: int,
    top_k: int,
    out_root: Path,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Distill the frozen ensemble on cumulative core/anchor positions."""
    all_rows = train_rows + monitor_rows
    references = build_stage_references(
        torch,
        prefix_model,
        all_rows,
        base_prefix=base_prefix,
        learners=learners,
        alphas=alphas,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
        top_k=top_k,
        desc="  Cache ensemble teacher",
    )
    # In distillation every selected position uses the final ensemble itself.
    for reference in references.values():
        ids, logp, residual = topk_reference_from_logits(
            torch,
            reference["fprev"].to(prefix_model.device),
            top_k=top_k,
        )
        reference["ensemble_topk_ids"] = ids.cpu()
        reference["ensemble_topk_logp"] = logp.cpu()
        reference["ensemble_residual"] = residual.cpu()
    install_prefix(torch, prefix_model, base_prefix)
    optimizer = torch.optim.AdamW(
        [prefix_model.prefix_embeddings], lr=learning_rate, weight_decay=0.0
    )

    def evaluate(rows: list[dict[str, Any]]) -> dict[str, float]:
        total = count = 0.0
        student = prefix_model.prefix_embeddings.detach().clone()
        for row in rows:
            ref = references[str(row["id"])]
            install_prefix(torch, prefix_model, student)
            logits = logits_for_positions(
                torch,
                prefix_model,
                row,
                ref["positions"],
                max_prompt_tokens=max_prompt_tokens,
                max_target_tokens=max_target_tokens,
                inference=True,
            )
            loss = topk_residual_kl_from_logits(
                torch,
                logits,
                reference_topk_ids=ref["ensemble_topk_ids"],
                reference_topk_logp=ref["ensemble_topk_logp"],
                reference_residual_log_mass=ref["ensemble_residual"],
            )
            total += float(loss.sum().cpu())
            count += int(loss.numel())
        return {"ensemble_kl": total / max(count, 1), "tokens": int(count)}

    baseline = evaluate(monitor_rows)
    best_loss = float(baseline["ensemble_kl"])
    best_prefix = base_prefix.clone()
    history = []
    order = list(range(len(train_rows)))
    for epoch in range(1, epochs + 1):
        random.Random(seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        for offset, index in enumerate(tqdm(order, desc=f"  Distill {epoch}", leave=False)):
            row = train_rows[index]
            ref = references[str(row["id"])]
            logits = logits_for_positions(
                torch,
                prefix_model,
                row,
                ref["positions"],
                max_prompt_tokens=max_prompt_tokens,
                max_target_tokens=max_target_tokens,
                inference=False,
            )
            loss = topk_residual_kl_from_logits(
                torch,
                logits,
                reference_topk_ids=ref["ensemble_topk_ids"],
                reference_topk_logp=ref["ensemble_topk_logp"],
                reference_residual_log_mass=ref["ensemble_residual"],
            ).mean()
            (loss / accumulation).backward()
            if (offset + 1) % accumulation == 0 or offset + 1 == len(order):
                torch.nn.utils.clip_grad_norm_([prefix_model.prefix_embeddings], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            del logits, loss
            gc.collect()
            torch.cuda.empty_cache()
        monitor = evaluate(monitor_rows)
        history.append({"epoch": epoch, **monitor})
        if float(monitor["ensemble_kl"]) < best_loss:
            best_loss = float(monitor["ensemble_kl"])
            best_prefix = prefix_model.prefix_embeddings.detach().clone()
            torch.save(prefix_model.state_dict(), out_root / "student_best_prefix.pt")
    install_prefix(torch, prefix_model, best_prefix)
    return best_prefix, {
        "baseline_monitor": baseline,
        "best_monitor_ensemble_kl": best_loss,
        "history": history,
    }


def main() -> None:
    args = parse_args()
    alphas_grid = parse_alphas(args.alpha_grid)
    if args.max_stages <= 0 or args.max_steps_per_stage <= 0:
        raise ValueError("stage counts and step budgets must be positive")
    if args.learner_rank < 0 or args.learner_rank > 8:
        raise ValueError("learner-rank must be between 0 and 8")
    set_seed(args.seed)
    import torch

    out_root = resolve(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve(args.initial_checkpoint)
    model_source = resolve_model_reference(args.model_path)
    rows = read_jsonl(resolve(args.manifest))
    if args.limit > 0:
        rows = rows[: args.limit]
    train_ids, monitor_ids = fixed_monitor_split(
        rows, monitor_count=args.monitor_trajectories, seed=args.seed
    )
    config = {
        **vars(args),
        "method": (
            "PRCB-v6-lowrank-functional-logit-boosting"
            if args.learner_rank > 0
            else "PRCB-v6-functional-logit-boosting"
        ),
        "model_path": model_source,
        "initial_checkpoint": str(checkpoint_path),
        "initial_checkpoint_sha256": sha256(checkpoint_path),
        "learner_prefix_length": 8,
        "weak_learner": (
            f"centered_logits(P_0+A@B,rank={args.learner_rank})-centered_logits(P_0)"
            if args.learner_rank > 0
            else "centered_logits(P_s)-centered_logits(P_0)"
        ),
        "ensemble": "F_0+sum(alpha_s*h_s)",
        "learner_checkpoint_selection": "monitor_core_skill_kl",
        "stage_acceptance": "alpha_line_search_global_core_history_anchor",
        "history_replay": "cumulative_prior_core_minus_current_core",
        "alpha_values": alphas_grid,
        "train_ids": train_ids,
        "monitor_ids": monitor_ids,
        "test_split_accessed": False,
    }
    config_path = out_root / "prcb_v6_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing PRCB-v6 config differs from this run")
    if not config_path.exists():
        atomic_json(config_path, config)

    print(f"Loading frozen Qwen from {model_source}", flush=True)
    prefix_model = SoftPrefixCausalLM(
        model_source,
        prefix_length=8,
        init_strategy="random",
        torch_dtype="bfloat16",
        device="cuda",
    )
    base_prefix = load_prefix(
        torch,
        checkpoint_path,
        prefix_model.device,
        prefix_model.prefix_embeddings.dtype,
    )
    install_prefix(torch, prefix_model, base_prefix)
    prefix_model.model.config.use_cache = False
    prefix_model.model.eval()

    accepted: list[Any] = []
    accepted_alphas: list[float] = []
    cumulative_history: dict[str, set[int]] = {}
    final_stage_rows: list[dict[str, Any]] = []
    stage_summaries = []
    for stage in range(1, args.max_stages + 1):
        stage_dir = out_root / f"stage_{stage:02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        summary_path = stage_dir / "summary.json"
        learner_path = stage_dir / "learner_prefix.pt"
        if summary_path.exists() and learner_path.exists():
            summary = json.loads(summary_path.read_text())
            if not summary.get("accepted"):
                print(f"[stage {stage}] cached rejection; stopping", flush=True)
                break
            accepted.append(
                load_prefix(
                    torch,
                    learner_path,
                    prefix_model.device,
                    prefix_model.prefix_embeddings.dtype,
                )
            )
            accepted_alphas.append(float(summary["selected_alpha"]))
            final_stage_rows = read_jsonl(stage_dir / "manifest.jsonl")
            for row in final_stage_rows:
                cumulative_history.setdefault(str(row["id"]), set()).update(
                    int(value) for value in row["core_indices"]
                )
            stage_summaries.append(summary)
            print(f"[stage {stage}] resumed accepted learner", flush=True)
            continue

        started = time.time()
        print(f"[stage {stage}/{args.max_stages}] locate functional residual", flush=True)
        score_dir = stage_dir / "scores"
        scores = {}
        for row in tqdm(rows, desc=f"  V6 locate {stage}", unit="traj"):
            path = score_dir / f"{slug(str(row['id']))}.npz"
            if not path.exists():
                values = score_ensemble(
                    torch,
                    prefix_model,
                    row,
                    base_prefix=base_prefix,
                    learners=accepted,
                    alphas=accepted_alphas,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_target_tokens=args.max_target_tokens,
                    chunk_size=args.score_chunk_size,
                    top_k=args.top_k,
                )
                atomic_npz(path, **values)
            with np.load(path) as cached:
                scores[str(row["id"])] = {
                    key: cached[key] for key in cached.files
                }
        stage_rows, new_history, locator = build_stage_rows(
            rows,
            scores,
            cumulative_history,
            ratio=args.selection_ratio,
        )
        atomic_jsonl(stage_dir / "manifest.jsonl", stage_rows)
        atomic_json(stage_dir / "locator_statistics.json", locator)
        train_set = set(train_ids)
        monitor_set = set(monitor_ids)
        train_rows = [row for row in stage_rows if str(row["id"]) in train_set]
        monitor_rows = [row for row in stage_rows if str(row["id"]) in monitor_set]
        print(f"[stage {stage}] cache frozen ensemble references", flush=True)
        references = build_stage_references(
            torch,
            prefix_model,
            stage_rows,
            base_prefix=base_prefix,
            learners=accepted,
            alphas=accepted_alphas,
            max_prompt_tokens=args.max_prompt_tokens,
            max_target_tokens=args.max_target_tokens,
            top_k=args.top_k,
            desc=f"  V6 refs {stage}",
        )
        train_refs = {str(row["id"]): references[str(row["id"])] for row in train_rows}
        monitor_refs = {
            str(row["id"]): references[str(row["id"])] for row in monitor_rows
        }
        prefix_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        prefix_model.model.train()
        low_rank_state = None
        training_kwargs = dict(
            base_prefix=base_prefix,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps_per_stage,
            accumulation=args.accumulation,
            monitor_interval=args.monitor_interval,
            min_steps=args.monitor_min_steps,
            patience=args.monitor_patience,
            min_relative_improvement=args.min_relative_improvement,
            delta_weight=args.delta_weight,
            max_prompt_tokens=args.max_prompt_tokens,
            max_target_tokens=args.max_target_tokens,
            seed=args.seed,
            schedule_offset=(stage - 1) * args.max_steps_per_stage * args.accumulation,
        )
        if args.learner_rank > 0:
            learner, training, low_rank_state = train_low_rank_learner(
                torch,
                prefix_model,
                train_rows,
                monitor_rows,
                train_refs,
                monitor_refs,
                rank=args.learner_rank,
                **training_kwargs,
            )
        else:
            learner, training = train_learner(
                torch,
                prefix_model,
                train_rows,
                monitor_rows,
                train_refs,
                monitor_refs,
                **training_kwargs,
            )
        prefix_model.model.gradient_checkpointing_disable()
        prefix_model.model.eval()
        alpha_rows = []
        for alpha in alphas_grid:
            metrics = evaluate_stage(
                torch,
                prefix_model,
                monitor_rows,
                monitor_refs,
                learner_prefix=learner,
                alpha=alpha,
                delta_weight=args.delta_weight,
                max_prompt_tokens=args.max_prompt_tokens,
                max_target_tokens=args.max_target_tokens,
            )
            alpha_rows.append(
                {
                    "alpha": alpha,
                    **metrics,
                    "safe": float(metrics["history"]) <= args.history_kl_limit,
                }
            )
        chosen = choose_stage_alpha(alpha_rows)
        baseline_loss = next(
            float(row["global_loss"]) for row in alpha_rows if row["alpha"] == 0
        )
        improvement = (
            baseline_loss - float(chosen["global_loss"])
        ) / max(abs(baseline_loss), 1e-12)
        epsilon = 0.125 if 0.125 in alphas_grid else min(
            value for value in alphas_grid if value > 0
        )
        epsilon_loss = next(
            float(row["global_loss"]) for row in alpha_rows if row["alpha"] == epsilon
        )
        directional_edge = (baseline_loss - epsilon_loss) / epsilon
        accepted_stage = (
            float(chosen["alpha"]) > 0
            and improvement >= args.min_relative_improvement
            and directional_edge > 0
        )
        install_prefix(torch, prefix_model, learner)
        torch.save(prefix_model.state_dict(), learner_path)
        factor_path = None
        if low_rank_state is not None:
            factor_path = stage_dir / "learner_low_rank_factors.pt"
            torch.save(low_rank_state, factor_path)
        summary = {
            "stage": stage,
            "accepted": accepted_stage,
            "selected_alpha": float(chosen["alpha"]) if accepted_stage else 0.0,
            "relative_global_improvement": improvement,
            "directional_functional_edge": directional_edge,
            "locator": locator,
            "training": training,
            "alpha_line_search": alpha_rows,
            "learner_checkpoint": str(learner_path),
            "learner_checkpoint_sha256": sha256(learner_path),
            "learner_factor_checkpoint": str(factor_path) if factor_path else None,
            "learner_factor_checkpoint_sha256": (
                sha256(factor_path) if factor_path else None
            ),
            "wall_time_s": round(time.time() - started, 1),
            "test_split_accessed": False,
        }
        atomic_json(summary_path, summary)
        stage_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        if not accepted_stage:
            print(f"[stage {stage}] rejected; boosting stops", flush=True)
            break
        accepted.append(learner.detach().clone())
        accepted_alphas.append(float(chosen["alpha"]))
        cumulative_history = new_history
        final_stage_rows = stage_rows

    ensemble_manifest = {
        "base_checkpoint": str(checkpoint_path),
        "base_checkpoint_sha256": sha256(checkpoint_path),
        "learner_checkpoints": [
            summary["learner_checkpoint"]
            for summary in stage_summaries
            if summary.get("accepted")
        ],
        "alphas": accepted_alphas,
        "accepted_learners": len(accepted),
        "test_split_accessed": False,
    }
    atomic_json(out_root / "ensemble_manifest.json", ensemble_manifest)
    summary: dict[str, Any] = {
        "accepted_learners": len(accepted),
        "alphas": accepted_alphas,
        "stages": stage_summaries,
        "test_split_accessed": False,
    }
    if not accepted or args.skip_distill:
        atomic_json(out_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return

    # Use cumulative cores and the final stage anchors for the fixed student.
    final_by_id = {str(row["id"]): dict(row) for row in final_stage_rows}
    for source in rows:
        identifier = str(source["id"])
        row = final_by_id[identifier]
        row["core_indices"] = sorted(cumulative_history.get(identifier, set()))
        row["history_indices"] = []
    final_rows = list(final_by_id.values())
    train_set = set(train_ids)
    monitor_set = set(monitor_ids)
    final_train = [row for row in final_rows if str(row["id"]) in train_set]
    final_monitor = [row for row in final_rows if str(row["id"]) in monitor_set]
    prefix_model.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    prefix_model.model.train()
    prefix_model.prefix_embeddings.requires_grad_(True)
    student, distillation = distill_student(
        torch,
        prefix_model,
        final_train,
        final_monitor,
        base_prefix=base_prefix,
        learners=accepted,
        alphas=accepted_alphas,
        epochs=args.student_epochs,
        learning_rate=args.student_learning_rate,
        accumulation=args.accumulation,
        max_prompt_tokens=args.max_prompt_tokens,
        max_target_tokens=args.max_target_tokens,
        top_k=args.top_k,
        out_root=out_root,
        seed=args.seed,
    )
    install_prefix(torch, prefix_model, student)
    final_student = out_root / "prcb_v6_student_prefix.pt"
    torch.save(prefix_model.state_dict(), final_student)
    summary["distillation"] = distillation
    summary["final_student"] = str(final_student)
    summary["final_student_sha256"] = sha256(final_student)
    if not args.skip_val:
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
        validation = dataloader.load_split_items(str(resolve(args.split_dir) / "val"))[:40]
        hard, soft, results = evaluate_spreadsheet_prefix(
            prefix_model,
            validation,
            out_dir=str(out_root / "eval" / "student" / "valid_seen"),
            data_root=str(resolve(args.data_root)),
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            exec_timeout=600,
            desc="PRCB-v6 Student Val",
            generator=None,
            injection_position="prompt_start",
            repair_turns=1,
            generation_batch_size=args.generation_batch_size,
        )
        summary["student_valid_seen_hard"] = hard
        summary["student_valid_seen_soft"] = soft
        summary["student_valid_seen_successes"] = sum(
            bool(row.get("hard")) for row in results
        )
    atomic_json(out_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
