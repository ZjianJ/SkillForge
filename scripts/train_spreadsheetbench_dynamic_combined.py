#!/usr/bin/env python3
"""Train a soft prefix with convergence-triggered dynamic Combined localization."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_spreadsheetbench_combined_full_vocab_kl import (  # noqa: E402
    _empty_cuda_cache,
    _load_rows,
    _redact,
    _sha256,
    _training_order,
)
from skillopt.config import flatten_config, is_structured, load_config  # noqa: E402
from skillopt.softprefix.distillation_losses import (  # noqa: E402
    chunked_full_vocab_forward_kl,
    chunked_weighted_full_vocab_forward_kl,
    topk_residual_forward_kl,
)
from skillopt.softprefix.dynamic_combined import (  # noqa: E402
    dynamic_additive_scores,
    dynamic_gain_competitor_scores,
    dynamic_skill_effect_scores,
    dynamic_stop_decision,
    full_vocab_dynamic_metrics,
    jaccard_indices,
    locator_loss_weights,
    select_dynamic_top_fraction,
)
from skillopt.softprefix.entropy_localization import select_top_fraction  # noqa: E402
from skillopt.softprefix.official_distillation import encode_trajectory, target_logits  # noqa: E402
from skillopt.softprefix.trainer import (  # noqa: E402
    SoftPrefixSettings,
    _build_dataloader,
    _build_prefix_model,
    _evaluate_prefix,
    _items_for_eval,
    _load_init_text,
    _set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--model_name", default="")
    parser.add_argument("--cfg-options", nargs="+", default=[])
    parser.add_argument("--no_val", action="store_true")
    return parser.parse_args()


def _prepare_records(rows, prefix_model, settings, skill_text, dynamic_cfg):
    preserve_field = str(dynamic_cfg.get("preservation_label_field", "preserve_indices"))
    records = []
    for row in tqdm(rows, desc="Encode Dynamic Combined train61", unit="ex"):
        example = encode_trajectory(
            tokenizer=prefix_model.tokenizer,
            row=row,
            skill_text=skill_text,
            max_prompt_tokens=settings.max_prompt_tokens,
            max_target_tokens=settings.max_target_tokens,
        )
        eos_index = len(example.target_ids) - 1
        preserve = sorted({int(index) for index in row.get(preserve_field, [])})
        if any(index < 0 or index >= eos_index for index in preserve):
            raise ValueError(f"Trajectory {example.task_id} has an out-of-range preservation index")
        with np.load(example.score_cache) as cached:
            cached_ids = cached["target_ids"].astype(np.int64).tolist()
            if cached_ids != example.target_ids:
                raise ValueError(f"Tokenizer/cache mismatch for trajectory {example.task_id}")
            original_beneficial = cached["positive_gain"][:eos_index].astype(np.float32) > 0
            clean_topk_ids = cached["clean_topk_ids"][preserve].astype(np.int64)
            clean_topk_logp = cached["clean_topk_logp"][preserve].astype(np.float32)
            clean_residual = cached["clean_residual_log_mass"][preserve].astype(np.float32)
        records.append(
            {
                "example": example,
                "selected": [],
                "preserve": preserve,
                "original_beneficial": original_beneficial,
                "clean_topk_ids": clean_topk_ids,
                "clean_topk_logp": clean_topk_logp,
                "clean_residual": clean_residual,
            }
        )
    if not records:
        raise ValueError("Dynamic Combined requires at least one successful trajectory")
    return records


def _dynamic_localize(prefix_model, records, *, round_index, out_root, dynamic_cfg, previous):
    torch = prefix_model.torch
    ratio = float(dynamic_cfg.get("core_ratio", 0.10))
    chunk_size = int(dynamic_cfg.get("kl_chunk_size", 8))
    exclude_preserve = bool(dynamic_cfg.get("exclude_preservation_from_core", True))
    locator_method = str(dynamic_cfg.get("locator_method", "combined")).strip().lower()
    if locator_method not in {
        "combined",
        "additive_skill",
        "gain_competitor",
        "skill_effect_additive",
        "skill_effect_multiplicative",
    }:
        raise ValueError(f"Unsupported dynamic locator method: {locator_method!r}")
    additive_alpha = float(dynamic_cfg.get("additive_alpha", 0.5))
    gain_weight = float(dynamic_cfg.get("gain_weight", 0.75))
    four_signal_weights = [
        float(value)
        for value in dynamic_cfg.get("four_signal_weights", [0.45, 0.45, 0.10, 0.0])
    ]
    loss_weighting = str(dynamic_cfg.get("loss_weighting", "equal")).strip().lower()
    if loss_weighting not in {"equal", "locator_score"}:
        raise ValueError(f"Unsupported dynamic loss weighting: {loss_weighting!r}")
    effective_core_ratio = float(dynamic_cfg.get("effective_core_ratio", ratio))
    if not 0.0 < effective_core_ratio < 1.0:
        raise ValueError("effective_core_ratio must be strictly between zero and one")
    locator_dir = out_root / "locators" / f"round_{round_index:02d}"
    array_dir = locator_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = locator_dir / "manifest.jsonl"

    global_mass = 0.0
    global_full_vocab_kl_mass = 0.0
    eligible_kl_sum = 0.0
    eligible_count = 0
    selected_mass = 0.0
    selected_kl_sum = 0.0
    selected_count = 0
    selected_effective_weight = 0.0
    preservation_sum = 0.0
    preservation_count = 0
    jaccards = []
    rows_out = []

    for record in tqdm(records, desc=f"Dynamic locator r{round_index}", unit="traj"):
        example = record["example"]
        count = len(example.target_ids) - 1
        indices = list(range(count))
        teacher_logits = target_logits(
            prefix_model, example, indices, use_prefix=False, with_grad=False
        )
        student_logits = target_logits(
            prefix_model, example, indices, use_prefix=True, with_grad=False
        )
        metrics = full_vocab_dynamic_metrics(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            target_ids=example.target_ids[:count],
            original_beneficial=record["original_beneficial"],
            chunk_size=chunk_size,
        )
        forbidden = record["preserve"] if exclude_preserve else []
        if locator_method == "combined":
            candidate = np.asarray(record["original_beneficial"], dtype=bool)
            score = metrics["combined"]
            selected = select_dynamic_top_fraction(
                score, ratio=ratio, forbidden_indices=forbidden
            )
            normalized_gain = np.zeros_like(score, dtype=np.float32)
            normalized_js = np.zeros_like(score, dtype=np.float32)
            normalized_competitor = np.zeros_like(score, dtype=np.float32)
        else:
            candidate = np.ones(count, dtype=bool)
            if forbidden:
                candidate[np.asarray(forbidden, dtype=np.int64)] = False
            normalized_resolved = np.zeros(count, dtype=np.float32)
            if locator_method == "additive_skill":
                additive = dynamic_additive_scores(
                    metrics["residual_gain"],
                    metrics["js"],
                    alpha=additive_alpha,
                    eligible_indices=np.flatnonzero(candidate),
                )
                score = additive["additive"]
                normalized_gain = additive["normalized_gain"]
                normalized_js = additive["normalized_js"]
                normalized_competitor = np.zeros_like(score, dtype=np.float32)
            elif locator_method == "gain_competitor":
                gain_competitor = dynamic_gain_competitor_scores(
                    metrics["residual_gain"],
                    metrics["competitor_suppression"],
                    gain_weight=gain_weight,
                    eligible_indices=np.flatnonzero(candidate),
                )
                score = gain_competitor["gain_competitor"]
                normalized_gain = gain_competitor["normalized_gain"]
                normalized_js = np.zeros_like(score, dtype=np.float32)
                normalized_competitor = gain_competitor["normalized_competitor"]
            else:
                four_signal = dynamic_skill_effect_scores(
                    metrics["residual_gain"],
                    metrics["js"],
                    metrics["competitor_suppression"],
                    metrics["resolved_uncertainty"],
                    weights=four_signal_weights,
                    mode=(
                        "multiplicative"
                        if locator_method == "skill_effect_multiplicative"
                        else "additive"
                    ),
                    eligible_indices=np.flatnonzero(candidate),
                )
                score = four_signal["skill_effect"]
                normalized_gain = four_signal["normalized_gain"]
                normalized_js = four_signal["normalized_js"]
                normalized_competitor = four_signal["normalized_competitor"]
                normalized_resolved = four_signal["normalized_resolved_uncertainty"]
            selected = select_top_fraction(score, ratio=ratio, forbidden=forbidden)
        if locator_method.startswith("skill_effect_"):
            base_effect = (
                four_signal_weights[0] * normalized_gain
                + four_signal_weights[1] * normalized_js
                + four_signal_weights[2] * normalized_competitor
            )
            uncertainty_expert = (
                base_effect * (1.0 + normalized_resolved)
                if locator_method == "skill_effect_multiplicative"
                else normalized_resolved
            )
            record["meta_expert_selected"] = {
                "gain": select_top_fraction(normalized_gain, ratio=ratio, forbidden=forbidden),
                "js": select_top_fraction(normalized_js, ratio=ratio, forbidden=forbidden),
                "competitor": select_top_fraction(
                    normalized_competitor, ratio=ratio, forbidden=forbidden
                ),
                "resolved": select_top_fraction(
                    uncertainty_expert, ratio=ratio, forbidden=forbidden
                ),
            }
        requested_effective_weight = min(
            len(selected), max(1, math.ceil(count * effective_core_ratio))
        )
        if loss_weighting == "locator_score":
            selected_weights = locator_loss_weights(
                score,
                selected,
                effective_weight=requested_effective_weight,
            )
        else:
            selected_weights = np.ones(len(selected), dtype=np.float32)
            requested_effective_weight = len(selected)
        previous_selected = [] if previous is None else previous.get(example.task_id, [])
        if previous is not None:
            jaccards.append(jaccard_indices(previous_selected, selected))

        if exclude_preserve and record["preserve"]:
            candidate[np.asarray(record["preserve"], dtype=np.int64)] = False
        global_mass += float(score[candidate].sum(dtype=np.float64))
        global_full_vocab_kl_mass += float(metrics["forward_kl"][candidate].sum(dtype=np.float64))
        eligible_kl_sum += float(metrics["forward_kl"][candidate].sum(dtype=np.float64))
        eligible_count += int(candidate.sum())
        if selected:
            selected_array = np.asarray(selected, dtype=np.int64)
            selected_mass += float(score[selected_array].sum(dtype=np.float64))
            selected_kl_sum += float(metrics["forward_kl"][selected_array].sum(dtype=np.float64))
            selected_count += len(selected)
            selected_effective_weight += float(selected_weights.sum(dtype=np.float64))

        preserve_loss = 0.0
        if record["preserve"]:
            preserve_rows = torch.as_tensor(
                record["preserve"], dtype=torch.long, device=student_logits.device
            )
            preserve_logits = student_logits[preserve_rows]
            preserve_tensor = topk_residual_forward_kl(
                student_logits=preserve_logits,
                reference_topk_ids=record["clean_topk_ids"],
                reference_topk_logp=record["clean_topk_logp"],
                reference_residual_log_mass=record["clean_residual"],
            )
            preserve_loss = float(preserve_tensor.detach().cpu())
            preservation_sum += preserve_loss * len(record["preserve"])
            preservation_count += len(record["preserve"])

        record["selected"] = selected
        record["selected_weights"] = selected_weights
        record["effective_skill_weight"] = float(selected_weights.sum(dtype=np.float64))
        slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in example.task_id)
        array_path = array_dir / f"{slug}.npz"
        np.savez_compressed(
            array_path,
            target_ids=np.asarray(example.target_ids[:count], dtype=np.int32),
            original_beneficial=np.asarray(record["original_beneficial"], dtype=np.bool_),
            selected_indices=np.asarray(selected, dtype=np.int32),
            selected_loss_weights=selected_weights,
            residual_gain=metrics["residual_gain"],
            full_vocab_kl=metrics["forward_kl"],
            full_vocab_js=metrics["js"],
            competitor_suppression=metrics["competitor_suppression"],
            teacher_entropy=metrics["teacher_entropy"],
            student_entropy=metrics["student_entropy"],
            resolved_uncertainty=metrics["resolved_uncertainty"],
            dynamic_combined=metrics["combined"],
            normalized_residual_gain=normalized_gain,
            normalized_dynamic_js=normalized_js,
            normalized_competitor_suppression=normalized_competitor,
            normalized_resolved_uncertainty=(
                normalized_resolved
                if locator_method.startswith("skill_effect_")
                else np.zeros_like(score, dtype=np.float32)
            ),
            dynamic_locator_score=score,
        )
        rows_out.append(
            {
                "id": example.task_id,
                "selectable_tokens": count,
                "eligible_tokens": int(candidate.sum()),
                "selected_indices": selected,
                "selected_tokens": len(selected),
                "loss_weighting": loss_weighting,
                "selected_loss_weights": selected_weights.tolist(),
                "effective_skill_weight": float(selected_weights.sum(dtype=np.float64)),
                "previous_jaccard": (
                    None if previous is None else jaccard_indices(previous_selected, selected)
                ),
                "locator_method": locator_method,
                "total_dynamic_mass": float(score[candidate].sum(dtype=np.float64)),
                "total_full_vocab_kl_mass": float(
                    metrics["forward_kl"][candidate].sum(dtype=np.float64)
                ),
                "selected_dynamic_mass": (
                    float(score[np.asarray(selected, dtype=np.int64)].sum(dtype=np.float64))
                    if selected
                    else 0.0
                ),
                "mean_eligible_full_vocab_kl": (
                    float(metrics["forward_kl"][candidate].mean()) if candidate.any() else 0.0
                ),
                "mean_selected_full_vocab_kl": (
                    float(metrics["forward_kl"][np.asarray(selected, dtype=np.int64)].mean())
                    if selected
                    else 0.0
                ),
                "preservation_kl": preserve_loss,
                "array_path": str(array_path),
            }
        )
        del teacher_logits, student_logits, metrics
        _empty_cuda_cache(torch)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if selected_count == 0:
        mean_selected_kl = 0.0
    else:
        mean_selected_kl = selected_kl_sum / selected_count
    summary = {
        "round": round_index,
        "locator_method": locator_method,
        "additive_alpha": additive_alpha if locator_method == "additive_skill" else None,
        "gain_weight": gain_weight if locator_method == "gain_competitor" else None,
        "four_signal_weights": (
            four_signal_weights if locator_method.startswith("skill_effect_") else None
        ),
        "trajectories": len(records),
        "selected_tokens": selected_count,
        "loss_weighting": loss_weighting,
        "effective_core_ratio": effective_core_ratio,
        "selected_effective_weight": selected_effective_weight,
        "global_dynamic_residual_mass": global_mass,
        "global_full_vocab_kl_mass": global_full_vocab_kl_mass,
        "selected_dynamic_residual_mass": selected_mass,
        "selected_mass_capture": selected_mass / max(global_mass, 1e-12),
        "eligible_tokens": eligible_count,
        "mean_eligible_full_vocab_kl": eligible_kl_sum / max(eligible_count, 1),
        "mean_selected_full_vocab_kl": mean_selected_kl,
        "mean_preservation_kl": preservation_sum / max(preservation_count, 1),
        "mean_previous_core_jaccard": None if not jaccards else float(np.mean(jaccards)),
        "manifest_path": str(manifest_path),
    }
    (locator_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary, {record["example"].task_id: list(record["selected"]) for record in records}


def _record_losses(prefix_model, record, *, dynamic_cfg, with_grad):
    torch = prefix_model.torch
    selected = record["selected"]
    example = record["example"]
    eos_index = len(example.target_ids) - 1
    teacher_logits = target_logits(
        prefix_model, example, selected, use_prefix=False, with_grad=False
    )
    student_logits = target_logits(
        prefix_model, example, selected + [eos_index], use_prefix=True, with_grad=with_grad
    )
    loss_weighting = str(dynamic_cfg.get("loss_weighting", "equal")).strip().lower()
    if loss_weighting == "locator_score":
        skill_kl = chunked_weighted_full_vocab_forward_kl(
            teacher_logits=teacher_logits,
            student_logits=student_logits[:-1],
            token_weights=record["selected_weights"],
            chunk_size=int(dynamic_cfg.get("kl_chunk_size", 8)),
        )
    else:
        skill_kl = chunked_full_vocab_forward_kl(
            teacher_logits=teacher_logits,
            student_logits=student_logits[:-1],
            chunk_size=int(dynamic_cfg.get("kl_chunk_size", 8)),
        )
    eos_target = torch.tensor([example.target_ids[-1]], dtype=torch.long, device=prefix_model.device)
    eos_ce = torch.nn.functional.cross_entropy(student_logits[-1:].float(), eos_target)
    preservation = student_logits.sum() * 0.0
    if record["preserve"]:
        preservation_logits = target_logits(
            prefix_model, example, record["preserve"], use_prefix=True, with_grad=with_grad
        )
        preservation = topk_residual_forward_kl(
            student_logits=preservation_logits,
            reference_topk_ids=record["clean_topk_ids"],
            reference_topk_logp=record["clean_topk_logp"],
            reference_residual_log_mass=record["clean_residual"],
        )
        del preservation_logits
    del teacher_logits, student_logits
    return skill_kl, eos_ce, preservation


def _monitor(prefix_model, records, monitor_indices, dynamic_cfg):
    if str(dynamic_cfg.get("monitor_scope", "selected")).strip().lower() == "global":
        return _global_monitor(prefix_model, records, monitor_indices, dynamic_cfg)
    skill_sum = eos_sum = preserve_sum = 0.0
    skill_count = eos_count = preserve_count = 0
    for index in monitor_indices:
        record = records[index]
        skill_kl, eos_ce, preservation = _record_losses(
            prefix_model, record, dynamic_cfg=dynamic_cfg, with_grad=False
        )
        selected_count = float(record.get("effective_skill_weight", len(record["selected"])))
        current_preserve = len(record["preserve"])
        skill_sum += float(skill_kl.detach().cpu()) * selected_count
        eos_sum += float(eos_ce.detach().cpu())
        preserve_sum += float(preservation.detach().cpu()) * current_preserve
        skill_count += selected_count
        eos_count += 1
        preserve_count += current_preserve
        del skill_kl, eos_ce, preservation
        _empty_cuda_cache(prefix_model.torch)
    core = (skill_sum + eos_sum) / max(skill_count + eos_count, 1)
    preserve = preserve_sum / max(preserve_count, 1)
    weight = float(dynamic_cfg.get("preservation_loss_weight", 1.0))
    return {
        "loss": core + weight * preserve,
        "core": core,
        "preservation": preserve,
        "skill_tokens": skill_count,
        "preservation_tokens": preserve_count,
    }


def _global_monitor(prefix_model, records, monitor_indices, dynamic_cfg):
    """Evaluate a locator-independent full-trajectory teacher-forced objective."""
    torch = prefix_model.torch
    kl_sum = nll_sum = preserve_sum = 0.0
    token_count = preserve_count = 0
    chunk_size = int(dynamic_cfg.get("kl_chunk_size", 8))
    for index in monitor_indices:
        record = records[index]
        example = record["example"]
        count = len(example.target_ids) - 1
        positions = list(range(count))
        teacher_logits = target_logits(
            prefix_model, example, positions, use_prefix=False, with_grad=False
        )
        student_logits = target_logits(
            prefix_model, example, positions, use_prefix=True, with_grad=False
        )
        skill_kl = chunked_full_vocab_forward_kl(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            chunk_size=chunk_size,
        )
        targets = torch.as_tensor(
            example.target_ids[:count], dtype=torch.long, device=student_logits.device
        )
        gold_nll = torch.nn.functional.cross_entropy(student_logits.float(), targets)
        kl_sum += float(skill_kl.detach().cpu()) * count
        nll_sum += float(gold_nll.detach().cpu()) * count
        token_count += count
        if record["preserve"]:
            preserve_logits = student_logits[
                torch.as_tensor(record["preserve"], dtype=torch.long, device=student_logits.device)
            ]
            preservation = topk_residual_forward_kl(
                student_logits=preserve_logits,
                reference_topk_ids=record["clean_topk_ids"],
                reference_topk_logp=record["clean_topk_logp"],
                reference_residual_log_mass=record["clean_residual"],
            )
            preserve_sum += float(preservation.detach().cpu()) * len(record["preserve"])
            preserve_count += len(record["preserve"])
            del preservation, preserve_logits
        del teacher_logits, student_logits, skill_kl, gold_nll
        _empty_cuda_cache(torch)
    global_kl = kl_sum / max(token_count, 1)
    gold_nll = nll_sum / max(token_count, 1)
    preservation = preserve_sum / max(preserve_count, 1)
    loss = (
        float(dynamic_cfg.get("monitor_skill_kl_weight", 1.0)) * global_kl
        + float(dynamic_cfg.get("monitor_preservation_weight", 1.0)) * preservation
        + float(dynamic_cfg.get("monitor_gold_nll_weight", 0.1)) * gold_nll
    )
    return {
        "loss": loss,
        "core": global_kl,
        "global_skill_kl": global_kl,
        "gold_nll": gold_nll,
        "preservation": preservation,
        "skill_tokens": token_count,
        "preservation_tokens": preserve_count,
        "monitor_scope": "global",
    }


def _prefix_gradient_vector(prefix_model):
    torch = prefix_model.torch
    pieces = []
    for parameter in prefix_model.trainable_parameters():
        if parameter.grad is None:
            pieces.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1))
        else:
            pieces.append(parameter.grad.detach().float().reshape(-1).clone())
    return torch.cat(pieces)


def _global_monitor_gradient(prefix_model, records, monitor_indices, dynamic_cfg):
    """Gradient of the locator-independent held-out monitor objective."""
    torch = prefix_model.torch
    parameters = list(prefix_model.trainable_parameters())
    for parameter in parameters:
        parameter.grad = None
    token_total = sum(len(records[index]["example"].target_ids) - 1 for index in monitor_indices)
    preserve_total = sum(len(records[index]["preserve"]) for index in monitor_indices)
    kl_weight = float(dynamic_cfg.get("monitor_skill_kl_weight", 1.0))
    preserve_weight = float(dynamic_cfg.get("monitor_preservation_weight", 1.0))
    nll_weight = float(dynamic_cfg.get("monitor_gold_nll_weight", 0.1))
    chunk_size = int(dynamic_cfg.get("kl_chunk_size", 8))
    for index in monitor_indices:
        record = records[index]
        example = record["example"]
        count = len(example.target_ids) - 1
        positions = list(range(count))
        teacher_logits = target_logits(
            prefix_model, example, positions, use_prefix=False, with_grad=False
        )
        student_logits = target_logits(
            prefix_model, example, positions, use_prefix=True, with_grad=True
        )
        skill_kl = chunked_full_vocab_forward_kl(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            chunk_size=chunk_size,
        )
        targets = torch.as_tensor(
            example.target_ids[:count], dtype=torch.long, device=student_logits.device
        )
        gold_nll = torch.nn.functional.cross_entropy(student_logits.float(), targets)
        loss = (kl_weight * skill_kl + nll_weight * gold_nll) * (
            count / max(token_total, 1)
        )
        if record["preserve"]:
            preserve_logits = student_logits[
                torch.as_tensor(record["preserve"], dtype=torch.long, device=student_logits.device)
            ]
            preservation = topk_residual_forward_kl(
                student_logits=preserve_logits,
                reference_topk_ids=record["clean_topk_ids"],
                reference_topk_logp=record["clean_topk_logp"],
                reference_residual_log_mass=record["clean_residual"],
            )
            loss = loss + preserve_weight * preservation * (
                len(record["preserve"]) / max(preserve_total, 1)
            )
            del preservation, preserve_logits
        loss.backward()
        del teacher_logits, student_logits, skill_kl, gold_nll, loss
        _empty_cuda_cache(torch)
    gradient = _prefix_gradient_vector(prefix_model)
    for parameter in parameters:
        parameter.grad = None
    return gradient


def _expert_gradient(prefix_model, records, record_indices, expert_name, dynamic_cfg):
    """Gradient induced by one token-localization expert on Train49 only."""
    torch = prefix_model.torch
    parameters = list(prefix_model.trainable_parameters())
    for parameter in parameters:
        parameter.grad = None
    selected_total = sum(
        len(records[index].get("meta_expert_selected", {}).get(expert_name, []))
        for index in record_indices
    )
    for index in record_indices:
        record = records[index]
        selected = record.get("meta_expert_selected", {}).get(expert_name, [])
        if not selected:
            continue
        example = record["example"]
        teacher_logits = target_logits(
            prefix_model, example, selected, use_prefix=False, with_grad=False
        )
        student_logits = target_logits(
            prefix_model, example, selected, use_prefix=True, with_grad=True
        )
        skill_kl = chunked_full_vocab_forward_kl(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            chunk_size=int(dynamic_cfg.get("kl_chunk_size", 8)),
        )
        (skill_kl * (len(selected) / max(selected_total, 1))).backward()
        del teacher_logits, student_logits, skill_kl
        _empty_cuda_cache(torch)
    gradient = _prefix_gradient_vector(prefix_model)
    for parameter in parameters:
        parameter.grad = None
    return gradient


def _cap_weight_change(new, old, maximum):
    proposed = np.asarray(new, dtype=np.float64)
    previous = np.asarray(old, dtype=np.float64)
    proposed = np.maximum(proposed, 1e-6)
    proposed /= proposed.sum()
    delta = proposed - previous
    largest = float(np.max(np.abs(delta)))
    scale = 1.0 if largest <= maximum else float(maximum) / largest
    updated = previous + scale * delta
    updated = np.maximum(updated, 1e-6)
    return updated / updated.sum()


def _adapt_locator_weights(
    prefix_model,
    records,
    *,
    train_indices,
    monitor_indices,
    dynamic_cfg,
    seed,
    round_index,
):
    """First-order meta update from Train49 expert/Monitor12 gradient alignment.

    For a virtual update along expert gradient ``g_i``, the first-order change
    in held-out monitor loss is proportional to ``-<g_monitor, g_i>``.  This is
    the inexpensive one-step approximation used here instead of retaining a
    second-order graph through the frozen 35B backbone.
    """
    torch = prefix_model.torch
    requested = min(int(dynamic_cfg.get("meta_train_trajectories", 8)), len(train_indices))
    relative = _training_order(torch, len(train_indices), seed + 17011 * round_index)[:requested]
    meta_train_indices = [train_indices[index] for index in relative]
    monitor_gradient = _global_monitor_gradient(
        prefix_model, records, monitor_indices, dynamic_cfg
    )
    monitor_norm = torch.linalg.vector_norm(monitor_gradient).clamp_min(1e-12)
    expert_names = ["gain", "js", "competitor", "resolved"]
    rewards = []
    gradient_norms = []
    for name in expert_names:
        gradient = _expert_gradient(
            prefix_model, records, meta_train_indices, name, dynamic_cfg
        )
        norm = torch.linalg.vector_norm(gradient)
        reward = (
            float(torch.dot(monitor_gradient, gradient) / (monitor_norm * norm).clamp_min(1e-12))
            if float(norm) > 0.0
            else -1.0
        )
        rewards.append(reward)
        gradient_norms.append(float(norm))
        del gradient
    del monitor_gradient
    _empty_cuda_cache(torch)

    old = np.asarray(dynamic_cfg["four_signal_weights"], dtype=np.float64)
    learning_rate = float(dynamic_cfg.get("meta_weight_learning_rate", 2.0))
    ema = float(dynamic_cfg.get("meta_weight_ema", 0.2))
    max_change = float(dynamic_cfg.get("meta_max_weight_change", 0.10))
    locator_method = str(dynamic_cfg.get("locator_method", "")).strip().lower()
    if locator_method == "skill_effect_additive":
        logits = np.log(np.maximum(old, 1e-6)) + learning_rate * np.asarray(rewards)
        proposal = np.exp(logits - logits.max())
        proposal /= proposal.sum()
        blended = (1.0 - ema) * old + ema * proposal
        updated = _cap_weight_change(blended, old, max_change)
    else:
        effect_old = old[:3] / max(old[:3].sum(), 1e-12)
        logits = np.log(np.maximum(effect_old, 1e-6)) + learning_rate * np.asarray(rewards[:3])
        effect_proposal = np.exp(logits - logits.max())
        effect_proposal /= effect_proposal.sum()
        effect_blended = (1.0 - ema) * effect_old + ema * effect_proposal
        effect_updated = _cap_weight_change(effect_blended, effect_old, max_change)
        base_reward = float(np.dot(effect_old, np.asarray(rewards[:3])))
        d_proposal = float(np.clip(old[3] + learning_rate * (rewards[3] - base_reward), 0.0, 1.0))
        d_updated = float(np.clip((1.0 - ema) * old[3] + ema * d_proposal, old[3] - max_change, old[3] + max_change))
        updated = np.concatenate([effect_updated, [d_updated]])
    dynamic_cfg["four_signal_weights"] = updated.tolist()
    return {
        "round": round_index,
        "meta_train_trajectory_ids": [records[index]["example"].task_id for index in meta_train_indices],
        "monitor_trajectory_ids": [records[index]["example"].task_id for index in monitor_indices],
        "expert_names": expert_names,
        "gradient_cosine_rewards": rewards,
        "expert_gradient_norms": gradient_norms,
        "old_weights": old.tolist(),
        "new_weights": updated.tolist(),
        "method": "first_order_monitor_gradient_alignment",
    }


def _train_group(prefix_model, records, group_ids, optimizer, dynamic_cfg):
    preservation_weight = float(dynamic_cfg.get("preservation_loss_weight", 1.0))
    core_total = sum(
        float(records[index].get("effective_skill_weight", len(records[index]["selected"]))) + 1.0
        for index in group_ids
    )
    preserve_total = sum(len(records[index]["preserve"]) for index in group_ids)
    optimizer.zero_grad(set_to_none=True)
    detached_total = 0.0
    for index in group_ids:
        record = records[index]
        skill_kl, eos_ce, preservation = _record_losses(
            prefix_model, record, dynamic_cfg=dynamic_cfg, with_grad=True
        )
        selected_count = float(record.get("effective_skill_weight", len(record["selected"])))
        core_count = selected_count + 1
        core = (skill_kl * selected_count + eos_ce) / core_count
        (core * (core_count / core_total)).backward()
        detached_total += float(core.detach().cpu()) * (core_count / core_total)
        preserve_count = len(record["preserve"])
        if preserve_count:
            scaled_preserve = preservation_weight * preservation * (
                preserve_count / max(preserve_total, 1)
            )
            scaled_preserve.backward()
            detached_total += float(scaled_preserve.detach().cpu())
        del skill_kl, eos_ce, preservation, core
        _empty_cuda_cache(prefix_model.torch)
    optimizer.step()
    return detached_total


def _stage_order(torch, count, *, seed, required):
    order = []
    cycle = 0
    while len(order) < required:
        order.extend(_training_order(torch, count, seed + cycle))
        cycle += 1
    return order[:required]


def _train_stage(
    prefix_model,
    records,
    *,
    stage,
    settings,
    dynamic_cfg,
    seed,
    monitor_indices,
    train_indices=None,
):
    torch = prefix_model.torch
    accumulation = 2
    max_steps = int(dynamic_cfg.get("max_steps_per_stage", 32))
    min_steps = int(dynamic_cfg.get("min_steps_per_stage", 8))
    interval = int(dynamic_cfg.get("monitor_interval_steps", 4))
    patience_limit = int(dynamic_cfg.get("monitor_patience", 3))
    min_improvement = float(dynamic_cfg.get("min_relative_monitor_improvement", 0.002))
    if min_steps < 1 or max_steps < min_steps or interval < 1 or patience_limit < 1:
        raise ValueError("Invalid Dynamic Combined stage stopping configuration")
    optimizer = torch.optim.AdamW(
        prefix_model.trainable_parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    required_trajectories = max_steps * accumulation
    allowed = list(range(len(records))) if train_indices is None else list(train_indices)
    if not allowed:
        raise ValueError("Dynamic training requires at least one non-monitor trajectory")
    relative_order = _stage_order(
        torch, len(allowed), seed=seed + stage * 1000, required=required_trajectories
    )
    order = [allowed[index] for index in relative_order]
    initial_monitor = _monitor(prefix_model, records, monitor_indices, dynamic_cfg)
    best_monitor = float(initial_monitor["loss"])
    best_state = copy.deepcopy(prefix_model.state_dict())
    history = [{"step": 0, **initial_monitor, "is_best": True}]
    stagnant = 0
    stopped_by = "max-steps"
    progress = tqdm(total=max_steps, desc=f"Dynamic Skill stage {stage}", unit="step")

    for step in range(1, max_steps + 1):
        start = (step - 1) * accumulation
        group_ids = order[start : start + accumulation]
        train_loss = _train_group(prefix_model, records, group_ids, optimizer, dynamic_cfg)
        progress.update(1)
        progress.set_postfix(train=f"{train_loss:.4f}", best=f"{best_monitor:.4f}")
        should_monitor = step % interval == 0 or step == max_steps
        if not should_monitor:
            continue
        current = _monitor(prefix_model, records, monitor_indices, dynamic_cfg)
        relative = (best_monitor - float(current["loss"])) / max(abs(best_monitor), 1e-12)
        improved = relative >= min_improvement
        if improved:
            best_monitor = float(current["loss"])
            best_state = copy.deepcopy(prefix_model.state_dict())
            stagnant = 0
        else:
            stagnant += 1
        history.append(
            {
                "step": step,
                **current,
                "relative_improvement_vs_best": relative,
                "is_best": improved,
                "stagnant_checks": stagnant,
            }
        )
        if step >= min_steps and stagnant >= patience_limit:
            stopped_by = "monitor-stagnant"
            break
    progress.close()
    prefix_model.load_state_dict(best_state)
    return {
        "stage": stage,
        "actual_optimizer_steps": int(history[-1]["step"]),
        "best_monitor_loss": best_monitor,
        "stopped_by": stopped_by,
        "monitor_trajectory_ids": [records[index]["example"].task_id for index in monitor_indices],
        "history": history,
    }


def main() -> None:
    args = parse_args()
    raw = load_config(args.config, overrides=args.cfg_options)
    flat = flatten_config(raw) if is_structured(raw) else dict(raw)
    soft_cfg = dict(raw.get("soft_prefix", {}))
    dynamic_cfg = dict(raw.get("dynamic_combined", {}))
    if args.model_name:
        soft_cfg["model_name"] = args.model_name
    elif os.environ.get("SPREADSHEETBENCH_MODEL"):
        soft_cfg["model_name"] = os.environ["SPREADSHEETBENCH_MODEL"]

    out_root = Path(args.out_root).resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    flat["out_root"] = str(out_root)
    seed = int(flat.get("seed", 1))
    _set_seed(seed)
    settings = SoftPrefixSettings.from_dict(soft_cfg)
    init_text = _load_init_text(settings.init_text_path or str(flat.get("skill_init", "")))
    if not init_text.strip():
        raise ValueError("Dynamic Combined requires the non-empty hard Skill text")
    prefix_model = _build_prefix_model("spreadsheetbench", settings, init_text)
    rows = _load_rows(settings.trajectory_examples_path)
    records = _prepare_records(rows, prefix_model, settings, init_text, dynamic_cfg)
    monitor_count = min(int(dynamic_cfg.get("monitor_trajectories", 12)), len(records))
    monitor_indices = _training_order(prefix_model.torch, len(records), seed + 7919)[:monitor_count]
    holdout_monitor = bool(dynamic_cfg.get("holdout_monitor_from_training", False))
    monitor_set = set(monitor_indices)
    train_indices = (
        [index for index in range(len(records)) if index not in monitor_set]
        if holdout_monitor
        else list(range(len(records)))
    )

    locator_method = str(dynamic_cfg.get("locator_method", "combined")).strip().lower()
    loss_weighting = str(dynamic_cfg.get("loss_weighting", "equal")).strip().lower()
    if locator_method.startswith("skill_effect_"):
        experiment_method = f"dynamic_{locator_method}_full_vocab_skill_kl_v1"
    elif locator_method == "gain_competitor":
        experiment_method = "dynamic_gain_competitor_full_vocab_skill_kl_v1"
    elif locator_method == "additive_skill":
        experiment_method = (
            "dynamic_additive_skill_locator_weighted_full_vocab_skill_kl_v1"
            if loss_weighting == "locator_score"
            else "dynamic_additive_skill_full_vocab_skill_kl_no_preservation_guard_v1"
        )
    else:
        experiment_method = "dynamic_combined_full_vocab_skill_kl_v1"
    config_record = {
        "method": experiment_method,
        "runtime": _redact(flat),
        "soft_prefix": _redact(soft_cfg),
        "dynamic_combined": dynamic_cfg,
        "fairness_contract": {
            "backbone_frozen": True,
            "prefix_length": settings.prefix_length,
            "trainable_parameters": int(prefix_model.prefix_embeddings.numel()),
            "training_support": len(records),
            "gradient_training_support": len(train_indices),
            "monitor_support": len(monitor_indices),
            "monitor_held_out_from_gradient_training": holdout_monitor,
            "validation_used_for_training_or_stopping": False,
            "test_accessed": False,
            "locator_context": "successful GPT-5.5 trajectories with gold teacher forcing",
            "locator_teacher": "Qwen plus full hard Skill",
            "locator_student": "Qwen plus current soft prefix",
            "locator_method": locator_method,
            "locator_additive_alpha": (
                float(dynamic_cfg.get("additive_alpha", 0.5))
                if locator_method == "additive_skill"
                else None
            ),
            "locator_gain_weight": (
                float(dynamic_cfg.get("gain_weight", 0.75))
                if locator_method == "gain_competitor"
                else None
            ),
            "locator_four_signal_weights": (
                [
                    float(value)
                    for value in dynamic_cfg.get(
                        "four_signal_weights", [0.45, 0.45, 0.10, 0.0]
                    )
                ]
                if locator_method.startswith("skill_effect_")
                else None
            ),
            "loss_weighting": loss_weighting,
            "effective_core_ratio": float(
                dynamic_cfg.get("effective_core_ratio", dynamic_cfg.get("core_ratio", 0.10))
            ),
            "core_objective": "full-vocabulary forward KL",
            "preservation": "fixed coverage-control Top-64 plus residual-bucket KL",
            "preservation_can_stop_or_rollback": bool(
                dynamic_cfg.get("preservation_guard_enabled", True)
            ),
        },
        "input_fingerprints": {
            "trajectory_manifest_sha256": _sha256(settings.trajectory_examples_path),
            "skill_sha256": _sha256(settings.init_text_path),
        },
    }
    (out_root / "config.json").write_text(
        json.dumps(config_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 76, flush=True)
    if locator_method == "skill_effect_multiplicative":
        title = "Dynamic G+JS+C with resolved-entropy modulation"
    elif locator_method == "skill_effect_additive":
        title = "Dynamic additive G+JS+C+R locator"
    elif locator_method == "gain_competitor":
        title = "Dynamic G+C locator / equal-weight KL"
    elif locator_method == "additive_skill":
        title = (
            "Dynamic Additive Skill / locator-weighted KL"
            if loss_weighting == "locator_score"
            else "Dynamic Additive Skill v1 / no preservation guard"
        )
    else:
        title = "Dynamic Combined v1"
    print(f"{title} / Full-Vocabulary Skill-KL / SpreadsheetBench", flush=True)
    print(f"model={settings.model_name}", flush=True)
    print(
        f"records={len(records)} gradient_train={len(train_indices)} "
        f"monitor={monitor_count} prefix={settings.prefix_length}",
        flush=True,
    )
    print("Val40 is evaluation-only; Test280 access is disabled", flush=True)
    print("=" * 76, flush=True)

    started = time.time()
    locator_history = []
    stage_history = []
    weight_adaptation_history = []
    previous_sets = None
    previous_mass = None
    initial_mass = None
    initial_preservation = None
    stagnant_rounds = 0
    relocations = 0
    best_global_mass = math.inf
    best_global_state = copy.deepcopy(prefix_model.state_dict())
    best_global_round = 0
    stop_reason = "unknown"
    stop_mass_metric = str(dynamic_cfg.get("stop_mass_metric", "locator_score")).strip().lower()
    if stop_mass_metric not in {"locator_score", "full_vocab_kl"}:
        raise ValueError(f"Unsupported stop_mass_metric: {stop_mass_metric!r}")
    preservation_guard_enabled = bool(dynamic_cfg.get("preservation_guard_enabled", True))

    while True:
        if (
            bool(dynamic_cfg.get("adaptive_locator_weights", False))
            and relocations > 0
            and locator_method.startswith("skill_effect_")
        ):
            adaptation = _adapt_locator_weights(
                prefix_model,
                records,
                train_indices=train_indices,
                monitor_indices=monitor_indices,
                dynamic_cfg=dynamic_cfg,
                seed=seed,
                round_index=relocations,
            )
            weight_adaptation_history.append(adaptation)
            print(
                f"[meta weights {relocations}] old={adaptation['old_weights']} "
                f"new={adaptation['new_weights']} "
                f"cos={adaptation['gradient_cosine_rewards']}",
                flush=True,
            )
        locator, current_sets = _dynamic_localize(
            prefix_model,
            records,
            round_index=relocations,
            out_root=out_root,
            dynamic_cfg=dynamic_cfg,
            previous=previous_sets,
        )
        mass = float(
            locator["global_full_vocab_kl_mass"]
            if stop_mass_metric == "full_vocab_kl"
            else locator["global_dynamic_residual_mass"]
        )
        if initial_mass is None:
            initial_mass = mass
            initial_preservation = float(locator["mean_preservation_kl"])
        preservation_limit = float(initial_preservation) * (
            1.0 + float(dynamic_cfg.get("max_preservation_degradation", 0.10))
        )
        preservation_bad = preservation_guard_enabled and (
            relocations > 0
            and float(locator["mean_preservation_kl"]) > preservation_limit
        )
        if not preservation_bad and mass < best_global_mass:
            best_global_mass = mass
            best_global_state = copy.deepcopy(prefix_model.state_dict())
            best_global_round = relocations
        decision = dynamic_stop_decision(
            current_mass=mass,
            initial_mass=float(initial_mass),
            previous_mass=previous_mass,
            mean_eligible_kl=float(locator["mean_eligible_full_vocab_kl"]),
            completed_relocations=relocations,
            max_relocations=int(dynamic_cfg.get("max_relocations", 4)),
            stagnant_rounds=stagnant_rounds,
            residual_mass_ratio_threshold=float(
                dynamic_cfg.get("residual_mass_ratio_threshold", 0.10)
            ),
            mean_eligible_kl_threshold=float(
                dynamic_cfg.get("mean_eligible_kl_threshold", 0.02)
            ),
            min_relative_mass_improvement=float(
                dynamic_cfg.get("min_relative_mass_improvement", 0.002)
            ),
            global_patience=int(dynamic_cfg.get("global_patience", 2)),
        )
        stagnant_rounds = decision.stagnant_rounds
        locator.update(
            {
                "residual_mass_ratio": decision.residual_mass_ratio,
                "relative_mass_improvement": decision.relative_mass_improvement,
                "global_stagnant_rounds": stagnant_rounds,
                "preservation_limit": preservation_limit,
                "preservation_guard_enabled": preservation_guard_enabled,
                "stop_mass_metric": stop_mass_metric,
                "preservation_guard_failed": preservation_bad,
                "stop_decision": "preservation-degraded" if preservation_bad else decision.reason,
            }
        )
        locator_history.append(locator)
        print(
            f"[locator {relocations}] {stop_mass_metric}={mass:.6f} "
            f"ratio={decision.residual_mass_ratio:.4f} "
            f"selected={locator['selected_tokens']} capture={locator['selected_mass_capture']:.2%} "
            f"jaccard={locator['mean_previous_core_jaccard']} preserve={locator['mean_preservation_kl']:.6f}",
            flush=True,
        )
        if preservation_bad:
            stop_reason = "preservation-degraded"
            break
        if decision.stop:
            stop_reason = decision.reason
            break

        stage_result = _train_stage(
            prefix_model,
            records,
            stage=relocations,
            settings=settings,
            dynamic_cfg=dynamic_cfg,
            seed=seed,
            monitor_indices=monitor_indices,
            train_indices=train_indices,
        )
        stage_history.append(stage_result)
        torch = prefix_model.torch
        torch.save(prefix_model.state_dict(), out_root / f"stage_{relocations:02d}_best_prefix.pt")
        previous_sets = current_sets
        previous_mass = mass
        relocations += 1

    prefix_model.load_state_dict(best_global_state)
    best_path = out_root / "best_prefix.pt"
    latest_path = out_root / "latest_prefix.pt"
    prefix_model.torch.save(prefix_model.state_dict(), best_path)
    prefix_model.torch.save(prefix_model.state_dict(), latest_path)
    if locator_method == "skill_effect_multiplicative":
        summary_method = "Dynamic-Skill-Effect-Multiplicative-Full-Vocabulary-Skill-KL"
    elif locator_method == "skill_effect_additive":
        summary_method = "Dynamic-Skill-Effect-Additive-Full-Vocabulary-Skill-KL"
    elif locator_method == "gain_competitor":
        summary_method = "Dynamic-Gain-Competitor-Equal-Weight-Full-Vocabulary-Skill-KL"
    elif locator_method == "additive_skill":
        summary_method = (
            "Dynamic-Additive-Skill-Locator-Weighted-Full-Vocabulary-Skill-KL"
            if loss_weighting == "locator_score"
            else "Dynamic-Additive-Skill-v1-Full-Vocabulary-Skill-KL"
        )
    else:
        summary_method = "Dynamic-Combined-v1-Full-Vocabulary-Skill-KL"
    summary = {
        "method": summary_method,
        "trajectories": len(records),
        "gradient_training_trajectories": len(train_indices),
        "monitor_trajectories": len(monitor_indices),
        "monitor_held_out_from_gradient_training": holdout_monitor,
        "prefix_length": settings.prefix_length,
        "relocations_completed": relocations,
        "stop_reason": stop_reason,
        "best_global_round": best_global_round,
        "best_global_residual_mass": best_global_mass,
        "initial_global_residual_mass": initial_mass,
        "final_residual_mass_ratio": best_global_mass / max(float(initial_mass), 1e-12),
        "stop_mass_metric": stop_mass_metric,
        "preservation_guard_enabled": preservation_guard_enabled,
        "total_optimizer_steps": sum(item["actual_optimizer_steps"] for item in stage_history),
        "locator_history": locator_history,
        "stage_history": stage_history,
        "weight_adaptation_history": weight_adaptation_history,
        "final_locator_weights": dynamic_cfg.get("four_signal_weights"),
        "checkpoint_path": str(best_path),
        "checkpoint_sha256": _sha256(best_path),
        "test_split_accessed": False,
        "wall_time_s": round(time.time() - started, 1),
    }

    if bool(dynamic_cfg.get("eval_after_train", True)) and not args.no_val:
        dataloader = _build_dataloader("spreadsheetbench", flat, seed)
        dataloader.setup(flat)
        val_items = _items_for_eval(
            dataloader, "valid_seen", int(flat.get("sel_env_num", 40)), seed
        )
        val_hard, val_soft, _ = _evaluate_prefix(
            "spreadsheetbench",
            prefix_model,
            val_items,
            cfg=flat,
            settings=settings,
            out_dir=str(out_root / "eval" / "final" / "valid_seen"),
            desc="Dynamic Skill v1 Val40",
        )
        summary["valid_seen_hard"] = val_hard
        summary["valid_seen_soft"] = val_soft
        print(
            f"Validation: {round(val_hard * len(val_items))}/{len(val_items)} ({val_hard:.2%})",
            flush=True,
        )
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"stop_reason={stop_reason}", flush=True)
    print(f"checkpoint={best_path}", flush=True)
    print(f"sha256={summary['checkpoint_sha256']}", flush=True)


if __name__ == "__main__":
    main()
