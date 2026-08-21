#!/usr/bin/env python3
"""Run the direct calibration stage of Future-Impact Locator v2.

FIL-v2 keeps candidate losses on successful gold trajectories but measures
their local utility against a held-out, learner-state outer objective.  This
driver implements the pre-registered Source39/Outer10/Cal12 gate.  It does not
touch Val40 or Test280.  A full prefix run is permitted only after the gate has
shown that the proxy predicts real AdamW group updates and free generation.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
import json
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

from scripts.evaluate_spreadsheetbench_hard_skill_baseline import HardSkillGenerator  # noqa: E402
from scripts.train_spreadsheetbench_combined_full_vocab_kl import (  # noqa: E402
    _empty_cuda_cache,
    _load_rows,
    _redact,
    _sha256,
    _training_order,
)
from scripts.train_spreadsheetbench_dynamic_combined import (  # noqa: E402
    _prefix_gradient_vector,
    _prepare_records,
    _record_losses,
    _train_group,
)
from skillopt.config import flatten_config, is_structured, load_config  # noqa: E402
from skillopt.softprefix.distillation_losses import (  # noqa: E402
    chunked_full_vocab_forward_kl,
    chunked_full_vocab_forward_kl_vector,
    topk_residual_forward_kl,
)
from skillopt.softprefix.future_impact import (  # noqa: E402
    adam_diagonal_direction,
    central_difference_scores,
    chunked_eager_attention,
    deterministic_task_partition,
    jaccard,
    per_token_forward_kl,
    select_fraction,
    spearman_correlation,
)
from skillopt.softprefix.official_distillation import (  # noqa: E402
    encode_on_policy_response,
    generation_mode,
    target_logits,
)
from skillopt.softprefix.trainer import (  # noqa: E402
    SoftPrefixSettings,
    _build_dataloader,
    _build_prefix_model,
    _evaluate_prefix,
    _load_init_text,
    _set_seed,
    evaluate_spreadsheet_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--model_name", default="")
    parser.add_argument("--cfg-options", nargs="+", default=[])
    parser.add_argument(
        "--phase",
        choices=("outer", "score", "calibrate", "all"),
        default="all",
        help="Resume-safe experiment phase.",
    )
    parser.add_argument(
        "--skip-calibration-generation",
        action="store_true",
        help="Debug only: measure cached outer loss but do not run Cal12 free generation.",
    )
    return parser.parse_args()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _load_completed_results(path: Path, expected_ids: list[str]) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    rows = _read_jsonl(path)
    if [str(row.get("id", "")) for row in rows] != expected_ids:
        raise ValueError(f"Cached FIL results have unexpected task order: {path}")
    return rows


def _prepare_experiment(raw: dict[str, Any], args: argparse.Namespace):
    flat = flatten_config(raw) if is_structured(raw) else dict(raw)
    soft_cfg = dict(raw.get("soft_prefix", {}))
    fil_cfg = dict(raw.get("future_impact", {}))
    if args.model_name:
        soft_cfg["model_name"] = args.model_name
    elif os.environ.get("SPREADSHEETBENCH_MODEL"):
        soft_cfg["model_name"] = os.environ["SPREADSHEETBENCH_MODEL"]
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    flat["out_root"] = str(out_root)
    seed = int(flat.get("seed", 1))
    _set_seed(seed)
    settings = SoftPrefixSettings.from_dict(soft_cfg)
    init_text = _load_init_text(settings.init_text_path or str(flat.get("skill_init", "")))
    if not init_text.strip():
        raise ValueError("FIL-v2 requires the non-empty hard Skill")
    prefix_model = _build_prefix_model("spreadsheetbench", settings, init_text)
    rows = _load_rows(settings.trajectory_examples_path)
    records = _prepare_records(rows, prefix_model, settings, init_text, fil_cfg)
    if len(records) != 61:
        raise ValueError(f"FIL-v2 registered pilot requires 61 successful trajectories, got {len(records)}")
    for row, record in zip(rows, records, strict=True):
        selected = sorted({int(index) for index in row.get("selected_indices", [])})
        if not selected:
            raise ValueError(f"FIL trajectory {record['example'].task_id} lacks Combined10 positions")
        record["combined_selected"] = selected
        record["selected"] = list(selected)
        record["selected_weights"] = np.ones(len(selected), dtype=np.float32)
        record["effective_skill_weight"] = float(len(selected))
    partition = deterministic_task_partition(
        [record["example"].task_id for record in records],
        seed=int(fil_cfg.get("partition_seed", seed + 24017)),
        source_count=int(fil_cfg.get("source_trajectories", 39)),
        outer_count=int(fil_cfg.get("outer_trajectories", 10)),
        calibration_count=int(fil_cfg.get("calibration_trajectories", 12)),
    )
    split_record = {
        "seed": int(fil_cfg.get("partition_seed", seed + 24017)),
        "source": list(partition.source),
        "outer": list(partition.outer),
        "calibration": list(partition.calibration),
        "val40_accessed": False,
        "test280_accessed": False,
    }
    split_path = out_root / "task_partition.json"
    if split_path.exists():
        if json.loads(split_path.read_text(encoding="utf-8")) != split_record:
            raise ValueError("Existing FIL task partition differs from the requested configuration")
    else:
        _json(split_path, split_record)
    by_id = {record["example"].task_id: record for record in records}
    dataloader = _build_dataloader("spreadsheetbench", flat, seed)
    dataloader.setup(flat)
    train_items = {str(item["id"]): item for item in dataloader.train_items}
    missing = set(by_id) - set(train_items)
    if missing:
        raise ValueError(f"FIL trajectory IDs absent from SpreadsheetBench train split: {sorted(missing)}")
    return flat, soft_cfg, fil_cfg, out_root, seed, settings, init_text, prefix_model, records, by_id, train_items, partition


def _run_outer_rollouts(
    prefix_model,
    *,
    ids: list[str],
    train_items: dict[str, dict[str, Any]],
    flat: dict[str, Any],
    settings: SoftPrefixSettings,
    init_text: str,
    out_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items = [train_items[task_id] for task_id in ids]
    soft_dir = out_root / "outer_rollouts" / "soft"
    hard_dir = out_root / "outer_rollouts" / "hard"
    soft_results = _load_completed_results(soft_dir / "results.jsonl", ids)
    if soft_results is None:
        print(f"[FIL outer] generating current soft-prefix responses for {len(items)} tasks", flush=True)
        _, _, soft_results = _evaluate_prefix(
            "spreadsheetbench",
            prefix_model,
            items,
            cfg=flat,
            settings=settings,
            out_dir=str(soft_dir),
            desc="FIL Outer10 Soft",
        )
    else:
        print(f"[FIL outer] reusing {soft_dir / 'results.jsonl'}", flush=True)

    hard_results = _load_completed_results(hard_dir / "results.jsonl", ids)
    if hard_results is None:
        print(f"[FIL outer] generating full-Hard-Skill responses for {len(items)} tasks", flush=True)
        generator = HardSkillGenerator(
            prefix_model,
            skill_text=init_text,
            batch_size=int(flat.get("generation_batch_size", 8) or 8),
        )
        with generation_mode(prefix_model):
            _, _, hard_results = evaluate_spreadsheet_prefix(
                prefix_model,
                items,
                out_dir=str(hard_dir),
                data_root=str(flat.get("data_root", "")),
                max_prompt_tokens=settings.max_prompt_tokens,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.generation_temperature,
                exec_timeout=int(flat.get("exec_timeout", 600) or 600),
                desc="FIL Outer10 Hard",
                generator=generator,
                injection_position=settings.injection_position,
                repair_turns=1,
                generation_batch_size=int(flat.get("generation_batch_size", 8) or 8),
            )
    else:
        print(f"[FIL outer] reusing {hard_dir / 'results.jsonl'}", flush=True)

    soft_by_id = {str(row["id"]): row for row in soft_results}
    hard_by_id = {str(row["id"]): row for row in hard_results}
    comparison = []
    for task_id in ids:
        soft = soft_by_id[task_id]
        hard = hard_by_id[task_id]
        advantage = max(0.0, float(hard.get("soft", 0.0)) - float(soft.get("soft", 0.0)))
        comparison.append(
            {
                "id": task_id,
                "soft_hard": int(soft.get("hard", 0)),
                "soft_score": float(soft.get("soft", 0.0)),
                "hard_skill_hard": int(hard.get("hard", 0)),
                "hard_skill_score": float(hard.get("soft", 0.0)),
                "teacher_advantage": advantage,
            }
        )
    _json(
        out_root / "outer_rollouts" / "summary.json",
        {
            "tasks": len(ids),
            "soft_successes": sum(row["soft_hard"] for row in comparison),
            "hard_skill_successes": sum(row["hard_skill_hard"] for row in comparison),
            "positive_teacher_advantage_tasks": sum(row["teacher_advantage"] > 0 for row in comparison),
            "teacher_advantage_mass": sum(row["teacher_advantage"] for row in comparison),
            "comparison": comparison,
        },
    )
    return soft_results, hard_results


def _encode_outer_examples(
    prefix_model,
    by_id: dict[str, dict[str, Any]],
    soft_results: list[dict[str, Any]],
    hard_results: list[dict[str, Any]],
    *,
    settings: SoftPrefixSettings,
) -> list[dict[str, Any]]:
    hard_by_id = {str(row["id"]): row for row in hard_results}
    prepared = []
    for soft in soft_results:
        task_id = str(soft["id"])
        hard = hard_by_id[task_id]
        advantage = max(0.0, float(hard.get("soft", 0.0)) - float(soft.get("soft", 0.0)))
        generated = encode_on_policy_response(
            by_id[task_id]["example"],
            tokenizer=prefix_model.tokenizer,
            response=str(soft.get("response", "")),
            max_target_tokens=settings.max_target_tokens,
        )
        prepared.append({"id": task_id, "example": generated, "weight": advantage})
    return prepared


def _outer_objective(prefix_model, prepared, *, chunk_size: int, with_grad: bool):
    torch = prefix_model.torch
    parameters = list(prefix_model.trainable_parameters())
    if with_grad:
        for parameter in parameters:
            parameter.grad = None
    total_weight = sum(float(row["weight"]) for row in prepared)
    if total_weight <= 0.0:
        raise RuntimeError("FIL teacher-competence gate has zero positive advantage mass")
    objective = 0.0
    details = []
    for row in tqdm(prepared, desc="FIL on-policy outer", unit="task", leave=False):
        weight = float(row["weight"])
        if weight <= 0.0:
            details.append({"id": row["id"], "weight": weight, "kl": None})
            continue
        example = row["example"]
        indices = list(range(len(example.target_ids)))
        teacher_logits = target_logits(
            prefix_model, example, indices, use_prefix=False, with_grad=False
        )
        student_logits = target_logits(
            prefix_model, example, indices, use_prefix=True, with_grad=with_grad
        )
        loss = chunked_full_vocab_forward_kl(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            chunk_size=chunk_size,
        )
        scaled = weight / total_weight
        if with_grad:
            (loss * scaled).backward()
        value = float(loss.detach().cpu())
        objective += scaled * value
        details.append(
            {"id": row["id"], "weight": weight, "tokens": len(indices), "kl": value}
        )
        del teacher_logits, student_logits, loss
        _empty_cuda_cache(torch)
    gradient = _prefix_gradient_vector(prefix_model) if with_grad else None
    if with_grad:
        for parameter in parameters:
            parameter.grad = None
    return objective, gradient, details


def _preservation_objective(prefix_model, records, *, with_grad: bool):
    torch = prefix_model.torch
    parameters = list(prefix_model.trainable_parameters())
    if with_grad:
        for parameter in parameters:
            parameter.grad = None
    total = sum(len(record["preserve"]) for record in records)
    weighted = 0.0
    for record in tqdm(records, desc="FIL preservation", unit="traj", leave=False):
        count = len(record["preserve"])
        if not count:
            continue
        logits = target_logits(
            prefix_model,
            record["example"],
            record["preserve"],
            use_prefix=True,
            with_grad=with_grad,
        )
        loss = topk_residual_forward_kl(
            student_logits=logits,
            reference_topk_ids=record["clean_topk_ids"],
            reference_topk_logp=record["clean_topk_logp"],
            reference_residual_log_mass=record["clean_residual"],
        )
        scale = count / max(total, 1)
        if with_grad:
            (loss * scale).backward()
        weighted += float(loss.detach().cpu()) * scale
        del logits, loss
        _empty_cuda_cache(torch)
    gradient = _prefix_gradient_vector(prefix_model) if with_grad else None
    if with_grad:
        for parameter in parameters:
            parameter.grad = None
    return weighted, gradient


def _reference_second_moment(prefix_model, records, *, settings, fil_cfg, seed):
    torch = prefix_model.torch
    requested = min(int(fil_cfg.get("moment_probe_trajectories", 8)), len(records))
    order = _training_order(torch, len(records), seed + 43103)[:requested]
    moment = None
    norms = []
    dynamic_cfg = {
        "loss_weighting": "equal",
        "kl_chunk_size": int(fil_cfg.get("kl_chunk_size", 8)),
        "preservation_loss_weight": float(fil_cfg.get("preservation_loss_weight", 1.0)),
    }
    for index in tqdm(order, desc="FIL Adam moment probe", unit="traj"):
        record = records[index]
        record["selected"] = list(record["combined_selected"])
        record["selected_weights"] = np.ones(len(record["selected"]), dtype=np.float32)
        for parameter in prefix_model.trainable_parameters():
            parameter.grad = None
        skill, eos, preserve = _record_losses(
            prefix_model, record, dynamic_cfg=dynamic_cfg, with_grad=True
        )
        core = (skill * len(record["selected"]) + eos) / (len(record["selected"]) + 1)
        loss = core + float(dynamic_cfg["preservation_loss_weight"]) * preserve
        loss.backward()
        gradient = _prefix_gradient_vector(prefix_model)
        moment = gradient.square() if moment is None else moment + gradient.square()
        norms.append(float(torch.linalg.vector_norm(gradient).cpu()))
        del skill, eos, preserve, core, loss, gradient
        _empty_cuda_cache(torch)
    for parameter in prefix_model.trainable_parameters():
        parameter.grad = None
    if moment is None:
        raise RuntimeError("FIL moment probe produced no gradients")
    return moment / requested, {"trajectory_ids": [records[index]["example"].task_id for index in order], "gradient_norms": norms}


def _set_flat_prefix(prefix_model, flat_value) -> None:
    torch = prefix_model.torch
    reshaped = flat_value.reshape_as(prefix_model.prefix_embeddings)
    with torch.no_grad():
        prefix_model.prefix_embeddings.copy_(
            reshaped.to(device=prefix_model.device, dtype=prefix_model.prefix_embeddings.dtype)
        )


def _score_source_records_finite_difference(
    prefix_model,
    records,
    *,
    direction,
    fil_cfg,
    seed: int,
    out_root: Path,
):
    torch = prefix_model.torch
    base = prefix_model.prefix_embeddings.detach().float().reshape(-1).clone()
    main_epsilon = float(fil_cfg.get("finite_difference_epsilon", 0.125))
    stability_factors = [float(value) for value in fil_cfg.get("stability_epsilon_factors", [0.5, 1.0, 2.0])]
    stability_count = min(int(fil_cfg.get("stability_trajectories", 6)), len(records))
    stability_ids = {
        records[index]["example"].task_id
        for index in _training_order(torch, len(records), seed + 61001)[:stability_count]
    }
    ratio = float(fil_cfg.get("core_ratio", 0.10))
    positive_only = bool(fil_cfg.get("positive_edge_only", True))
    chunk_size = int(fil_cfg.get("kl_chunk_size", 8))
    array_dir = out_root / "locator" / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    stability = []
    all_quantized = []
    try:
        for record in tqdm(records, desc="FIL directional scoring", unit="traj"):
            example = record["example"]
            count = len(example.target_ids) - 1
            indices = list(range(count))
            teacher_logits = target_logits(
                prefix_model, example, indices, use_prefix=False, with_grad=False
            )
            factors = stability_factors if example.task_id in stability_ids else [1.0]
            factor_scores: dict[str, np.ndarray] = {}
            quantized = {}
            for factor in factors:
                epsilon = main_epsilon * factor
                requested_plus = base + epsilon * direction
                _set_flat_prefix(prefix_model, requested_plus)
                actual_plus = prefix_model.prefix_embeddings.detach().float().reshape(-1).clone()
                plus_unchanged = float(actual_plus.eq(base).float().mean().cpu())
                plus_logits = target_logits(
                    prefix_model, example, indices, use_prefix=True, with_grad=False
                )
                plus = per_token_forward_kl(
                    teacher_logits=teacher_logits,
                    student_logits=plus_logits,
                    chunk_size=chunk_size,
                )
                del plus_logits
                _empty_cuda_cache(torch)

                requested_minus = base - epsilon * direction
                _set_flat_prefix(prefix_model, requested_minus)
                actual_minus = prefix_model.prefix_embeddings.detach().float().reshape(-1).clone()
                minus_unchanged = float(actual_minus.eq(base).float().mean().cpu())
                minus_logits = target_logits(
                    prefix_model, example, indices, use_prefix=True, with_grad=False
                )
                minus = per_token_forward_kl(
                    teacher_logits=teacher_logits,
                    student_logits=minus_logits,
                    chunk_size=chunk_size,
                )
                del minus_logits
                _empty_cuda_cache(torch)
                factor_scores[f"{factor:g}"] = central_difference_scores(
                    plus, minus, epsilon=epsilon
                )
                quantized[f"{factor:g}"] = {
                    "plus_unchanged_fraction": plus_unchanged,
                    "minus_unchanged_fraction": minus_unchanged,
                    "actual_plus_l2": float(torch.linalg.vector_norm(actual_plus - base).cpu()),
                    "actual_minus_l2": float(torch.linalg.vector_norm(actual_minus - base).cpu()),
                }
                _set_flat_prefix(prefix_model, base)
            scores = factor_scores["1"]
            forbidden = record["preserve"]
            top_forced = select_fraction(
                scores, ratio=ratio, forbidden=forbidden, largest=True, positive_only=False
            )
            top_positive = select_fraction(
                scores,
                ratio=ratio,
                forbidden=forbidden,
                largest=True,
                positive_only=positive_only,
            )
            bottom = select_fraction(
                scores, ratio=ratio, forbidden=forbidden, largest=False, positive_only=False
            )
            eligible = [index for index in indices if index not in set(forbidden)]
            budget = len(top_forced)
            task_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{example.task_id}".encode()).digest()[:8], "big"
            )
            random_indices = sorted(random.Random(task_seed).sample(eligible, budget))
            combined = list(record["combined_selected"])
            if set(combined) & set(forbidden):
                raise ValueError(f"Combined/preservation overlap for {example.task_id}")
            slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in example.task_id)
            array_path = array_dir / f"{slug}.npz"
            payload = {
                "target_ids": np.asarray(example.target_ids[:count], dtype=np.int32),
                "future_impact": scores,
                "fil_top_forced": np.asarray(top_forced, dtype=np.int32),
                "fil_top_positive": np.asarray(top_positive, dtype=np.int32),
                "fil_bottom": np.asarray(bottom, dtype=np.int32),
                "combined": np.asarray(combined, dtype=np.int32),
                "random": np.asarray(random_indices, dtype=np.int32),
                "preserve": np.asarray(forbidden, dtype=np.int32),
            }
            for factor, values in factor_scores.items():
                payload[f"future_impact_eps_factor_{factor}"] = values
            np.savez_compressed(array_path, **payload)
            row = {
                "id": example.task_id,
                "tokens": count,
                "positive_score_tokens": int((scores > 0).sum()),
                "negative_score_tokens": int((scores < 0).sum()),
                "top_forced": top_forced,
                "top_positive": top_positive,
                "bottom": bottom,
                "combined": combined,
                "random": random_indices,
                "top_forced_positive_fraction": float(
                    np.mean(scores[np.asarray(top_forced, dtype=np.int64)] > 0)
                ),
                "top_combined_jaccard": jaccard(top_forced, combined),
                "quantization": quantized,
                "array_path": str(array_path),
            }
            manifest.append(row)
            all_quantized.extend(quantized.values())
            if len(factor_scores) > 1:
                base_scores = factor_scores["1"]
                base_top = top_forced
                for factor, other in factor_scores.items():
                    if factor == "1":
                        continue
                    other_top = select_fraction(
                        other, ratio=ratio, forbidden=forbidden, largest=True
                    )
                    stability.append(
                        {
                            "id": example.task_id,
                            "factor": float(factor),
                            "spearman": spearman_correlation(base_scores, other),
                            "top_jaccard": jaccard(base_top, other_top),
                        }
                    )
            del teacher_logits
            _empty_cuda_cache(torch)
    finally:
        _set_flat_prefix(prefix_model, base)
    _write_jsonl(out_root / "locator" / "manifest.jsonl", manifest)
    summary = {
        "trajectories": len(manifest),
        "tokens": sum(row["tokens"] for row in manifest),
        "selected_tokens_forced": sum(len(row["top_forced"]) for row in manifest),
        "selected_tokens_positive": sum(len(row["top_positive"]) for row in manifest),
        "mean_top_forced_positive_fraction": float(
            np.mean([row["top_forced_positive_fraction"] for row in manifest])
        ),
        "mean_combined_jaccard": float(np.mean([row["top_combined_jaccard"] for row in manifest])),
        "mean_stability_spearman": float(np.nanmean([row["spearman"] for row in stability])),
        "mean_stability_top_jaccard": float(np.mean([row["top_jaccard"] for row in stability])),
        "mean_plus_unchanged_fraction": float(
            np.mean([row["plus_unchanged_fraction"] for row in all_quantized])
        ),
        "mean_minus_unchanged_fraction": float(
            np.mean([row["minus_unchanged_fraction"] for row in all_quantized])
        ),
        "stability": stability,
        "manifest_path": str(out_root / "locator" / "manifest.jsonl"),
    }
    _json(out_root / "locator" / "summary.json", summary)
    return manifest, summary


@contextlib.contextmanager
def _jvp_compatible_backend(prefix_model, *, attention_query_chunk_size: int = 256):
    """Temporarily select Qwen kernels with forward-AD support."""
    model = prefix_model.model
    was_training = bool(model.training)
    was_checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    had_input_grad_hooks = bool(getattr(model, "_require_grads_hooks", None))
    experts_implementation = getattr(model.config, "_experts_implementation", None)
    attention_implementation = getattr(model.config, "_attn_implementation", None)
    old_chunk_size = getattr(model.config, "_fil_jvp_attention_query_chunk_size", None)
    attention_registry = None
    had_eager_override = False
    old_eager_override = None
    try:
        if was_checkpointing:
            model.gradient_checkpointing_disable()
        if had_input_grad_hooks:
            model.disable_input_require_grads()
        if experts_implementation not in {None, "eager"}:
            model.set_experts_implementation("eager")
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        attention_registry = ALL_ATTENTION_FUNCTIONS
        had_eager_override = "eager" in attention_registry
        if had_eager_override:
            old_eager_override = attention_registry["eager"]
        # Keep the implementation name exactly ``eager`` so Transformers
        # constructs the same 4-D causal mask.  Only its scoped kernel is
        # replaced by the query-chunked algebraic equivalent.
        attention_registry["eager"] = chunked_eager_attention
        model.config._fil_jvp_attention_query_chunk_size = max(
            1, int(attention_query_chunk_size)
        )
        model.set_attn_implementation("eager")
        model.eval()
        yield {
            "attention": attention_implementation,
            "experts": experts_implementation,
            "jvp_attention": getattr(model.config, "_attn_implementation", None),
            "jvp_experts": getattr(model.config, "_experts_implementation", None),
        }
    finally:
        if attention_implementation is not None:
            model.set_attn_implementation(attention_implementation)
        if attention_registry is not None:
            if had_eager_override:
                attention_registry["eager"] = old_eager_override
            else:
                del attention_registry["eager"]
        if old_chunk_size is None:
            if hasattr(model.config, "_fil_jvp_attention_query_chunk_size"):
                delattr(model.config, "_fil_jvp_attention_query_chunk_size")
        else:
            model.config._fil_jvp_attention_query_chunk_size = old_chunk_size
        if experts_implementation not in {None, "eager"}:
            model.set_experts_implementation(experts_implementation)
        if was_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if had_input_grad_hooks and not bool(getattr(model, "_require_grads_hooks", None)):
            model.enable_input_require_grads()
        model.train(was_training)


def _jvp_losses_and_scores(
    prefix_model,
    *,
    example,
    indices,
    teacher_logits,
    direction,
    chunk_size: int,
):
    torch = prefix_model.torch
    base = prefix_model.prefix_embeddings.detach().clone()
    tangent = direction.reshape_as(base).to(device=base.device, dtype=base.dtype)

    def token_losses(prefix_value):
        student_logits = target_logits(
            prefix_model,
            example,
            indices,
            use_prefix=True,
            with_grad=True,
            prefix_embeddings_override=prefix_value,
        )
        return chunked_full_vocab_forward_kl_vector(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            chunk_size=chunk_size,
        )

    primal, directional = torch.func.jvp(
        token_losses,
        (base,),
        (tangent,),
        strict=True,
    )
    return (
        primal.detach().float().cpu().numpy(),
        directional.detach().float().cpu().numpy(),
    )


def _score_source_records_jvp(
    prefix_model,
    records,
    *,
    direction,
    fil_cfg,
    seed: int,
    out_root: Path,
):
    """Score every gold token with one forward-mode pass per trajectory."""
    torch = prefix_model.torch
    ratio = float(fil_cfg.get("core_ratio", 0.10))
    positive_only = bool(fil_cfg.get("positive_edge_only", True))
    chunk_size = int(fil_cfg.get("kl_chunk_size", 8))
    audit_count = min(int(fil_cfg.get("jvp_backend_audit_trajectories", 6)), len(records))
    repeat_count = min(int(fil_cfg.get("jvp_repeat_trajectories", 2)), audit_count)
    audit_order = _training_order(torch, len(records), seed + 61001)
    audit_ids = {records[index]["example"].task_id for index in audit_order[:audit_count]}
    repeat_ids = {records[index]["example"].task_id for index in audit_order[:repeat_count]}
    array_dir = out_root / "locator" / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    repeat_stability: list[dict[str, Any]] = []
    backend_audits: list[dict[str, Any]] = []
    backend_record: dict[str, Any] | None = None
    normalized_direction = direction.float().reshape(-1)
    normalized_direction = normalized_direction / torch.linalg.vector_norm(
        normalized_direction
    ).clamp_min(1e-30)

    for record in tqdm(records, desc="FIL exact directional JVP", unit="traj"):
        example = record["example"]
        count = len(example.target_ids) - 1
        indices = list(range(count))
        slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in example.task_id)
        array_path = array_dir / f"{slug}.npz"
        if array_path.exists():
            with np.load(array_path) as cached:
                required = {
                    "target_ids",
                    "future_impact",
                    "jvp_eager_primal_kl",
                    "fil_top_forced",
                    "fil_top_positive",
                    "fil_bottom",
                    "combined",
                    "random",
                    "preserve",
                }
                reusable = required.issubset(cached.files)
                reusable = reusable and (
                    example.task_id not in repeat_ids or "jvp_repeat_scores" in cached.files
                )
                reusable = reusable and (
                    example.task_id not in audit_ids or "default_backend_primal_kl" in cached.files
                )
                if reusable:
                    target_ids = cached["target_ids"].astype(np.int64)
                    if not np.array_equal(
                        target_ids, np.asarray(example.target_ids[:count], dtype=np.int64)
                    ):
                        raise ValueError(f"Cached FIL target IDs changed for {example.task_id}")
                    scores = cached["future_impact"].astype(np.float32)
                    eager_primal = cached["jvp_eager_primal_kl"].astype(np.float32)
                    top_forced = cached["fil_top_forced"].astype(np.int64).tolist()
                    top_positive = cached["fil_top_positive"].astype(np.int64).tolist()
                    bottom = cached["fil_bottom"].astype(np.int64).tolist()
                    combined = cached["combined"].astype(np.int64).tolist()
                    random_indices = cached["random"].astype(np.int64).tolist()
                    forbidden = cached["preserve"].astype(np.int64).tolist()
                    if example.task_id in repeat_ids:
                        repeated_scores = cached["jvp_repeat_scores"].astype(np.float32)
                        repeated_top = select_fraction(
                            repeated_scores,
                            ratio=ratio,
                            forbidden=forbidden,
                            largest=True,
                            positive_only=False,
                        )
                        repeat_stability.append(
                            {
                                "id": example.task_id,
                                "spearman": spearman_correlation(scores, repeated_scores),
                                "top_jaccard": jaccard(top_forced, repeated_top),
                                "max_absolute_difference": float(
                                    np.max(
                                        np.abs(
                                            scores.astype(np.float64)
                                            - repeated_scores.astype(np.float64)
                                        )
                                    )
                                ),
                            }
                        )
                    if example.task_id in audit_ids:
                        default_primal = cached["default_backend_primal_kl"].astype(np.float32)
                        eager_top = select_fraction(
                            eager_primal, ratio=ratio, forbidden=forbidden, largest=True
                        )
                        default_top = select_fraction(
                            default_primal, ratio=ratio, forbidden=forbidden, largest=True
                        )
                        backend_audits.append(
                            {
                                "id": example.task_id,
                                "primal_spearman": spearman_correlation(
                                    eager_primal, default_primal
                                ),
                                "primal_top_jaccard": jaccard(eager_top, default_top),
                                "primal_relative_l2_error": float(
                                    np.linalg.norm(
                                        eager_primal.astype(np.float64)
                                        - default_primal.astype(np.float64)
                                    )
                                    / max(
                                        float(
                                            np.linalg.norm(default_primal.astype(np.float64))
                                        ),
                                        1e-30,
                                    )
                                ),
                            }
                        )
                    manifest.append(
                        {
                            "id": example.task_id,
                            "tokens": count,
                            "positive_score_tokens": int((scores > 0).sum()),
                            "negative_score_tokens": int((scores < 0).sum()),
                            "top_forced": top_forced,
                            "top_positive": top_positive,
                            "bottom": bottom,
                            "combined": combined,
                            "random": random_indices,
                            "top_forced_positive_fraction": float(
                                np.mean(scores[np.asarray(top_forced, dtype=np.int64)] > 0)
                            ),
                            "top_combined_jaccard": jaccard(top_forced, combined),
                            "array_path": str(array_path),
                            "resumed_from_cache": True,
                        }
                    )
                    continue
        teacher_logits = target_logits(
            prefix_model,
            example,
            indices,
            use_prefix=False,
            with_grad=False,
        ).detach().clone()
        with _jvp_compatible_backend(
            prefix_model,
            attention_query_chunk_size=int(
                fil_cfg.get("jvp_attention_query_chunk_size", 256)
            ),
        ) as current_backend:
            if backend_record is None:
                backend_record = current_backend
            eager_primal, scores = _jvp_losses_and_scores(
                prefix_model,
                example=example,
                indices=indices,
                teacher_logits=teacher_logits,
                direction=normalized_direction,
                chunk_size=chunk_size,
            )
            repeated_scores = None
            if example.task_id in repeat_ids:
                _, repeated_scores = _jvp_losses_and_scores(
                    prefix_model,
                    example=example,
                    indices=indices,
                    teacher_logits=teacher_logits,
                    direction=normalized_direction,
                    chunk_size=chunk_size,
                )

        forbidden = list(record["preserve"])
        top_forced = select_fraction(
            scores, ratio=ratio, forbidden=forbidden, largest=True, positive_only=False
        )
        top_positive = select_fraction(
            scores,
            ratio=ratio,
            forbidden=forbidden,
            largest=True,
            positive_only=positive_only,
        )
        bottom = select_fraction(
            scores, ratio=ratio, forbidden=forbidden, largest=False, positive_only=False
        )
        eligible = [index for index in indices if index not in set(forbidden)]
        budget = len(top_forced)
        task_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{example.task_id}".encode()).digest()[:8], "big"
        )
        random_indices = sorted(random.Random(task_seed).sample(eligible, budget))
        combined = list(record["combined_selected"])
        if set(combined) & set(forbidden):
            raise ValueError(f"Combined/preservation overlap for {example.task_id}")

        if repeated_scores is not None:
            repeated_top = select_fraction(
                repeated_scores,
                ratio=ratio,
                forbidden=forbidden,
                largest=True,
                positive_only=False,
            )
            repeat_stability.append(
                {
                    "id": example.task_id,
                    "spearman": spearman_correlation(scores, repeated_scores),
                    "top_jaccard": jaccard(top_forced, repeated_top),
                    "max_absolute_difference": float(
                        np.max(np.abs(scores.astype(np.float64) - repeated_scores.astype(np.float64)))
                    ),
                }
            )

        default_primal = None
        if example.task_id in audit_ids:
            default_student = target_logits(
                prefix_model,
                example,
                indices,
                use_prefix=True,
                with_grad=False,
            )
            default_primal = per_token_forward_kl(
                teacher_logits=teacher_logits,
                student_logits=default_student,
                chunk_size=chunk_size,
            )
            eager_top = select_fraction(
                eager_primal, ratio=ratio, forbidden=forbidden, largest=True
            )
            default_top = select_fraction(
                default_primal, ratio=ratio, forbidden=forbidden, largest=True
            )
            backend_audits.append(
                {
                    "id": example.task_id,
                    "primal_spearman": spearman_correlation(eager_primal, default_primal),
                    "primal_top_jaccard": jaccard(eager_top, default_top),
                    "primal_relative_l2_error": float(
                        np.linalg.norm(eager_primal.astype(np.float64) - default_primal.astype(np.float64))
                        / max(float(np.linalg.norm(default_primal.astype(np.float64))), 1e-30)
                    ),
                }
            )
            del default_student

        payload = {
            "target_ids": np.asarray(example.target_ids[:count], dtype=np.int32),
            "future_impact": scores.astype(np.float32),
            "jvp_eager_primal_kl": eager_primal.astype(np.float32),
            "fil_top_forced": np.asarray(top_forced, dtype=np.int32),
            "fil_top_positive": np.asarray(top_positive, dtype=np.int32),
            "fil_bottom": np.asarray(bottom, dtype=np.int32),
            "combined": np.asarray(combined, dtype=np.int32),
            "random": np.asarray(random_indices, dtype=np.int32),
            "preserve": np.asarray(forbidden, dtype=np.int32),
        }
        if default_primal is not None:
            payload["default_backend_primal_kl"] = default_primal.astype(np.float32)
        if repeated_scores is not None:
            payload["jvp_repeat_scores"] = repeated_scores.astype(np.float32)
        np.savez_compressed(array_path, **payload)
        manifest.append(
            {
                "id": example.task_id,
                "tokens": count,
                "positive_score_tokens": int((scores > 0).sum()),
                "negative_score_tokens": int((scores < 0).sum()),
                "top_forced": top_forced,
                "top_positive": top_positive,
                "bottom": bottom,
                "combined": combined,
                "random": random_indices,
                "top_forced_positive_fraction": float(
                    np.mean(scores[np.asarray(top_forced, dtype=np.int64)] > 0)
                ),
                "top_combined_jaccard": jaccard(top_forced, combined),
                "array_path": str(array_path),
            }
        )
        del teacher_logits
        _empty_cuda_cache(torch)

    _write_jsonl(out_root / "locator" / "manifest.jsonl", manifest)
    summary = {
        "score_method": "exact_forward_ad_jvp_with_eager_qwen_kernels",
        "backend": backend_record,
        "trajectories": len(manifest),
        "tokens": sum(row["tokens"] for row in manifest),
        "selected_tokens_forced": sum(len(row["top_forced"]) for row in manifest),
        "selected_tokens_positive": sum(len(row["top_positive"]) for row in manifest),
        "mean_top_forced_positive_fraction": float(
            np.mean([row["top_forced_positive_fraction"] for row in manifest])
        ),
        "mean_combined_jaccard": float(
            np.mean([row["top_combined_jaccard"] for row in manifest])
        ),
        "mean_stability_spearman": float(
            np.nanmean([row["spearman"] for row in repeat_stability])
        ),
        "mean_stability_top_jaccard": float(
            np.mean([row["top_jaccard"] for row in repeat_stability])
        ),
        "max_repeat_absolute_difference": float(
            max((row["max_absolute_difference"] for row in repeat_stability), default=0.0)
        ),
        "mean_backend_primal_spearman": float(
            np.nanmean([row["primal_spearman"] for row in backend_audits])
        ),
        "mean_backend_primal_top_jaccard": float(
            np.mean([row["primal_top_jaccard"] for row in backend_audits])
        ),
        "mean_backend_primal_relative_l2_error": float(
            np.mean([row["primal_relative_l2_error"] for row in backend_audits])
        ),
        "repeat_stability": repeat_stability,
        "backend_audits": backend_audits,
        "manifest_path": str(out_root / "locator" / "manifest.jsonl"),
    }
    _json(out_root / "locator" / "summary.json", summary)
    return manifest, summary


def _score_source_records(
    prefix_model,
    records,
    *,
    direction,
    fil_cfg,
    seed: int,
    out_root: Path,
):
    method = str(fil_cfg.get("score_method", "finite_difference")).strip().lower()
    if method in {"jvp", "forward_ad", "jvp_eager"}:
        return _score_source_records_jvp(
            prefix_model,
            records,
            direction=direction,
            fil_cfg=fil_cfg,
            seed=seed,
            out_root=out_root,
        )
    if method not in {"finite_difference", "central_difference"}:
        raise ValueError(f"Unknown FIL score_method: {method}")
    return _score_source_records_finite_difference(
        prefix_model,
        records,
        direction=direction,
        fil_cfg=fil_cfg,
        seed=seed,
        out_root=out_root,
    )


def _load_locator_masks(out_root: Path) -> dict[str, dict[str, list[int]]]:
    rows = _read_jsonl(out_root / "locator" / "manifest.jsonl")
    methods = {
        "fil_top10": "top_forced",
        "combined10": "combined",
        "random10": "random",
        "fil_bottom10": "bottom",
    }
    return {
        method: {str(row["id"]): list(row[field]) for row in rows}
        for method, field in methods.items()
    }


def _apply_mask(records, selected_by_id: dict[str, list[int]]) -> None:
    for record in records:
        selected = list(selected_by_id[record["example"].task_id])
        record["selected"] = selected
        record["selected_weights"] = np.ones(len(selected), dtype=np.float32)
        record["effective_skill_weight"] = float(len(selected))


def _train_calibration_steps(prefix_model, records, optimizer, *, steps, seed, fil_cfg):
    torch = prefix_model.torch
    accumulation = int(fil_cfg.get("calibration_accumulation", 2))
    required = int(steps) * accumulation
    order = []
    cycle = 0
    while len(order) < required:
        order.extend(_training_order(torch, len(records), seed + cycle))
        cycle += 1
    order = order[:required]
    dynamic_cfg = {
        "loss_weighting": "equal",
        "kl_chunk_size": int(fil_cfg.get("kl_chunk_size", 8)),
        "preservation_loss_weight": float(fil_cfg.get("preservation_loss_weight", 1.0)),
    }
    history = []
    for step in tqdm(range(int(steps)), desc="FIL actual Adam gate", unit="step"):
        group = order[step * accumulation : (step + 1) * accumulation]
        loss = _train_group(prefix_model, records, group, optimizer, dynamic_cfg)
        history.append(
            {
                "step": step + 1,
                "loss": loss,
                "trajectory_ids": [records[index]["example"].task_id for index in group],
            }
        )
    return history


def _calibrate_groups(
    prefix_model,
    *,
    source_records,
    outer_prepared,
    calibration_ids,
    train_items,
    masks,
    settings,
    flat,
    fil_cfg,
    seed,
    out_root,
    skip_generation,
):
    torch = prefix_model.torch
    initial_state = copy.deepcopy(prefix_model.state_dict())
    initial_outer, _, _ = _outer_objective(
        prefix_model,
        outer_prepared,
        chunk_size=int(fil_cfg.get("kl_chunk_size", 8)),
        with_grad=False,
    )
    initial_preservation, _ = _preservation_objective(
        prefix_model, source_records, with_grad=False
    )
    results = {}
    calibration_items = [train_items[task_id] for task_id in calibration_ids]

    initial_dir = out_root / "calibration" / "no_update"
    initial_summary_path = initial_dir / "summary.json"
    if initial_summary_path.exists():
        results["no_update"] = json.loads(
            initial_summary_path.read_text(encoding="utf-8")
        )
        print("[FIL gate] reusing completed no_update", flush=True)
    else:
        base_hard = base_soft = None
        base_task_results = []
        if not skip_generation:
            base_hard, base_soft, base_task_results = _evaluate_prefix(
                "spreadsheetbench",
                prefix_model,
                calibration_items,
                cfg=flat,
                settings=settings,
                out_dir=str(initial_dir / "free_generation"),
                desc="FIL Cal12 no_update",
            )
        base_summary = {
            "method": "no_update",
            "optimizer_steps": 0,
            "initial_outer_loss": initial_outer,
            "post_outer_loss": initial_outer,
            "outer_loss_change": 0.0,
            "initial_preservation_kl": initial_preservation,
            "post_preservation_kl": initial_preservation,
            "preservation_change": 0.0,
            "calibration_tasks": len(calibration_items),
            "calibration_hard": base_hard,
            "calibration_soft": base_soft,
            "calibration_successes": (
                None
                if base_hard is None
                else round(float(base_hard) * len(calibration_items))
            ),
            "task_results": base_task_results,
            "val40_accessed": False,
            "test280_accessed": False,
        }
        _json(initial_summary_path, base_summary)
        results["no_update"] = base_summary
        print(
            f"[FIL gate] no_update: Cal12={base_summary['calibration_successes']}/"
            f"{len(calibration_items)}",
            flush=True,
        )

    for method in ("fil_top10", "combined10", "random10", "fil_bottom10"):
        method_dir = out_root / "calibration" / method
        summary_path = method_dir / "summary.json"
        if summary_path.exists():
            results[method] = json.loads(summary_path.read_text(encoding="utf-8"))
            print(f"[FIL gate] reusing completed {method}", flush=True)
            continue
        prefix_model.load_state_dict(initial_state)
        _apply_mask(source_records, masks[method])
        optimizer = torch.optim.AdamW(
            prefix_model.trainable_parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
        history = _train_calibration_steps(
            prefix_model,
            source_records,
            optimizer,
            steps=int(fil_cfg.get("calibration_optimizer_steps", 4)),
            seed=seed + 70001,
            fil_cfg=fil_cfg,
        )
        post_outer, _, outer_details = _outer_objective(
            prefix_model,
            outer_prepared,
            chunk_size=int(fil_cfg.get("kl_chunk_size", 8)),
            with_grad=False,
        )
        post_preservation, _ = _preservation_objective(
            prefix_model, source_records, with_grad=False
        )
        checkpoint = method_dir / "prefix_after_4_steps.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(prefix_model.state_dict(), checkpoint)
        cal_hard = cal_soft = None
        task_results = []
        if not skip_generation:
            cal_hard, cal_soft, task_results = _evaluate_prefix(
                "spreadsheetbench",
                prefix_model,
                calibration_items,
                cfg=flat,
                settings=settings,
                out_dir=str(method_dir / "free_generation"),
                desc=f"FIL Cal12 {method}",
            )
        summary = {
            "method": method,
            "optimizer_steps": len(history),
            "initial_outer_loss": initial_outer,
            "post_outer_loss": post_outer,
            "outer_loss_change": post_outer - initial_outer,
            "predicted_benefit_sign": "beneficial" if post_outer < initial_outer else "harmful",
            "initial_preservation_kl": initial_preservation,
            "post_preservation_kl": post_preservation,
            "preservation_change": post_preservation - initial_preservation,
            "calibration_tasks": len(calibration_items),
            "calibration_hard": cal_hard,
            "calibration_soft": cal_soft,
            "calibration_successes": (
                None if cal_hard is None else round(float(cal_hard) * len(calibration_items))
            ),
            "history": history,
            "outer_details": outer_details,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "task_results": task_results,
            "val40_accessed": False,
            "test280_accessed": False,
        }
        _json(summary_path, summary)
        results[method] = summary
        print(
            f"[FIL gate] {method}: outer_delta={summary['outer_loss_change']:+.6f} "
            f"preserve_delta={summary['preservation_change']:+.6f} "
            f"Cal12={summary['calibration_successes']}/{len(calibration_items)}",
            flush=True,
        )
    prefix_model.load_state_dict(initial_state)
    fil = results["fil_top10"]
    combined = results["combined10"]
    random_result = results["random10"]
    bottom = results["fil_bottom10"]
    no_update = results["no_update"]
    surrogate_pass = (
        float(fil["outer_loss_change"]) < 0.0
        and float(fil["outer_loss_change"]) < float(combined["outer_loss_change"])
        and float(fil["outer_loss_change"]) < float(random_result["outer_loss_change"])
        and float(fil["outer_loss_change"]) < float(bottom["outer_loss_change"])
    )
    preservation_limit = max(
        float(combined["post_preservation_kl"]) * 1.10,
        float(combined["post_preservation_kl"]) + 1e-8,
    )
    preservation_pass = float(fil["post_preservation_kl"]) <= preservation_limit
    execution_pass = True

    def execution_tuple(summary):
        rows = summary.get("task_results", [])
        return (
            int(summary.get("calibration_successes") or 0),
            float(sum(float(row.get("soft", 0.0)) for row in rows)),
            int(sum(int(row.get("n_exec_pass", 0)) for row in rows)),
            int(
                sum(
                    bool(str(row.get("response", "")).strip())
                    and "```" in str(row.get("response", ""))
                    for row in rows
                )
            ),
        )

    if not skip_generation:
        fil_execution = execution_tuple(fil)
        execution_pass = all(
            fil_execution >= execution_tuple(other)
            for other in (no_update, combined, random_result, bottom)
        )
    no_update_by_id = {
        str(row["id"]): int(row.get("hard", 0))
        for row in no_update.get("task_results", [])
    }
    paired_vs_no_update = {}
    for method, summary in results.items():
        current = {
            str(row["id"]): int(row.get("hard", 0))
            for row in summary.get("task_results", [])
        }
        paired_vs_no_update[method] = {
            "gained": sorted(
                task_id
                for task_id, value in current.items()
                if value > no_update_by_id.get(task_id, 0)
            ),
            "lost": sorted(
                task_id
                for task_id, value in current.items()
                if value < no_update_by_id.get(task_id, 0)
            ),
            "execution_tuple": execution_tuple(summary),
        }
    gate = {
        "surrogate_pass": surrogate_pass,
        "preservation_pass": preservation_pass,
        "execution_pass": execution_pass,
        "passed": surrogate_pass and preservation_pass and execution_pass,
        "rule": (
            "FIL outer delta must be negative and beat Combined/Random/Bottom; preservation "
            "must be within 10% of Combined; Cal12 lexicographic execution must not trail "
            "No-update/Combined/Random/Bottom"
        ),
        "methods": results,
        "paired_vs_no_update": paired_vs_no_update,
        "val40_accessed": False,
        "test280_accessed": False,
    }
    _json(out_root / "calibration" / "gate_summary.json", gate)
    return gate


def main() -> None:
    args = parse_args()
    raw = load_config(args.config, overrides=args.cfg_options)
    (
        flat,
        soft_cfg,
        fil_cfg,
        out_root,
        seed,
        settings,
        init_text,
        prefix_model,
        records,
        by_id,
        train_items,
        partition,
    ) = _prepare_experiment(raw, args)
    config_record = {
        "method": "FIL-v2-direct-gate",
        "runtime": _redact(flat),
        "soft_prefix": _redact(soft_cfg),
        "future_impact": fil_cfg,
        "fairness_contract": {
            "backbone_frozen": True,
            "prefix_length": settings.prefix_length,
            "trainable_parameters": int(prefix_model.prefix_embeddings.numel()),
            "candidate_context": "successful GPT-5.5 gold trajectories",
            "outer_context": "held-out current-soft-prefix learner states",
            "outer_teacher": "Qwen plus full hard Skill, outcome-gated",
            "outer_execution_gradient": False,
            "source_tasks": len(partition.source),
            "outer_tasks": len(partition.outer),
            "calibration_tasks": len(partition.calibration),
            "val40_accessed": False,
            "test280_accessed": False,
        },
        "input_fingerprints": {
            "trajectory_manifest_sha256": _sha256(settings.trajectory_examples_path),
            "skill_sha256": _sha256(settings.init_text_path),
        },
    }
    config_path = out_root / "config.json"
    if not config_path.exists():
        _json(config_path, config_record)

    print("=" * 76, flush=True)
    print("Future-Impact Locator v2 / direct reliability gate", flush=True)
    print(f"model={settings.model_name}", flush=True)
    print(
        f"Source={len(partition.source)} Outer={len(partition.outer)} "
        f"Cal={len(partition.calibration)} prefix={settings.prefix_length}",
        flush=True,
    )
    print("Val40 and Test280 are inaccessible in this driver", flush=True)
    print("=" * 76, flush=True)

    started = time.time()
    soft_results, hard_results = _run_outer_rollouts(
        prefix_model,
        ids=list(partition.outer),
        train_items=train_items,
        flat=flat,
        settings=settings,
        init_text=init_text,
        out_root=out_root,
    )
    if args.phase == "outer":
        return
    outer_prepared = _encode_outer_examples(
        prefix_model,
        by_id,
        soft_results,
        hard_results,
        settings=settings,
    )
    source_records = [by_id[task_id] for task_id in partition.source]
    locator_summary_path = out_root / "locator" / "summary.json"
    direction_path = out_root / "locator" / "direction.pt"
    if not locator_summary_path.exists():
        reuse_direction = bool(fil_cfg.get("reuse_direction_if_present", False)) and direction_path.exists()
        if reuse_direction:
            print(f"[FIL score] reusing frozen direction {direction_path}", flush=True)
            direction_bundle = prefix_model.torch.load(
                direction_path, map_location="cpu", weights_only=False
            )
            direction = direction_bundle["direction"].float()
            outer_gradient = direction_bundle["outer_gradient"].float()
            preservation_gradient = direction_bundle["preservation_gradient"].float()
            second_moment = direction_bundle["second_moment"].float()
            outer_loss, _, outer_details = _outer_objective(
                prefix_model,
                outer_prepared,
                chunk_size=int(fil_cfg.get("kl_chunk_size", 8)),
                with_grad=False,
            )
            preservation_loss, _ = _preservation_objective(
                prefix_model, source_records, with_grad=False
            )
            direction_stats = {
                "gradient_l2": float(prefix_model.torch.linalg.vector_norm(outer_gradient).cpu()),
                "second_moment_mean": float(second_moment.mean().cpu()),
                "reused_frozen_direction": True,
            }
            moment_probe = {"reused_frozen_direction": True}
        else:
            outer_loss, outer_gradient, outer_details = _outer_objective(
                prefix_model,
                outer_prepared,
                chunk_size=int(fil_cfg.get("kl_chunk_size", 8)),
                with_grad=True,
            )
            preservation_loss, preservation_gradient = _preservation_objective(
                prefix_model, source_records, with_grad=True
            )
            second_moment, moment_probe = _reference_second_moment(
                prefix_model,
                source_records,
                settings=settings,
                fil_cfg=fil_cfg,
                seed=seed,
            )
            direction, direction_stats = adam_diagonal_direction(
                outer_gradient,
                second_moment,
                adam_epsilon=float(fil_cfg.get("adam_epsilon", 1e-8)),
                floor_fraction=float(fil_cfg.get("moment_floor_fraction", 1e-3)),
            )
            direction_path.parent.mkdir(parents=True, exist_ok=True)
            prefix_model.torch.save(
                {
                    "direction": direction.cpu(),
                    "outer_gradient": outer_gradient.cpu(),
                    "preservation_gradient": preservation_gradient.cpu(),
                    "second_moment": second_moment.cpu(),
                },
                direction_path,
            )
        _, locator_summary = _score_source_records(
            prefix_model,
            source_records,
            direction=direction.to(prefix_model.device),
            fil_cfg=fil_cfg,
            seed=seed,
            out_root=out_root,
        )
        locator_summary.update(
            {
                "outer_loss": outer_loss,
                "outer_details": outer_details,
                "preservation_loss": preservation_loss,
                "direction_stats": direction_stats,
                "moment_probe": moment_probe,
                "direction_path": str(direction_path),
                "direction_sha256": _sha256(direction_path),
            }
        )
        _json(locator_summary_path, locator_summary)
    else:
        locator_summary = json.loads(locator_summary_path.read_text(encoding="utf-8"))
        print(f"[FIL score] reusing {locator_summary_path}", flush=True)
    print(
        f"[FIL score] repeat Spearman={locator_summary['mean_stability_spearman']:.4f} "
        f"Top10 Jaccard={locator_summary['mean_stability_top_jaccard']:.4f} "
        f"positive-in-Top10={locator_summary['mean_top_forced_positive_fraction']:.2%}",
        flush=True,
    )
    stability_pass = (
        float(locator_summary["mean_stability_spearman"])
        >= float(fil_cfg.get("min_stability_spearman", 0.80))
        and float(locator_summary["mean_stability_top_jaccard"])
        >= float(fil_cfg.get("min_stability_jaccard", 0.60))
    )
    score_method = str(locator_summary.get("score_method", "finite_difference"))
    backend_pass = True
    oracle_pass = True
    oracle_record = None
    if score_method.startswith("exact_forward_ad_jvp"):
        backend_pass = (
            float(locator_summary["mean_backend_primal_spearman"])
            >= float(fil_cfg.get("min_backend_primal_spearman", 0.90))
            and float(locator_summary["mean_backend_primal_top_jaccard"])
            >= float(fil_cfg.get("min_backend_primal_jaccard", 0.70))
        )
        oracle_path = Path(str(fil_cfg.get("jvp_oracle_path", "")))
        if not oracle_path.is_absolute():
            oracle_path = PROJECT_ROOT / oracle_path
        if not oracle_path.exists():
            raise FileNotFoundError(f"FIL JVP oracle is required but missing: {oracle_path}")
        oracle_record = json.loads(oracle_path.read_text(encoding="utf-8"))
        oracle_pass = bool(oracle_record.get("passed", False))
    numerical_pass = stability_pass and backend_pass and oracle_pass
    _json(
        out_root / "locator" / "numerical_gate.json",
        {
            "passed": numerical_pass,
            "stability_pass": stability_pass,
            "backend_pass": backend_pass,
            "oracle_pass": oracle_pass,
            "score_method": score_method,
            "min_stability_spearman": float(fil_cfg.get("min_stability_spearman", 0.80)),
            "min_stability_jaccard": float(fil_cfg.get("min_stability_jaccard", 0.60)),
            "actual_stability_spearman": locator_summary["mean_stability_spearman"],
            "actual_stability_jaccard": locator_summary["mean_stability_top_jaccard"],
            "min_backend_primal_spearman": float(
                fil_cfg.get("min_backend_primal_spearman", 0.90)
            ),
            "min_backend_primal_jaccard": float(
                fil_cfg.get("min_backend_primal_jaccard", 0.70)
            ),
            "actual_backend_primal_spearman": locator_summary.get(
                "mean_backend_primal_spearman"
            ),
            "actual_backend_primal_jaccard": locator_summary.get(
                "mean_backend_primal_top_jaccard"
            ),
            "jvp_oracle": oracle_record,
        },
    )
    if args.phase == "score":
        return
    if not numerical_pass:
        print("[FIL] numerical gate failed; calibration and full training are intentionally blocked", flush=True)
        return

    masks = _load_locator_masks(out_root)
    gate = _calibrate_groups(
        prefix_model,
        source_records=source_records,
        outer_prepared=outer_prepared,
        calibration_ids=list(partition.calibration),
        train_items=train_items,
        masks=masks,
        settings=settings,
        flat=flat,
        fil_cfg=fil_cfg,
        seed=seed,
        out_root=out_root,
        skip_generation=args.skip_calibration_generation,
    )
    summary = {
        "method": "FIL-v2-direct-gate",
        "numerical_gate_passed": numerical_pass,
        "group_gate_passed": gate["passed"],
        "group_gate": gate,
        "locator": locator_summary,
        "wall_time_s": round(time.time() - started, 1),
        "val40_accessed": False,
        "test280_accessed": False,
        "next_action": (
            "implement residual 5%+5% and matched full training"
            if gate["passed"]
            else "stop: direct future-impact proxy failed its pre-registered reliability gate"
        ),
    }
    _json(out_root / "summary.json", summary)
    print(f"FIL-v2 gate passed={gate['passed']}", flush=True)
    print(f"summary={out_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
