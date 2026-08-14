#!/usr/bin/env python3
"""Task-specific soft-prefix oracle for 61 successful SpreadsheetBench tasks.

Each successful training trajectory gets an independent length-8 prefix.  The
existing Combined Top-5% locator is frozen.  On its selected positions the
prefix matches the full text-Skill Qwen Top-64 distribution; on its matched
preservation positions it retains the no-Skill Qwen distribution.  Training
uses the successful GPT-5.5 trajectory as gold context and a fixed step budget.

The optional evaluation compares, on the same 61 training tasks:
  * task-specific soft prefix + clean task prompt;
  * full text Skill + clean task prompt; and
  * no Skill + clean task prompt.

This is deliberately an oracle/capacity experiment, not a generalization
result.  Val40 and Test280 are never loaded.
"""
from __future__ import annotations

import argparse
import gc
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
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skillopt.envs.spreadsheetbench.codegen_agent import _build_system
from skillopt.envs.spreadsheetbench.dataloader import SpreadsheetBenchDataLoader
from skillopt.softprefix.model import SoftPrefixCausalLM
from skillopt.softprefix.prcb_v6 import topk_residual_kl_from_logits
from skillopt.softprefix.trainer import evaluate_spreadsheet_prefix
from scripts.train_spreadsheetbench_prcb_v1 import (
    atomic_json,
    atomic_jsonl,
    encode_trajectory,
    read_jsonl,
    resolve,
    resolve_model_reference,
    sha256,
    slug,
)
from scripts.train_spreadsheetbench_prcb_v6 import (
    install_prefix,
    load_prefix,
    logits_for_positions,
)


DEFAULT_MODEL = os.environ.get("SPREADSHEETBENCH_MODEL", "Qwen/Qwen3.6-35B-A3B")
DEFAULT_MANIFEST = (
    "outputs/SpreadsheetBench_selective_stage2_manifests/"
    "combined_top0.05_core_shared_preserve.jsonl"
)
DEFAULT_OUT = "outputs/SpreadsheetBench_task_specific_selective_skillkl_len8_seed1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--skill-path", default="ckpt/spreadsheetbench/gpt5.5_skill.md")
    parser.add_argument("--out-root", default=DEFAULT_OUT)
    parser.add_argument("--split-dir", default="data/spreadsheetbench_split")
    parser.add_argument("--data-root", default="data/spreadsheetbench_verified_400")
    parser.add_argument("--phase", choices=("train", "eval", "all"), default="all")
    parser.add_argument(
        "--eval-conditions",
        default="soft,hard,plain",
        help="Comma-separated subset of soft,hard,plain.",
    )
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument(
        "--checkpoint-steps",
        default="1,4,8,16,32",
        help="Fixed diagnostic snapshots; the max-step checkpoint is evaluated.",
    )
    parser.add_argument("--preservation-weight", type=float, default=1.0)
    parser.add_argument("--delta-weight", type=float, default=1e-4)
    parser.add_argument("--max-prompt-tokens", type=int, default=16384)
    parser.add_argument("--max-target-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--exec-timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain task checkpoints even when complete summaries exist.",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Regenerate a condition/task result even when it already exists.",
    )
    return parser.parse_args()


def stable_seed(identifier: str, seed: int) -> int:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int(seed) + int.from_bytes(digest[:8], "little") % (2**31 - 1)


def reference_arrays(row: dict[str, Any], target_ids: list[int]) -> dict[str, np.ndarray]:
    with np.load(resolve(str(row["score_cache"]))) as cached:
        arrays = {name: cached[name] for name in cached.files}
    required = {
        "target_ids",
        "skill_target_logp",
        "skill_topk_ids",
        "skill_topk_logp",
        "skill_residual_log_mass",
        "clean_topk_ids",
        "clean_topk_logp",
        "clean_residual_log_mass",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"Teacher cache for {row['id']} is missing {sorted(missing)}")
    if arrays["target_ids"].astype(np.int64).tolist() != target_ids:
        raise ValueError(f"Tokenizer/cache mismatch for task {row['id']!r}")
    return arrays


def reference_tensors(
    torch: Any,
    arrays: dict[str, np.ndarray],
    positions: list[int],
    device: Any,
    *,
    distribution: str,
) -> tuple[Any, Any, Any]:
    if distribution not in {"skill", "clean"}:
        raise ValueError(distribution)
    index = np.asarray(positions, dtype=np.int64)
    return (
        torch.from_numpy(arrays[f"{distribution}_topk_ids"][index].astype(np.int64)).to(device),
        torch.from_numpy(arrays[f"{distribution}_topk_logp"][index].astype(np.float32)).to(device),
        torch.from_numpy(
            arrays[f"{distribution}_residual_log_mass"][index].astype(np.float32)
        ).to(device),
    )


def selective_objective(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    row: dict[str, Any],
    arrays: dict[str, np.ndarray],
    core_positions: list[int],
    preserve_positions: list[int],
    *,
    prefix: Any,
    max_prompt_tokens: int,
    max_target_tokens: int,
    preservation_weight: float,
    inference: bool,
) -> dict[str, Any]:
    positions = sorted(set(core_positions) | set(preserve_positions))
    offset = {position: index for index, position in enumerate(positions)}
    core_offsets = [offset[position] for position in core_positions]
    preserve_offsets = [offset[position] for position in preserve_positions]
    logits = logits_for_positions(
        torch,
        prefix_model,
        row,
        positions,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
        inference=inference,
        prefix_value=prefix,
    )
    device = logits.device
    core_index = torch.tensor(core_offsets, dtype=torch.long, device=device)
    preserve_index = torch.tensor(preserve_offsets, dtype=torch.long, device=device)
    skill_ids, skill_logp, skill_residual = reference_tensors(
        torch, arrays, core_positions, device, distribution="skill"
    )
    clean_ids, clean_logp, clean_residual = reference_tensors(
        torch, arrays, preserve_positions, device, distribution="clean"
    )
    core_kl = topk_residual_kl_from_logits(
        torch,
        logits[core_index],
        reference_topk_ids=skill_ids,
        reference_topk_logp=skill_logp,
        reference_residual_log_mass=skill_residual,
    ).mean()
    preservation_kl = topk_residual_kl_from_logits(
        torch,
        logits[preserve_index],
        reference_topk_ids=clean_ids,
        reference_topk_logp=clean_logp,
        reference_residual_log_mass=clean_residual,
    ).mean()
    total = core_kl + float(preservation_weight) * preservation_kl
    return {
        "loss": total,
        "core_skill_kl": core_kl,
        "preservation_clean_kl": preservation_kl,
        "core_positions": len(core_positions),
        "preserve_positions": len(preserve_positions),
    }


def detached_metrics(losses: dict[str, Any]) -> dict[str, float | int]:
    return {
        "loss": float(losses["loss"].detach().cpu()),
        "core_skill_kl": float(losses["core_skill_kl"].detach().cpu()),
        "preservation_clean_kl": float(
            losses["preservation_clean_kl"].detach().cpu()
        ),
        "core_positions": int(losses["core_positions"]),
        "preserve_positions": int(losses["preserve_positions"]),
    }


def evaluate_selective_objective(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    row: dict[str, Any],
    arrays: dict[str, np.ndarray],
    core_positions: list[int],
    preserve_positions: list[int],
    *,
    prefix: Any,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    losses = selective_objective(
        torch,
        prefix_model,
        row,
        arrays,
        core_positions,
        preserve_positions,
        prefix=prefix,
        max_prompt_tokens=args.max_prompt_tokens,
        max_target_tokens=args.max_target_tokens,
        preservation_weight=args.preservation_weight,
        inference=True,
    )
    metrics = detached_metrics(losses)
    del losses
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def train_one_task(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    row: dict[str, Any],
    *,
    base_prefix: Any,
    args: argparse.Namespace,
    task_dir: Path,
) -> dict[str, Any]:
    checkpoint_path = task_dir / "final_prefix.pt"
    summary_path = task_dir / "training_summary.json"
    if summary_path.exists() and checkpoint_path.exists() and not args.force_train:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("complete") and summary.get("checkpoint_sha256") == sha256(checkpoint_path):
            return summary

    task_dir.mkdir(parents=True, exist_ok=True)
    identifier = str(row["id"])
    _, target_ids = encode_trajectory(
        prefix_model,
        row,
        max_prompt_tokens=args.max_prompt_tokens,
        max_target_tokens=args.max_target_tokens,
    )
    arrays = reference_arrays(row, target_ids)
    core_positions = sorted({int(value) for value in row["selected_indices"]})
    # Match the previous selective training convention: always supervise EOS.
    core_positions = sorted(set(core_positions) | {len(target_ids) - 1})
    preserve_positions = sorted(
        int(value)
        for value in row["preserve_indices"]
        if int(value) not in set(core_positions)
    )
    if not core_positions or not preserve_positions:
        raise ValueError(f"Task {identifier} has an empty core/preservation set")
    if max(core_positions + preserve_positions) >= len(target_ids):
        raise ValueError(f"Task {identifier} contains an out-of-range selected position")

    install_prefix(torch, prefix_model, base_prefix)
    prefix_model.prefix_embeddings.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [prefix_model.prefix_embeddings],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    baseline = evaluate_selective_objective(
        torch,
        prefix_model,
        row,
        arrays,
        core_positions,
        preserve_positions,
        prefix=base_prefix,
        args=args,
    )
    history: list[dict[str, Any]] = [{"step": 0, **baseline}]
    checkpoint_steps = {
        int(value.strip())
        for value in str(args.checkpoint_steps).split(",")
        if value.strip()
    }
    checkpoint_steps = {value for value in checkpoint_steps if 0 < value <= args.max_steps}
    checkpoint_steps.add(args.max_steps)
    steps_executed = 0
    started = time.time()
    progress = tqdm(
        range(1, args.max_steps + 1),
        desc=f"Task {identifier}",
        unit="step",
        leave=False,
    )
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        losses = selective_objective(
            torch,
            prefix_model,
            row,
            arrays,
            core_positions,
            preserve_positions,
            prefix=prefix_model.prefix_embeddings,
            max_prompt_tokens=args.max_prompt_tokens,
            max_target_tokens=args.max_target_tokens,
            preservation_weight=args.preservation_weight,
            inference=False,
        )
        delta = prefix_model.prefix_embeddings.float() - base_prefix.float()
        delta_loss = delta.square().mean()
        loss = losses["loss"] + float(args.delta_weight) * delta_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([prefix_model.prefix_embeddings], max_norm=1.0)
        optimizer.step()
        steps_executed = step
        train_metrics = detached_metrics(losses)
        progress.set_postfix(
            core=f"{train_metrics['core_skill_kl']:.4f}",
            keep=f"{train_metrics['preservation_clean_kl']:.4f}",
        )
        del losses, delta, delta_loss, loss
        gc.collect()
        torch.cuda.empty_cache()
        if step in checkpoint_steps:
            current = prefix_model.prefix_embeddings.detach().clone()
            metrics = evaluate_selective_objective(
                torch,
                prefix_model,
                row,
                arrays,
                core_positions,
                preserve_positions,
                prefix=current,
                args=args,
            )
            history.append({"step": step, **metrics})
            snapshot = task_dir / f"prefix_step_{step:03d}.pt"
            torch.save(
                {
                    "prefix_embeddings": current.cpu(),
                    "prefix_length": args.prefix_length,
                    "task_id": identifier,
                    "step": step,
                },
                snapshot,
            )
            install_prefix(torch, prefix_model, current)
    progress.close()
    final_prefix = prefix_model.prefix_embeddings.detach().clone()
    final_metrics = history[-1]
    if int(final_metrics["step"]) != args.max_steps:
        raise AssertionError("Final fixed-step metrics were not recorded")
    torch.save(
        {
            "prefix_embeddings": final_prefix.cpu(),
            "prefix_length": args.prefix_length,
            "task_id": identifier,
            "step": args.max_steps,
        },
        checkpoint_path,
    )
    baseline_kl = float(baseline["core_skill_kl"])
    final_kl = float(final_metrics["core_skill_kl"])
    summary = {
        "complete": True,
        "task_id": identifier,
        "objective": "selected_skill_top64_kl_plus_clean_preservation_kl",
        "trajectory_target_tokens": len(target_ids),
        "core_positions": len(core_positions),
        "manifest_selected_positions": len(row["selected_indices"]),
        "always_supervised_eos": True,
        "preserve_positions": len(preserve_positions),
        "steps_executed": steps_executed,
        "fixed_final_step": args.max_steps,
        "baseline": baseline,
        "final": {key: value for key, value in final_metrics.items() if key != "step"},
        "core_skill_kl_closure": (
            (baseline_kl - final_kl) / baseline_kl if baseline_kl > 0 else 0.0
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "history": history,
        "wall_time_s": round(time.time() - started, 1),
        "test_split_accessed": False,
    }
    atomic_json(summary_path, summary)
    return summary


class PrefixDisabledGenerator:
    """Generate from text prompts without installing the soft prefix."""

    def __init__(self, model: SoftPrefixCausalLM, *, skill_text: str = "") -> None:
        self.model = model
        self.clean_system = _build_system("")
        self.hard_system = _build_system(skill_text) if skill_text.strip() else ""

    def generate_from_prompts(
        self,
        prompts: list[str],
        *,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float,
        **_kwargs: Any,
    ) -> list[str]:
        rendered = prompts
        if self.hard_system:
            rendered = []
            for prompt in prompts:
                if self.clean_system not in prompt:
                    raise ValueError("Could not locate clean SpreadsheetBench system prompt")
                rendered.append(prompt.replace(self.clean_system, self.hard_system, 1))
        return self.model.generate_from_prompts(
            rendered,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_prefix=False,
        )


def load_single_result(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one result in {path}, found {len(rows)}")
    return rows[0]


def evaluate_one_condition(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    item: dict[str, Any],
    *,
    condition: str,
    checkpoint_path: Path,
    skill_text: str,
    args: argparse.Namespace,
    task_out: Path,
) -> dict[str, Any]:
    result_path = task_out / "results.jsonl"
    if result_path.exists() and not args.force_eval:
        return load_single_result(result_path)
    if result_path.exists() and args.force_eval:
        raise FileExistsError(
            f"Refusing destructive overwrite of {result_path}; choose a new --out-root"
        )
    if condition == "soft":
        prefix = load_prefix(
            torch,
            checkpoint_path,
            prefix_model.device,
            prefix_model.prefix_embeddings.dtype,
        )
        install_prefix(torch, prefix_model, prefix)
        generator = None
    elif condition == "hard":
        generator = PrefixDisabledGenerator(prefix_model, skill_text=skill_text)
    elif condition == "plain":
        generator = PrefixDisabledGenerator(prefix_model)
    else:
        raise ValueError(condition)
    hard, soft, results = evaluate_spreadsheet_prefix(
        prefix_model,
        [item],
        out_dir=str(task_out),
        data_root=str(resolve(args.data_root)),
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        exec_timeout=args.exec_timeout,
        desc=f"{condition}:{item['id']}",
        generator=generator,
        injection_position="prompt_start",
        repair_turns=1,
        generation_batch_size=1,
    )
    if len(results) != 1:
        raise ValueError(f"Expected one {condition} result for {item['id']}")
    if float(hard) != float(results[0]["hard"]) or float(soft) != float(results[0]["soft"]):
        raise AssertionError("Single-task aggregate mismatch")
    return results[0]


def aggregate_results(
    rows: list[dict[str, Any]],
    training: list[dict[str, Any]],
    conditions: list[str],
) -> dict[str, Any]:
    by_condition: dict[str, dict[str, dict[str, Any]]] = {value: {} for value in conditions}
    for row in rows:
        by_condition[str(row["condition"])][str(row["id"])] = row
    ids = sorted({str(row["task_id"]) for row in training})
    summary: dict[str, Any] = {
        "scope": "same-task oracle on 61 successful training trajectories",
        "tasks": len(ids),
        "conditions": {},
        "test_split_accessed": False,
    }
    for condition in conditions:
        present = by_condition[condition]
        successes = sum(bool(present[identifier].get("hard")) for identifier in ids if identifier in present)
        summary["conditions"][condition] = {
            "evaluated": sum(identifier in present for identifier in ids),
            "successes": successes,
            "hard_rate": successes / len(ids) if ids else 0.0,
            "soft_case_rate": (
                sum(float(present[identifier].get("soft", 0.0)) for identifier in ids if identifier in present)
                / len(ids)
                if ids
                else 0.0
            ),
        }
    if {"soft", "hard"}.issubset(conditions):
        complete = [
            identifier
            for identifier in ids
            if identifier in by_condition["soft"] and identifier in by_condition["hard"]
        ]
        hard_success = {i for i in complete if by_condition["hard"][i].get("hard")}
        soft_success = {i for i in complete if by_condition["soft"][i].get("hard")}
        summary["soft_vs_hard"] = {
            "paired_tasks": len(complete),
            "both_success": len(hard_success & soft_success),
            "soft_only_success": len(soft_success - hard_success),
            "hard_only_success": len(hard_success - soft_success),
            "neither_success": len(set(complete) - hard_success - soft_success),
            "hard_success_replacement_rate": (
                len(hard_success & soft_success) / len(hard_success) if hard_success else None
            ),
        }
    closures = [float(row["core_skill_kl_closure"]) for row in training]
    summary["teacher_forced"] = {
        "mean_core_skill_kl_closure": float(np.mean(closures)) if closures else 0.0,
        "median_core_skill_kl_closure": float(np.median(closures)) if closures else 0.0,
        "positive_closure_tasks": sum(value > 0 for value in closures),
        "fixed_training_steps": sorted({int(row["fixed_final_step"]) for row in training}),
    }
    return summary


def validate_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(resolve(args.manifest))
    if len(rows) != 61:
        raise ValueError(f"Expected the registered 61 successful trajectories, got {len(rows)}")
    identifiers = [str(row["id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Trajectory IDs are not unique")
    loader = SpreadsheetBenchDataLoader(
        split_dir=str(resolve(args.split_dir)),
        split_mode="split_dir",
        split_seed=42,
        data_root=str(resolve(args.data_root)),
        seed=args.seed,
    )
    train_items = loader.load_split_items(str(resolve(args.split_dir) / "train"))
    items_by_id = {str(item["id"]): item for item in train_items}
    missing = sorted(set(identifiers) - set(items_by_id))
    if missing:
        raise ValueError(f"Successful trajectory IDs absent from train split: {missing}")
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows, items_by_id


def main() -> None:
    args = parse_args()
    if args.prefix_length != 8:
        raise ValueError("This registered oracle experiment fixes prefix length at 8")
    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if args.preservation_weight < 0:
        raise ValueError("preservation-weight cannot be negative")
    conditions = [value.strip() for value in args.eval_conditions.split(",") if value.strip()]
    if len(conditions) != len(set(conditions)) or not set(conditions) <= {"soft", "hard", "plain"}:
        raise ValueError("eval-conditions must be a unique subset of soft,hard,plain")
    rows, items_by_id = validate_inputs(args)
    out_root = resolve(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    skill_path = resolve(args.skill_path)
    skill_text = skill_path.read_text(encoding="utf-8")
    model_source = resolve_model_reference(args.model_path)
    config = {
        **vars(args),
        "model_path": model_source,
        "initialization": "first-8 embeddings of full hard-Skill text",
        "manifest": str(resolve(args.manifest)),
        "manifest_sha256": sha256(resolve(args.manifest)),
        "skill_path": str(skill_path),
        "skill_sha256": sha256(skill_path),
        "trajectory_ids": [str(row["id"]) for row in rows],
        "scope": "train-success tasks only",
        "test_split_accessed": False,
    }
    atomic_json(out_root / "experiment_config.json", config)

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"Loading frozen Qwen: {model_source}", flush=True)
    prefix_model = SoftPrefixCausalLM(
        model_source,
        prefix_length=args.prefix_length,
        init_strategy="text",
        init_text=skill_text,
        torch_dtype="bfloat16",
        device="cuda",
    )
    prefix_model.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    prefix_model.model.train()
    base_prefix = prefix_model.prefix_embeddings.detach().clone()
    training: list[dict[str, Any]] = []
    if args.phase in {"train", "all"}:
        for index, row in enumerate(rows, 1):
            identifier = str(row["id"])
            print(f"[train {index}/{len(rows)}] task={identifier}", flush=True)
            summary = train_one_task(
                torch,
                prefix_model,
                row,
                base_prefix=base_prefix,
                args=args,
                task_dir=out_root / "training" / slug(identifier),
            )
            training.append(summary)
            print(
                f"  fixed_step={summary['fixed_final_step']} "
                f"core-KL-closure={100 * float(summary['core_skill_kl_closure']):.2f}%",
                flush=True,
            )
        atomic_jsonl(out_root / "training_results.jsonl", training)
    else:
        for row in rows:
            path = out_root / "training" / slug(str(row["id"])) / "training_summary.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing task-specific training result: {path}")
            training.append(json.loads(path.read_text(encoding="utf-8")))

    evaluation_rows: list[dict[str, Any]] = []
    if args.phase in {"eval", "all"}:
        prefix_model.model.gradient_checkpointing_disable()
        prefix_model.model.config.use_cache = True
        prefix_model.model.eval()
        prefix_model.prefix_embeddings.requires_grad_(False)
        for condition in conditions:
            successes = 0
            print(f"[eval] condition={condition} tasks={len(rows)}", flush=True)
            for index, row in enumerate(rows, 1):
                identifier = str(row["id"])
                checkpoint = out_root / "training" / slug(identifier) / "final_prefix.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                result = evaluate_one_condition(
                    torch,
                    prefix_model,
                    items_by_id[identifier],
                    condition=condition,
                    checkpoint_path=checkpoint,
                    skill_text=skill_text,
                    args=args,
                    task_out=out_root / "eval" / condition / "tasks" / slug(identifier),
                )
                result = {**result, "condition": condition}
                evaluation_rows.append(result)
                successes += int(bool(result.get("hard")))
                print(
                    f"  [{condition} {index}/{len(rows)}] task={identifier} "
                    f"hard={int(bool(result.get('hard')))} cumulative={successes}",
                    flush=True,
                )
            condition_rows = [row for row in evaluation_rows if row["condition"] == condition]
            atomic_jsonl(out_root / "eval" / condition / "results.jsonl", condition_rows)
        atomic_jsonl(out_root / "evaluation_results.jsonl", evaluation_rows)
    else:
        evaluation_path = out_root / "evaluation_results.jsonl"
        if evaluation_path.exists():
            evaluation_rows = read_jsonl(evaluation_path)

    summary = aggregate_results(evaluation_rows, training, conditions)
    summary.update(
        {
            "method": "task-specific-selective-skill-top64-distillation",
            "prefix_length": args.prefix_length,
            "initialization": config["initialization"],
            "skill_sha256": config["skill_sha256"],
            "source_successful_trajectories": len(rows),
        }
    )
    atomic_json(out_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
