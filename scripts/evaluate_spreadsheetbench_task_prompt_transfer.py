#!/usr/bin/env python3
"""Stage-C held-out task-prompt transfer pilot for SpreadsheetBench.

The pilot freezes one Stage-B outer fold. Prompts for its held-out tasks are
constructed using only the other tasks' task-specific Combined-10% prompts:
shared prompt, train-prompt mean, nearest-neighbor retrieval, Top-3 attentive
mixture, and a fold-local PCA-ridge low-rank generator. The held-out task's own
prompt is never used to construct a candidate.

This is a Train80-internal generalization experiment. Val40 and Test280 are
never loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_spreadsheetbench_task_prompt_manifold import (  # noqa: E402
    add_bias,
    checkpoint_matrix,
    choose_ridge_alpha,
    read_jsonl,
    ridge_predict,
    stratified_folds,
)
from scripts.analyze_spreadsheetbench_task_representations import normalize_train_test  # noqa: E402
from scripts.train_spreadsheetbench_prcb_v1 import atomic_json, atomic_jsonl, resolve_model_reference, sha256, slug  # noqa: E402
from scripts.train_spreadsheetbench_task_specific_prefixes import evaluate_one_condition  # noqa: E402
from skillopt.envs.spreadsheetbench.dataloader import SpreadsheetBenchDataLoader  # noqa: E402
from skillopt.softprefix.model import SoftPrefixCausalLM  # noqa: E402


DEFAULT_TASK_ROOT = PROJECT_ROOT / "outputs/SpreadsheetBench_task_specific_combined_core10_len8_seed1_coverage_ablation"
DEFAULT_REP_ROOT = PROJECT_ROOT / "outputs/SpreadsheetBench_task_representation_stage_b"
DEFAULT_SHARED = PROJECT_ROOT / "outputs/SpreadsheetBench_combined_core10_coverage_shared_preserve_len8_seed1/best_prefix.pt"
DEFAULT_OUT = PROJECT_ROOT / "outputs/SpreadsheetBench_task_prompt_transfer_stage_c_fold1"
DEFAULT_MODEL = PROJECT_ROOT.parent / "model_cache/huggingface/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--representation-root", type=Path, default=DEFAULT_REP_ROOT)
    parser.add_argument("--shared-checkpoint", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get("SPREADSHEETBENCH_MODEL", DEFAULT_MODEL)))
    parser.add_argument("--skill-path", type=Path, default=PROJECT_ROOT / "ckpt/spreadsheetbench/gpt5.5_skill.md")
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "data/spreadsheetbench_split")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/spreadsheetbench_verified_400")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=24018)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--mixture-topk", type=int, default=3)
    parser.add_argument("--mixture-temperature", type=float, default=0.1)
    parser.add_argument(
        "--conditions",
        default="shared,mean49,nn_task_spec,top3_task_spec,ridge_task_spec,nn_instruction",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--exec-timeout", type=int, default=600)
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def save_prefix(path: Path, array: np.ndarray, *, task_id: str, condition: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prefix_embeddings": torch.from_numpy(array).to(torch.bfloat16),
            "prefix_length": int(array.shape[0]),
            "task_id": task_id,
            "condition": condition,
        },
        path,
    )


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = values / temperature
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    return exp / np.sum(exp)


def prepare_candidates(args: argparse.Namespace) -> tuple[list[str], dict[str, dict[str, Path]], dict[str, Any]]:
    task_root = args.task_root.resolve()
    rep_root = args.representation_root.resolve()
    specs = read_jsonl(rep_root / "task_specs.jsonl")
    task_ids = [str(row["id"]) for row in specs]
    checkpoint_paths = [task_root / "training" / task_id / "final_prefix.pt" for task_id in task_ids]
    prompts_flat, _ = checkpoint_matrix(checkpoint_paths)
    prompts_flat = prompts_flat.astype(np.float64)
    prompt_shape = tuple(torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)["prefix_embeddings"].shape)
    evaluations = {
        str(row["id"]): row
        for row in read_jsonl(task_root / "evaluation_results.jsonl")
        if row.get("condition") == "soft"
    }
    successes = np.asarray([int(bool(evaluations[task_id]["hard"])) for task_id in task_ids])
    strata = [f"{row['instruction_type']}|success={successes[idx]}" for idx, row in enumerate(specs)]
    folds = stratified_folds(strata, args.folds, args.seed)
    if not 0 <= args.fold < len(folds):
        raise ValueError(f"fold must be in [0,{len(folds) - 1}]")
    test_indices = folds[args.fold]
    all_indices = set(range(len(task_ids)))
    train_indices = sorted(all_indices - set(test_indices))

    embedding_payload = np.load(rep_root / "qwen_task_embeddings.npz")
    embedding_ids = [str(value) for value in embedding_payload["task_ids"].tolist()]
    if embedding_ids != task_ids:
        raise RuntimeError("TaskSpec and Qwen embedding task order differ")
    instruction_embedding = embedding_payload["qwen_instruction_last"].astype(np.float64)
    task_embedding = embedding_payload["qwen_task_spec_last"].astype(np.float64)
    instruction_train, instruction_test = normalize_train_test(
        instruction_embedding[train_indices], instruction_embedding[test_indices]
    )
    task_train, task_test = normalize_train_test(task_embedding[train_indices], task_embedding[test_indices])

    train_prompts = prompts_flat[train_indices]
    fold_mean = np.mean(train_prompts, axis=0, keepdims=True)
    centered_train = train_prompts - fold_mean
    _, _, vt = np.linalg.svd(centered_train, full_matrices=False)
    n_components = min(args.components, len(train_indices) - 1)
    basis = vt[:n_components]
    y_train = centered_train @ basis.T
    ridge_alpha = choose_ridge_alpha(
        task_train,
        y_train,
        [strata[index] for index in train_indices],
        seed=args.seed + args.fold,
    )
    ridge_prediction = ridge_predict(add_bias(task_train), y_train, add_bias(task_test), ridge_alpha)
    ridge_prompts = fold_mean + ridge_prediction @ basis

    task_similarity = task_test @ task_train.T
    instruction_similarity = instruction_test @ instruction_train.T
    nearest_task = np.argmax(task_similarity, axis=1)
    nearest_instruction = np.argmax(instruction_similarity, axis=1)

    shared_payload = torch.load(args.shared_checkpoint.resolve(), map_location="cpu", weights_only=False)
    shared_prompt = shared_payload["prefix_embeddings"].float().numpy().reshape(-1).astype(np.float64)
    condition_paths: dict[str, dict[str, Path]] = {}
    provenance_rows: list[dict[str, Any]] = []
    for local_idx, test_idx in enumerate(test_indices):
        task_id = task_ids[test_idx]
        top_order = np.argsort(task_similarity[local_idx])[::-1][: args.mixture_topk]
        weights = softmax(task_similarity[local_idx, top_order], args.mixture_temperature)
        mixture = np.sum(train_prompts[top_order] * weights[:, None], axis=0)
        arrays = {
            "shared": shared_prompt,
            "mean49": fold_mean[0],
            "nn_task_spec": train_prompts[nearest_task[local_idx]],
            "top3_task_spec": mixture,
            "ridge_task_spec": ridge_prompts[local_idx],
            "nn_instruction": train_prompts[nearest_instruction[local_idx]],
        }
        condition_paths[task_id] = {}
        for condition, flat in arrays.items():
            checkpoint = args.out_root.resolve() / "candidates" / condition / task_id / "prefix.pt"
            save_prefix(checkpoint, flat.reshape(prompt_shape).astype(np.float32), task_id=task_id, condition=condition)
            condition_paths[task_id][condition] = checkpoint
        provenance_rows.append(
            {
                "id": task_id,
                "fold": args.fold,
                "train_task_count": len(train_indices),
                "held_out": True,
                "nn_task_spec_id": task_ids[train_indices[int(nearest_task[local_idx])]],
                "nn_task_spec_similarity": float(task_similarity[local_idx, nearest_task[local_idx]]),
                "nn_instruction_id": task_ids[train_indices[int(nearest_instruction[local_idx])]],
                "nn_instruction_similarity": float(
                    instruction_similarity[local_idx, nearest_instruction[local_idx]]
                ),
                "top3_task_spec_ids": [task_ids[train_indices[int(value)]] for value in top_order],
                "top3_task_spec_similarities": [float(task_similarity[local_idx, value]) for value in top_order],
                "top3_task_spec_weights": weights.tolist(),
                "ridge_components": n_components,
                "ridge_alpha": ridge_alpha,
                "candidate_norms": {
                    condition: float(np.linalg.norm(flat - shared_prompt)) for condition, flat in arrays.items()
                },
            }
        )
    atomic_jsonl(args.out_root.resolve() / "candidate_provenance.jsonl", provenance_rows)
    manifest = {
        "stage": "C_task_prompt_transfer_pilot",
        "scope": "Train80 successful-trajectory tasks only",
        "outer_fold": args.fold,
        "folds": args.folds,
        "seed": args.seed,
        "train_ids": [task_ids[index] for index in train_indices],
        "held_out_ids": [task_ids[index] for index in test_indices],
        "conditions": list(next(iter(condition_paths.values())).keys()),
        "components": n_components,
        "ridge_alpha": ridge_alpha,
        "mixture_topk": args.mixture_topk,
        "mixture_temperature": args.mixture_temperature,
        "shared_checkpoint": str(args.shared_checkpoint.resolve()),
        "shared_checkpoint_sha256": sha256(args.shared_checkpoint.resolve()),
        "held_out_oracle_prompts_used_for_candidate_construction": False,
        "val_or_test_accessed": False,
    }
    atomic_json(args.out_root.resolve() / "candidate_manifest.json", manifest)
    return manifest["held_out_ids"], condition_paths, manifest


def cached_baselines(task_ids: list[str], task_root: Path) -> dict[str, Any]:
    oracle_rows = {
        str(row["id"]): row
        for row in read_jsonl(task_root / "evaluation_results.jsonl")
        if row.get("condition") == "soft"
    }
    reference_root = PROJECT_ROOT / "outputs/SpreadsheetBench_task_specific_selective_skillkl_len8_seed1/eval"
    baseline = {}
    for condition in ("plain", "hard"):
        rows = {str(row["id"]): row for row in read_jsonl(reference_root / condition / "results.jsonl")}
        baseline[condition] = {
            "successes": sum(bool(rows[task_id].get("hard")) for task_id in task_ids),
            "evaluated": len(task_ids),
            "source": str(reference_root / condition / "results.jsonl"),
        }
    baseline["same_task_oracle"] = {
        "successes": sum(bool(oracle_rows[task_id].get("hard")) for task_id in task_ids),
        "evaluated": len(task_ids),
        "source": str(task_root / "evaluation_results.jsonl"),
        "note": "Upper bound only; held-out oracle prompts were not used to construct candidates.",
    }
    return baseline


def main() -> None:
    args = parse_args()
    args.out_root.resolve().mkdir(parents=True, exist_ok=True)
    task_ids, condition_paths, manifest = prepare_candidates(args)
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    unknown = set(conditions) - set(manifest["conditions"])
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    if args.prepare_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return

    loader = SpreadsheetBenchDataLoader(
        split_dir=str(args.split_dir.resolve()),
        split_mode="split_dir",
        split_seed=42,
        data_root=str(args.data_root.resolve()),
        seed=args.seed,
    )
    items = loader.load_split_items(str(args.split_dir.resolve() / "train"))
    items_by_id = {str(item["id"]): item for item in items}
    if any(task_id not in items_by_id for task_id in task_ids):
        raise RuntimeError("At least one held-out pilot task is absent from Train80")

    model_source = resolve_model_reference(str(args.model_path.resolve()))
    skill_text = args.skill_path.resolve().read_text(encoding="utf-8")
    print(f"Loading frozen Qwen: {model_source}", flush=True)
    model = SoftPrefixCausalLM(
        model_source,
        prefix_length=8,
        init_strategy="text",
        init_text=skill_text,
        torch_dtype="bfloat16",
        device="cuda",
    )
    model.model.eval()
    model.model.config.use_cache = True
    model.prefix_embeddings.requires_grad_(False)
    eval_args = SimpleNamespace(
        force_eval=args.force_eval,
        data_root=str(args.data_root.resolve()),
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        exec_timeout=args.exec_timeout,
    )
    all_results: list[dict[str, Any]] = []
    for condition in conditions:
        successes = 0
        print(f"[condition] {condition}: {len(task_ids)} held-out tasks", flush=True)
        condition_results: list[dict[str, Any]] = []
        for index, task_id in enumerate(task_ids, 1):
            result = evaluate_one_condition(
                torch,
                model,
                items_by_id[task_id],
                condition="soft",
                checkpoint_path=condition_paths[task_id][condition],
                skill_text=skill_text,
                args=eval_args,
                task_out=args.out_root.resolve() / "eval" / condition / "tasks" / slug(task_id),
            )
            result = {**result, "condition": condition}
            condition_results.append(result)
            all_results.append(result)
            successes += int(bool(result.get("hard")))
            print(
                f"  [{condition} {index}/{len(task_ids)}] task={task_id} "
                f"hard={int(bool(result.get('hard')))} cumulative={successes}",
                flush=True,
            )
        atomic_jsonl(args.out_root.resolve() / "eval" / condition / "results.jsonl", condition_results)
    atomic_jsonl(args.out_root.resolve() / "evaluation_results.jsonl", all_results)

    by_condition = {
        condition: {str(row["id"]): row for row in all_results if row["condition"] == condition}
        for condition in conditions
    }
    shared_success = {
        task_id for task_id, row in by_condition.get("shared", {}).items() if bool(row.get("hard"))
    }
    condition_summary = {}
    for condition in conditions:
        success_ids = {task_id for task_id, row in by_condition[condition].items() if bool(row.get("hard"))}
        condition_summary[condition] = {
            "evaluated": len(task_ids),
            "successes": len(success_ids),
            "hard_rate": len(success_ids) / len(task_ids),
            "soft_case_rate": sum(float(by_condition[condition][task_id].get("soft", 0.0)) for task_id in task_ids)
            / len(task_ids),
            "gains_vs_shared": sorted(success_ids - shared_success) if condition != "shared" else [],
            "losses_vs_shared": sorted(shared_success - success_ids) if condition != "shared" else [],
        }
    summary = {
        **manifest,
        "conditions": condition_summary,
        "cached_baselines_on_same_held_out_tasks": cached_baselines(task_ids, args.task_root.resolve()),
        "gate": {
            "criterion": "A transfer method must have more paired gains than losses versus shared before full 5-fold evaluation.",
            "passing_conditions": [
                condition
                for condition, row in condition_summary.items()
                if condition != "shared" and len(row["gains_vs_shared"]) > len(row["losses_vs_shared"])
            ],
        },
        "val_or_test_accessed": False,
    }
    atomic_json(args.out_root.resolve() / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
