#!/usr/bin/env python3
"""Task-specific SE-KD-Prefix oracle on the 61 successful SpreadsheetBench tasks.

Each task gets its own length-8 prefix trained only on that task's gold
trajectory with the *official* SE-KD objective: per-step student-entropy
``ceil(Top-k%)`` selection followed by full-vocabulary forward KL against the
frozen Qwen + hard-Skill teacher.  The prefix is then asked to solve the same
task by free generation.

This is the SE-KD counterpart of the Combined coverage ablation
(``train_spreadsheetbench_task_specific_prefixes.py``), which reached 28/35/33
of 61 at 5%/10%/20% core coverage.  It is an oracle/capacity probe, not a
generalization result: Val40 and Test280 are never loaded.

Declared differences from the Combined coverage ablation — SE-KD's native
objective has none of these, so they are properties of the method, not of this
harness:
  * no preservation term (the ablation matches 2,777 no-Skill positions);
  * no delta regularizer (the ablation adds 1e-4 * ||prefix - base||^2);
  * no gradient clipping (the official shared SE-KD run does not clip);
  * the selected core is recomputed from student entropy every step and
    therefore drifts, whereas the ablation freezes a Combined locator.

To keep the two comparable on one axis, teacher-forced closure is measured on a
*fixed* probe: the Combined Top-10% core positions from the coverage-ablation
manifest, scored against the cached hard-Skill Top-64 reference.  Training never
reads those positions.
"""
from __future__ import annotations

import argparse
import gc
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

from skillopt.envs.spreadsheetbench.dataloader import SpreadsheetBenchDataLoader
from skillopt.softprefix.model import SoftPrefixCausalLM
from skillopt.softprefix.official_distillation import (
    SEKD_COMMIT,
    SEKD_REPOSITORY,
    chunked_forward_kl,
    encode_trajectory,
    expected_topk_count,
    load_official_sekd,
    official_sekd_select_hidden,
    target_hidden_states,
    target_logits,
)
from skillopt.softprefix.prcb_v6 import topk_residual_kl_from_logits
from skillopt.softprefix.trainer import evaluate_spreadsheet_prefix
from scripts.train_spreadsheetbench_prcb_v1 import (
    atomic_json,
    atomic_jsonl,
    read_jsonl,
    resolve,
    resolve_model_reference,
    sha256,
    slug,
)
from scripts.train_spreadsheetbench_prcb_v6 import install_prefix, load_prefix
from scripts.train_spreadsheetbench_task_specific_prefixes import (
    PrefixDisabledGenerator,
    reference_arrays,
    reference_tensors,
)

DEFAULT_MODEL = os.environ.get("SPREADSHEETBENCH_MODEL", "Qwen/Qwen3.6-35B-A3B")
DEFAULT_PROBE_MANIFEST = (
    "outputs/SpreadsheetBench_selective_stage2_manifests/"
    "combined_top0.10_core_coverage_ablation.jsonl"
)
DEFAULT_OUT = "outputs/SpreadsheetBench_task_specific_sekd_len8_seed1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--probe-manifest",
        default=DEFAULT_PROBE_MANIFEST,
        help="Task list plus the FIXED Combined Top-10%% closure probe. Not a training input.",
    )
    parser.add_argument("--skill-path", default="ckpt/spreadsheetbench/gpt5.5_skill.md")
    parser.add_argument("--out-root", default=DEFAULT_OUT)
    parser.add_argument("--split-dir", default="data/spreadsheetbench_split")
    parser.add_argument("--data-root", default="data/spreadsheetbench_verified_400")
    parser.add_argument("--official-root", default="third_party/official")
    parser.add_argument("--phase", choices=("train", "eval", "all"), default="all")
    parser.add_argument(
        "--eval-conditions",
        default="soft",
        help="Subset of soft,hard,plain. The coverage ablation evaluated soft only; "
        "hard (32/61) and plain (26/61) are already recorded.",
    )
    # Training budget matches the Combined coverage ablation exactly.
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=32)
    # Official SE-KD hyper-parameters, from configs/spreadsheetbench/sekd_prefix_official.yaml.
    parser.add_argument("--k-percent", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--entropy-chunk-size", type=int, default=16)
    parser.add_argument("--kl-chunk-size", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=16384)
    parser.add_argument("--max-target-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--exec-timeout", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Smoke mode: first N tasks only.")
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def sekd_step_loss(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    example: Any,
    official: dict[str, Any],
    args: argparse.Namespace,
    teacher_hidden: Any,
) -> tuple[Any, dict[str, float]]:
    """One official SE-KD update signal for a single gold trajectory."""
    n_target = len(example.target_ids)

    # Match upstream's selective-lm-head path: one student transformer forward,
    # entropy selection from detached hidden states, and a differentiable lm_head
    # only on selected positions.  The frozen hard-Skill teacher hidden states
    # are cached once per task because neither their context nor weights change.
    student_hidden = target_hidden_states(
        prefix_model, example, use_prefix=True, with_grad=True
    )
    selected, entropy = official_sekd_select_hidden(
        hidden_states=student_hidden,
        lm_head=prefix_model.model.get_output_embeddings(),
        compute_student_entropy_and_select=official["compute_student_entropy_and_select"],
        k_percent=args.k_percent,
        chunk_size=args.entropy_chunk_size,
    )
    expected = expected_topk_count(n_target, args.k_percent)
    if int(selected.numel()) != expected:
        raise RuntimeError(
            f"SE-KD selected {int(selected.numel())} != {expected} for {example.task_id}"
        )
    mean_entropy = float(entropy.mean().cpu())
    lm_head = prefix_model.model.get_output_embeddings()
    with torch.inference_mode():
        teacher = lm_head(teacher_hidden[selected])
    student = lm_head(student_hidden[selected])
    loss = chunked_forward_kl(
        teacher_logits=teacher / args.temperature,
        student_logits=student / args.temperature,
        official_forward_kl=official["forward_kl"],
        chunk_size=args.kl_chunk_size,
    ) * (args.temperature**2)
    metrics = {
        "forward_kl": float(loss.detach().cpu()),
        "selected_tokens": int(selected.numel()),
        "target_tokens": n_target,
        "mean_student_entropy": mean_entropy,
    }
    del teacher, student, student_hidden, selected, entropy
    return loss, metrics


def probe_core_kl(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    example: Any,
    arrays: dict[str, np.ndarray],
    probe_positions: list[int],
    *,
    use_prefix: bool,
) -> float:
    """Mean hard-Skill Top-64 KL on the FIXED Combined Top-10% probe positions."""
    indices = torch.tensor(probe_positions, dtype=torch.long, device=prefix_model.device)
    logits = target_logits(prefix_model, example, indices, use_prefix=use_prefix, with_grad=False)
    skill_ids, skill_logp, skill_residual = reference_tensors(
        torch, arrays, probe_positions, logits.device, distribution="skill"
    )
    value = float(
        topk_residual_kl_from_logits(
            torch,
            logits,
            reference_topk_ids=skill_ids,
            reference_topk_logp=skill_logp,
            reference_residual_log_mass=skill_residual,
        )
        .mean()
        .detach()
        .cpu()
    )
    del logits, skill_ids, skill_logp, skill_residual
    gc.collect()
    torch.cuda.empty_cache()
    return value


def train_one_task(
    torch: Any,
    prefix_model: SoftPrefixCausalLM,
    example: Any,
    row: dict[str, Any],
    *,
    base_prefix: Any,
    official: dict[str, Any],
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

    arrays = reference_arrays(row, example.target_ids)
    # Two frozen probes, neither of which contributes any gradient.
    #   core10 : the Combined Top-10% positions -- directly comparable with the
    #            coverage-ablation arm's recorded closure, but NOT neutral, since
    #            that arm trained on exactly these positions and SE-KD did not.
    #   all    : every gold target position -- neutral ground; no method in the
    #            comparison is trained on this set as such.
    # The coverage ablation unions the EOS position into its core before measuring
    # closure, so the probe must do the same to stay numerically comparable.
    probe_core10 = sorted(
        {int(v) for v in row["selected_indices"]} | {len(example.target_ids) - 1}
    )
    probe_all = list(range(len(example.target_ids)))
    if not probe_core10 or max(probe_core10) >= len(example.target_ids):
        raise ValueError(f"Task {example.task_id} has an invalid Combined Top-10% probe")

    install_prefix(torch, prefix_model, base_prefix)
    prefix_model.prefix_embeddings.requires_grad_(True)
    # Official SE-KD optimizer: AdamW, no weight decay, no gradient clipping.
    optimizer = torch.optim.AdamW(
        [prefix_model.prefix_embeddings], lr=args.learning_rate, weight_decay=0.0
    )

    # Frozen teacher context and weights are constant across all 32 repeated
    # optimization steps for this task.  Caching hidden states is exact and
    # avoids one full 35B-model forward per step.
    teacher_hidden = target_hidden_states(
        prefix_model, example, use_prefix=False, with_grad=False
    )

    baseline_core10 = probe_core_kl(
        torch, prefix_model, example, arrays, probe_core10, use_prefix=True
    )
    baseline_all = probe_core_kl(
        torch, prefix_model, example, arrays, probe_all, use_prefix=True
    )
    history: list[dict[str, Any]] = []
    started = time.time()
    progress = tqdm(range(1, args.max_steps + 1), desc=f"SE-KD {example.task_id}", unit="step", leave=False)
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = sekd_step_loss(
            torch,
            prefix_model,
            example,
            official,
            args,
            teacher_hidden,
        )
        loss.backward()
        optimizer.step()
        history.append({"step": step, **metrics})
        progress.set_postfix(kl=f"{metrics['forward_kl']:.4f}", sel=metrics["selected_tokens"])
        del loss
        gc.collect()
        torch.cuda.empty_cache()
    progress.close()
    del teacher_hidden

    final_core10 = probe_core_kl(
        torch, prefix_model, example, arrays, probe_core10, use_prefix=True
    )
    final_all = probe_core_kl(
        torch, prefix_model, example, arrays, probe_all, use_prefix=True
    )
    final_prefix = prefix_model.prefix_embeddings.detach().clone()
    torch.save(
        {
            "prefix_embeddings": final_prefix.cpu(),
            "prefix_length": args.prefix_length,
            "task_id": example.task_id,
            "step": args.max_steps,
        },
        checkpoint_path,
    )
    summary = {
        "complete": True,
        "task_id": example.task_id,
        "objective": "official SE-KD student-entropy top-k% + full-vocabulary forward KL",
        "trajectory_target_tokens": len(example.target_ids),
        "selected_tokens_per_step": history[-1]["selected_tokens"],
        "selected_fraction": history[-1]["selected_tokens"] / max(len(example.target_ids), 1),
        "steps_executed": args.max_steps,
        "fixed_final_step": args.max_steps,
        "first_step_forward_kl": history[0]["forward_kl"],
        "final_step_forward_kl": history[-1]["forward_kl"],
        "mean_student_entropy": history[-1]["mean_student_entropy"],
        "probe_core10": {
            "positions": len(probe_core10),
            "source": "fixed Combined Top-10% coverage-ablation core (NOT neutral)",
            "baseline_core_skill_kl": baseline_core10,
            "final_core_skill_kl": final_core10,
        },
        "probe_all": {
            "positions": len(probe_all),
            "source": "all gold target positions (neutral)",
            "baseline_core_skill_kl": baseline_all,
            "final_core_skill_kl": final_all,
        },
        # Same definition as the coverage ablation's core_skill_kl_closure, so the
        # core10 value is directly comparable with that arm's recorded 56.86%.
        "core_skill_kl_closure": (
            (baseline_core10 - final_core10) / baseline_core10 if baseline_core10 > 0 else 0.0
        ),
        "all_position_skill_kl_closure": (
            (baseline_all - final_all) / baseline_all if baseline_all > 0 else 0.0
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "history": history,
        "wall_time_s": round(time.time() - started, 1),
        "test_split_accessed": False,
    }
    atomic_json(summary_path, summary)
    return summary


def evaluate_one(
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
    if result_path.exists():
        rows = read_jsonl(result_path)
        if len(rows) != 1:
            raise ValueError(f"Expected one result in {result_path}, found {len(rows)}")
        return rows[0]
    if condition == "soft":
        prefix = load_prefix(
            torch, checkpoint_path, prefix_model.device, prefix_model.prefix_embeddings.dtype
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


def main() -> None:
    args = parse_args()
    if args.prefix_length != 8:
        raise ValueError("This oracle experiment fixes prefix length at 8")
    conditions = [v.strip() for v in args.eval_conditions.split(",") if v.strip()]
    if len(conditions) != len(set(conditions)) or not set(conditions) <= {"soft", "hard", "plain"}:
        raise ValueError("eval-conditions must be a unique subset of soft,hard,plain")

    rows = read_jsonl(resolve(args.probe_manifest))
    if len(rows) != 61:
        raise ValueError(f"Expected the registered 61 successful trajectories, got {len(rows)}")
    identifiers = [str(r["id"]) for r in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Trajectory IDs are not unique")
    if args.limit > 0:
        rows = rows[: args.limit]

    split_dir = resolve(args.split_dir)
    loader = SpreadsheetBenchDataLoader(
        split_dir=str(split_dir),
        split_mode="split_dir",
        split_seed=42,
        data_root=str(resolve(args.data_root)),
        seed=args.seed,
    )
    items_by_id = {
        str(i["id"]): i for i in loader.load_split_items(str(split_dir / "train"))
    }
    missing = sorted({str(r["id"]) for r in rows} - set(items_by_id))
    if missing:
        raise ValueError(f"Trajectory IDs absent from train split: {missing}")

    out_root = resolve(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    skill_path = resolve(args.skill_path)
    skill_text = skill_path.read_text(encoding="utf-8")
    model_source = resolve_model_reference(args.model_path)
    official = load_official_sekd(resolve(args.official_root))

    atomic_json(
        out_root / "experiment_config.json",
        {
            **vars(args),
            "model_path": model_source,
            "method": "task-specific SE-KD-Prefix oracle",
            "initialization": "first-8 embeddings of full hard-Skill text",
            "probe_manifest_sha256": sha256(resolve(args.probe_manifest)),
            "skill_sha256": sha256(skill_path),
            "official": {"repository": SEKD_REPOSITORY, "commit": SEKD_COMMIT},
            "execution_optimization": (
                "exact hidden-state implementation: cache the frozen hard-Skill teacher "
                "states once per task and reuse the same student transformer forward for "
                "official entropy selection and selected-position full-vocabulary KL"
            ),
            "declared_differences_vs_combined_ablation": [
                "no preservation term",
                "no delta regularizer",
                "no gradient clipping",
                "core recomputed from student entropy every step (drifts)",
            ],
            "trajectory_ids": [str(r["id"]) for r in rows],
            "test_split_accessed": False,
        },
    )

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

    examples = {}
    for row in tqdm(rows, desc="Encode", unit="ex"):
        example = encode_trajectory(
            tokenizer=prefix_model.tokenizer,
            row=row,
            skill_text=skill_text,
            max_prompt_tokens=args.max_prompt_tokens,
            max_target_tokens=args.max_target_tokens,
        )
        if not example.score_cache:
            raise ValueError(f"Trajectory {example.task_id} lacks score_cache")
        examples[str(row["id"])] = example

    training: list[dict[str, Any]] = []
    if args.phase in {"train", "all"}:
        for index, row in enumerate(rows, 1):
            identifier = str(row["id"])
            print(f"[train {index}/{len(rows)}] task={identifier}", flush=True)
            summary = train_one_task(
                torch,
                prefix_model,
                examples[identifier],
                row,
                base_prefix=base_prefix,
                official=official,
                args=args,
                task_dir=out_root / "training" / slug(identifier),
            )
            training.append(summary)
            print(
                f"  KL {summary['first_step_forward_kl']:.4f} -> "
                f"{summary['final_step_forward_kl']:.4f} | "
                f"probe closure={100 * float(summary['core_skill_kl_closure']):.2f}%",
                flush=True,
            )
        atomic_jsonl(out_root / "training_results.jsonl", training)
    else:
        for row in rows:
            path = out_root / "training" / slug(str(row["id"])) / "training_summary.json"
            if not path.exists():
                raise FileNotFoundError(path)
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
                result = evaluate_one(
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
            atomic_jsonl(
                out_root / "eval" / condition / "results.jsonl",
                [r for r in evaluation_rows if r["condition"] == condition],
            )
        atomic_jsonl(out_root / "evaluation_results.jsonl", evaluation_rows)
    else:
        path = out_root / "evaluation_results.jsonl"
        if path.exists():
            evaluation_rows = read_jsonl(path)

    ids = sorted({str(r["task_id"]) for r in training})
    closures = [float(r["core_skill_kl_closure"]) for r in training]
    summary: dict[str, Any] = {
        "scope": "same-task oracle on 61 successful training trajectories",
        "method": "task-specific-official-sekd-prefix",
        "tasks": len(ids),
        "conditions": {},
        "test_split_accessed": False,
        "prefix_length": args.prefix_length,
        "initialization": "first-8 embeddings of full hard-Skill text",
        "skill_sha256": sha256(skill_path),
        "official": {"repository": SEKD_REPOSITORY, "commit": SEKD_COMMIT},
        "k_percent": args.k_percent,
        "source_successful_trajectories": len(rows),
    }
    for condition in conditions:
        present = {
            str(r["id"]): r for r in evaluation_rows if str(r.get("condition")) == condition
        }
        successes = sum(bool(present[i].get("hard")) for i in ids if i in present)
        summary["conditions"][condition] = {
            "evaluated": sum(i in present for i in ids),
            "successes": successes,
            "hard_rate": successes / len(ids) if ids else 0.0,
            "soft_case_rate": (
                sum(float(present[i].get("soft", 0.0)) for i in ids if i in present) / len(ids)
                if ids
                else 0.0
            ),
        }
    closures_all = [float(r["all_position_skill_kl_closure"]) for r in training]
    summary["teacher_forced"] = {
        "probe_core10": "fixed Combined Top-10% core, hard-Skill Top-64 reference "
        "(comparable with the coverage ablation, but Combined trained on these positions)",
        "mean_core_skill_kl_closure": float(np.mean(closures)) if closures else 0.0,
        "median_core_skill_kl_closure": float(np.median(closures)) if closures else 0.0,
        "positive_closure_tasks": sum(v > 0 for v in closures),
        "probe_all": "all gold target positions, hard-Skill Top-64 reference (neutral)",
        "mean_all_position_skill_kl_closure": float(np.mean(closures_all)) if closures_all else 0.0,
        "median_all_position_skill_kl_closure": (
            float(np.median(closures_all)) if closures_all else 0.0
        ),
        "positive_all_position_closure_tasks": sum(v > 0 for v in closures_all),
        "fixed_training_steps": sorted({int(r["fixed_final_step"]) for r in training}),
    }
    summary["mean_selected_fraction"] = float(
        np.mean([float(r["selected_fraction"]) for r in training])
    ) if training else 0.0
    atomic_json(out_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
