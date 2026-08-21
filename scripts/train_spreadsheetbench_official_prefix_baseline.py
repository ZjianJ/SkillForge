#!/usr/bin/env python3
"""Train Official-Adapted SE-KD-Prefix or OPCD-Prefix on SpreadsheetBench."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skillopt.config import flatten_config, is_structured, load_config  # noqa: E402
from skillopt.softprefix.official_distillation import (  # noqa: E402
    OPCD_COMMIT,
    OPCD_REPOSITORY,
    SEKD_COMMIT,
    SEKD_REPOSITORY,
    chunked_forward_kl,
    chunked_opcd_reverse_kl,
    encode_on_policy_response,
    encode_trajectory,
    expected_topk_count,
    gather_log_probs,
    generation_mode,
    load_official_opcd_kl,
    load_official_sekd,
    official_sekd_select,
    student_topk_support,
    target_logits,
    verify_official_sources,
)
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


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "messages" not in row or "target" not in row:
                raise ValueError(f"{path}:{line_no} lacks messages/target")
            rows.append(row)
    return rows


def _empty_cuda_cache(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_checkpoint(torch, prefix_model, path: Path) -> None:
    torch.save(prefix_model.state_dict(), path)


def _prepare_examples(rows, prefix_model, settings, skill_text):
    examples = [
        encode_trajectory(
            tokenizer=prefix_model.tokenizer,
            row=row,
            skill_text=skill_text,
            max_prompt_tokens=settings.max_prompt_tokens,
            max_target_tokens=settings.max_target_tokens,
        )
        for row in tqdm(rows, desc="Encode train61", unit="ex")
    ]
    for row, example in zip(rows, examples, strict=True):
        if not example.score_cache:
            raise ValueError(f"Trajectory {example.task_id} lacks score_cache")
        import numpy as np

        with np.load(example.score_cache) as cached:
            cached_ids = cached["target_ids"].astype(np.int64).tolist()
        if cached_ids != example.target_ids:
            raise ValueError(f"Tokenizer/cache mismatch for {example.task_id}")
    return examples


def _run_sekd(prefix_model, examples, optimizer, baseline_cfg, accumulation, seed):
    torch = prefix_model.torch
    official = load_official_sekd(baseline_cfg["official_root"])
    k_percent = float(baseline_cfg.get("k_percent", 20.0))
    temperature = float(baseline_cfg.get("temperature", 1.0))
    entropy_chunk = int(baseline_cfg.get("entropy_chunk_size", 16))
    kl_chunk = int(baseline_cfg.get("kl_chunk_size", 16))
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    token_loss_sum = 0.0
    selected_total = 0
    entropy_sum = 0.0
    target_total = 0
    started = time.time()

    progress = tqdm(total=len(order), desc="SE-KD-Prefix train", unit="traj")
    for group_start in range(0, len(order), accumulation):
        group_ids = order[group_start : group_start + accumulation]
        selected_records = []
        for index in group_ids:
            example = examples[index]
            all_indices = torch.arange(len(example.target_ids), dtype=torch.long, device=prefix_model.device)
            student_entropy_logits = target_logits(prefix_model, example, all_indices, use_prefix=True, with_grad=False)
            selected, entropy = official_sekd_select(
                logits=student_entropy_logits / temperature,
                compute_student_entropy_and_select=official["compute_student_entropy_and_select"],
                k_percent=k_percent,
                chunk_size=entropy_chunk,
            )
            expected = expected_topk_count(len(example.target_ids), k_percent)
            if int(selected.numel()) != expected:
                raise RuntimeError(f"SE-KD selected {int(selected.numel())} != {expected} for {example.task_id}")
            selected_records.append((example, selected, entropy))
            del student_entropy_logits
            _empty_cuda_cache(torch)

        group_selected = sum(int(record[1].numel()) for record in selected_records)
        optimizer.zero_grad(set_to_none=True)
        group_loss = 0.0
        for example, selected, entropy in selected_records:
            teacher_logits = target_logits(prefix_model, example, selected, use_prefix=False, with_grad=False)
            student_logits = target_logits(prefix_model, example, selected, use_prefix=True, with_grad=True)
            loss = chunked_forward_kl(
                teacher_logits=teacher_logits / temperature,
                student_logits=student_logits / temperature,
                official_forward_kl=official["forward_kl"],
                chunk_size=kl_chunk,
            ) * (temperature**2)
            weight = int(selected.numel()) / max(group_selected, 1)
            (loss * weight).backward()
            value = float(loss.detach().cpu())
            group_loss += value * weight
            token_loss_sum += value * int(selected.numel())
            selected_total += int(selected.numel())
            target_total += len(example.target_ids)
            entropy_sum += float(entropy.sum().cpu())
            progress.update(1)
            progress.set_postfix(loss=f"{group_loss:.4f}", selected=selected_total)
            del teacher_logits, student_logits, loss, entropy
            _empty_cuda_cache(torch)
        optimizer.step()
        optimizer_steps += 1
        history.append(
            {
                "optimizer_step": optimizer_steps,
                "trajectory_ids": [record[0].task_id for record in selected_records],
                "loss": group_loss,
                "selected_tokens": group_selected,
            }
        )
    progress.close()
    return {
        "method": "SE-KD-Prefix",
        "optimizer_steps": optimizer_steps,
        "trajectories": len(order),
        "target_tokens": target_total,
        "selected_tokens": selected_total,
        "selected_fraction": selected_total / max(target_total, 1),
        "mean_forward_kl": token_loss_sum / max(selected_total, 1),
        "mean_student_entropy": entropy_sum / max(target_total, 1),
        "wall_time_s": round(time.time() - started, 1),
        "history": history,
        "official": {
            "repository": SEKD_REPOSITORY,
            "commit": SEKD_COMMIT,
            "selection": f"student entropy top {k_percent:g}% per trajectory (ceil)",
            "objective": "full-vocabulary forward KL",
            "temperature": temperature,
        },
    }


def _rollout(prefix_model, example, baseline_cfg):
    with generation_mode(prefix_model):
        return prefix_model.generate_from_prompt(
            example.clean_prompt,
            max_prompt_tokens=len(example.clean_prompt_ids),
            max_new_tokens=int(baseline_cfg.get("rollout_max_new_tokens", 4096)),
            temperature=float(baseline_cfg.get("rollout_temperature", 1.0)),
            use_prefix=True,
        )


def _run_opcd(prefix_model, examples, optimizer, baseline_cfg, accumulation, seed, out_root):
    torch = prefix_model.torch
    official = load_official_opcd_kl(baseline_cfg["official_root"])
    topk = int(baseline_cfg.get("kl_topk", 256))
    renorm = bool(baseline_cfg.get("kl_renorm_topk", False))
    kl_chunk = int(baseline_cfg.get("kl_chunk_size", 8))
    max_target = int(baseline_cfg.get("rollout_max_new_tokens", 4096))
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    history: list[dict[str, Any]] = []
    rollout_path = out_root / "on_policy_rollouts.jsonl"
    rollout_path.write_text("", encoding="utf-8")
    optimizer_steps = 0
    token_loss_sum = 0.0
    token_total = 0
    student_mass_sum = 0.0
    started = time.time()

    progress = tqdm(total=len(order), desc="OPCD-Prefix train", unit="rollout")
    for group_start in range(0, len(order), accumulation):
        group_ids = order[group_start : group_start + accumulation]
        group = []
        for index in group_ids:
            source = examples[index]
            response = _rollout(prefix_model, source, baseline_cfg)
            generated = encode_on_policy_response(
                source,
                tokenizer=prefix_model.tokenizer,
                response=response,
                max_target_tokens=max_target,
            )
            all_indices = torch.arange(len(generated.target_ids), dtype=torch.long, device=prefix_model.device)
            student_reference_logits = target_logits(
                prefix_model, generated, all_indices, use_prefix=True, with_grad=False
            )
            topk_indices, _, student_mass = student_topk_support(student_reference_logits, k=topk)
            teacher_logits = target_logits(prefix_model, generated, all_indices, use_prefix=False, with_grad=False)
            teacher_topk_logp = gather_log_probs(teacher_logits, topk_indices).to(torch.bfloat16)
            group.append((generated, topk_indices, teacher_topk_logp, student_mass))
            with rollout_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "id": source.task_id,
                            "optimizer_step_before_update": optimizer_steps,
                            "response": response,
                            "target_tokens": len(generated.target_ids),
                            "student_topk_mass_mean": float(student_mass.mean().cpu()),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            del student_reference_logits, teacher_logits
            _empty_cuda_cache(torch)

        group_tokens = sum(len(record[0].target_ids) for record in group)
        optimizer.zero_grad(set_to_none=True)
        group_loss = 0.0
        for generated, topk_indices, teacher_topk_logp, student_mass in group:
            all_indices = torch.arange(len(generated.target_ids), dtype=torch.long, device=prefix_model.device)
            student_logits = target_logits(prefix_model, generated, all_indices, use_prefix=True, with_grad=True)
            loss = chunked_opcd_reverse_kl(
                student_logits=student_logits,
                teacher_topk_logp=teacher_topk_logp,
                topk_indices=topk_indices,
                official_kl_penalty=official["kl_penalty"],
                chunk_size=kl_chunk,
                renorm_topk=renorm,
            )
            count = len(generated.target_ids)
            weight = count / max(group_tokens, 1)
            (loss * weight).backward()
            value = float(loss.detach().cpu())
            group_loss += value * weight
            token_loss_sum += value * count
            token_total += count
            student_mass_sum += float(student_mass.sum().cpu())
            progress.update(1)
            progress.set_postfix(loss=f"{group_loss:.4f}", tokens=token_total)
            del student_logits, loss, topk_indices, teacher_topk_logp, student_mass
            _empty_cuda_cache(torch)
        optimizer.step()
        optimizer_steps += 1
        history.append(
            {
                "optimizer_step": optimizer_steps,
                "trajectory_ids": [record[0].task_id for record in group],
                "loss": group_loss,
                "response_tokens": group_tokens,
            }
        )
    progress.close()
    return {
        "method": "OPCD-Prefix",
        "optimizer_steps": optimizer_steps,
        "trajectories": len(order),
        "on_policy_tokens": token_total,
        "mean_reverse_kl": token_loss_sum / max(token_total, 1),
        "mean_student_topk_mass": student_mass_sum / max(token_total, 1),
        "wall_time_s": round(time.time() - started, 1),
        "history": history,
        "rollout_path": str(rollout_path),
        "official": {
            "repository": OPCD_REPOSITORY,
            "commit": OPCD_COMMIT,
            "state_distribution": "student on-policy responses",
            "objective": "student Top-256 non-renormalized reverse KL",
            "kl_topk": topk,
            "kl_renorm_topk": renorm,
            "rollout_temperature": float(baseline_cfg.get("rollout_temperature", 1.0)),
            "source_file": official["source_file"],
            "source_lineno": official["source_lineno"],
        },
    }


def main() -> None:
    args = parse_args()
    raw = load_config(args.config, overrides=args.cfg_options)
    flat = flatten_config(raw) if is_structured(raw) else dict(raw)
    soft_cfg = dict(raw.get("soft_prefix", {}))
    baseline_cfg = dict(raw.get("official_baseline", {}))
    method = str(baseline_cfg.get("method", "")).strip().lower()
    if method not in {"sekd", "opcd"}:
        raise ValueError("official_baseline.method must be sekd or opcd")
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

    official_root = Path(str(baseline_cfg.get("official_root", "third_party/official")))
    if not official_root.is_absolute():
        official_root = PROJECT_ROOT / official_root
    baseline_cfg["official_root"] = str(official_root.resolve())
    source_paths = verify_official_sources(baseline_cfg["official_root"])

    settings = SoftPrefixSettings.from_dict(soft_cfg)
    init_text = _load_init_text(settings.init_text_path or str(flat.get("skill_init", "")))
    if not init_text.strip():
        raise ValueError("Official prefix baselines require the non-empty hard Skill text")
    prefix_model = _build_prefix_model("spreadsheetbench", settings, init_text)
    torch = prefix_model.torch

    rows = _load_rows(settings.trajectory_examples_path)
    max_examples = int(baseline_cfg.get("max_train_examples", 0) or 0)
    if max_examples > 0:
        rows = rows[:max_examples]
    examples = _prepare_examples(rows, prefix_model, settings, init_text)
    accumulation = int(flat.get("accumulation", 2))
    if int(flat.get("batch_size", 1)) != 1:
        raise ValueError("Official prefix adapters currently require train.batch_size=1")
    optimizer = torch.optim.AdamW(
        prefix_model.trainable_parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    config_record = {
        "method": method,
        "runtime": flat,
        "soft_prefix": soft_cfg,
        "official_baseline": baseline_cfg,
        "official_sources": {
            "SE-KD3x": {"path": source_paths["SE-KD3x"], "commit": SEKD_COMMIT},
            "LMOps": {"path": source_paths["LMOps"], "commit": OPCD_COMMIT},
        },
        "fairness_contract": {
            "backbone_frozen": True,
            "prefix_length": settings.prefix_length,
            "trainable_parameters": int(prefix_model.prefix_embeddings.numel()),
            "training_support": len(examples),
            "validation_tasks": int(flat.get("sel_env_num", 40)),
            "test_accessed": False,
            "initialization": "first prefix-length hard-Skill token embeddings",
            "batch_size": 1,
            "accumulation": accumulation,
            "seed": seed,
        },
        "input_fingerprints": {
            "trajectory_manifest_sha256": _sha256(settings.trajectory_examples_path),
            "skill_sha256": _sha256(settings.init_text_path),
        },
    }
    (out_root / "config.json").write_text(json.dumps(config_record, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72, flush=True)
    print(f"Official-Adapted {method.upper()}-Prefix / SpreadsheetBench", flush=True)
    print(f"model={settings.model_name}", flush=True)
    print(f"train={len(examples)} val={flat.get('sel_env_num', 40)} prefix={settings.prefix_length}", flush=True)
    print(f"seed={seed} batch=1 accumulation={accumulation}", flush=True)
    print("test split access: disabled", flush=True)
    print("=" * 72, flush=True)

    if method == "sekd":
        train_summary = _run_sekd(prefix_model, examples, optimizer, baseline_cfg, accumulation, seed)
    else:
        train_summary = _run_opcd(prefix_model, examples, optimizer, baseline_cfg, accumulation, seed, out_root)

    best_path = out_root / "best_prefix.pt"
    latest_path = out_root / "latest_prefix.pt"
    _save_checkpoint(torch, prefix_model, best_path)
    _save_checkpoint(torch, prefix_model, latest_path)
    train_summary["best_prefix_path"] = str(best_path)
    train_summary["checkpoint_sha256"] = _sha256(best_path)
    train_summary["test_split_accessed"] = False

    if bool(baseline_cfg.get("eval_after_train", True)) and not args.no_val:
        dataloader = _build_dataloader("spreadsheetbench", flat, seed)
        dataloader.setup(flat)
        val_items = _items_for_eval(dataloader, "valid_seen", int(flat.get("sel_env_num", 40)), seed)
        val_dir = out_root / "eval" / "final" / "valid_seen"
        val_hard, val_soft, _ = _evaluate_prefix(
            "spreadsheetbench",
            prefix_model,
            val_items,
            cfg=flat,
            settings=settings,
            out_dir=str(val_dir),
            desc=f"{method.upper()} Val40",
        )
        train_summary["valid_seen_hard"] = val_hard
        train_summary["valid_seen_soft"] = val_soft
        print(f"Validation: {round(val_hard * len(val_items))}/{len(val_items)} ({val_hard:.2%})", flush=True)

    (out_root / "summary.json").write_text(json.dumps(train_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"checkpoint={best_path}", flush=True)
    print(f"sha256={train_summary['checkpoint_sha256']}", flush=True)
    print(f"summary={out_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
