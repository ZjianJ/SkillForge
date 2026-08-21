#!/usr/bin/env python3
"""Train Combined-selected soft prefixes with full-vocabulary Skill KL."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skillopt.config import flatten_config, is_structured, load_config  # noqa: E402
from skillopt.softprefix.distillation_losses import (  # noqa: E402
    chunked_full_vocab_forward_kl,
    topk_residual_forward_kl,
)
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


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in ("api_key", "password", "secret", "access_token")):
                result[key] = "<redacted>" if item else item
            else:
                result[key] = _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


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


def _training_order(torch, count: int, seed: int) -> list[int]:
    """Match the seeded shuffled DataLoader order used by the CE trainer."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        list(range(count)),
        batch_size=1,
        shuffle=True,
        generator=generator,
    )
    return [int(batch.item()) for batch in loader]


def _prepare_records(rows, prefix_model, settings, skill_text, experiment_cfg):
    core_field = str(experiment_cfg.get("core_label_field", "selected_indices"))
    preserve_field = str(experiment_cfg.get("preservation_label_field", "preserve_indices"))
    records = []
    for row in tqdm(rows, desc="Encode Combined train61", unit="ex"):
        example = encode_trajectory(
            tokenizer=prefix_model.tokenizer,
            row=row,
            skill_text=skill_text,
            max_prompt_tokens=settings.max_prompt_tokens,
            max_target_tokens=settings.max_target_tokens,
        )
        selected = sorted({int(index) for index in row.get(core_field, [])})
        preserve = sorted({int(index) for index in row.get(preserve_field, [])})
        eos_index = len(example.target_ids) - 1
        if not selected:
            raise ValueError(f"Trajectory {example.task_id} has no Combined core positions")
        if any(index < 0 or index >= eos_index for index in selected + preserve):
            raise ValueError(f"Trajectory {example.task_id} has an out-of-range or EOS locator index")
        if set(selected) & set(preserve):
            raise ValueError(f"Trajectory {example.task_id} core/preservation positions overlap")
        with np.load(example.score_cache) as cached:
            cached_ids = cached["target_ids"].astype(np.int64).tolist()
            if cached_ids != example.target_ids:
                raise ValueError(f"Tokenizer/cache mismatch for trajectory {example.task_id}")
            clean_topk_ids = cached["clean_topk_ids"][preserve].astype(np.int64)
            clean_topk_logp = cached["clean_topk_logp"][preserve].astype(np.float32)
            clean_residual = cached["clean_residual_log_mass"][preserve].astype(np.float32)
        records.append(
            {
                "example": example,
                "selected": selected,
                "preserve": preserve,
                "clean_topk_ids": clean_topk_ids,
                "clean_topk_logp": clean_topk_logp,
                "clean_residual": clean_residual,
            }
        )
    return records


def _run_training(prefix_model, records, optimizer, *, accumulation, seed, experiment_cfg):
    torch = prefix_model.torch
    preservation_weight = float(experiment_cfg.get("preservation_loss_weight", 1.0))
    kl_chunk_size = int(experiment_cfg.get("kl_chunk_size", 8))
    order = _training_order(torch, len(records), seed)
    selected_total = sum(len(record["selected"]) for record in records)
    eos_total = len(records)
    preservation_total = sum(len(record["preserve"]) for record in records)
    skill_kl_sum = 0.0
    eos_ce_sum = 0.0
    preservation_sum = 0.0
    optimizer_steps = 0
    history = []
    started = time.time()

    progress = tqdm(total=len(order), desc="Combined Full-Vocab Skill-KL", unit="traj")
    for group_start in range(0, len(order), accumulation):
        group_ids = order[group_start : group_start + accumulation]
        group_core_tokens = sum(len(records[index]["selected"]) + 1 for index in group_ids)
        group_preservation_tokens = sum(len(records[index]["preserve"]) for index in group_ids)
        optimizer.zero_grad(set_to_none=True)
        group_objective = 0.0
        group_skill_tokens = 0
        group_preserve_seen = 0

        for index in group_ids:
            record = records[index]
            example = record["example"]
            selected = record["selected"]
            eos_index = len(example.target_ids) - 1
            student_indices = selected + [eos_index]

            teacher_logits = target_logits(
                prefix_model,
                example,
                selected,
                use_prefix=False,
                with_grad=False,
            )
            student_logits = target_logits(
                prefix_model,
                example,
                student_indices,
                use_prefix=True,
                with_grad=True,
            )
            student_core_logits = student_logits[:-1]
            skill_kl = chunked_full_vocab_forward_kl(
                teacher_logits=teacher_logits,
                student_logits=student_core_logits,
                chunk_size=kl_chunk_size,
            )
            eos_target = torch.tensor([example.target_ids[-1]], dtype=torch.long, device=prefix_model.device)
            eos_ce = torch.nn.functional.cross_entropy(student_logits[-1:].float(), eos_target)
            core_count = len(selected) + 1
            core_objective = (skill_kl * len(selected) + eos_ce) / core_count
            (core_objective * (core_count / group_core_tokens)).backward()

            skill_value = float(skill_kl.detach().cpu())
            eos_value = float(eos_ce.detach().cpu())
            skill_kl_sum += skill_value * len(selected)
            eos_ce_sum += eos_value
            group_objective += (skill_value * len(selected) + eos_value) / group_core_tokens
            group_skill_tokens += len(selected)
            del teacher_logits, student_logits, student_core_logits, skill_kl, eos_ce, core_objective
            _empty_cuda_cache(torch)

            preserve = record["preserve"]
            if preserve:
                preservation_logits = target_logits(
                    prefix_model,
                    example,
                    preserve,
                    use_prefix=True,
                    with_grad=True,
                )
                preservation_loss = topk_residual_forward_kl(
                    student_logits=preservation_logits,
                    reference_topk_ids=record["clean_topk_ids"],
                    reference_topk_logp=record["clean_topk_logp"],
                    reference_residual_log_mass=record["clean_residual"],
                )
                preserve_count = len(preserve)
                (
                    preservation_weight
                    * preservation_loss
                    * (preserve_count / max(group_preservation_tokens, 1))
                ).backward()
                preserve_value = float(preservation_loss.detach().cpu())
                preservation_sum += preserve_value * preserve_count
                group_objective += preservation_weight * preserve_value * (
                    preserve_count / max(group_preservation_tokens, 1)
                )
                group_preserve_seen += preserve_count
                del preservation_logits, preservation_loss
                _empty_cuda_cache(torch)

            progress.update(1)
            progress.set_postfix(loss=f"{group_objective:.4f}", skill=group_skill_tokens)

        optimizer.step()
        optimizer_steps += 1
        history.append(
            {
                "optimizer_step": optimizer_steps,
                "trajectory_ids": [records[index]["example"].task_id for index in group_ids],
                "full_vocab_skill_tokens": group_skill_tokens,
                "preservation_tokens": group_preserve_seen,
                "loss": group_objective,
            }
        )
    progress.close()

    core_objective_loss = (skill_kl_sum + eos_ce_sum) / max(selected_total + eos_total, 1)
    preservation_loss = preservation_sum / max(preservation_total, 1)
    return {
        "method": "Combined-10%-Full-Vocabulary-Skill-KL",
        "optimizer_steps": optimizer_steps,
        "trajectories": len(records),
        "full_vocab_skill_tokens": selected_total,
        "eos_ce_tokens": eos_total,
        "preservation_tokens": preservation_total,
        "mean_full_vocab_skill_kl": skill_kl_sum / max(selected_total, 1),
        "mean_eos_ce": eos_ce_sum / max(eos_total, 1),
        "mean_core_objective": core_objective_loss,
        "mean_preservation_kl": preservation_loss,
        "mean_total_objective": core_objective_loss + preservation_weight * preservation_loss,
        "preservation_loss_weight": preservation_weight,
        "wall_time_s": round(time.time() - started, 1),
        "history": history,
        "training_order": [records[index]["example"].task_id for index in order],
        "objective": {
            "core": "full-vocabulary forward KL(Qwen+Hard-Skill || Qwen+Soft-Prefix)",
            "eos": "one-hot CE, unchanged from the Combined CE baseline",
            "preservation": "no-Skill Top-64 plus residual-bucket forward KL, unchanged",
        },
    }


def main() -> None:
    args = parse_args()
    raw = load_config(args.config, overrides=args.cfg_options)
    flat = flatten_config(raw) if is_structured(raw) else dict(raw)
    soft_cfg = dict(raw.get("soft_prefix", {}))
    experiment_cfg = dict(raw.get("combined_full_vocab_kl", {}))
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
    if int(flat.get("num_epochs", 1)) != 1:
        raise ValueError("The matched Combined full-vocabulary experiment requires exactly one epoch")
    if int(flat.get("batch_size", 1)) != 1:
        raise ValueError("The matched Combined full-vocabulary experiment requires batch_size=1")
    accumulation = int(flat.get("accumulation", 2))

    settings = SoftPrefixSettings.from_dict(soft_cfg)
    init_text = _load_init_text(settings.init_text_path or str(flat.get("skill_init", "")))
    if not init_text.strip():
        raise ValueError("Combined full-vocabulary training requires the non-empty hard Skill text")
    prefix_model = _build_prefix_model("spreadsheetbench", settings, init_text)
    torch = prefix_model.torch
    rows = _load_rows(settings.trajectory_examples_path)
    records = _prepare_records(rows, prefix_model, settings, init_text, experiment_cfg)
    optimizer = torch.optim.AdamW(
        prefix_model.trainable_parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )

    config_record = {
        "method": "combined_core10_full_vocab_skill_kl",
        "runtime": _redact(flat),
        "soft_prefix": _redact(soft_cfg),
        "combined_full_vocab_kl": experiment_cfg,
        "fairness_contract": {
            "backbone_frozen": True,
            "prefix_length": settings.prefix_length,
            "trainable_parameters": int(prefix_model.prefix_embeddings.numel()),
            "training_support": len(records),
            "validation_tasks": int(flat.get("sel_env_num", 40)),
            "test_accessed": False,
            "initialization": "first prefix-length hard-Skill token embeddings",
            "batch_size": 1,
            "accumulation": accumulation,
            "seed": seed,
            "only_changed_term": "selected core one-hot CE -> full-vocabulary hard-Skill forward KL",
        },
        "input_fingerprints": {
            "trajectory_manifest_sha256": _sha256(settings.trajectory_examples_path),
            "skill_sha256": _sha256(settings.init_text_path),
        },
    }
    (out_root / "config.json").write_text(
        json.dumps(config_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 72, flush=True)
    print("Combined 10% + Full-Vocabulary Skill-KL / SpreadsheetBench", flush=True)
    print(f"model={settings.model_name}", flush=True)
    print(f"train={len(records)} val={flat.get('sel_env_num', 40)} prefix={settings.prefix_length}", flush=True)
    print(f"seed={seed} batch=1 accumulation={accumulation}", flush=True)
    print("test split access: disabled", flush=True)
    print("=" * 72, flush=True)

    train_summary = _run_training(
        prefix_model,
        records,
        optimizer,
        accumulation=accumulation,
        seed=seed,
        experiment_cfg=experiment_cfg,
    )
    best_path = out_root / "best_prefix.pt"
    latest_path = out_root / "latest_prefix.pt"
    torch.save(prefix_model.state_dict(), best_path)
    torch.save(prefix_model.state_dict(), latest_path)
    train_summary["best_prefix_path"] = str(best_path)
    train_summary["checkpoint_sha256"] = _sha256(best_path)
    train_summary["test_split_accessed"] = False

    if bool(experiment_cfg.get("eval_after_train", True)) and not args.no_val:
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
            desc="Combined Full-Vocab-KL Val40",
        )
        train_summary["valid_seen_hard"] = val_hard
        train_summary["valid_seen_soft"] = val_soft
        print(f"Validation: {round(val_hard * len(val_items))}/{len(val_items)} ({val_hard:.2%})", flush=True)

    (out_root / "summary.json").write_text(
        json.dumps(train_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"checkpoint={best_path}", flush=True)
    print(f"sha256={train_summary['checkpoint_sha256']}", flush=True)
    print(f"summary={out_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
