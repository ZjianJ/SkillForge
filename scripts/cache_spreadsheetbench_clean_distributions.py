#!/usr/bin/env python3
"""Add no-Skill Top-k output distributions to existing stage-1 score caches."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _clean_topk(logits: Any, *, top_k: int, chunk_size: int) -> dict[str, np.ndarray]:
    import torch

    n_tokens = int(logits.shape[1])
    ids = np.empty((n_tokens, top_k), dtype=np.int32)
    logp = np.empty((n_tokens, top_k), dtype=np.float16)
    residual = np.empty(n_tokens, dtype=np.float16)
    for start in range(0, n_tokens, chunk_size):
        end = min(start + chunk_size, n_tokens)
        distribution = torch.log_softmax(logits[0, start:end].float(), dim=-1)
        values, indices = torch.topk(distribution, k=top_k, dim=-1)
        ids[start:end] = indices.cpu().numpy().astype(np.int32)
        logp[start:end] = values.cpu().numpy().astype(np.float16)
        mass = values.exp().sum(dim=-1).clamp(max=1.0 - 1e-7)
        residual[start:end] = torch.log1p(-mass).cpu().numpy().astype(np.float16)
    return {
        "clean_topk_ids": ids,
        "clean_topk_logp": logp,
        "clean_residual_log_mass": residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=(
            "outputs/SpreadsheetBench_selective_stage1_qwen36_gpt55/"
            "training_manifests/positive_gain_top0.05_L2_R8.jsonl"
        ),
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("SPREADSHEETBENCH_MODEL", "Qwen/Qwen3.6-35B-A3B"),
    )
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from skillopt.softprefix.data import _apply_text_chat_template

    requested_model = Path(args.model_path).expanduser()
    model_path: str | Path = (
        _resolve(args.model_path)
        if requested_model.is_absolute() or (PROJECT_ROOT / requested_model).exists()
        else args.model_path
    )
    rows = _read_jsonl(_resolve(args.manifest))
    if args.limit > 0:
        rows = rows[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pending: list[dict[str, Any]] = []
    for row in rows:
        cache_path = _resolve(str(row["score_cache"]))
        with np.load(cache_path) as cached:
            complete = {
                "clean_topk_ids",
                "clean_topk_logp",
                "clean_residual_log_mass",
            }.issubset(cached.files)
        if args.force or not complete:
            pending.append(row)
    print(f"Clean-distribution caches: {len(rows) - len(pending)} ready, {len(pending)} pending", flush=True)
    if not pending:
        return

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map={"": "cuda"},
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for index, row in enumerate(pending, 1):
        prompt = _apply_text_chat_template(
            tokenizer,
            list(row["messages"]),
            enable_thinking=False,
            add_generation_prompt=True,
        )
        target = str(row["target"]).strip()
        eos = tokenizer.eos_token
        if eos and not target.endswith(eos):
            target += eos
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=16384,
        )["input_ids"]
        target_ids = tokenizer(
            target,
            add_special_tokens=False,
            truncation=True,
            max_length=8192,
        )["input_ids"]
        cache_path = _resolve(str(row["score_cache"]))
        with np.load(cache_path) as cached:
            arrays = {name: cached[name] for name in cached.files}
        if arrays["target_ids"].astype(np.int64).tolist() != target_ids:
            raise ValueError(f"Tokenizer/cache mismatch for {row['id']!r}")

        sequence = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device="cuda")
        attention = torch.ones_like(sequence)
        prediction_positions = torch.arange(
            len(prompt_ids) - 1,
            len(prompt_ids) + len(target_ids) - 1,
            dtype=torch.long,
            device="cuda",
        )
        print(
            f"[{index}/{len(pending)}] {row['id']}: prompt={len(prompt_ids)}, target={len(target_ids)}",
            flush=True,
        )
        with torch.inference_mode():
            output = model(
                input_ids=sequence,
                attention_mask=attention,
                use_cache=False,
                output_router_logits=False,
                logits_to_keep=prediction_positions,
                return_dict=True,
            )
        arrays.update(
            _clean_topk(output.logits, top_k=args.top_k, chunk_size=args.chunk_size)
        )
        _atomic_npz(cache_path, arrays)
        del output, sequence, attention
        torch.cuda.empty_cache()

    print(f"Updated {len(pending)} clean-distribution caches", flush=True)


if __name__ == "__main__":
    main()
