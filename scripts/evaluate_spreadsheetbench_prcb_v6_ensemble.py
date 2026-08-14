#!/usr/bin/env python3
"""Evaluate a frozen PRCB-v6 logit ensemble on SpreadsheetBench Val40."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skillopt.envs.spreadsheetbench.dataloader import SpreadsheetBenchDataLoader
from skillopt.softprefix.data import _apply_text_chat_template
from skillopt.softprefix.logit_ensemble import LogitBoostedPrefixGenerator
from skillopt.softprefix.model import SoftPrefixCausalLM
from skillopt.softprefix.trainer import evaluate_spreadsheet_prefix
from scripts.train_spreadsheetbench_prcb_v1 import atomic_json, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ensemble-manifest",
        default="outputs/SpreadsheetBench_prcb_v6_functional_len8_seed1/ensemble_manifest.json",
    )
    parser.add_argument(
        "--config",
        default="outputs/SpreadsheetBench_prcb_v6_functional_len8_seed1/prcb_v6_config.json",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/SpreadsheetBench_prcb_v6_functional_len8_seed1/eval/ensemble/valid_seen",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--skip-alpha-zero-check", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prefix(torch: Any, path: Path) -> Any:
    return torch.load(path, map_location="cpu")["prefix_embeddings"].detach().clone()


def install_prefix(torch: Any, model: SoftPrefixCausalLM, prefix: Any) -> None:
    with torch.no_grad():
        model.prefix_embeddings.copy_(
            prefix.to(device=model.device, dtype=model.prefix_embeddings.dtype)
        )


def main() -> None:
    args = parse_args()
    manifest_path = resolve(args.ensemble_manifest)
    config_path = resolve(args.config)
    out_dir = resolve(args.out_dir)
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    if bool(manifest.get("test_split_accessed")) or bool(config.get("test_split_accessed")):
        raise ValueError("Refusing a manifest marked as having accessed the test split")
    if int(args.limit) != 40:
        raise ValueError("The registered V6 ensemble evaluation is exactly Val40")
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        raise FileExistsError(f"Frozen ensemble evaluation already exists: {results_path}")

    import torch

    model_path = resolve(str(config["model_path"]))
    print(f"Loading frozen Qwen: {model_path}", flush=True)
    prefix_model = SoftPrefixCausalLM(
        str(model_path),
        prefix_length=8,
        init_strategy="random",
        torch_dtype="bfloat16",
        device="cuda",
    )
    prefix_model.model.eval()
    prefix_model.model.config.use_cache = True
    prefix_model.prefix_embeddings.requires_grad_(False)
    base_path = resolve(str(manifest["base_checkpoint"]))
    learner_paths = [resolve(str(value)) for value in manifest["learner_checkpoints"]]
    if sha256(base_path) != str(manifest["base_checkpoint_sha256"]):
        raise ValueError("Base checkpoint SHA-256 mismatch")
    base = load_prefix(torch, base_path)
    learners = [load_prefix(torch, path) for path in learner_paths]
    alphas = [float(value) for value in manifest["alphas"]]

    alpha_zero_match = None
    if not args.skip_alpha_zero_check:
        check_prompt = _apply_text_chat_template(
            prefix_model.tokenizer,
            [
                {"role": "system", "content": "Answer concisely."},
                {"role": "user", "content": "Return the word validation."},
            ],
            enable_thinking=False,
            add_generation_prompt=True,
        )
        install_prefix(torch, prefix_model, base)
        expected = prefix_model.generate_from_prompt(
            check_prompt,
            max_prompt_tokens=512,
            max_new_tokens=32,
            temperature=0.0,
        )
        zero_generator = LogitBoostedPrefixGenerator(
            prefix_model,
            base_prefix=base,
            learner_prefixes=learners,
            alphas=[0.0] * len(learners),
        )
        actual = zero_generator.generate_from_prompt(
            check_prompt,
            max_prompt_tokens=512,
            max_new_tokens=32,
            temperature=0.0,
        )
        alpha_zero_match = expected == actual
        print(f"alpha=0 decoder equivalence: {alpha_zero_match}", flush=True)
        if not alpha_zero_match:
            raise AssertionError(
                f"Custom decoder disagrees with standard greedy generation: {expected!r} != {actual!r}"
            )

    dataloader = SpreadsheetBenchDataLoader(
        split_dir=str(resolve(str(config["split_dir"]))),
        split_mode="split_dir",
        split_seed=42,
        data_root=str(resolve(str(config["data_root"]))),
        seed=int(config["seed"]),
    )
    validation = dataloader.load_split_items(
        str(resolve(str(config["split_dir"])) / "val")
    )[:40]
    generator = LogitBoostedPrefixGenerator(
        prefix_model,
        base_prefix=base,
        learner_prefixes=learners,
        alphas=alphas,
        response_cache_path=out_dir / "generation_cache.jsonl",
    )
    hard, soft, results = evaluate_spreadsheet_prefix(
        prefix_model,
        validation,
        out_dir=str(out_dir),
        data_root=str(resolve(str(config["data_root"]))),
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        exec_timeout=600,
        desc="PRCB-v6 Ensemble Val",
        generator=generator,
        injection_position="prompt_start",
        repair_turns=1,
        generation_batch_size=1,
    )
    summary = {
        "method": "PRCB-v6-frozen-logit-ensemble",
        "ensemble_manifest": str(manifest_path),
        "ensemble_manifest_sha256": sha256(manifest_path),
        "base_checkpoint_sha256": sha256(base_path),
        "learner_checkpoint_sha256": [sha256(path) for path in learner_paths],
        "alphas": alphas,
        "alpha_zero_decoder_equivalence": alpha_zero_match,
        "split": "val40",
        "test_split_accessed": False,
        "hard": hard,
        "soft": soft,
        "successes": sum(bool(row.get("hard")) for row in results),
        "tasks": len(results),
    }
    atomic_json(out_dir / "ensemble_eval_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
