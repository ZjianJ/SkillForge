#!/usr/bin/env python3
"""Measure a PRCB margin locator once more after its final pair update."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_spreadsheetbench_prcb_v1 import (
    atomic_json,
    atomic_jsonl,
    atomic_npz,
    build_round_rows,
    read_jsonl,
    score_current_prefix,
    sha256,
    slug,
)
from skillopt.softprefix.model import SoftPrefixCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        default="outputs/SpreadsheetBench_prcb_v3_margin_head_to_tail_len8_seed1",
    )
    args = parser.parse_args()
    root = Path(args.run_root).expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    config = json.loads((root / "prcb_config.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    completed_rounds = int(summary["completed_rounds"])
    checkpoint = Path(summary["final_checkpoint"])
    if sha256(checkpoint) != summary["final_checkpoint_sha256"]:
        raise ValueError("Final checkpoint hash does not match summary.json")
    rows = read_jsonl(Path(config["manifest"]))
    previous_manifest = read_jsonl(
        root / f"round_{completed_rounds:02d}" / "manifest.jsonl"
    )
    previous_core = {
        str(row["id"]): [int(index) for index in row["selected_core_indices"]]
        for row in previous_manifest
    }

    import torch

    model = SoftPrefixCausalLM(
        config["model_path"],
        prefix_length=8,
        init_strategy="random",
        torch_dtype="bfloat16",
        device="cuda",
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.prefix_embeddings.requires_grad_(False)
    model.model.eval()
    model.model.config.use_cache = False

    output_dir = root / "final_locator"
    score_dir = output_dir / "scores"
    scores = {}
    for row in tqdm(rows, desc="Final locator", unit="traj"):
        identifier = str(row["id"])
        cache_path = score_dir / f"{slug(identifier)}.npz"
        if not cache_path.exists():
            values = score_current_prefix(
                model,
                row,
                max_prompt_tokens=int(config["max_prompt_tokens"]),
                max_target_tokens=int(config["max_target_tokens"]),
                chunk_size=int(config["score_chunk_size"]),
                locator_policy="margin_decision",
            )
            atomic_npz(cache_path, **values)
        with np.load(cache_path) as cached:
            scores[identifier] = {name: cached[name] for name in cached.files}

    final_rows, _, statistics = build_round_rows(
        rows,
        scores,
        previous_core,
        ratio=float(config["selection_ratio"]),
        locator_policy="margin_decision",
    )
    statistics["checkpoint"] = str(checkpoint)
    statistics["checkpoint_sha256"] = sha256(checkpoint)
    statistics["measurement"] = (
        f"post_round_{completed_rounds:02d}_teacher_forced_success_trajectories"
    )
    statistics["test_split_accessed"] = False
    atomic_jsonl(output_dir / "manifest.jsonl", final_rows)
    atomic_json(output_dir / "locator_statistics.json", statistics)
    print(json.dumps({k: v for k, v in statistics.items() if k != "per_trajectory"}, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
