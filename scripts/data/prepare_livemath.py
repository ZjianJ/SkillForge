#!/usr/bin/env python3
"""Prepare LiveMathematicianBench splits from Hugging Face."""
from __future__ import annotations

import argparse
import os
import shutil

from huggingface_hub import hf_hub_download, list_repo_files

from skillopt.envs.livemathematicianbench.dataloader import LiveMathematicianBenchDataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LiveMathematicianBench train/val/test splits.")
    parser.add_argument(
        "--repo_id",
        default="LiveMathematicianBench/LiveMathematicianBench",
        help="Hugging Face dataset repository.",
    )
    parser.add_argument(
        "--raw_dir",
        default="data/livemathematicianbench/raw",
        help="Where to store downloaded raw monthly JSON files.",
    )
    parser.add_argument(
        "--out_split_dir",
        default="data/livemathematicianbench_split",
        help="Where to write the materialized train/val/test split.",
    )
    parser.add_argument("--split_ratio", default="2:1:7", help="Deterministic train:val:test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_files = list_repo_files(args.repo_id, repo_type="dataset")
    monthly_files = sorted(
        path
        for path in repo_files
        if path.startswith("data/") and path.endswith("_final.json")
    )
    if not monthly_files:
        raise ValueError(f"No monthly qa_*_final.json files found in {args.repo_id}")

    raw_dir = os.path.abspath(args.raw_dir)
    os.makedirs(raw_dir, exist_ok=True)
    for repo_path in monthly_files:
        src = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=repo_path,
        )
        dst = os.path.join(raw_dir, repo_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"downloaded {repo_path} -> {os.path.relpath(dst)}")

    split_dir = os.path.abspath(args.out_split_dir)
    loader = LiveMathematicianBenchDataLoader(
        split_mode="ratio",
        data_path=raw_dir,
        split_ratio=args.split_ratio,
        split_seed=args.seed,
        split_output_dir=split_dir,
        seed=args.seed,
    )
    loader.setup(
        {
            "env": "livemathematicianbench",
            "out_root": os.getcwd(),
        }
    )
    print(
        "Done: "
        f"{os.path.relpath(split_dir)} "
        f"(train={len(loader.train_items)} val={len(loader.val_items)} test={len(loader.test_items)})"
    )


if __name__ == "__main__":
    main()
