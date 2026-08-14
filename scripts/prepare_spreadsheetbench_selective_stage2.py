#!/usr/bin/env python3
"""Prepare clean, matched-random, and selective manifests for stage-2 training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _seed_for(identifier: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _expand(core: list[int], selectable: int, left: int, right: int) -> list[int]:
    selected: set[int] = set()
    for position in core:
        selected.update(range(max(0, position - left), min(selectable, position + right + 1)))
    return sorted(selected)


def _shared_preservation_indices(
    first: list[int],
    second: list[int],
    *,
    selectable: int,
    count: int,
    seed: int,
) -> list[int]:
    """Sample one preservation set disjoint from both paired selectors."""
    candidates = sorted(set(range(selectable)) - set(first) - set(second))
    if len(candidates) < count:
        raise ValueError(
            f"Shared preservation pool has {len(candidates)} positions, fewer than {count}"
        )
    rng = np.random.default_rng(seed)
    return sorted(
        int(index)
        for index in rng.choice(candidates, size=count, replace=False).tolist()
    )


def _clean_copy(
    row: dict,
    *,
    selected: list[int] | None,
    selector: str,
    preserve_indices: list[int] | None = None,
    core_indices: list[int] | None = None,
) -> dict:
    result = {
        "id": str(row["id"]),
        "messages": row["messages"],
        "target": row["target"],
        "score_cache": row["score_cache"],
        "stage2_selector": selector,
    }
    if selected is not None:
        result["selected_indices"] = selected
    if preserve_indices is not None:
        result["preserve_indices"] = preserve_indices
    if core_indices is not None:
        result["selected_core_indices"] = core_indices
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-root",
        default="outputs/SpreadsheetBench_selective_stage1_qwen36_gpt55",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("SPREADSHEETBENCH_MODEL", "Qwen/Qwen3.6-35B-A3B"),
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/SpreadsheetBench_selective_stage2_manifests",
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    stage1_root = _resolve(args.stage1_root)
    source_paths = {
        "gain5": stage1_root / "training_manifests/positive_gain_top0.05_L2_R8.jsonl",
        "gain10": stage1_root / "training_manifests/positive_gain_top0.1_L2_R8.jsonl",
        "combined5": stage1_root / "training_manifests/combined_top0.05_L2_R8.jsonl",
        "combined10": stage1_root / "training_manifests/combined_top0.1_L2_R8.jsonl",
        "combined20": stage1_root / "training_manifests/combined_top0.2_L2_R8.jsonl",
    }
    sources = {name: _read_jsonl(path) for name, path in source_paths.items()}
    identifiers = [[str(row["id"]) for row in rows] for rows in sources.values()]
    if not identifiers or any(ids != identifiers[0] for ids in identifiers[1:]):
        raise ValueError("Stage-1 manifests do not contain the same ordered trajectories")

    requested_model = Path(args.model_path).expanduser()
    model_source = (
        str(_resolve(args.model_path))
        if requested_model.is_absolute() or (PROJECT_ROOT / requested_model).exists()
        else args.model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    eos = tokenizer.eos_token
    output_rows = {
        "clean_full": [],
        "random_top0.05_core": [],
        "positive_gain_top0.05_core": [],
        "combined_top0.05_core": [],
        "positive_gain_top0.1_core": [],
        "combined_top0.05_L1_R2": [],
        "positive_gain_top0.05_core_preserve": [],
        "positive_gain_top0.05_core_shared_preserve": [],
        "combined_top0.05_core_shared_preserve": [],
        "combined_top0.05_core_coverage_ablation": [],
        "combined_top0.10_core_coverage_ablation": [],
        "combined_top0.20_core_coverage_ablation": [],
        "positive_gain_top0.05_L2_R8_preserve": [],
        "random_top0.05_core_preserve": [],
    }
    statistics = {name: {"tokens": 0, "selectable": 0} for name in output_rows}

    for gain5, gain10, combined5, combined10, combined20 in zip(
        sources["gain5"],
        sources["gain10"],
        sources["combined5"],
        sources["combined10"],
        sources["combined20"],
        strict=True,
    ):
        identifier = str(gain5["id"])
        messages = gain5["messages"]
        systems = [str(message.get("content", "")) for message in messages if message.get("role") == "system"]
        if any(re.search(r"(?m)^\s*##\s+Skill\s*$", text) for text in systems):
            raise ValueError(f"Trajectory {identifier} still contains a text Skill")

        target_text = str(gain5["target"]).strip()
        if eos and not target_text.endswith(eos):
            target_text += eos
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        with np.load(gain5["score_cache"]) as cached:
            cached_ids = cached["target_ids"].astype(np.int64).tolist()
        if target_ids != cached_ids:
            raise ValueError(f"Tokenizer/cache mismatch for trajectory {identifier}")
        selectable = len(target_ids) - 1

        gain5_core = [int(index) for index in gain5["selected_core_indices"]]
        gain5_window = [int(index) for index in gain5["selected_token_indices"]]
        combined5_core = [int(index) for index in combined5["selected_core_indices"]]
        combined10_core = [int(index) for index in combined10["selected_core_indices"]]
        combined20_core = [int(index) for index in combined20["selected_core_indices"]]
        if len(combined5_core) != len(gain5_core):
            raise ValueError(
                f"Paired Top-5% selectors differ in size for trajectory {identifier}: "
                f"positive={len(gain5_core)} combined={len(combined5_core)}"
            )
        gain10_core = [int(index) for index in gain10["selected_core_indices"]]
        rng = np.random.default_rng(_seed_for(identifier, args.seed))
        random5 = sorted(
            int(index)
            for index in rng.choice(selectable, size=len(gain5_core), replace=False).tolist()
        )
        combined5_window = _expand(combined5_core, selectable, left=1, right=2)
        shared_preserve = _shared_preservation_indices(
            gain5_core,
            combined5_core,
            selectable=selectable,
            count=len(gain5_core),
            seed=_seed_for(f"{identifier}:positive-combined-shared-preserve", args.seed),
        )
        # Coverage ablation: the only treatment variable is the Combined core
        # budget.  All three variants use the same Top-5%-sized preservation
        # set, sampled outside the largest (Top-20%) core.  Since Combined
        # ranking is nested, this set is disjoint from every treatment core.
        coverage_preserve = _shared_preservation_indices(
            combined20_core,
            combined20_core,
            selectable=selectable,
            count=len(combined5_core),
            seed=_seed_for(f"{identifier}:combined-coverage-ablation-preserve", args.seed),
        )

        variants = {
            "clean_full": None,
            "random_top0.05_core": random5,
            "positive_gain_top0.05_core": gain5_core,
            "combined_top0.05_core": combined5_core,
            "positive_gain_top0.1_core": gain10_core,
            "combined_top0.05_L1_R2": combined5_window,
            "positive_gain_top0.05_core_preserve": gain5_core,
            "positive_gain_top0.05_core_shared_preserve": gain5_core,
            "combined_top0.05_core_shared_preserve": combined5_core,
            "combined_top0.05_core_coverage_ablation": combined5_core,
            "combined_top0.10_core_coverage_ablation": combined10_core,
            "combined_top0.20_core_coverage_ablation": combined20_core,
            "positive_gain_top0.05_L2_R8_preserve": gain5_window,
            "random_top0.05_core_preserve": random5,
        }
        for name, selected in variants.items():
            source = combined5 if name.startswith("combined") else gain5
            preserve_indices = None
            if name.endswith("_coverage_ablation"):
                preserve_indices = coverage_preserve
            elif name.endswith("_shared_preserve"):
                preserve_indices = shared_preserve
            elif name.endswith("_preserve"):
                candidates = sorted(set(range(selectable)) - set(selected or []))
                # Keep the preservation budget matched to the original Top-5%
                # core count. Expanding CE to L2/R8 is the only intended
                # treatment change; matching preservation to the expanded
                # window would increase it by roughly sevenfold as a confound.
                requested_preserve_count = (
                    len(gain5_core)
                    if name == "positive_gain_top0.05_L2_R8_preserve"
                    else len(selected or [])
                )
                preserve_count = min(requested_preserve_count, len(candidates))
                preserve_indices = sorted(
                    int(index)
                    for index in rng.choice(candidates, size=preserve_count, replace=False).tolist()
                )
            output_rows[name].append(
                _clean_copy(
                    source,
                    selected=selected,
                    selector=name,
                    preserve_indices=preserve_indices,
                    core_indices=(
                        gain5_core
                        if name == "positive_gain_top0.05_L2_R8_preserve"
                        else None
                    ),
                )
            )
            count = selectable if selected is None else len(selected)
            statistics[name]["tokens"] += count + 1  # EOS is always supervised.
            statistics[name]["selectable"] += selectable
            if preserve_indices is not None:
                statistics[name]["preservation_tokens"] = (
                    statistics[name].get("preservation_tokens", 0) + len(preserve_indices)
                )

    out_dir = _resolve(args.out_dir)
    for name, rows in output_rows.items():
        _write_jsonl(out_dir / f"{name}.jsonl", rows)
        stats = statistics[name]
        stats["trajectories"] = len(rows)
        stats["mean_supervised_tokens"] = stats["tokens"] / len(rows)
        stats["non_eos_coverage"] = (
            (stats["tokens"] - len(rows)) / stats["selectable"]
        )
        if "preservation_tokens" in stats:
            stats["mean_preservation_tokens"] = stats["preservation_tokens"] / len(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "model_path": model_source,
                "always_supervise_eos": True,
                "variants": statistics,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(statistics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
