#!/usr/bin/env python3
"""Evaluate one frozen SpreadsheetBench soft-prefix checkpoint without training."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_spreadsheetbench_combined_full_vocab_kl import _sha256  # noqa: E402
from skillopt.config import flatten_config, is_structured, load_config  # noqa: E402
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--cfg-options", nargs="+", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_config(args.config, overrides=args.cfg_options)
    flat = flatten_config(raw) if is_structured(raw) else dict(raw)
    soft_cfg = dict(raw.get("soft_prefix", {}))
    if os.environ.get("SPREADSHEETBENCH_MODEL"):
        soft_cfg["model_name"] = os.environ["SPREADSHEETBENCH_MODEL"]
    out_root = Path(args.out_root).resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    seed = int(flat.get("seed", 1))
    _set_seed(seed)
    settings = SoftPrefixSettings.from_dict(soft_cfg)
    init_text = _load_init_text(settings.init_text_path or str(flat.get("skill_init", "")))
    prefix_model = _build_prefix_model("spreadsheetbench", settings, init_text)
    checkpoint = Path(args.checkpoint).resolve()
    state = prefix_model.torch.load(checkpoint, map_location="cpu", weights_only=True)
    prefix_model.load_state_dict(state)
    dataloader = _build_dataloader("spreadsheetbench", flat, seed)
    dataloader.setup(flat)
    items = _items_for_eval(dataloader, "valid_seen", args.count, seed)
    hard_rate, soft_rate, _ = _evaluate_prefix(
        "spreadsheetbench",
        prefix_model,
        items,
        cfg=flat,
        settings=settings,
        out_dir=str(out_root / "valid_seen"),
        desc=f"Frozen prefix Val{len(items)}",
    )
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "split": "valid_seen",
        "evaluated": len(items),
        "hard_rate": hard_rate,
        "soft_rate": soft_rate,
        "successes": round(soft_rate * len(items)),
        "test_split_accessed": False,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
