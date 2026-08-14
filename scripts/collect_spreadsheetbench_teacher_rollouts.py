#!/usr/bin/env python3
"""Collect SpreadsheetBench teacher trajectories without loading the student LM."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local/spreadsheetbench_paper_gpt55.local.yaml")
    parser.add_argument("--split_dir", default="data/spreadsheetbench_split")
    parser.add_argument("--out_root", default="outputs/SpreadsheetBench_teacher_gpt55_collection")
    parser.add_argument("--limit", type=int, default=0, help="Limit training tasks; 0 means all 80.")
    return parser.parse_args()


def _reject_placeholder_credentials(cfg: dict) -> None:
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    required = (
        "target_azure_openai_endpoint",
        "target_azure_openai_api_key",
        "target_azure_openai_auth_mode",
    )
    invalid = [
        key
        for key in required
        if not str(model_cfg.get(key, "")).strip()
        or str(model_cfg.get(key, "")).strip() == "REPLACE_ME"
    ]
    if invalid:
        raise ValueError(
            "Local GPT-5.5 config is incomplete; set these model fields before making API calls: "
            + ", ".join(invalid)
        )


def main() -> None:
    args = parse_args()
    os.chdir(_PROJECT_ROOT)

    from scripts.train_soft_prefix import _configure_rollout_model
    from skillopt.config import flatten_config, is_structured, load_config
    from skillopt.softprefix.trainer import (
        SoftPrefixSettings,
        _collect_spreadsheet_trajectory_examples,
        _load_init_text,
    )

    raw_cfg = load_config(args.config)
    _reject_placeholder_credentials(raw_cfg)
    flat_cfg = flatten_config(raw_cfg) if is_structured(raw_cfg) else dict(raw_cfg)
    settings = SoftPrefixSettings.from_dict(dict(raw_cfg.get("soft_prefix", {})))
    _configure_rollout_model(flat_cfg, [])

    items_path = Path(args.split_dir) / "train" / "items.json"
    with items_path.open(encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or not items:
        raise ValueError(f"Expected a non-empty JSON array in {items_path}")
    if args.limit > 0:
        items = items[: args.limit]

    init_text = _load_init_text(settings.init_text_path or str(flat_cfg.get("skill_init", "")))
    examples = _collect_spreadsheet_trajectory_examples(
        items=items,
        cfg=flat_cfg,
        settings=settings,
        out_root=os.path.abspath(args.out_root),
        init_text=init_text,
    )
    print(json.dumps({
        "candidate_tasks": len(items),
        "sft_examples": len(examples),
        "trajectory_rollout_dir": os.path.abspath(settings.trajectory_rollout_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
