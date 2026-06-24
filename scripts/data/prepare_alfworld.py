#!/usr/bin/env python3
"""Prepare or validate ALFWorld data for the released path split."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_SPLIT_DIR = "data/alfworld_path_split"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_split_items(split_dir: Path, split: str) -> list[dict]:
    path = split_dir / split / "items.json"
    items = load_json(path)
    if not isinstance(items, list):
        raise ValueError(f"Expected JSON array in {path}")
    return items


def resolve_gamefile(data_root: Path, gamefile: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(gamefile))))
    if path.is_absolute():
        return path
    return data_root / path


def validate_split(split_dir: Path, data_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    missing: list[str] = []
    for split in ("train", "val", "test"):
        items = load_split_items(split_dir, split)
        counts[split] = len(items)
        for item in items:
            gamefile = str(item.get("gamefile") or "")
            if not gamefile:
                missing.append(f"{split}:<missing gamefile>")
                continue
            if not resolve_gamefile(data_root, gamefile).is_file():
                missing.append(f"{split}:{gamefile}")
    if missing:
        examples = "\n".join(f"  - {item}" for item in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} ALFWorld gamefiles from {split_dir} do not exist under {data_root}.\n"
            f"Examples:\n{examples}"
        )
    return counts


def maybe_run_alfworld_download(command: str) -> None:
    executable = command.split()[0]
    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"{executable!r} is not on PATH. Install ALFWorld first, e.g. `pip install -e .[alfworld]`."
        )
    subprocess.run(command, shell=True, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ALFWorld raw data against the released path split.")
    parser.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR, help="Released ALFWorld path split directory.")
    parser.add_argument(
        "--data_root",
        default=os.environ.get("ALFWORLD_DATA", ""),
        help="ALFWorld data root containing json_2.1.1. Defaults to $ALFWORLD_DATA.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Run alfworld-download before validation. This requires ALFWorld to be installed.",
    )
    parser.add_argument(
        "--download_command",
        default="alfworld-download",
        help="Command used when --download is set.",
    )
    parser.add_argument(
        "--write_manifest",
        action="store_true",
        help="Record validation metadata in split_manifest.materialized.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    if args.download:
        maybe_run_alfworld_download(args.download_command)

    if not args.data_root:
        raise ValueError("Set --data_root or ALFWORLD_DATA to the ALFWorld data root containing json_2.1.1")
    data_root = Path(args.data_root).expanduser().resolve()
    counts = validate_split(split_dir, data_root)

    if args.write_manifest:
        base_manifest_path = split_dir / "split_manifest.json"
        base_manifest = load_json(base_manifest_path) if base_manifest_path.exists() else {}
        write_json(
            split_dir / "split_manifest.materialized.json",
            {
                **base_manifest,
                "data_root": str(data_root),
                "counts": counts,
                "validated": True,
            },
        )
    print(f"Done: {split_dir} resolves under {data_root} (train={counts['train']} val={counts['val']} test={counts['test']})")


if __name__ == "__main__":
    main()
