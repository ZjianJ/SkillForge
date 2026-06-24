#!/usr/bin/env python3
"""Materialize the released SpreadsheetBench split manifest into runnable local files."""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "KAKA22/SpreadsheetBench"
DEFAULT_ARCHIVE = "spreadsheetbench_verified_400.tar.gz"
DEFAULT_ID_SPLIT_DIR = "data/spreadsheetbench_id_split"
DEFAULT_DATA_ROOT = "data/spreadsheetbench_verified_400"
DEFAULT_OUT_SPLIT_DIR = "data/spreadsheetbench_split"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_manifest_items(id_split_dir: Path, split: str) -> list[dict]:
    path = id_split_dir / split / "items.json"
    items = load_json(path)
    if not isinstance(items, list):
        raise ValueError(f"Expected JSON array in {path}")
    return items


def safe_extract_verified_archive(archive_path: Path, data_root: Path) -> None:
    expected_prefix = "spreadsheetbench_verified_400/"
    if data_root.exists():
        shutil.rmtree(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.name == "spreadsheetbench_verified_400":
                continue
            if not member.name.startswith(expected_prefix):
                raise ValueError(f"Unexpected archive member outside verified-400 root: {member.name}")
            relative_name = member.name[len(expected_prefix) :]
            if not relative_name or Path(relative_name).is_absolute() or ".." in Path(relative_name).parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            member.name = relative_name
            archive.extract(member, data_root)


def materialize_split(split: str, manifest_items: list[dict], source_rows: dict[str, dict]) -> list[dict]:
    items: list[dict] = []
    for manifest_item in manifest_items:
        item_id = str(manifest_item["id"])
        source = source_rows.get(item_id)
        if source is None:
            raise KeyError(f"SpreadsheetBench id {item_id!r} from {split} manifest is missing from dataset.json")
        row = dict(source)
        row["id"] = item_id
        row["spreadsheet_path"] = manifest_item.get("spreadsheet_path") or row.get("spreadsheet_path")
        row["instruction_type"] = manifest_item.get("instruction_type") or row.get("instruction_type")
        items.append(row)
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SpreadsheetBench Verified 400 splits and workbooks.")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repository.")
    parser.add_argument("--archive_path", default="", help="Existing local verified-400 tarball. If omitted, download from HF.")
    parser.add_argument("--archive_name", default=DEFAULT_ARCHIVE, help="Archive filename in the HF dataset repo.")
    parser.add_argument("--id_split_dir", default=DEFAULT_ID_SPLIT_DIR, help="Released SpreadsheetBench ID manifest directory.")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="Where to extract SpreadsheetBench Verified 400.")
    parser.add_argument("--out_split_dir", default=DEFAULT_OUT_SPLIT_DIR, help="Output train/val/test split directory.")
    parser.add_argument("--skip_extract", action="store_true", help="Use an existing data_root instead of extracting the tarball.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    id_split_dir = Path(args.id_split_dir)
    out_split_dir = Path(args.out_split_dir)
    data_root = Path(args.data_root)
    manifest = load_json(id_split_dir / "split_manifest.json")

    if not args.skip_extract:
        archive_path = Path(args.archive_path) if args.archive_path else Path(
            _download_hf_archive(args.repo_id, args.archive_name)
        )
        safe_extract_verified_archive(archive_path, data_root)

    dataset_path = data_root / "dataset.json"
    source_items = load_json(dataset_path)
    if not isinstance(source_items, list):
        raise ValueError(f"Expected JSON array in {dataset_path}")
    source_rows = {str(item["id"]): item for item in source_items}

    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        items = materialize_split(split, load_manifest_items(id_split_dir, split), source_rows)
        write_json(out_split_dir / split / "items.json", items)
        counts[split] = len(items)
        print(f"{split}: wrote {len(items)} items")

    write_json(
        out_split_dir / "split_manifest.json",
        {
            **manifest,
            "materialized_split_dir": str(out_split_dir),
            "data_root": str(data_root),
            "counts": counts,
        },
    )
    print(f"Done: {out_split_dir}")


def _download_hf_archive(repo_id: str, archive_name: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=archive_name)


if __name__ == "__main__":
    main()
