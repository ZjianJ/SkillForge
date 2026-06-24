#!/usr/bin/env python3
"""Materialize the released OfficeQA split manifest into runnable local files."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "databricks/officeqa"
DEFAULT_ID_SPLIT_DIR = "data/officeqa_id_split"
DEFAULT_OUT_SPLIT_DIR = "data/officeqa_split"
DEFAULT_DOCS_DIR = "data/officeqa_docs_official"
DEFAULT_CSV_FILENAME = "officeqa_full.csv"


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


def load_officeqa_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {str(row["uid"]): row for row in reader}


def download_hf_file(repo_id: str, filename: str, destination: Path) -> bool:
    from huggingface_hub import hf_hub_download

    try:
        source = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
    except Exception as exc:  # noqa: BLE001 - keep prep robust across optional OfficeQA assets.
        print(f"warning: could not download {filename}: {exc}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return True


def source_file_names(items: list[dict]) -> set[str]:
    names: set[str] = set()
    for item in items:
        raw = str(item.get("source_files") or "")
        for part in raw.replace("\n", ",").split(","):
            name = Path(part.strip()).name
            if name:
                names.add(name)
    return names


def materialize_split(split: str, manifest_items: list[dict], source_rows: dict[str, dict[str, str]]) -> list[dict]:
    materialized: list[dict] = []
    for manifest_item in manifest_items:
        uid = str(manifest_item["uid"])
        source = source_rows.get(uid)
        if source is None:
            raise KeyError(f"UID {uid!r} from {split} manifest is missing from OfficeQA CSV")
        category = str(manifest_item.get("category") or source.get("difficulty") or "")
        source_files = str(manifest_item.get("source_files") or source.get("source_files") or "")
        source_docs = str(manifest_item.get("source_docs") or source.get("source_docs") or "")
        answer = str(source.get("answer") or "")
        materialized.append(
            {
                "id": uid,
                "uid": uid,
                "question": str(source.get("question") or ""),
                "answer": answer,
                "ground_truth": answer,
                "answers": [answer] if answer else [],
                "category": category,
                "difficulty": category,
                "source_files": source_files,
                "source_docs": source_docs,
                "split": split,
            }
        )
    return materialized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OfficeQA split files and local document corpus.")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repository.")
    parser.add_argument("--csv_path", default="", help="Existing local officeqa_full.csv. If omitted, download from HF.")
    parser.add_argument("--csv_filename", default=DEFAULT_CSV_FILENAME, help="CSV filename in the HF dataset repo.")
    parser.add_argument("--id_split_dir", default=DEFAULT_ID_SPLIT_DIR, help="Released OfficeQA ID manifest directory.")
    parser.add_argument("--out_split_dir", default=DEFAULT_OUT_SPLIT_DIR, help="Output train/val/test split directory.")
    parser.add_argument("--docs_dir", default=DEFAULT_DOCS_DIR, help="Output OfficeQA document root.")
    parser.add_argument("--skip_docs", action="store_true", help="Only write split files; do not download document text/JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    id_split_dir = Path(args.id_split_dir)
    out_split_dir = Path(args.out_split_dir)
    docs_dir = Path(args.docs_dir)
    manifest = load_json(id_split_dir / "split_manifest.json")

    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        from huggingface_hub import hf_hub_download

        csv_path = Path(hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=args.csv_filename))
    source_rows = load_officeqa_rows(csv_path)

    all_manifest_items: list[dict] = []
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        manifest_items = load_manifest_items(id_split_dir, split)
        all_manifest_items.extend(manifest_items)
        items = materialize_split(split, manifest_items, source_rows)
        write_json(out_split_dir / split / "items.json", items)
        counts[split] = len(items)
        print(f"{split}: wrote {len(items)} items")

    downloaded_docs = 0
    downloaded_jsons = 0
    if not args.skip_docs:
        for name in sorted(source_file_names(all_manifest_items)):
            if download_hf_file(
                args.repo_id,
                f"treasury_bulletins_parsed/transformed/{name}",
                docs_dir / "transformed" / name,
            ):
                downloaded_docs += 1
            json_name = f"{Path(name).stem}.json"
            if download_hf_file(
                args.repo_id,
                f"treasury_bulletins_parsed/jsons/{json_name}",
                docs_dir / "jsons" / json_name,
            ):
                downloaded_jsons += 1

    write_json(
        out_split_dir / "split_manifest.json",
        {
            **manifest,
            "materialized_split_dir": str(out_split_dir),
            "docs_dir": str(docs_dir),
            "counts": counts,
            "downloaded_transformed_text_files": downloaded_docs,
            "downloaded_parsed_json_files": downloaded_jsons,
        },
    )
    print(f"Done: {out_split_dir}")


if __name__ == "__main__":
    main()
