#!/usr/bin/env python3
"""Prepare GPQA-Diamond as a LiveMath-style MCQ split for soft-prefix eval."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import quote
from urllib.request import urlretrieve


DEFAULT_REPO_ID = "Idavidrein/gpqa"
DEFAULT_CONFIG = "gpqa_diamond"
DEFAULT_FILENAME = "gpqa_diamond.csv"
DEFAULT_SOURCE_URL = "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
CHOICE_LABELS = ("A", "B", "C", "D")


def hf_resolve_url(repo_id: str, filename: str) -> str:
    repo_path = "/".join(quote(part, safe="") for part in repo_id.split("/"))
    file_path = quote(filename, safe="/")
    return f"https://huggingface.co/datasets/{repo_path}/resolve/main/{file_path}"


def download_gpqa_csv(
    *,
    repo_id: str,
    filename: str,
    raw_dir: str | os.PathLike[str],
    source_url: str = DEFAULT_SOURCE_URL,
    retrieve=urlretrieve,
) -> Path:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    downloaded = raw_path / filename
    if source_url:
        retrieve(source_url, downloaded)
        return downloaded

    try:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=str(raw_path),
            )
        )
    except ImportError:
        retrieve(hf_resolve_url(repo_id, filename), downloaded)
        return downloaded


def _row_get(row: dict[str, str], *names: str) -> str:
    by_normalized = {
        key.strip().lower().replace("_", " ").replace("-", " "): value
        for key, value in row.items()
    }
    for name in names:
        value = by_normalized.get(name.strip().lower().replace("_", " ").replace("-", " "))
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def convert_gpqa_row(row: dict[str, str], *, row_idx: int) -> dict:
    """Convert one GPQA CSV row into the existing LiveMath MCQ item schema."""
    question = _row_get(row, "Question")
    correct = _row_get(row, "Correct Answer")
    incorrect = [
        _row_get(row, "Incorrect Answer 1"),
        _row_get(row, "Incorrect Answer 2"),
        _row_get(row, "Incorrect Answer 3"),
    ]
    if not question or not correct or any(not answer for answer in incorrect):
        raise ValueError(f"GPQA row {row_idx} is missing question or answer fields")

    answers = [correct, *incorrect]
    choices = [
        {"label": label, "text": answer}
        for label, answer in zip(CHOICE_LABELS, answers, strict=True)
    ]
    theorem_type = [
        value
        for value in (
            _row_get(row, "High-level domain", "High Level Domain", "Domain"),
            _row_get(row, "Subdomain", "Sub-domain"),
        )
        if value
    ]

    return {
        "id": _row_get(row, "Record ID", "RecordID", "id") or f"gpqa_diamond:{row_idx}",
        "question": question,
        "choices": choices,
        "correct_choice": {"label": "A", "text": correct},
        "theorem_type": theorem_type,
        "source": DEFAULT_REPO_ID,
        "source_config": DEFAULT_CONFIG,
    }


def load_gpqa_csv(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_gpqa_split(rows: Iterable[dict[str, str]], out_split_dir: str | os.PathLike[str]) -> None:
    out_dir = Path(out_split_dir)
    items = [
        convert_gpqa_row(row, row_idx=idx)
        for idx, row in enumerate(rows)
    ]

    for split_name, split_items in (
        ("train", []),
        ("val", []),
        ("test", items),
    ):
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "items.json").open("w", encoding="utf-8") as f:
            json.dump(split_items, f, ensure_ascii=False, indent=2)

    manifest = {
        "benchmark": "GPQA-Diamond",
        "source_repo": DEFAULT_REPO_ID,
        "source_config": DEFAULT_CONFIG,
        "source_file": DEFAULT_FILENAME,
        "layout": "LiveMathematicianBench MCQ split schema",
        "counts": {
            "train": 0,
            "val": 0,
            "test": len(items),
        },
        "notes": [
            "All GPQA-Diamond rows are placed in test because this split is for cross-task evaluation only.",
            "Choices are written with the correct answer as A; the LiveMath dataloader can remap labels via env.shuffle_choices=true.",
        ],
    }
    with (out_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GPQA-Diamond for soft-prefix MCQ evaluation.")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repository.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Dataset config name, used for metadata.")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="CSV file to download from the dataset repo.")
    parser.add_argument(
        "--source_url",
        default=DEFAULT_SOURCE_URL,
        help="Direct CSV URL. Set to an empty string to use Hugging Face instead.",
    )
    parser.add_argument("--raw_dir", default="data/gpqa/raw", help="Where to store the downloaded CSV.")
    parser.add_argument(
        "--out_split_dir",
        default="data/gpqa_diamond_split",
        help="Where to write train/val/test items.json files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downloaded = download_gpqa_csv(
        repo_id=args.repo_id,
        filename=args.filename,
        raw_dir=args.raw_dir,
        source_url=args.source_url,
    )
    rows = load_gpqa_csv(downloaded)
    write_gpqa_split(rows, args.out_split_dir)
    print(
        f"Done: {args.out_split_dir} "
        f"(train=0 val=0 test={len(rows)})"
    )


if __name__ == "__main__":
    main()
