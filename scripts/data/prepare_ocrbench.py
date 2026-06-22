#!/usr/bin/env python3
"""Prepare OCRBench as a DocVQA-style split for local vision soft-prefix eval."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlencode
from urllib.request import urlopen, urlretrieve


DEFAULT_DATASET = "echo840/OCRBench"
DEFAULT_CONFIG = "default"
DEFAULT_SPLIT = "test"
DEFAULT_PAGE_SIZE = 100
DATASETS_SERVER_ROWS = "https://datasets-server.huggingface.co/rows"


def ocrbench_rows_url(dataset: str, config: str, split: str, *, offset: int, length: int) -> str:
    query = urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": int(offset),
            "length": int(length),
        }
    )
    return f"{DATASETS_SERVER_ROWS}?{query}"


def fetch_ocrbench_rows(
    *,
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_SPLIT,
    page_size: int = DEFAULT_PAGE_SIZE,
    opener: Callable[..., Any] = urlopen,
) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        url = ocrbench_rows_url(dataset, config, split, offset=offset, length=page_size)
        with opener(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        page_rows = payload.get("rows") or []
        if total is None:
            total = int(payload.get("num_rows_total") or len(page_rows))
        if not page_rows:
            break
        rows.extend(page_rows)
        offset += len(page_rows)
    return rows


def _as_answers(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(answer).strip() for answer in raw if str(answer).strip()]
    text = str(raw or "").strip()
    return [text] if text else []


def _image_extension(image_url: str) -> str:
    path = image_url.split("?", 1)[0]
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return suffix
    return ".jpg"


def convert_ocrbench_row(entry: dict, *, image_path: str) -> dict:
    row = entry.get("row") or {}
    row_idx = int(entry.get("row_idx", 0))
    answers = _as_answers(row.get("answer") or row.get("answers") or row.get("target"))
    task_type = str(row.get("question_type") or row.get("type") or row.get("subset_key") or "ocrbench").strip()
    dataset_name = str(row.get("dataset") or row.get("dataset_name") or "").strip()
    return {
        "id": f"ocrbench:{row_idx}",
        "question": str(row.get("question") or "").strip(),
        "answer": answers[0] if answers else "",
        "answers": answers,
        "task_type": task_type or "ocrbench",
        "subtask": task_type or "ocrbench",
        "image_path": image_path,
        "image_paths": [image_path],
        "questionId": f"ocrbench:{row_idx}",
        "source_dataset": dataset_name,
        "source_repo": DEFAULT_DATASET,
        "source_config": DEFAULT_CONFIG,
        "source_split": DEFAULT_SPLIT,
        "source_row_idx": row_idx,
    }


def download_url_to_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, path)


def write_ocrbench_split(
    entries: Iterable[dict],
    out_split_dir: str | os.PathLike[str],
    *,
    image_dir: str | os.PathLike[str] | None = None,
    download_image: Callable[[str, Path], None] = download_url_to_file,
) -> None:
    out_dir = Path(out_split_dir)
    image_root = Path(image_dir) if image_dir is not None else out_dir.parent / "ocrbench_images"
    items: list[dict] = []
    entries_list = list(entries)

    for entry in entries_list:
        row = entry.get("row") or {}
        image = row.get("image") or {}
        image_url = str(image.get("src") or "").strip() if isinstance(image, dict) else ""
        if not image_url:
            raise ValueError(f"OCRBench row {entry.get('row_idx', '<unknown>')} is missing image.src")
        row_idx = int(entry.get("row_idx", len(items)))
        image_path = image_root / f"{row_idx:06d}{_image_extension(image_url)}"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        download_image(image_url, image_path)
        items.append(convert_ocrbench_row(entry, image_path=str(image_path)))

    for split_name, split_items in (
        ("train", []),
        ("val", []),
        ("test", items),
    ):
        split_path = out_dir / split_name
        split_path.mkdir(parents=True, exist_ok=True)
        with (split_path / "items.json").open("w", encoding="utf-8") as f:
            json.dump(split_items, f, ensure_ascii=False, indent=2)

    manifest = {
        "benchmark": "OCRBench",
        "source_repo": DEFAULT_DATASET,
        "source_config": DEFAULT_CONFIG,
        "source_split": DEFAULT_SPLIT,
        "layout": "DocVQA split schema",
        "image_dir": str(image_root),
        "counts": {
            "train": 0,
            "val": 0,
            "test": len(items),
        },
        "notes": [
            "All OCRBench rows are placed in test because this split is for cross-task evaluation only.",
            "Images are downloaded locally because in-process transformers DocVQA soft-prefix eval expects file paths.",
        ],
    }
    with (out_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OCRBench for DocVQA soft-prefix evaluation.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Dataset config name.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split name.")
    parser.add_argument("--page_size", type=int, default=DEFAULT_PAGE_SIZE, help="Rows to fetch per API page.")
    parser.add_argument("--out_split_dir", default="data/ocrbench_split", help="Output train/val/test split dir.")
    parser.add_argument("--image_dir", default="data/ocrbench_images", help="Where to save OCRBench images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = fetch_ocrbench_rows(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        page_size=args.page_size,
    )
    write_ocrbench_split(entries, args.out_split_dir, image_dir=args.image_dir)
    print(f"Done: {args.out_split_dir} (train=0 val=0 test={len(entries)})")


if __name__ == "__main__":
    main()
