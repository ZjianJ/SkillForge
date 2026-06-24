#!/usr/bin/env python3
"""Materialize the released DocVQA split manifest into runnable local files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "lmms-lab/DocVQA"
DEFAULT_CONFIG = "DocVQA"
DEFAULT_SOURCE_SPLIT = "validation"
DEFAULT_ID_SPLIT_DIR = "data/docvqa_id_split"
DEFAULT_OUT_SPLIT_DIR = "data/docvqa/splits"
DEFAULT_IMAGE_DIR = "data/docvqa_images"


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


def row_by_question_id(dataset: Any) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in dataset:
        rows[str(row["questionId"])] = row
    return rows


def materialize_split(
    *,
    split: str,
    manifest_items: list[dict],
    source_rows: dict[str, dict],
    image_dir: Path,
    save_images: bool,
) -> list[dict]:
    materialized: list[dict] = []
    for manifest_item in manifest_items:
        question_id = str(manifest_item["questionId"])
        source = source_rows.get(question_id)
        if source is None:
            raise KeyError(f"QuestionId {question_id!r} from {split} manifest is missing from DocVQA source split")

        image_path = Path(str(manifest_item.get("image_path") or "")).as_posix()
        if not image_path:
            image_path = (image_dir / f"q{question_id}_d{source['docId']}.png").as_posix()
        if save_images:
            local_image_path = Path(image_path)
            local_image_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_image_path.exists():
                source["image"].save(local_image_path)

        answers = [str(answer) for answer in (source.get("answers") or [])]
        question_types = source.get("question_types") or []
        materialized.append(
            {
                "id": str(manifest_item.get("id") or question_id),
                "questionId": question_id,
                "docId": str(source.get("docId") or manifest_item.get("docId") or ""),
                "question": str(source.get("question") or ""),
                "answer": answers[0] if answers else "",
                "answers": answers,
                "image_path": image_path,
                "image_paths": [image_path],
                "topic": ";".join(str(item) for item in question_types),
                "ucsf_document_id": str(source.get("ucsf_document_id") or manifest_item.get("ucsf_document_id") or ""),
                "ucsf_document_page_no": str(
                    source.get("ucsf_document_page_no") or manifest_item.get("ucsf_document_page_no") or ""
                ),
                "source_dataset": manifest_item.get("source_dataset", DEFAULT_REPO_ID),
                "source_config": manifest_item.get("source_config", DEFAULT_CONFIG),
                "source_split": manifest_item.get("source_split", DEFAULT_SOURCE_SPLIT),
                "sample_seed": manifest_item.get("sample_seed", ""),
            }
        )
    return materialized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DocVQA split files and images from the released manifest.")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repository.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Hugging Face dataset config.")
    parser.add_argument("--source_split", default=DEFAULT_SOURCE_SPLIT, help="Source split to load from Hugging Face.")
    parser.add_argument("--revision", default="", help="Optional Hugging Face dataset revision.")
    parser.add_argument("--id_split_dir", default=DEFAULT_ID_SPLIT_DIR, help="Released DocVQA ID manifest directory.")
    parser.add_argument("--out_split_dir", default=DEFAULT_OUT_SPLIT_DIR, help="Output train/val/test split directory.")
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR, help="Directory for downloaded DocVQA images.")
    parser.add_argument("--skip_images", action="store_true", help="Write split files without saving image files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    id_split_dir = Path(args.id_split_dir)
    out_split_dir = Path(args.out_split_dir)
    image_dir = Path(args.image_dir)
    manifest = load_json(id_split_dir / "split_manifest.json")
    revision = args.revision or manifest.get("source_revision") or None

    dataset = load_dataset(args.repo_id, args.config, split=args.source_split, revision=revision)
    source_rows = row_by_question_id(dataset)

    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        items = materialize_split(
            split=split,
            manifest_items=load_manifest_items(id_split_dir, split),
            source_rows=source_rows,
            image_dir=image_dir,
            save_images=not args.skip_images,
        )
        write_json(out_split_dir / split / "items.json", items)
        counts[split] = len(items)
        print(f"{split}: wrote {len(items)} items")

    write_json(
        out_split_dir / "split_manifest.json",
        {
            **manifest,
            "materialized_split_dir": str(out_split_dir),
            "image_dir": str(image_dir),
            "images_saved": not args.skip_images,
            "counts": counts,
        },
    )
    print(f"Done: {out_split_dir}")


if __name__ == "__main__":
    main()
