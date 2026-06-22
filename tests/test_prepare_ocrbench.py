from __future__ import annotations

import json


def test_ocrbench_rows_url_uses_dataset_server_pagination() -> None:
    from scripts.data.prepare_ocrbench import ocrbench_rows_url

    assert (
        ocrbench_rows_url("echo840/OCRBench", "default", "test", offset=100, length=50)
        == "https://datasets-server.huggingface.co/rows?dataset=echo840%2FOCRBench&config=default&split=test&offset=100&length=50"
    )


def test_ocrbench_row_converts_to_docvqa_item() -> None:
    from scripts.data.prepare_ocrbench import convert_ocrbench_row

    item = convert_ocrbench_row(
        {
            "row_idx": 3,
            "row": {
                "dataset": "IIIT5K",
                "question": "what is written in the image?",
                "question_type": "Regular Text Recognition",
                "answer": ["CENTRE"],
                "image": {"src": "https://example.test/image.jpg"},
            },
        },
        image_path="data/ocrbench_images/000003.jpg",
    )

    assert item["id"] == "ocrbench:3"
    assert item["question"] == "what is written in the image?"
    assert item["answer"] == "CENTRE"
    assert item["answers"] == ["CENTRE"]
    assert item["image_path"] == "data/ocrbench_images/000003.jpg"
    assert item["image_paths"] == ["data/ocrbench_images/000003.jpg"]
    assert item["task_type"] == "Regular Text Recognition"
    assert item["source_dataset"] == "IIIT5K"
    assert item["source_repo"] == "echo840/OCRBench"


def test_write_ocrbench_split_puts_all_items_in_test_and_downloads_images(tmp_path) -> None:
    from scripts.data.prepare_ocrbench import write_ocrbench_split

    entries = [
        {
            "row_idx": 0,
            "row": {
                "dataset": "IIIT5K",
                "question": "Q1",
                "question_type": "Regular Text Recognition",
                "answer": ["A1"],
                "image": {"src": "https://example.test/0/image.jpg"},
            },
        },
        {
            "row_idx": 1,
            "row": {
                "dataset": "DocVQA",
                "question": "Q2",
                "question_type": "Document VQA",
                "answer": ["A2", "Alt"],
                "image": {"src": "https://example.test/1/image.jpg"},
            },
        },
    ]
    downloads = []

    def fake_download(url, path):
        downloads.append((url, path))
        path.write_bytes(b"image-bytes")

    write_ocrbench_split(entries, tmp_path / "split", image_dir=tmp_path / "images", download_image=fake_download)

    assert json.loads((tmp_path / "split" / "train" / "items.json").read_text()) == []
    assert json.loads((tmp_path / "split" / "val" / "items.json").read_text()) == []
    test_items = json.loads((tmp_path / "split" / "test" / "items.json").read_text())
    assert [item["id"] for item in test_items] == ["ocrbench:0", "ocrbench:1"]
    assert test_items[1]["answers"] == ["A2", "Alt"]
    assert len(downloads) == 2
    assert all(path.exists() for _url, path in downloads)

    manifest = json.loads((tmp_path / "split" / "split_manifest.json").read_text())
    assert manifest["source_repo"] == "echo840/OCRBench"
    assert manifest["counts"] == {"train": 0, "val": 0, "test": 2}
