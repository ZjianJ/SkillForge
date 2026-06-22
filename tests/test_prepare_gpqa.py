from __future__ import annotations

import json


def test_hf_resolve_url_points_to_dataset_file() -> None:
    from scripts.data.prepare_gpqa import hf_resolve_url

    assert (
        hf_resolve_url("Idavidrein/gpqa", "gpqa_diamond.csv")
        == "https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/gpqa_diamond.csv"
    )


def test_download_gpqa_csv_prefers_explicit_source_url(tmp_path) -> None:
    from scripts.data.prepare_gpqa import download_gpqa_csv

    calls = []

    def fake_retrieve(url, filename):
        calls.append((url, filename))
        with open(filename, "w", encoding="utf-8") as f:
            f.write("Question,Correct Answer,Incorrect Answer 1,Incorrect Answer 2,Incorrect Answer 3\n")

    downloaded = download_gpqa_csv(
        repo_id="Idavidrein/gpqa",
        filename="gpqa_diamond.csv",
        raw_dir=tmp_path,
        source_url="https://example.test/gpqa_diamond.csv",
        retrieve=fake_retrieve,
    )

    assert downloaded == tmp_path / "gpqa_diamond.csv"
    assert calls == [("https://example.test/gpqa_diamond.csv", tmp_path / "gpqa_diamond.csv")]


def test_gpqa_row_converts_to_livemath_mcq_item() -> None:
    from scripts.data.prepare_gpqa import convert_gpqa_row

    item = convert_gpqa_row(
        {
            "Record ID": "gpqa:abc",
            "Question": "Which particle is neutral?",
            "Correct Answer": "neutron",
            "Incorrect Answer 1": "electron",
            "Incorrect Answer 2": "proton",
            "Incorrect Answer 3": "muon",
            "High-level domain": "Physics",
            "Subdomain": "Particle physics",
        },
        row_idx=7,
    )

    assert item["id"] == "gpqa:abc"
    assert item["question"] == "Which particle is neutral?"
    assert item["choices"] == [
        {"label": "A", "text": "neutron"},
        {"label": "B", "text": "electron"},
        {"label": "C", "text": "proton"},
        {"label": "D", "text": "muon"},
    ]
    assert item["correct_choice"] == {"label": "A", "text": "neutron"}
    assert item["theorem_type"] == ["Physics", "Particle physics"]


def test_write_gpqa_split_puts_all_items_in_test(tmp_path) -> None:
    from scripts.data.prepare_gpqa import write_gpqa_split

    rows = [
        {
            "Record ID": "one",
            "Question": "Q1",
            "Correct Answer": "gold",
            "Incorrect Answer 1": "bad1",
            "Incorrect Answer 2": "bad2",
            "Incorrect Answer 3": "bad3",
        },
        {
            "Record ID": "two",
            "Question": "Q2",
            "Correct Answer": "gold2",
            "Incorrect Answer 1": "bad4",
            "Incorrect Answer 2": "bad5",
            "Incorrect Answer 3": "bad6",
        },
    ]

    write_gpqa_split(rows, tmp_path)

    assert json.loads((tmp_path / "train" / "items.json").read_text()) == []
    assert json.loads((tmp_path / "val" / "items.json").read_text()) == []
    test_items = json.loads((tmp_path / "test" / "items.json").read_text())
    assert [item["id"] for item in test_items] == ["one", "two"]

    manifest = json.loads((tmp_path / "split_manifest.json").read_text())
    assert manifest["source_repo"] == "Idavidrein/gpqa"
    assert manifest["source_config"] == "gpqa_diamond"
    assert manifest["counts"] == {"train": 0, "val": 0, "test": 2}
