"""Tests for ALFWorld split item loading."""
from __future__ import annotations

import json
from pathlib import Path

from skillopt.envs.alfworld.dataloader import ALFWorldDataLoader


def _write_split(split_dir: Path, split: str, gamefile: str) -> None:
    split_path = split_dir / split
    split_path.mkdir(parents=True)
    with (split_path / "items.json").open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "id": f"{split}:0000",
                    "gamefile": gamefile,
                    "task_type": "pick_and_place_simple",
                }
            ],
            f,
        )


def test_split_gamefiles_relative_to_alfworld_data_are_expanded(tmp_path: Path, monkeypatch) -> None:
    alfworld_data = tmp_path / "alfworld_data"
    split_dir = tmp_path / "split"
    relative_gamefile = "json_2.1.1/train/task/trial/game.tw-pddl"
    for split in ("train", "val", "test"):
        _write_split(split_dir, split, relative_gamefile)

    monkeypatch.setenv("ALFWORLD_DATA", str(alfworld_data))

    loader = ALFWorldDataLoader(split_dir=str(split_dir), split_mode="split_dir")
    loader.setup({})

    assert loader.train_items[0]["gamefile"] == str(alfworld_data / relative_gamefile)


def test_split_item_ids_are_filesystem_safe(tmp_path: Path) -> None:
    split_dir = tmp_path / "split"
    for split in ("train", "val", "test"):
        _write_split(split_dir, split, "/tmp/alfworld/game.tw-pddl")

    loader = ALFWorldDataLoader(split_dir=str(split_dir), split_mode="split_dir")
    loader.setup({})
    batch = loader.build_train_batch(batch_size=1, seed=1)

    assert loader.train_items[0]["id"] == "train_0000"
    assert batch.metadata["result_ids"] == ["train_0000"]
