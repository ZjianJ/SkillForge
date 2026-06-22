"""ALFWorld task dataloader."""
from __future__ import annotations

import os
import re

from skillopt.datasets.base import BatchSpec, SplitDataLoader


class ALFWorldDataLoader(SplitDataLoader):
    """ALFWorld batch planner.

    In split_dir mode, batches are fixed gamefile items so ablations differ
    only in how the same training set is batched.
    """

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        seed: int = 42,
        limit: int = 0,
        train_size: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )
        self.train_size_override = int(train_size or 0)

    @staticmethod
    def _resolve_gamefile_path(gamefile: str) -> str:
        path = os.path.expanduser(os.path.expandvars(str(gamefile or "").strip()))
        if not path or os.path.isabs(path):
            return path

        data_root = os.path.expanduser(os.path.expandvars(os.environ.get("ALFWORLD_DATA", ""))).strip()
        if data_root:
            return os.path.abspath(os.path.join(data_root, path))
        return path

    @staticmethod
    def _resolve_item_id(item_id: str) -> str:
        safe_id = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(item_id or "").strip())
        safe_id = safe_id.strip(" .")
        return safe_id or "item"

    @classmethod
    def _normalize_items(cls, items: list[dict]) -> list[dict]:
        normalized: list[dict] = []
        for item in items:
            if not isinstance(item, dict):
                normalized.append(item)
                continue
            row = dict(item)
            if "id" in row:
                row["id"] = cls._resolve_item_id(str(row.get("id") or ""))
            if "gamefile" in row:
                row["gamefile"] = cls._resolve_gamefile_path(str(row.get("gamefile") or ""))
            normalized.append(row)
        return normalized

    def load_raw_items(self, data_path: str) -> list[dict]:
        return self._normalize_items(super().load_raw_items(data_path))

    def load_split_items(self, split_path: str) -> list[dict]:
        return self._normalize_items(super().load_split_items(split_path))

    @staticmethod
    def _metadata_for_items(items: list[dict], split: str, phase: str) -> dict:
        gamefiles = [str(item.get("gamefile") or "") for item in items]
        if any(not gamefile for gamefile in gamefiles):
            raise ValueError("ALFWorld split items must contain non-empty gamefile paths.")
        eval_dataset = "train"
        is_train = phase == "train"
        first = gamefiles[0] if gamefiles else ""
        if "/valid_seen/" in first:
            eval_dataset = "eval_in_distribution"
            is_train = False
        elif "/valid_unseen/" in first:
            eval_dataset = "eval_out_of_distribution"
            is_train = False
        return {
            "eval_dataset": eval_dataset,
            "is_train": is_train,
            "gamefiles": gamefiles,
            "result_ids": [str(item.get("id") or idx) for idx, item in enumerate(items)],
        }

    def get_train_size(self) -> int:
        if self.train_size_override > 0:
            return self.train_size_override
        return super().get_train_size()

    def build_train_batch(self, batch_size: int, seed: int, **kwargs) -> BatchSpec:
        batch = super().build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        items = list(batch.payload or [])
        batch.metadata.update(self._metadata_for_items(items, "train", "train"))
        return BatchSpec(
            phase="train",
            split="train",
            seed=seed,
            batch_size=len(items),
            payload=items,
            metadata=batch.metadata,
        )

    def plan_train_epoch(
        self,
        *,
        epoch: int,
        steps_per_epoch: int,
        accumulation: int,
        batch_size: int,
        seed: int,
        **kwargs,
    ) -> list[BatchSpec]:
        batches = super().plan_train_epoch(
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
            accumulation=accumulation,
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )
        for batch in batches:
            items = list(batch.payload or [])
            batch.metadata.update(self._metadata_for_items(items, "train", "train"))
        return batches

    def build_eval_batch(
        self,
        env_num: int,
        split: str,
        seed: int,
        **kwargs,
    ) -> BatchSpec:
        batch = super().build_eval_batch(
            env_num=env_num,
            split=split,
            seed=seed,
            **kwargs,
        )
        items = list(batch.payload or [])
        batch.metadata.update(self._metadata_for_items(items, split, "eval"))
        return BatchSpec(
            phase="eval",
            split=split,
            seed=seed,
            batch_size=len(items),
            payload=items,
            metadata=batch.metadata,
        )
