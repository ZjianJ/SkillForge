"""Filesystem-safe path helpers."""
from __future__ import annotations

import re


def safe_prediction_dir_name(item_id: str) -> str:
    """Return a filesystem-safe directory name for a task/item id."""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(item_id or "").strip())
    safe = safe.strip(" .")
    return safe or "item"
