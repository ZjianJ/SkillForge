#!/usr/bin/env python3
"""Run the registered PRCB-v6 experiment with rank-2 weak learners."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_spreadsheetbench_prcb_v6 import main


def _has_option(name: str) -> bool:
    return name in sys.argv or any(value.startswith(name + "=") for value in sys.argv)


if __name__ == "__main__":
    if not _has_option("--learner-rank"):
        sys.argv.extend(["--learner-rank", "2"])
    if not _has_option("--out-root"):
        sys.argv.extend(
            [
                "--out-root",
                "outputs/SpreadsheetBench_prcb_v6_lowrank_r2_len8_seed1",
            ]
        )
    main()
