#!/usr/bin/env python3
"""Run PRCB-v2 with a fixed head-to-tail pair schedule."""
from __future__ import annotations

from train_spreadsheetbench_prcb_v1 import main


if __name__ == "__main__":
    main(
        default_pair_policy="head_to_tail",
        default_out_root="outputs/SpreadsheetBench_prcb_v2_head_to_tail_len8_seed1",
    )
