#!/usr/bin/env python3
"""Run PRCB-v2 with a fixed tail-to-head pair schedule."""
from __future__ import annotations

from train_spreadsheetbench_prcb_v1 import main


if __name__ == "__main__":
    main(
        default_pair_policy="tail_to_head",
        default_out_root="outputs/SpreadsheetBench_prcb_v2_tail_to_head_len8_seed1",
    )
