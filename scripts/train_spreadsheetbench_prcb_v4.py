#!/usr/bin/env python3
"""Run PRCB-v4 with stride-one overlapping two-row prefix updates."""
from train_spreadsheetbench_prcb_v1 import main


if __name__ == "__main__":
    main(
        default_pair_policy="sliding_head_to_tail",
        default_out_root="outputs/SpreadsheetBench_prcb_v4_overlap1_len8_seed1",
        default_locator_policy="margin_decision",
        default_method_version="v4",
        default_teacher_kl_weight=0.5,
        default_teacher_margin_weight=0.5,
        default_rounds=7,
        # 32 total optimizer steps, matching PRCB-v3 exactly.
        default_round_step_pattern="5,4,5,4,5,4,5",
    )
