#!/usr/bin/env python3
"""Run low-compute PRCB-v5 with replay retention and adaptive stage commit."""
from train_spreadsheetbench_prcb_v1 import main


if __name__ == "__main__":
    main(
        default_pair_policy="sliding_head_to_tail",
        default_out_root="outputs/SpreadsheetBench_prcb_v5_retention_alpha_len8_seed1",
        default_locator_policy="margin_decision",
        default_method_version="v5",
        default_teacher_kl_weight=0.5,
        default_teacher_margin_weight=0.5,
        default_rounds=7,
        default_round_step_pattern="32,32,32,32,32,32,32",
        default_early_stop=True,
        default_sliding_window_size=2,
        default_retention_weight=1.0,
        # 0 rejects a harmful stage; 0.25 reuses the ordinary ES monitor.
        # Only 0.125 and 0.5 need extra monitor passes.
        default_stage_alpha_grid="0,0.125,0.25,0.5",
    )
