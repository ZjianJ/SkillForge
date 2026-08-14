#!/usr/bin/env python3
"""Run PRCB-v4-ES with all four stride-one windows of five prefix rows."""
from train_spreadsheetbench_prcb_v1 import main


if __name__ == "__main__":
    main(
        default_pair_policy="sliding_head_to_tail",
        default_out_root=(
            "outputs/SpreadsheetBench_prcb_v4_es_window5_stride1_all4_len8_seed1"
        ),
        default_locator_policy="margin_decision",
        default_method_version="v4_es",
        default_teacher_kl_weight=0.5,
        default_teacher_margin_weight=0.5,
        default_rounds=4,
        default_round_step_pattern="32,32,32,32",
        default_early_stop=True,
        default_sliding_window_size=5,
    )
