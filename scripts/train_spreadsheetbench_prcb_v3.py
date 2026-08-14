#!/usr/bin/env python3
"""Run PRCB-v3: successful-trajectory decision margins, head-to-tail pairs."""
from train_spreadsheetbench_prcb_v1 import main


if __name__ == "__main__":
    main(
        default_pair_policy="head_to_tail",
        default_out_root="outputs/SpreadsheetBench_prcb_v3_margin_head_to_tail_len8_seed1",
        default_locator_policy="margin_decision",
        default_method_version="v3",
        # KL and margin together retain unit total teacher-distillation weight.
        default_teacher_kl_weight=0.5,
        default_teacher_margin_weight=0.5,
    )
