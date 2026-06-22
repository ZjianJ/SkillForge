#!/usr/bin/env bash
# Interpret a trained soft prefix through nearest-token decoding.
set -euo pipefail

RUN_DIR="${RUN_DIR:-outputs/SoftSkill_searchqa_example}"
CHECKPOINT="${CHECKPOINT:-best_prefix.pt}"
TOP_K="${TOP_K:-5}"
METRIC="${METRIC:-cosine}"
DEVICE="${DEVICE:-cpu}"

python scripts/analysis/interpret_soft_prefix.py   --run_dir "${RUN_DIR}"   --checkpoint "${CHECKPOINT}"   --top_k "${TOP_K}"   --metric "${METRIC}"   --device "${DEVICE}"
