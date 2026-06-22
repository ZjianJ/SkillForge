#!/usr/bin/env bash
# Evaluate a trained soft prefix without running additional training epochs.
set -euo pipefail

CONFIG="${CONFIG:-configs/searchqa/soft_prefix.yaml}"
SPLIT_DIR="${SPLIT_DIR:-data/searchqa_split}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-outputs/SoftSkill_searchqa_example/best_prefix.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/SoftSkill_searchqa_eval}"
GPU_IDS="${GPU_IDS:-0}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-local_hf}"
INFERENCE_BASE_URL="${INFERENCE_BASE_URL:-}"

cfg_options=(
  train.num_epochs=0
  evaluation.eval_test=true
  soft_prefix.checkpoint_path="${CHECKPOINT_PATH}"
  soft_prefix.inference_backend="${INFERENCE_BACKEND}"
)
if [[ -n "${INFERENCE_BASE_URL}" ]]; then
  cfg_options+=(soft_prefix.inference_base_url="${INFERENCE_BASE_URL}")
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python scripts/train_soft_prefix.py   --config "${CONFIG}"   --split_dir "${SPLIT_DIR}"   --model_name "${MODEL_NAME}"   --cfg-options "${cfg_options[@]}"   --out_root "${OUTPUT_DIR}"
