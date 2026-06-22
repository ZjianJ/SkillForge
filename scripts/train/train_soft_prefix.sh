#!/usr/bin/env bash
# Minimal soft-prefix training launcher. Configure through env vars.
set -euo pipefail

CONFIG="${CONFIG:-configs/searchqa/soft_prefix.yaml}"
SPLIT_DIR="${SPLIT_DIR:-data/searchqa_split}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/SoftSkill_searchqa_example}"
GPU_IDS="${GPU_IDS:-0}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-local_hf}"
INFERENCE_BASE_URL="${INFERENCE_BASE_URL:-}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

cfg_options=("soft_prefix.inference_backend=${INFERENCE_BACKEND}")
if [[ -n "${INFERENCE_BASE_URL}" ]]; then
  cfg_options+=("soft_prefix.inference_base_url=${INFERENCE_BASE_URL}")
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python scripts/train_soft_prefix.py   --config "${CONFIG}"   --split_dir "${SPLIT_DIR}"   --model_name "${MODEL_NAME}"   --cfg-options "${cfg_options[@]}"   --out_root "${OUTPUT_DIR}"
