#!/usr/bin/env bash
# Curated soft-prefix experiment runner for public examples.
# This intentionally covers a small matrix; paper/internal sweeps belong in scripts/internal/.
set -euo pipefail

TASKS="${TASKS:-searchqa livemathematicianbench docvqa}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
GPU_IDS="${GPU_IDS:-0}"
OUTPUT_BASE="${OUTPUT_BASE:-outputs/SoftSkill_experiments}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-local_hf}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

split_dir_for_task() {
  case "$1" in
    searchqa) echo "data/searchqa_split" ;;
    livemathematicianbench|livemath) echo "data/livemathematicianbench_split" ;;
    docvqa) echo "data/docvqa/splits" ;;
    *) echo "Unknown task: $1" >&2; return 1 ;;
  esac
}

for task in ${TASKS}; do
  config_task="${task}"
  if [[ "${task}" == "livemath" ]]; then
    config_task="livemathematicianbench"
  fi
  out_root="${OUTPUT_BASE}/${task}"
  if [[ -f "${out_root}/summary.json" ]]; then
    echo "Skipping ${task}; summary exists at ${out_root}/summary.json"
    continue
  fi
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" python scripts/train_soft_prefix.py     --config "configs/${config_task}/soft_prefix.yaml"     --split_dir "$(split_dir_for_task "${task}")"     --model_name "${MODEL_NAME}"     --cfg-options "soft_prefix.inference_backend=${INFERENCE_BACKEND}"     --out_root "${out_root}"
done
