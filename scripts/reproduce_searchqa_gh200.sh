#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/SoftSkill_searchqa_example}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}" >&2
  echo "Create the project environment before running this launcher." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

if [[ ! -f data/searchqa_split/train/items.json ]] || \
   [[ ! -f data/searchqa_split/val/items.json ]] || \
   [[ ! -f data/searchqa_split/test/items.json ]]; then
  "${PYTHON_BIN}" scripts/data/prepare_searchqa.py
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
"${PYTHON_BIN}" scripts/train_soft_prefix.py \
  --config configs/searchqa/soft_prefix.yaml \
  --split_dir data/searchqa_split \
  --model_name Qwen/Qwen3.5-4B \
  --out_root "${OUTPUT_DIR}" \
  --cfg-options \
    soft_prefix.inference_backend=local_hf \
    train.batch_size=2 \
    train.accumulation=4
