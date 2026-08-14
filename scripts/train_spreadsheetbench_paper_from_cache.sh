#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/spreadsheetbench/soft_prefix_paper_cached.yaml}"
SPLIT_DIR="${SPLIT_DIR:-${PROJECT_DIR}/data/spreadsheetbench_split}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/SoftSkill_spreadsheetbench_paper_len8_seed1}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${PROJECT_DIR}/rollouts/teacher_gpt55_spreadsheetbench_rollouts}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
INFERENCE_BASE_URL="${INFERENCE_BASE_URL:-http://127.0.0.1:8020}"

if [[ ! -f "${ROLLOUT_DIR}/results.jsonl" ]]; then
  echo "GPT-5.5 trajectory cache is missing: ${ROLLOUT_DIR}" >&2
  exit 1
fi

result_count="$(wc -l < "${ROLLOUT_DIR}/results.jsonl")"
if [[ "${result_count}" -lt 80 ]]; then
  echo "GPT-5.5 trajectory cache is incomplete: ${result_count}/80 candidates" >&2
  exit 1
fi

if [[ ! -f "${SPLIT_DIR}/train/items.json" ]]; then
  echo "Materialized SpreadsheetBench split is missing: ${SPLIT_DIR}" >&2
  exit 1
fi

if ! curl --fail --silent "${INFERENCE_BASE_URL}/health" >/dev/null; then
  echo "Soft-prefix vLLM service is not healthy: ${INFERENCE_BASE_URL}/health" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" scripts/train_soft_prefix.py \
  --config "${CONFIG_PATH}" \
  --split_dir "${SPLIT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --out_root "${OUTPUT_DIR}"
