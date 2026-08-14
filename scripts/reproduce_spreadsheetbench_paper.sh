#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/local/spreadsheetbench_paper_gpt55.local.yaml}"
SPLIT_DIR="${SPLIT_DIR:-${PROJECT_DIR}/data/spreadsheetbench_split}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/SoftSkill_spreadsheetbench_paper_len8_seed1}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
COLLECTION_OUTPUT_DIR="${COLLECTION_OUTPUT_DIR:-${PROJECT_DIR}/outputs/SpreadsheetBench_teacher_gpt55_collection}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing local config: ${CONFIG_PATH}" >&2
  exit 1
fi

if grep -q "REPLACE_ME" "${CONFIG_PATH}"; then
  echo "Local GPT-5.5 config still contains REPLACE_ME values: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${SPLIT_DIR}/train/items.json" ]]; then
  echo "Materialized SpreadsheetBench split is missing: ${SPLIT_DIR}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" scripts/collect_spreadsheetbench_teacher_rollouts.py \
  --config "${CONFIG_PATH}" \
  --split_dir "${SPLIT_DIR}" \
  --out_root "${COLLECTION_OUTPUT_DIR}"

exec env \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  SPLIT_DIR="${SPLIT_DIR}" \
  "${PROJECT_DIR}/scripts/train_spreadsheetbench_paper_from_cache.sh"
