#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/u6ow/zijian.u6ow/softskill"
MODEL_PATH="/home/u6ow/zijian.u6ow/model_cache/huggingface/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
CONFIG="configs/spreadsheetbench/entropy_localization_stage2.yaml"
MANIFEST_ROOT="outputs/SpreadsheetBench_entropy_localization/manifests"
OUTPUT_ROOT="outputs/SpreadsheetBench_entropy_localization_train_len8_seed1"

cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

methods=(
  random
  entropy
  positive_gain
  js
  skill_additive
  eac_lambda0.25
  eac_lambda0.5
  eac_lambda1
)

mkdir -p "$OUTPUT_ROOT"
for method in "${methods[@]}"; do
  manifest="$MANIFEST_ROOT/${method}_top0.05.jsonl"
  output="$OUTPUT_ROOT/$method"
  test -f "$manifest" || { echo "Missing manifest: $manifest" >&2; exit 1; }
  if test -f "$output/summary.json"; then
    echo "[$method] summary exists; skipping"
    continue
  fi
  test ! -e "$output" || { echo "[$method] incomplete output exists: $output" >&2; exit 1; }
  echo "[$method] starting $(date -u +%FT%TZ)"
  mkdir -p "$output"
  python -u scripts/train_soft_prefix.py \
    --config "$CONFIG" \
    --out_root "$output" \
    --model_name "$MODEL_PATH" \
    --cfg-options \
      soft_prefix.trajectory_examples_path="$manifest" \
    2>&1 | tee "$output/train.log"
  echo "[$method] finished $(date -u +%FT%TZ)"
done
