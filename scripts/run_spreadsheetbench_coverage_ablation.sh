#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0}"
MODEL="${SPREADSHEETBENCH_MODEL:-Qwen/Qwen3.6-35B-A3B}"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" scripts/prepare_spreadsheetbench_selective_stage2.py \
  --model-path "$MODEL"

for rate in 05 10 20; do
  manifest="outputs/SpreadsheetBench_selective_stage2_manifests/combined_top0.${rate}_core_coverage_ablation.jsonl"
  output="outputs/SpreadsheetBench_task_specific_combined_core${rate}_len8_seed1_coverage_ablation"
  test -f "$manifest" || {
    echo "Missing coverage-ablation manifest: $manifest" >&2
    exit 1
  }
  mkdir -p "$output"
  echo "[coverage] Combined Top-${rate}% core-only -> $output"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" -u \
    scripts/train_spreadsheetbench_task_specific_prefixes.py \
    --model-path "$MODEL" \
    --manifest "$manifest" \
    --out-root "$output" \
    --phase all \
    --eval-conditions soft \
    --prefix-length 8 \
    --learning-rate 0.001 \
    --max-steps 32 \
    --checkpoint-steps 1,4,8,16,32 \
    --preservation-weight 1.0 \
    --delta-weight 0.0001 \
    --max-prompt-tokens 16384 \
    --max-target-tokens 8192 \
    --max-new-tokens 4096 \
    --exec-timeout 600 \
    --seed 1 \
    2>&1 | tee -a "$output/run.log"
done
