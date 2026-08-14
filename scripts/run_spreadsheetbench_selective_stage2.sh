#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python scripts/prepare_spreadsheetbench_selective_stage2.py

CONFIG="configs/spreadsheetbench/selective_distillation_stage2.yaml"
MANIFEST_ROOT="outputs/SpreadsheetBench_selective_stage2_manifests"
RUN_ROOT="outputs/SpreadsheetBench_selective_stage2_len8_seed1_safe"
ARCHIVE="${SPREADSHEETBENCH_ARCHIVE:-}"

mkdir -p "$RUN_ROOT"
if [[ -f "$ARCHIVE" ]]; then
  python scripts/audit_spreadsheetbench_inputs.py --archive "$ARCHIVE" \
    | tee "$RUN_ROOT/input_integrity_before.json"
fi

run_variant() {
  local name="$1"
  local label_field="$2"
  local output="$RUN_ROOT/$name"
  if [[ -f "$output/summary.json" ]]; then
    echo "[stage2] skip completed $name"
    return
  fi
  mkdir -p "$output"
  python scripts/train_soft_prefix.py \
    --config "$CONFIG" \
    --out_root "$output" \
    --cfg-options \
      "soft_prefix.trajectory_examples_path=$MANIFEST_ROOT/$name.jsonl" \
      "soft_prefix.selective_label_field=$label_field" \
    2>&1 | tee "$output/train.log"
}

run_variant clean_full ""
run_variant random_top0.05_core selected_indices
run_variant positive_gain_top0.05_core selected_indices
run_variant combined_top0.05_core selected_indices
run_variant positive_gain_top0.1_core selected_indices
run_variant combined_top0.05_L1_R2 selected_indices

if [[ -f "$ARCHIVE" ]]; then
  python scripts/audit_spreadsheetbench_inputs.py --archive "$ARCHIVE" \
    | tee "$RUN_ROOT/input_integrity_after.json"
fi

python scripts/report_spreadsheetbench_selective_stage2.py \
  --run-root "$RUN_ROOT" \
  --manifest-summary "$MANIFEST_ROOT/summary.json"
