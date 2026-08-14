#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/spreadsheetbench/selective_distillation_stage2.yaml"
MANIFEST_ROOT="outputs/SpreadsheetBench_selective_stage2_manifests"
RUN_ROOT="outputs/SpreadsheetBench_selective_stage2_len8_seed1_safe"
NAME="positive_gain_top0.05_L2_R8_preserve"
OUTPUT="$RUN_ROOT/$NAME"
GPU_IDS="${GPU_IDS:-0}"
ARCHIVE="${SPREADSHEETBENCH_ARCHIVE:-}"

# Existing stage-1 score caches are reused. This command is a no-op when all
# no-Skill Top-64 references are already present.
python scripts/cache_spreadsheetbench_clean_distributions.py
python scripts/prepare_spreadsheetbench_selective_stage2.py

mkdir -p "$OUTPUT"
if [[ -f "$ARCHIVE" ]]; then
  python scripts/audit_spreadsheetbench_inputs.py --archive "$ARCHIVE" \
    | tee "$OUTPUT/input_integrity_before.json"
fi

if [[ -f "$OUTPUT/summary.json" ]]; then
  echo "[window experiment] completed summary already exists: $OUTPUT/summary.json"
else
  CUDA_VISIBLE_DEVICES="$GPU_IDS" python scripts/train_soft_prefix.py \
    --config "$CONFIG" \
    --out_root "$OUTPUT" \
    --cfg-options \
      "train.num_epochs=1" \
      "soft_prefix.trajectory_examples_path=$MANIFEST_ROOT/$NAME.jsonl" \
      "soft_prefix.selective_label_field=selected_indices" \
      "soft_prefix.preservation_loss_weight=1.0" \
      "soft_prefix.preservation_label_field=preserve_indices" \
    2>&1 | tee "$OUTPUT/train.log"
fi

if [[ -f "$ARCHIVE" ]]; then
  python scripts/audit_spreadsheetbench_inputs.py --archive "$ARCHIVE" \
    | tee "$OUTPUT/input_integrity_after.json"
fi

python scripts/report_spreadsheetbench_positive_preservation.py \
  --run-root "$RUN_ROOT" \
  --variant "$NAME"
