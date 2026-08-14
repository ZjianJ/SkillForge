#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/spreadsheetbench/selective_distillation_stage2.yaml"
MANIFEST_ROOT="outputs/SpreadsheetBench_selective_stage2_manifests"
RUN_ROOT="outputs/SpreadsheetBench_full_distribution_locator_len8_seed1_shared"
GPU_IDS="${GPU_IDS:-0}"
ARCHIVE="${SPREADSHEETBENCH_ARCHIVE:-}"

mkdir -p "$RUN_ROOT"
python scripts/prepare_spreadsheetbench_selective_stage2.py \
  > "$RUN_ROOT/manifest_preparation.log"

if [[ -f "$ARCHIVE" ]]; then
  python scripts/audit_spreadsheetbench_inputs.py --archive "$ARCHIVE" \
    | tee "$RUN_ROOT/input_integrity_before.json"
fi

variants=(
  positive_gain_top0.05_core_shared_preserve
  combined_top0.05_core_shared_preserve
)

for name in "${variants[@]}"; do
  output="$RUN_ROOT/$name"
  mkdir -p "$output"
  if [[ -f "$output/summary.json" ]]; then
    echo "[shared-preserve locator] completed summary exists: $output/summary.json"
    continue
  fi
  CUDA_VISIBLE_DEVICES="$GPU_IDS" python scripts/train_soft_prefix.py \
    --config "$CONFIG" \
    --out_root "$output" \
    --cfg-options \
      "train.num_epochs=1" \
      "evaluation.sel_env_num=40" \
      "evaluation.test_env_num=0" \
      "evaluation.eval_test=false" \
      "env.checkpoint_eval_val=false" \
      "soft_prefix.prefix_length=8" \
      "soft_prefix.trajectory_examples_path=$MANIFEST_ROOT/$name.jsonl" \
      "soft_prefix.selective_label_field=selected_indices" \
      "soft_prefix.preservation_loss_weight=1.0" \
      "soft_prefix.preservation_label_field=preserve_indices" \
    2>&1 | tee "$output/train.log"
done

if [[ -f "$ARCHIVE" ]]; then
  python scripts/audit_spreadsheetbench_inputs.py --archive "$ARCHIVE" \
    | tee "$RUN_ROOT/input_integrity_after.json"
fi
