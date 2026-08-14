#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export FLA_USE_FUSED_CONV="0"
export FLA_USE_FAST_CONV1D="0"

cd "$PROJECT_ROOT"
args=(--config configs/spreadsheetbench/selective_distillation_stage1.yaml)
if [[ -n "${SPREADSHEETBENCH_MODEL:-}" ]]; then
  args+=(--model-path "$SPREADSHEETBENCH_MODEL")
fi
exec "${PYTHON_BIN:-python}" scripts/analyze_selective_distillation_tokens.py "${args[@]}" "$@"
