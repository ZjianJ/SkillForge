#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server for soft-prefix prompt embeddings.
# Override values with environment variables instead of editing this file.
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
PORT="${PORT:-8010}"
GPU_IDS="${GPU_IDS:-0}"
DTYPE="${DTYPE:-bfloat16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

args=(
  --model_name "${MODEL_NAME}"
  --port "${PORT}"
  --dtype "${DTYPE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --enable-auto-tool-choice
  --tool-call-parser "${TOOL_CALL_PARSER}"
)

if [[ -n "${REASONING_PARSER}" ]]; then
  args+=(--reasoning-parser "${REASONING_PARSER}")
fi
if [[ "${LANGUAGE_MODEL_ONLY}" == "1" ]]; then
  args+=(--language-model-only)
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  args+=(--enable-chunked-prefill)
fi

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -m skillopt.softprefix.vllm_prompt_embeds "${args[@]}" ${EXTRA_ARGS}
