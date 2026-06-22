#!/usr/bin/env bash
set -euo pipefail

# Cross-model transfer eval for seed-1 soft-prefix skills.
# For hidden-size mismatches, converts the source prefix through vocab space:
#   source_prefix -> softmax(source_vocab) -> target_vocab_embeddings.
# Text-only tasks default to prompt-embeds vLLM. Set INFERENCE_BACKEND=local_hf
# to use in-process transformers instead. MAX_NEW_TOKENS controls generation
# length for both text backends. DocVQA always uses local HF/VLM.

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"

OUT_BASE="${OUT_BASE:-outputs/skill_section/cross_model_transfer_seed1}"
CONVERTED_PREFIX_BASE="${CONVERTED_PREFIX_BASE:-outputs/skill_section/cross_model_transfer_prefixes}"
TRANSFER_TEMPERATURE="${TRANSFER_TEMPERATURE:-0.05}"
TRANSFER_TOP_K="${TRANSFER_TOP_K:-128}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm_prompt_embeds}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
PROMPT_EMBEDS_4B_BASE_URL="${PROMPT_EMBEDS_4B_BASE_URL:-http://127.0.0.1:8010}"
PROMPT_EMBEDS_35BA3B_BASE_URL="${PROMPT_EMBEDS_35BA3B_BASE_URL:-http://127.0.0.1:8020}"

case "${INFERENCE_BACKEND}" in
    local | local_hf | local-hf | vllm | vllm_prompt_embeds | prompt_embeds | prompt-embeds) ;;
    *)
        echo "INFERENCE_BACKEND must be one of local_hf or vllm_prompt_embeds, got: ${INFERENCE_BACKEND}" >&2
        exit 1
        ;;
esac

run_eval() {
    local out_root=""
    local args=()

    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--out_root" ]]; then
            out_root="$2"
            args+=("$1" "$2")
            shift 2
        else
            args+=("$1")
            shift
        fi
    done

    if [[ -n "${out_root}" && -f "${out_root}/summary.json" ]]; then
        echo "Skipping (summary.json exists): ${out_root}"
        return 0
    fi

    python scripts/train_soft_prefix.py "${args[@]}"
}

model_short_name() {
    local model="$1"
    model="${model#Qwen/}"
    model="${model//\//-}"
    echo "${model}"
}

prefix_checkpoint() {
    local task="$1"
    local model="$2"

    case "${model}:${task}" in
        Qwen/Qwen3.5-4B:livemath) echo "outputs/skill_section/main_livemath_seed1/best_prefix.pt" ;;
        Qwen/Qwen3.5-4B:searchqa) echo "outputs/skill_section/main_searchqa_seed1/best_prefix.pt" ;;
        Qwen/Qwen3.5-4B:docvqa) echo "outputs/skill_section/main_docvqa_seed1/best_prefix.pt" ;;
        Qwen/Qwen3.6-35B-A3B:livemath) echo "outputs/skill_section/model_Qwen3.6-35B-A3B_livemath/best_prefix.pt" ;;
        Qwen/Qwen3.6-35B-A3B:searchqa) echo "outputs/skill_section/model_Qwen3.6-35B-A3B_searchqa/best_prefix.pt" ;;
        Qwen/Qwen3.6-35B-A3B:docvqa) echo "outputs/skill_section/model_Qwen3.6-35B-A3B_docvqa/best_prefix.pt" ;;
        *) echo "Unknown source model/task: ${model} ${task}" >&2; return 1 ;;
    esac
}

prompt_embeds_base_url() {
    case "$1" in
        Qwen/Qwen3.5-4B) echo "${PROMPT_EMBEDS_4B_BASE_URL}" ;;
        Qwen/Qwen3.6-35B-A3B) echo "${PROMPT_EMBEDS_35BA3B_BASE_URL}" ;;
        *) echo "Unknown prompt-embeds model: $1" >&2; return 1 ;;
    esac
}

text_inference_cfg_options() {
    local model="$1"
    case "${INFERENCE_BACKEND}" in
        local | local_hf | local-hf)
            echo "soft_prefix.inference_backend=local_hf"
            echo "soft_prefix.max_new_tokens=${MAX_NEW_TOKENS}"
            ;;
        vllm | vllm_prompt_embeds | prompt_embeds | prompt-embeds)
            echo "soft_prefix.inference_backend=vllm_prompt_embeds"
            echo "soft_prefix.inference_base_url=$(prompt_embeds_base_url "${model}")"
            echo "soft_prefix.inference_timeout_seconds=600"
            echo "soft_prefix.max_new_tokens=${MAX_NEW_TOKENS}"
            ;;
        *)
            echo "INFERENCE_BACKEND must be one of local_hf or vllm_prompt_embeds, got: ${INFERENCE_BACKEND}" >&2
            return 1
            ;;
    esac
}

task_architecture() {
    case "$1" in
        docvqa) echo "vision_lm" ;;
        livemath | searchqa) echo "causal_lm" ;;
        *) echo "Unknown task: $1" >&2; return 1 ;;
    esac
}

task_injection_position() {
    case "$1" in
        searchqa) echo "skill_section" ;;
        docvqa | livemath) echo "skill_section" ;;
        *) echo "Unknown task: $1" >&2; return 1 ;;
    esac
}

converted_checkpoint_path() {
    local task="$1"
    local source_model="$2"
    local target_model="$3"
    local source_short
    local target_short
    source_short="$(model_short_name "${source_model}")"
    target_short="$(model_short_name "${target_model}")"
    echo "${CONVERTED_PREFIX_BASE}/${task}_seed1_${source_short}_to_${target_short}.pt"
}

checkpoint_for_target() {
    local task="$1"
    local source_model="$2"
    local target_model="$3"
    local checkpoint="$4"
    local architecture
    local converted

    if [[ "${target_model}" == "${source_model}" ]]; then
        echo "${checkpoint}"
        return 0
    fi

    architecture="$(task_architecture "${task}")"
    converted="$(converted_checkpoint_path "${task}" "${source_model}" "${target_model}")"
    if [[ ! -f "${converted}" ]]; then
        echo "Converting ${task} prefix ${source_model} -> ${target_model}: ${converted}" >&2
        python scripts/analysis/convert_soft_prefix_vocab.py \
            --source_checkpoint "${checkpoint}" \
            --output_path "${converted}" \
            --source_model_name "${source_model}" \
            --target_model_name "${target_model}" \
            --source_architecture "${architecture}" \
            --target_architecture "${architecture}" \
            --temperature "${TRANSFER_TEMPERATURE}" \
            --top_k "${TRANSFER_TOP_K}" \
            >&2
    else
        echo "Using converted checkpoint: ${converted}" >&2
    fi
    echo "${converted}"
}

tasks=(livemath searchqa docvqa)
source_models=(Qwen/Qwen3.5-4B Qwen/Qwen3.6-35B-A3B)
# Qwen/Qwen3.5-4B 
target_models=(Qwen/Qwen3.5-4B Qwen/Qwen3.6-35B-A3B)

for task in "${tasks[@]}"; do
    for source_model in "${source_models[@]}"; do
        source_checkpoint="$(prefix_checkpoint "${task}" "${source_model}")"
        if [[ ! -f "${source_checkpoint}" ]]; then
            echo "Missing checkpoint for ${task} ${source_model}: ${source_checkpoint}" >&2
            exit 1
        fi

        for target_model in "${target_models[@]}"; do
            if [[ "${target_model}" == "${source_model}" ]]; then
                echo "Skipping same-model eval: ${task} ${source_model}" >&2
                continue
            fi

            source_short="$(model_short_name "${source_model}")"
            target_short="$(model_short_name "${target_model}")"
            out_root="${OUT_BASE}/${task}_seed1_${source_short}_to_${target_short}"
            eval_checkpoint="$(checkpoint_for_target "${task}" "${source_model}" "${target_model}" "${source_checkpoint}")"
            injection_position="$(task_injection_position "${task}")"
            inference_cfg_options=()
            if [[ "${task}" != "docvqa" ]]; then
                mapfile -t inference_cfg_options < <(text_inference_cfg_options "${target_model}")
            fi

            echo "=== ${task}: seed-1 prefix ${source_model} -> ${target_model} ==="

            case "${task}" in
                livemath)
                    run_eval \
                        --config configs/livemathematicianbench/soft_prefix.yaml \
                        --split_dir data/livemathematicianbench_split \
                        --model_name "${target_model}" \
                        --cfg-options \
                            train.num_epochs=0 \
                            evaluation.sel_env_num=0 \
                            evaluation.eval_test=true \
                            evaluation.test_env_num=0 \
                            soft_prefix.checkpoint_path="${eval_checkpoint}" \
                            "${inference_cfg_options[@]}" \
                            soft_prefix.injection_position="${injection_position}" \
                            soft_prefix.prefix_length=32 \
                        --out_root "${out_root}"
                    ;;
                searchqa)
                    run_eval \
                        --config configs/searchqa/soft_prefix.yaml \
                        --split_dir data/searchqa_split \
                        --model_name "${target_model}" \
                        --cfg-options \
                            train.num_epochs=0 \
                            evaluation.sel_env_num=0 \
                            evaluation.eval_test=true \
                            evaluation.test_env_num=0 \
                            soft_prefix.checkpoint_path="${eval_checkpoint}" \
                            "${inference_cfg_options[@]}" \
                            soft_prefix.injection_position="${injection_position}" \
                            soft_prefix.prefix_length=32 \
                        --out_root "${out_root}"
                    ;;
                docvqa)
                    run_eval \
                        --config configs/docvqa/soft_prefix.yaml \
                        --split_dir data/docvqa/splits \
                        --model_name "${target_model}" \
                        --cfg-options \
                            train.num_epochs=0 \
                            evaluation.sel_env_num=0 \
                            evaluation.eval_test=true \
                            evaluation.test_env_num=0 \
                            soft_prefix.checkpoint_path="${eval_checkpoint}" \
                            soft_prefix.docvqa_max_image_tokens=10000 \
                            soft_prefix.max_new_tokens="${MAX_NEW_TOKENS}" \
                            soft_prefix.injection_position="${injection_position}" \
                            soft_prefix.prefix_length=32 \
                        --out_root "${out_root}"
                    ;;
            esac

            sleep 1
        done
    done
done
