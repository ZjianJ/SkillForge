#!/usr/bin/env bash
# Batch runner for agentic soft-prefix experiments.
#
# Recommended 8x H100 layout:
#   GPU 0      : Qwen/Qwen3.5-4B vllm_prompt_embeds server on port 8010
#                used for prompt-embeds inference/evaluation
#   GPUs 4-7   : one training job at a time; 4 cards fit the 4B agentic runs well
#   GPUs 1-3   : leave free for monitoring, data prep, or manual ablations
#
# Start the 4B server separately, for example:
#   CUDA_VISIBLE_DEVICES=0 python -m skillopt.softprefix.vllm_prompt_embeds \
#     --model_name Qwen/Qwen3.5-4B \
#     --port 8010 \
#     --dtype bfloat16 \
#     --enable-auto-tool-choice \
#     --tool-call-parser qwen3_coder \
#     --gpu-memory-utilization 0.9 \
#     --max-model-len 65536 \
#     --language-model-only \
#     --enable-chunked-prefill
#
# Override defaults as needed:
#   TRAIN_GPUS=0,1,2,3 PROMPT_EMBEDS_BASE_URL=http://127.0.0.1:8010 bash run_agent_experiments.sh
#   RUN_MODEL_ABLATIONS=1 bash run_agent_experiments.sh
#   RUN_DEFERRED_TESTS=0 bash run_agent_experiments.sh
#   TEST_ENV_NUM=128 bash run_agent_experiments.sh
#   DRY_RUN=1 bash run_agent_experiments.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -f "${HOME}/xijia/xj_config" ]]; then
    # Keep parity with train.sh without requiring the file on every machine.
    # shellcheck disable=SC1090
    source "${HOME}/xijia/xj_config"
fi

export ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_GPUS="${TRAIN_GPUS:-1,2}"
PROMPT_EMBEDS_BASE_URL="${PROMPT_EMBEDS_BASE_URL:-http://127.0.0.1:8010}"
OUT_BASE="${OUT_BASE:-outputs/agentic}"
RUN_MODEL_ABLATIONS="${RUN_MODEL_ABLATIONS:-0}"
RUN_DEFERRED_TESTS="${RUN_DEFERRED_TESTS:-1}"
TEST_ENV_NUM="${TEST_ENV_NUM:-64}"
TEST_OUT_SUFFIX="${TEST_OUT_SUFFIX:-_test}"
DRY_RUN="${DRY_RUN:-0}"

run() {
    local out_root=""
    local cuda_devices="${TRAIN_GPUS}"
    local args=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --out_root)
                out_root="$2"
                args+=("$1" "$2")
                shift 2
                ;;
            --cuda_devices)
                cuda_devices="$2"
                shift 2
                ;;
            *)
                args+=("$1")
                shift
                ;;
        esac
    done

    if [[ -n "${out_root}" && -f "${out_root}/summary.json" ]]; then
        echo "Skipping (summary.json exists): ${out_root}"
        return 0
    fi

    echo "CUDA_VISIBLE_DEVICES=${cuda_devices} python scripts/train_soft_prefix.py ${args[*]}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    CUDA_VISIBLE_DEVICES="${cuda_devices}" python scripts/train_soft_prefix.py "${args[@]}"
}

pause_between_runs() {
    if [[ "${DRY_RUN}" != "1" ]]; then
        sleep 5
    fi
}

task_config() {
    case "$1" in
        officeqa) echo "configs/officeqa/soft_prefix.yaml" ;;
        alfworld) echo "configs/alfworld/soft_prefix_rej.yaml" ;;
        spreadsheetbench) echo "configs/spreadsheetbench/soft_prefix.yaml" ;;
        *) echo "Unknown task: $1" >&2; return 1 ;;
    esac
}

task_split() {
    case "$1" in
        officeqa) echo "data/officeqa_split" ;;
        alfworld) echo "data/alfworld_path_split" ;;
        spreadsheetbench) echo "data/spreadsheetbench_split" ;;
        *) echo "Unknown task: $1" >&2; return 1 ;;
    esac
}

task_skill_path() {
    case "$1" in
        officeqa) echo "ckpt/officeqa/gpt5.5_skill.md" ;;
        alfworld) echo "ckpt/alfworld/gpt5.5_skill.md" ;;
        spreadsheetbench) echo "ckpt/spreadsheetbench/gpt5.5_skill.md" ;;
        *) echo "Unknown task: $1" >&2; return 1 ;;
    esac
}

task_gpt55_rollout_dir() {
    case "$1" in
        officeqa) echo "rollouts/teacher_gpt55_officeqa_rollouts" ;;
        alfworld) echo "rollouts/teacher_gpt55_alfworld_rollouts" ;;
        spreadsheetbench) echo "rollouts/teacher_gpt55_spreadsheetbench_rollouts" ;;
        *) echo "Unknown task: $1" >&2; return 1 ;;
    esac
}

task_opts() {
    local task="$1"
    local rollout_dir="$2"
    case "${task}" in
        officeqa)
            echo \
                "train.batch_size=1" \
                "soft_prefix.trajectory_rollout_backend=openai_compatible" \
                "soft_prefix.trajectory_rollout_dir=${rollout_dir}" \
                "soft_prefix.trajectory_use_skill=true" \
                "soft_prefix.trajectory_rollouts_per_task=1"
            ;;
        alfworld)
            echo \
                "train.batch_size=4" \
                "soft_prefix.max_prompt_tokens=8192" \
                "soft_prefix.strip_trajectory_thoughts=false" \
                "soft_prefix.trajectory_rollout_backend=openai_compatible" \
                "soft_prefix.trajectory_rollout_dir=${rollout_dir}" \
                "soft_prefix.trajectory_use_skill=false" \
                "soft_prefix.trajectory_rollouts_per_task=1" \
                "soft_prefix.trajectory_max_new_tokens=16384"
            ;;
        spreadsheetbench)
            echo \
                "train.batch_size=2" \
                "soft_prefix.trajectory_rollout_backend=openai_compatible" \
                "soft_prefix.trajectory_rollout_dir=${rollout_dir}" \
                "soft_prefix.trajectory_use_skill=true" \
                "soft_prefix.trajectory_rollouts_per_task=1"
            ;;
        *)
            echo "Unknown task: ${task}" >&2
            return 1
            ;;
    esac
}

run_agent_task() {
    local task="$1"
    local out_name="$2"
    shift 2

    local config split rollout_dir train_out_root out_root
    config="$(task_config "${task}")"
    split="$(task_split "${task}")"
    rollout_dir="$(task_gpt55_rollout_dir "${task}")"
    train_out_root="${OUT_PREFIX}/${out_name}_${task}"
    out_root="${train_out_root}"

    read -r -a task_cfg_options <<<"$(task_opts "${task}" "${rollout_dir}")"

    local phase_options=()
    case "${RUN_PHASE:-train}" in
        train)
            phase_options=(evaluation.eval_test=false)
            ;;
        test)
            local checkpoint_path="${train_out_root}/best_prefix.pt"
            out_root="${train_out_root}${TEST_OUT_SUFFIX}"
            if [[ "${DRY_RUN}" != "1" && ! -f "${checkpoint_path}" ]]; then
                echo "Skipping test (missing best_prefix.pt): ${checkpoint_path}"
                return 0
            fi
            phase_options=(
                train.num_epochs=0
                evaluation.sel_env_num=0
                "evaluation.test_env_num=${TEST_ENV_NUM}"
                evaluation.eval_test=true
                "soft_prefix.checkpoint_path=${checkpoint_path}"
            )
            ;;
        *)
            echo "Unknown RUN_PHASE: ${RUN_PHASE}" >&2
            return 1
            ;;
    esac

    run --config "${config}" \
        --split_dir "${split}" \
        --model_name Qwen/Qwen3.5-4B \
        --cfg-options \
        model.target=gpt-5.5 \
        model.target_backend=openai_chat \
        model.target_qwen_chat_base_url= \
        model.target_qwen_chat_api_key= \
        model.target_qwen_chat_enable_thinking=false \
        soft_prefix.inference_backend=vllm_prompt_embeds \
        "soft_prefix.inference_base_url=${PROMPT_EMBEDS_BASE_URL}" \
        soft_prefix.inference_timeout_seconds=600 \
        "${task_cfg_options[@]}" \
        "$@" \
        "${phase_options[@]}" \
        --out_root "${out_root}"

    pause_between_runs
}

TASKS=(officeqa alfworld spreadsheetbench)

run_agent_experiment_matrix() {
    for INJECTION_POSITION in prompt_start skill_section; do
        OUT_PREFIX="${OUT_BASE}/${INJECTION_POSITION}"
        echo "=== Injection position: ${INJECTION_POSITION} (${RUN_PHASE}) ==="

        echo "=== EXP 1: main Qwen3.5-4B agentic runs, 3 seeds (${RUN_PHASE}) ==="
        for SEED in 1 2 3; do
            for TASK in "${TASKS[@]}"; do
                run_agent_task "${TASK}" "main_seed${SEED}" \
                    "train.seed=${SEED}" \
                    "soft_prefix.injection_position=${INJECTION_POSITION}"
            done
        done

        echo "=== EXP 2: prefix length ablations, seed=1 (${RUN_PHASE}) ==="
        for LEN in 8 32 auto; do
            for TASK in "${TASKS[@]}"; do
                run_agent_task "${TASK}" "len${LEN}" \
                    train.seed=1 \
                    "soft_prefix.prefix_length=${LEN}" \
                    "soft_prefix.injection_position=${INJECTION_POSITION}"
            done
        done

        echo "=== EXP 3: initialization ablations, seed=1 (${RUN_PHASE}) ==="
        for TASK in "${TASKS[@]}"; do
            run_agent_task "${TASK}" "random_init_prefix" \
                train.seed=1 \
                soft_prefix.init_strategy=vocab_mean \
                "soft_prefix.injection_position=${INJECTION_POSITION}"

            SKILL_PATH="$(task_skill_path "${TASK}")"
            run_agent_task "${TASK}" "skillopt_prefix" \
                train.seed=1 \
                soft_prefix.init_strategy=text \
                "soft_prefix.init_text_path=${SKILL_PATH}" \
                soft_prefix.prefix_length=auto \
                soft_prefix.max_new_tokens=16384 \
                "soft_prefix.injection_position=${INJECTION_POSITION}"
        done
    done
}

run_model_ablation() {
    local model="$1"
    local mshort="${model#Qwen/}"
    local train_out_root="${OUT_PREFIX}/model_${mshort}_officeqa"
    local out_root="${train_out_root}"
    local phase_options=()

    case "${RUN_PHASE:-train}" in
        train)
            phase_options=(evaluation.eval_test=false)
            ;;
        test)
            local checkpoint_path="${train_out_root}/best_prefix.pt"
            out_root="${train_out_root}${TEST_OUT_SUFFIX}"
            if [[ "${DRY_RUN}" != "1" && ! -f "${checkpoint_path}" ]]; then
                echo "Skipping test (missing best_prefix.pt): ${checkpoint_path}"
                return 0
            fi
            phase_options=(
                train.num_epochs=0
                evaluation.sel_env_num=0
                "evaluation.test_env_num=${TEST_ENV_NUM}"
                evaluation.eval_test=true
                "soft_prefix.checkpoint_path=${checkpoint_path}"
            )
            ;;
        *)
            echo "Unknown RUN_PHASE: ${RUN_PHASE}" >&2
            return 1
            ;;
    esac

    run --config configs/officeqa/soft_prefix.yaml \
        --split_dir data/officeqa_split \
        --model_name "${model}" \
        --cfg-options \
            train.seed=1 \
            train.batch_size=1 \
            soft_prefix.inference_backend=local_hf \
            soft_prefix.trajectory_rollout_backend=local_hf \
            "soft_prefix.trajectory_rollout_dir=rollouts/local_hf_${mshort}_officeqa" \
            "${phase_options[@]}" \
            --out_root "${out_root}"
    pause_between_runs
}

run_model_ablations() {
    echo "=== Optional EXP 4: OfficeQA local_hf model ablations (${RUN_PHASE}) ==="
    echo "ALFWorld and SpreadsheetBench reject local_hf trajectory rollouts; use a separate qwen_chat/vLLM server or cached rollouts for those model ablations."
    OUT_PREFIX="${OUT_BASE}/model_ablation"
    for MODEL in Qwen/Qwen3.5-9B Qwen/Qwen3.6-35B-A3B; do
        run_model_ablation "${MODEL}"
    done
}

RUN_PHASE=train
echo "=== Training phase: final test evaluation disabled ==="
run_agent_experiment_matrix
if [[ "${RUN_MODEL_ABLATIONS}" == "1" ]]; then
    run_model_ablations
fi

if [[ "${RUN_DEFERRED_TESTS}" == "1" ]]; then
    RUN_PHASE=test
    echo "=== Deferred test phase: evaluating best prefixes on valid_unseen ==="
    echo "=== TEST_ENV_NUM=${TEST_ENV_NUM}, TEST_OUT_SUFFIX=${TEST_OUT_SUFFIX} ==="
    run_agent_experiment_matrix
    if [[ "${RUN_MODEL_ABLATIONS}" == "1" ]]; then
        run_model_ablations
    fi
fi

echo "=== All agentic experiments complete ==="
