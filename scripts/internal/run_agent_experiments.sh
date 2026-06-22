#!/usr/bin/env bash
# Batch runner for agentic soft-prefix SkillOpt checkpoint-skill reproduction.
#
# Recommended 8x H100 layout:
#   GPU 0      : Qwen/Qwen3.6-35B-A3B vllm_prompt_embeds server on port 8020
#   GPUs 4-7   : one experiment job at a time
#
# Start the 35B-A3B prompt-embeds server separately, for example:
#   CUDA_VISIBLE_DEVICES=0 python -m skillopt.softprefix.vllm_prompt_embeds \
#     --model_name Qwen/Qwen3.6-35B-A3B \
#     --port 8020 \
#     --dtype bfloat16 \
#     --enable-auto-tool-choice \
#     --tool-call-parser qwen3_coder \
#     --gpu-memory-utilization 0.9 \
#     --max-model-len 65536 \
#     --language-model-only \
#     --enable-chunked-prefill
#
# Default soft-prefix setting:
#   - initialize from the GPT 5.5 SkillOpt checkpoint skill markdown
#   - inject at the skill section
#   - use prefix_length=auto
#   - evaluate the initialized prefix to reproduce checkpoint-skill performance
#
# Override defaults as needed:
#   TRAIN_GPUS=0,1,2,3 QWEN36_35BA3B_BASE_URL=http://127.0.0.1:8020 bash run_agent_experiments.sh
#   EVAL_INIT_ONLY=0 bash run_agent_experiments.sh
#   EVAL_TEST=1 TEST_ENV_NUM=64 bash run_agent_experiments.sh
#   EVAL_VAL=1 bash run_agent_experiments.sh
#   RUN_MODEL_ABLATIONS=1 bash run_agent_experiments.sh
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

TRAIN_GPUS="${TRAIN_GPUS:-4,5,6,7}"
QWEN36_35BA3B_BASE_URL="${QWEN36_35BA3B_BASE_URL:-http://127.0.0.1:8020}"
PROMPT_EMBEDS_BASE_URL="${PROMPT_EMBEDS_BASE_URL:-${QWEN36_35BA3B_BASE_URL}}"
OUT_BASE="${OUT_BASE:-outputs/agentic}"
RUN_MODEL_ABLATIONS="${RUN_MODEL_ABLATIONS:-0}"
EVAL_INIT_ONLY="${EVAL_INIT_ONLY:-0}"
EVAL_VAL="${EVAL_VAL:-1}"
EVAL_TEST="${EVAL_TEST:-1}"
EVAL_PREFIX="${EVAL_PREFIX:-false}"
TEST_ENV_NUM="${TEST_ENV_NUM:-0}"
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

task_initial_path() {
    case "$1" in
        officeqa) echo "skillopt/envs/officeqa/skills/initial.md" ;;
        alfworld) echo "skillopt/envs/alfworld/skills/initial.md" ;;
        spreadsheetbench) echo "skillopt/envs/spreadsheetbench/skills/initial.md" ;;
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

task_qwen36_rollout_dir() {
    case "$1" in
        officeqa) echo "rollouts/qwen36_officeqa_rollouts" ;;
        alfworld) echo "rollouts/qwen36_alfworld_rollouts" ;;
        spreadsheetbench) echo "rollouts/qwen36_spreadsheetbench_rollouts" ;;
        *) echo "Unknown task: $1" >&2; return 1 ;;
    esac
}

task_opts() {
    local task="$1"
    local rollout_dir="$2"
    local rollouts_per_task="${3:-1}"
    case "${task}" in
        officeqa)
            echo \
                "train.batch_size=1" \
                "soft_prefix.trajectory_rollout_backend=openai_compatible" \
                "soft_prefix.trajectory_rollout_dir=${rollout_dir}" \
                "soft_prefix.trajectory_use_skill=true" \
                "soft_prefix.trajectory_rollouts_per_task=${rollouts_per_task}"
            ;;
        alfworld)
            echo \
                "train.batch_size=4" \
                "soft_prefix.max_prompt_tokens=8192" \
                "soft_prefix.strip_trajectory_thoughts=false" \
                "soft_prefix.trajectory_rollout_backend=openai_compatible" \
                "soft_prefix.trajectory_rollout_dir=${rollout_dir}" \
                "soft_prefix.trajectory_use_skill=false" \
                "soft_prefix.trajectory_rollouts_per_task=${rollouts_per_task}" \
                "soft_prefix.trajectory_max_new_tokens=16384"
            ;;
        spreadsheetbench)
            echo \
                "train.batch_size=2" \
                "env.max_turns=5" \
                "soft_prefix.max_new_tokens=2048" \
                "soft_prefix.trajectory_max_new_tokens=4096" \
                "soft_prefix.trajectory_rollout_backend=openai_compatible" \
                "soft_prefix.trajectory_rollout_dir=${rollout_dir}" \
                "soft_prefix.trajectory_use_skill=true" \
                "soft_prefix.trajectory_rollouts_per_task=${rollouts_per_task}"
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

    local config split skill_path rollout_dir rollouts_per_task out_root model_name inference_base_url init_text_path prefix_length
    config="$(task_config "${task}")"
    split="$(task_split "${task}")"
    skill_path="$(task_skill_path "${task}")"
    rollout_dir="${AGENT_TRAJECTORY_ROLLOUT_DIR:-$(task_gpt55_rollout_dir "${task}")}"
    if [[ -n "${AGENT_TRAJECTORY_ROLLOUTS_PER_TASK:-}" ]]; then
        rollouts_per_task="${AGENT_TRAJECTORY_ROLLOUTS_PER_TASK}"
    elif [[ "${rollout_dir}" == rollouts/qwen36_* ]]; then
        rollouts_per_task=3
    else
        rollouts_per_task=1
    fi
    out_root="${OUT_PREFIX}/${out_name}_${task}"
    model_name="${AGENT_MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}"
    inference_base_url="${AGENT_PROMPT_EMBEDS_BASE_URL:-${PROMPT_EMBEDS_BASE_URL}}"
    init_text_path="${AGENT_INIT_TEXT_PATH:-${skill_path}}"
    prefix_length="${AGENT_PREFIX_LENGTH:-auto}"

    read -r -a task_cfg_options <<<"$(task_opts "${task}" "${rollout_dir}" "${rollouts_per_task}")"

    local eval_only_options=()
    if [[ "${EVAL_INIT_ONLY}" == "1" ]]; then
        eval_only_options=(train.num_epochs=0)
    fi

    run --config "${config}" \
        --split_dir "${split}" \
        --model_name "${model_name}" \
        --cfg-options \
        model.target=gpt-5.5 \
        model.target_backend=openai_chat \
        model.target_qwen_chat_base_url= \
        model.target_qwen_chat_api_key= \
        model.target_qwen_chat_enable_thinking=false \
        soft_prefix.inference_backend=vllm_prompt_embeds \
        "soft_prefix.inference_base_url=${inference_base_url}" \
        soft_prefix.inference_timeout_seconds=600 \
        soft_prefix.init_strategy=text \
        "soft_prefix.init_text_path=${init_text_path}" \
        soft_prefix.injection_position=skill_section \
        "soft_prefix.prefix_length=${prefix_length}" \
        soft_prefix.eval_init_prefix="${EVAL_PREFIX}" \
        "soft_prefix.eval_init_val=${EVAL_VAL}" \
        soft_prefix.max_new_tokens=16384 \
        "evaluation.eval_test=${EVAL_TEST}" \
        "evaluation.test_env_num=${TEST_ENV_NUM}" \
        "${task_cfg_options[@]}" \
        "${eval_only_options[@]}" \
        "$@" \
        --out_root "${out_root}"

    pause_between_runs
}

run_model_ablation() {
    local model="$1"
    local task=officeqa
    local mshort="${model#Qwen/}"
    local skill_path
    skill_path="$(task_skill_path "${task}")"

    local eval_only_options=()
    if [[ "${EVAL_INIT_ONLY}" == "1" ]]; then
        eval_only_options=(train.num_epochs=0)
    fi

    run --config configs/officeqa/soft_prefix.yaml \
        --split_dir data/officeqa_split \
        --model_name "${model}" \
        --cfg-options \
        train.seed=1 \
        train.batch_size=1 \
        soft_prefix.inference_backend=local_hf \
        soft_prefix.trajectory_rollout_backend=local_hf \
        "soft_prefix.trajectory_rollout_dir=rollouts/local_hf_${mshort}_officeqa" \
        soft_prefix.init_strategy=text \
        "soft_prefix.init_text_path=${skill_path}" \
        soft_prefix.injection_position=skill_section \
        soft_prefix.prefix_length=auto \
        soft_prefix.eval_init_prefix="${EVAL_PREFIX}" \
        "soft_prefix.eval_init_val=${EVAL_VAL}" \
        soft_prefix.max_new_tokens=16384 \
        "evaluation.eval_test=${EVAL_TEST}" \
        "evaluation.test_env_num=${TEST_ENV_NUM}" \
        "${eval_only_options[@]}" \
        --out_root "${OUT_PREFIX}/model_${mshort}_officeqa"
    pause_between_runs
}

TASKS=(alfworld officeqa spreadsheetbench)
OUT_PREFIX="${OUT_BASE}/skill_section"

echo "=== SkillOpt checkpoint-skill soft-prefix reproduction ==="
echo "=== default model=Qwen/Qwen3.6-35B-A3B, prompt-embeds server=${QWEN36_35BA3B_BASE_URL} ==="
echo "=== injection_position=skill_section, prefix_length=auto, eval_init_prefix=true ==="
echo "=== EVAL_INIT_ONLY=${EVAL_INIT_ONLY}, EVAL_VAL=${EVAL_VAL}, EVAL_TEST=${EVAL_TEST}, TEST_ENV_NUM=${TEST_ENV_NUM} ==="

echo "=== EXP 1: default GPT 5.5 checkpoint init, prefix_length=auto ==="
for TASK in "${TASKS[@]}"; do
    run_agent_task "${TASK}" "skillopt_qwen36_35ba3b" \
        train.seed=1
done

# no need for prefix_length sweep when training-free
# echo "=== EXP 2: prefix_length sweep, GPT 5.5 checkpoint init ==="
# 8 32 
for LEN in 8 32 256; do
    for TASK in "${TASKS[@]}"; do
        AGENT_PREFIX_LENGTH="${LEN}" \
        run_agent_task "${TASK}" "prefix_len${LEN}" \
            train.seed=1
    done
done

echo "=== EXP 3: init_text_path sweep, prefix_length=auto ==="
for TASK in "${TASKS[@]}"; do
    INITIAL_SKILL_PATH="$(task_initial_path "${TASK}")"

    AGENT_INIT_TEXT_PATH="${INITIAL_SKILL_PATH}" \
    AGENT_PREFIX_LENGTH=auto \
    run_agent_task "${TASK}" "init_initial" \
        train.seed=1
done

echo "=== EXP 4: Qwen36 rollouts, GPT 5.5 checkpoint init, prefix_length=auto ==="
for TASK in "${TASKS[@]}"; do
    AGENT_TRAJECTORY_ROLLOUT_DIR="$(task_qwen36_rollout_dir "${TASK}")" \
    run_agent_task "${TASK}" "qwen36_rollouts_skillopt_qwen36_35ba3b" \
        train.seed=1
done

echo "=== EXP 5: Qwen36 rollouts, prefix_length sweep, GPT 5.5 checkpoint init ==="
for LEN in 256; do
    for TASK in "${TASKS[@]}"; do
        AGENT_PREFIX_LENGTH="${LEN}" \
        AGENT_TRAJECTORY_ROLLOUT_DIR="$(task_qwen36_rollout_dir "${TASK}")" \
        run_agent_task "${TASK}" "qwen36_rollouts_prefix_len${LEN}" \
            train.seed=1
    done
done

echo "=== EXP 6: Qwen36 rollouts, init_text_path sweep, prefix_length=auto ==="
for TASK in "${TASKS[@]}"; do
    INITIAL_SKILL_PATH="$(task_initial_path "${TASK}")"

    AGENT_INIT_TEXT_PATH="${INITIAL_SKILL_PATH}" \
    AGENT_PREFIX_LENGTH=auto \
    AGENT_TRAJECTORY_ROLLOUT_DIR="$(task_qwen36_rollout_dir "${TASK}")" \
    run_agent_task "${TASK}" "qwen36_rollouts_init_initial" \
        train.seed=1
done

if [[ "${RUN_MODEL_ABLATIONS}" == "1" ]]; then
    echo "=== Optional OfficeQA local_hf model ablations ==="
    echo "ALFWorld and SpreadsheetBench reject local_hf trajectory rollouts; use a separate qwen_chat/vLLM server or cached rollouts for those model ablations."
    OUT_PREFIX="${OUT_BASE}/skill_section/model_ablation"
    for MODEL in Qwen/Qwen3.5-9B Qwen/Qwen3.6-35B-A3B; do
        run_model_ablation "${MODEL}"
    done
fi

echo "=== All agentic experiments complete ==="

# OUT_BASE=outputs/agentic_init_eval_modified EVAL_INIT_ONLY=1 bash run_agent_experiments.sh