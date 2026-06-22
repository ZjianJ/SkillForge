export ALFWORLD_DATA="${ALFWORLD_DATA:-${HOME}/.cache/alfworld}"

for TASK in alfworld; do
  case "$TASK" in
    alfworld)
      CONFIG=configs/alfworld/soft_prefix_rej.yaml
      SPLIT=data/alfworld_path_split
      EXTRA="soft_prefix.max_prompt_tokens=8192 soft_prefix.max_new_tokens=16384"
      ;;
    officeqa)
      CONFIG=configs/officeqa/soft_prefix.yaml
      SPLIT=data/officeqa_split
      EXTRA="soft_prefix.max_new_tokens=16384"
      ;;
    spreadsheetbench)
      CONFIG=configs/spreadsheetbench/soft_prefix.yaml
      SPLIT=data/spreadsheetbench_split
      EXTRA="env.max_turns=5 soft_prefix.max_new_tokens=2048"
      ;;
  esac

  CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/train_soft_prefix.py \
    --config "$CONFIG" \
    --split_dir "$SPLIT" \
    --model_name Qwen/Qwen3.6-35B-A3B \
    --cfg-options \
      model.target=gpt-5.5 \
      model.target_backend=openai_chat \
      soft_prefix.inference_backend=vllm_prompt_embeds \
      soft_prefix.inference_base_url=http://127.0.0.1:8020 \
      soft_prefix.inference_timeout_seconds=600 \
      soft_prefix.prefix_length=1 \
      soft_prefix.init_strategy=random \
      soft_prefix.init_text_path= \
      soft_prefix.injection_position=prompt_start \
      soft_prefix.eval_init_prefix=false \
      soft_prefix.eval_plain_baseline=true \
      evaluation.eval_test=true \
      evaluation.test_env_num=0 \
      train.num_epochs=0 \
      train.seed=1 \
      $EXTRA \
    --out_root "outputs/skill_section/no_skill_${TASK}"
done