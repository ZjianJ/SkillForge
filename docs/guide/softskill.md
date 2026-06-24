# Soft-Prefix Workflow

SoftSkill keeps the SkillOpt-compatible import path `skillopt` while adding a
soft-prefix training stack under `skillopt/softprefix/`.

## Install

```bash
pip install -e ".[dev,softprefix]"
```

Install `.[qwen]` as well when using local vLLM/Qwen serving.

## Serve Prompt Embeddings

```bash
MODEL_NAME=Qwen/Qwen3.5-4B GPU_IDS=0 PORT=8010 bash scripts/train/start_server.sh
```

The server exposes an OpenAI-compatible endpoint at
`http://127.0.0.1:8010/v1`.

## Train

```bash
CONFIG=configs/searchqa/soft_prefix.yaml \
SPLIT_DIR=data/searchqa_split \
MODEL_NAME=Qwen/Qwen3.5-4B \
OUTPUT_DIR=outputs/SoftSkill_searchqa_example \
bash scripts/train/train_soft_prefix.sh
```

You can also call the Python entry point directly:

```bash
python scripts/train_soft_prefix.py \
  --config configs/searchqa/soft_prefix.yaml \
  --split_dir data/searchqa_split \
  --model_name Qwen/Qwen3.5-4B \
  --out_root outputs/SoftSkill_searchqa_example
```

## Evaluate

```bash
CHECKPOINT_PATH=outputs/SoftSkill_searchqa_example/best_prefix.pt \
OUTPUT_DIR=outputs/SoftSkill_searchqa_eval \
bash scripts/train/eval_soft_prefix.sh
```

Use `scripts/eval_only.py` with `ckpt/*/gpt5.5_skill.md` when comparing
against retained hard-skill SkillOpt reference artifacts.
