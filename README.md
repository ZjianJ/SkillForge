# SoftSkill

SoftSkill is a research codebase for training **soft-prefix skills** for frozen open-weight language and vision-language models. It builds on the SkillOpt codebase and keeps the Python package name `skillopt` for compatibility, while adding soft-prefix and LoRA training, prompt-embedding vLLM serving, transfer utilities, and benchmark integrations.

This repository is derived from Microsoft SkillOpt, released under the MIT License. The original hard-skill training loop is still available; the SoftSkill release focuses on the soft-prefix stack in `skillopt/softprefix/`.

## What Is Included

- Core package: `skillopt/`
- Soft-prefix and LoRA training: `skillopt/softprefix/`, `scripts/train_soft_prefix.py`
- Supported configs: `configs/*/soft_prefix.yaml` and `configs/*/lora.yaml`
- Public launch examples: `scripts/train/`
- Research experiment helpers: `scripts/experiments/`
- Analysis utilities: `scripts/analysis/`
- Dataset split manifests and data setup notes: `data/README.md`
- Tests: `tests/`

Generated rollouts, outputs, local corpora, checkpoints, private environment files, and the separate `soft-skill/` paper repo are intentionally excluded from the public code release.

## Install

```bash
git clone https://github.com/xijia-tao/SoftSkill.git
cd SoftSkill
pip install -e .
```

For soft-prefix training and local serving:

```bash
pip install -e ".[softprefix,qwen]"
```

For development:

```bash
pip install -e ".[dev,softprefix]"
```

## Configure Credentials

```bash
cp .env.example .env
set -a
source .env
set +a
```

Only configure the backends you use. The OpenAI-compatible mode reuses `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_AUTH_MODE=openai_compatible`.

## Start A Prompt-Embeds Server

```bash
MODEL_NAME=Qwen/Qwen3.5-4B GPU_IDS=0 PORT=8010 bash scripts/train/start_server.sh
```

The server exposes an OpenAI-compatible endpoint at `http://127.0.0.1:8010/v1`.

## Train A Soft Prefix

```bash
CONFIG=configs/searchqa/soft_prefix.yaml SPLIT_DIR=data/searchqa_split MODEL_NAME=Qwen/Qwen3.5-4B OUTPUT_DIR=outputs/SoftSkill_searchqa_example bash scripts/train/train_soft_prefix.sh
```

You can also call the Python entry point directly:

```bash
python scripts/train_soft_prefix.py   --config configs/searchqa/soft_prefix.yaml   --split_dir data/searchqa_split   --model_name Qwen/Qwen3.5-4B   --out_root outputs/SoftSkill_searchqa_example
```

## Evaluate A Soft Prefix

```bash
CHECKPOINT_PATH=outputs/SoftSkill_searchqa_example/best_prefix.pt OUTPUT_DIR=outputs/SoftSkill_searchqa_eval bash scripts/train/eval_soft_prefix.sh
```

Hard-skill SkillOpt evaluation remains available through:

```bash
python scripts/eval_only.py   --config configs/searchqa/default.yaml   --skill ckpt/searchqa/gpt5.5_skill.md   --split valid_unseen   --split_dir data/searchqa_split
```

## Repository Hygiene

Before publishing or opening a PR, check that generated artifacts are not staged:

```bash
git status --short
python -m pytest tests/test_softprefix_data.py tests/test_softprefix_vllm_inference.py -q
```

See `docs/release.md` for the release boundary and `docs/guide/softskill.md` for the soft-prefix workflow.

## License And Attribution

This project is released under the MIT License. It is derived from Microsoft SkillOpt, whose copyright notice is preserved in `LICENSE`.
