# SoftSkill

SoftSkill is a research codebase for training soft-prefix skills for frozen
language and vision-language models.

The public project is SoftSkill. The Python import package remains `skillopt`
for compatibility with the SkillOpt-derived codebase, configs, scripts, and
checkpoint skills.

## Start Here

- Install with `pip install -e ".[dev,softprefix]"`.
- Train soft prefixes with `scripts/train_soft_prefix.py` or the scripts in
  `scripts/train/`.
- Evaluate retained hard-skill baselines with `scripts/eval_only.py` and the
  reference skills in `ckpt/`.
