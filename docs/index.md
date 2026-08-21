# SkillForge

SkillForge is a research codebase for selectively distilling natural-language
skills into short soft prompts for frozen language models and agent tasks.

The repository extends the SoftSkill training framework with full-distribution
token localization, preservation losses, dynamic and entropy-aware locators,
PRCB residual learners, Future-Impact Locator, official-adapted distillation
baselines, and task-conditioned prompt diagnostics. The Python import package
remains `skillopt` for compatibility with the inherited environment code.

## Start Here

- Install with `pip install -e ".[dev,softprefix]"`.
- Train soft prefixes with `scripts/train_soft_prefix.py` or the scripts in
  `scripts/train/`.
- Evaluate retained hard-skill baselines with `scripts/eval_only.py` and the
  reference skills in `ckpt/`.
- See `experiment_results_overview.md` for the complete result inventory and
  `spreadsheetbench_official_prefix_baselines.md` for the matched SE-KD-Prefix
  and OPCD-Prefix comparison. The strict Combined 5%/10% coverage-only study is
  documented in `spreadsheetbench_combined_coverage_ablation.md`; the controlled
  replacement of Combined10 one-hot CE by full-vocabulary Skill-KL is documented
  in `spreadsheetbench_combined_full_vocab_skill_kl.md`. The convergence-triggered
  dynamic re-localization version is documented in
  `spreadsheetbench_dynamic_combined_v1.md`.
- Entropy/EAC, continuous locator weighting, strongest-competitor localization,
  and four-signal adaptive experiments are documented in
  `spreadsheetbench_eac_full_sequence_weight_analysis.md`,
  `spreadsheetbench_locator_weighted_kl_experiment.md`,
  `spreadsheetbench_dynamic_gain_competitor_locator.md`, and
  `spreadsheetbench_meta_adaptive_four_signal_locator.md`.
- Future-Impact Locator and task-representation stages are summarized in the
  root `README.md`; their machine-readable summaries are intentionally retained
  in local `outputs/` and are not committed with model outputs.
- `spreadsheetbench_hard_skill_teacher_baseline.md` measures the distillation
  target itself — frozen Qwen plus the full text Skill, no soft prefix — on the
  matched Test280. It supplies the ceiling (118/280) that pairs with the Plain
  Qwen floor (97/280), so every compression claim can be reported as a recovery
  rate rather than as a bare delta over Original SoftSkill.
