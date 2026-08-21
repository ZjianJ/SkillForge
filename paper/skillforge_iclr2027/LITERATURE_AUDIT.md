# Literature and claim audit

This note records the closest primary sources used to position the draft. It is not compiled into the submission PDF.

## Closest work

| Work | Primary source | Relation to SkillForge | Claim boundary |
|---|---|---|---|
| SoftSkill (2026) | https://arxiv.org/abs/2606.20333 | Direct baseline: compresses successful trajectories into a soft prompt with full-trajectory next-token training. | SkillForge changes token localization and adds clean-position preservation; it does not introduce behavioral compression itself. |
| KFD (2026) | https://doi.org/10.3390/sym18040667 | Selects tokens by teacher/base distribution divergence and distills Top-k targets. | Do not claim the first teacher/base-difference token selector. SkillForge differs in using one frozen agent under hard/no-skill contexts and training only an input prefix for executable trajectories. |
| Rethinking Selective KD (2026) | https://arxiv.org/abs/2602.01395 | Separates position, vocabulary-class, and sample selection in LLM distillation. | Do not claim the first systematic selective token distillation method. |
| Delta-KD (2025) | https://arxiv.org/abs/2509.14526 | Transfers a teacher's fine-tuning-induced distribution shift. | SkillForge compresses a context-induced behavioral delta, not a weight-update delta. |
| On-Policy Context Distillation (2026) | https://arxiv.org/abs/2602.12275 | Distills privileged context on student-generated trajectories. | The current SkillForge method is off-policy/gold-context; OPCD motivates the proposed next step. |
| Unified On-Policy Self-Distillation (2026) | https://arxiv.org/abs/2608.08176 | Couples token selection and privileged-information strength to student capacity. | Treat as very recent/concurrent work; update positioning before submission. |
| GKD (ICLR 2024) | https://arxiv.org/abs/2306.13649 | Corrects student/teacher state mismatch using student-generated outputs. | Supports the exposure-bias diagnosis; the current experiments do not implement GKD. |

## Supporting lines of work

- Soft prompt adaptation: Prefix-Tuning, Prompt Tuning, P-Tuning v2, SPoT, Residual Prompt Tuning, InfoPrompt, and LoRA.
- Context compression: Gist Tokens, AutoCompressors, LLMLingua, LongLLMLingua, LLMLingua-2, and MEND.
- Distillation: classical KD, sequence-level KD, MiniLLM, GKD, and selective KD for NMT.
- Agents and prompt optimization: ReAct, AgentTuning, ProTeGi, OPRO, TextGrad, SkillOpt, and SpreadsheetBench.
- Optimization analogy: functional gradient boosting is cited only to explain why prefix-coordinate blocking lacks weak-learner independence.

## Safe central claim

> In the audited single-model SpreadsheetBench setting, locating hard-skill-induced behavioral changes on successful gold contexts and training a length-8 prefix on a selected 5% core with no-skill preservation improves execution success over a matched full-trajectory SoftSkill baseline.

The current evidence does **not** establish superiority over the no-skill base model, broad cross-benchmark generalization, multi-seed robustness, or the first use of selective token distillation.

## ICLR 2027 policy sources

- Author guidelines: https://iclr.cc/Conferences/2027/AuthorGuidelines
- AI policy for authors: https://iclr.cc/Conferences/2027/AIPolicyForAuthors
- Official style bundle: https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip

The draft remains anonymous, keeps `\iclrfinalcopy` disabled, places the main text within nine pages, and includes the mandatory AI use statement plus recommended ethics and reproducibility statements.
