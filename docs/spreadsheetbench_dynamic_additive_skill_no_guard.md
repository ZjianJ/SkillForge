# Dynamic Additive Skill without a preservation guard

## Question

Does dynamic re-localization continue to improve soft-prefix distillation when
the preservation-KL stop/rollback threshold is removed and the locator is
changed from multiplicative Combined to Additive Skill?

## Controlled setup

- Backbone: frozen Qwen3.6-35B-A3B.
- Prefix: one shared length-8 soft prefix, initialized from the first eight
  embeddings of the full hard Skill.
- Training support: the same 61 successful GPT-5.5 SpreadsheetBench
  trajectories; all localization and loss computation use gold
  teacher-forced contexts.
- Locator per trajectory:
  `0.5 * minmax(positive current target residual) + 0.5 * minmax(full-vocabulary JS)`.
- Core budget: Top 10%, 5,522 positions per relocation round.
- Core loss: full-vocabulary forward KL from Qwen + hard Skill to Qwen +
  current soft prefix, plus EOS CE.
- Preservation: the existing 2,777 fixed no-Skill Top-64 plus residual-bucket
  KL positions, weight 1.0. It remains in the training loss and reporting but
  cannot stop or roll back training.
- Cross-round stop/checkpoint metric: unnormalized full-vocabulary Skill-KL
  mass, because per-trajectory min-max normalized Additive scores do not have
  a comparable scale across rounds.
- Budget: four relocation/training stages, at most 32 optimizer steps per
  stage; all four stages used 32 steps (128 total).
- Evaluation: Val40 once after selecting the lowest training residual
  checkpoint. Test280 was not accessed.

## Dynamic trajectory results

| Round | Full-vocab Skill-KL mass | Drop vs. initial | Drop vs. prior | Mean eligible KL | Mean selected KL | Additive mass capture | Core Jaccard vs. prior | Preservation KL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2852.302 | 0.00% | -- | 0.054692 | 0.428760 | 84.58% | -- | 0.002802 |
| 1 | 2599.974 | 8.85% | 8.85% | 0.049854 | 0.386348 | 83.84% | 0.7447 | 0.004937 |
| 2 | 2334.093 | 18.17% | 10.23% | 0.044756 | 0.343446 | 83.28% | 0.6902 | 0.007049 |
| 3 | 2126.991 | 25.43% | 8.87% | 0.040784 | 0.308183 | 81.26% | 0.6504 | 0.009116 |
| 4 | 2021.805 | 29.12% | 4.95% | 0.038768 | 0.290124 | 80.81% | 0.6557 | 0.009405 |

The fixed monitor loss fell in every stage:

| Stage | Initial monitor | Best monitor | Relative reduction | Steps | Stop |
|---:|---:|---:|---:|---:|---|
| 0 | 0.370000 | 0.315879 | 14.63% | 32 | max steps |
| 1 | 0.325843 | 0.277030 | 14.98% | 32 | max steps |
| 2 | 0.286437 | 0.249484 | 12.90% | 32 | max steps |
| 3 | 0.264981 | 0.232609 | 12.22% | 32 | max steps |

Monitor losses across stages are not directly comparable because each stage
uses a newly localized core. The same 12 task IDs are also still members of
the 61-task optimizer sampling pool, so this is a fixed training-support
monitor rather than a held-out generalization estimate.

## Val40 free-generation result

| Method | Success | Cell-level | Sheet-level | Execution failures | Semantic failures |
|---|---:|---:|---:|---:|---:|
| Dynamic Additive, no guard, 4 stages | **16/40 (40.0%)** | 13/29 | 3/11 | 7 | 17 |
| Static Additive 5% + CE + preserve | 17/40 (42.5%) | 14/29 | 3/11 | 7 | 16 |
| Dynamic Combined rollback | 15/40 (37.5%) | 10/29 | 5/11 | 5 | 20 |
| Static Combined 10% full-vocab KL | 15/40 (37.5%) | 11/29 | 4/11 | 7 | 18 |
| Legacy Combined 5% + CE + preserve | 16/40 (40.0%) | 11/29 | 5/11 | 9 | 15 |
| Full Hard Skill teacher | **21/40 (52.5%)** | 17/29 | 4/11 | 4 | 15 |

Paired against static Additive, Dynamic Additive had 13 common successes,
three Dynamic-only successes, four Static-only successes, and a two-sided
exact McNemar p-value of 1.0. Against static Combined 10% full-vocabulary KL,
it gained four tasks and lost three (p=1.0). Against legacy Combined 5%, it
gained five and lost five. None of the observed Val40 differences is
statistically distinguishable from zero.

The subsequently completed exact-protocol Hard-Skill control is five tasks
above Dynamic Additive. They share 14 successes; Hard Skill uniquely solves
seven tasks and Dynamic Additive uniquely solves two (exact paired p=0.1797).
Thus the dynamic prefix has not reached the empirical teacher condition.

These baselines are informative but not all single-variable controls: the
static Additive run used a 5% core, one-hot CE, and one epoch, while the new
dynamic run used a 10% core, full-vocabulary KL, and 128 optimizer steps.

## Conclusion

Removing the guard demonstrates that dynamic Additive localization can keep
reducing hard-Skill teacher-forced residuals: the global full-vocabulary KL
mass fell by 29.12%, and the selected core continued to move. The original
preservation stop was therefore preventing additional teacher-distribution
fitting, not stopping an already converged optimizer.

However, the extra fitting did not improve free generation over the best
static Additive validation result. Preservation KL grew from 0.002802 to
0.009405 (3.36x), while Val40 reached only 40.0%, below static Additive's
42.5% and equal to legacy Combined. The threshold was not the main reason for
the limited task success. The stronger explanation is an objective/state
mismatch: repeated KL fitting on gold trajectory states changes many future
distributions but does not train recovery from the model's own generated
states, and a single shared length-8 prefix trades performance between task
types.

Do not access Test280 for this candidate. It did not exceed the frozen Val40
selection baseline.

## Artifacts

- Config: `configs/spreadsheetbench/dynamic_additive_skill_full_vocab_skillkl_no_preservation_guard.yaml`
- Summary: `outputs/SpreadsheetBench_dynamic_additive_skill_no_guard_full_vocab_len8_seed1/summary.json`
- Checkpoint SHA-256:
  `300891e6fad8cf848fb4a7fd48dbd9f0ba3b91d1b7bb81903209da74618ea2ef`
- Test split accessed: `false`
