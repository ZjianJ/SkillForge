# SpreadsheetBench Dynamic Locator-Weighted KL Experiment

## Question and preregistered decision rule

This experiment asks whether the dynamic Additive locator score should enter the
distillation loss as a detached continuous token weight.  The progression was
fixed before running:

1. Compare Top-10% locator-weighted KL with the existing Top-10% equal-weight KL.
2. Run weighted Top-20% with Top-10% effective mass only if step 1 improves Val40.
3. Run full-sequence Skill/preservation soft routing only if the preceding steps
   are effective.

The go/no-go criterion for step 1 was therefore strictly greater than the frozen
equal-weight baseline of 16/40 on Val40.

## Matched setup

Both conditions use the same Qwen3.6-35B-A3B snapshot, frozen backbone, shared
length-8 soft prefix, Train61 successful GPT-5.5 trajectories, dynamic Additive
locator, per-trajectory Top-10% support, four relocation stages, 32 optimizer
steps per stage, full-vocabulary forward Skill-KL, EOS CE, fixed preservation
KL, initialization, seed, and Val40 execution protocol.  Test280 was not
accessed.

The baseline uses equal selected-token weights.  The treatment uses

\[
L_A=\frac{\sum_{t\in S}\operatorname{sg}(w_t)
\operatorname{KL}(q_t^H\Vert p_t^\theta)}
{\sum_{t\in S}\operatorname{sg}(w_t)},
\qquad
w_t\propto 0.5\widetilde G_t+0.5\widetilde{\operatorname{JS}}_t.
\]

Weights are detached and normalized separately in every trajectory so their
sum equals the original number of selected Top-10% tokens.  Consequently the
relative strength of EOS and preservation losses is unchanged.

## Primary result

| Condition | Selected tokens / round | Effective weight / round | Optimizer steps | Val40 | Rate |
|---|---:|---:|---:|---:|---:|
| Top-10% equal-weight KL | 5,522 | 5,522 | 128 | 16/40 | 40.0% |
| Top-10% locator-weighted KL | 5,522 | 5,522 | 128 | 13/40 | 32.5% |

The paired difference is -3/40, or -7.5 percentage points.  A task-paired
bootstrap 95% interval is [-20.0, +5.0] percentage points.  Thus Val40 is too
small to establish a statistically conclusive negative effect, but the
preregistered improvement criterion was not met.

## Teacher-forced dynamics

| Relocation | Equal global full-vocab KL | Weighted global full-vocab KL | Equal preservation KL | Weighted preservation KL |
|---:|---:|---:|---:|---:|
| 0 | 2852.302 | 2852.302 | 0.002802 | 0.002802 |
| 1 | 2599.974 | 2642.950 | 0.004937 | 0.006051 |
| 2 | 2334.093 | 2396.288 | 0.007049 | 0.011385 |
| 3 | 2126.991 | 2165.346 | 0.009116 | 0.009968 |
| 4 | 2021.805 | 2104.694 | 0.009405 | 0.011133 |

At the final relocation, weighted training leaves 4.10% more global
Hard-Skill/student KL and 18.37% more preservation KL than equal weighting.
This agrees in direction with the lower execution success rate.

The weighted monitor losses are not directly comparable with baseline monitor
losses because they optimize different token averages.

## Weight concentration

| Round | Median weight | P90 | Maximum | Effective sample size | ESS / 5,522 | Weight--KL correlation |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.603 | 2.211 | 29.589 | 2,198 | 39.8% | 0.729 |
| 1 | 0.609 | 2.110 | 28.472 | 2,250 | 40.8% | 0.720 |
| 2 | 0.635 | 2.050 | 34.690 | 2,243 | 40.6% | 0.697 |
| 3 | 0.661 | 2.069 | 32.591 | 2,425 | 43.9% | 0.649 |
| 4 | 0.679 | 2.092 | 14.637 | 2,786 | 50.5% | 0.639 |

Although nominal support remains Top-10%, round-0 effective sample size falls
to 2,198 tokens, approximately 4% of the full selectable sequence.  The locator
weight also correlates strongly with full-vocabulary KL.  Since KL already
produces larger gradients for larger teacher/student residuals, multiplying it
by the locator score double-emphasizes a small set of high-residual positions.
The result is stronger local concentration but weaker global closure and
preservation.

## Paired task changes

The weighted model gains two tasks and loses five relative to equal weighting.

| Direction | Task IDs |
|---|---|
| Equal fail -> weighted pass | 38462, 402-43 |
| Equal pass -> weighted fail | 55049, 55979, 59595, 8942, 9569 |
| Pass in both | 12864, 142-12, 37378, 382-10, 463-17, 47798, 48588, 48921, 50768, 53383, 9726 |

Execution errors decrease from 7 to 5, but evaluation mismatches increase from
17 to 22.  The weighting therefore makes slightly more programs executable
without improving their semantic correctness.

## Decision

Step 1 fails the preregistered gate: 13/40 is below 16/40.  The Top-20%
effective-mass experiment and full-sequence soft-routing experiment are not
run.  Their configuration can remain prepared for a future explicitly revised
hypothesis, but running them in this experiment chain would turn a conditional
follow-up into post-hoc searching.

The immediate technical conclusion is not that continuous weighting can never
work.  It is that raw dynamic Additive scores are too concentrated to use as
linear KL multipliers.  Any later study should first test bounded weights
(clipping, temperature flattening, or rank weights) while preserving an
effective sample size close to the full 5,522-token support.

## Artifacts

- Equal baseline: `outputs/SpreadsheetBench_dynamic_additive_skill_no_guard_full_vocab_len8_seed1`
- Weighted run: `outputs/SpreadsheetBench_dynamic_additive_top10_locator_weighted_skillkl_len8_seed1`
- Weighted checkpoint SHA-256: `ed5f546b7d13560fb945e3d1aa5c11b18f57f1b57e4b57fe5e3878c4241177bb`
- Weighted config: `configs/spreadsheetbench/dynamic_additive_top10_locator_weighted_skillkl.yaml`
- Conditional, not run: `configs/spreadsheetbench/dynamic_additive_top20_locator_weighted_effective10_skillkl.yaml`
