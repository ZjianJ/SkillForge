# PRCB-v4 overlap-size ablation on SpreadsheetBench

## Controlled comparison

Every stage is constrained to update exactly two soft-prefix vectors.  Under
this constraint, consecutive update intervals can have only two overlap sizes:

- overlap 0 (PRCB-v3): `[01,23,45,67]`;
- overlap 1 (PRCB-v4): `[01,12,23,34,45,56,67]`.

Both runs start from the same Combined length-8 checkpoint and use the same 61
successful GPT-5.5 trajectories, successful-trajectory margin locator, Top-5%
token budget, replay rule, loss weights, learning rate, accumulation, shrinkage,
and seed.  PRCB-v4 uses per-stage optimizer steps `[5,4,5,4,5,4,5]`, for 32
total steps and 64 trajectory presentations.  These exactly match PRCB-v3's
`4 * 8 = 32` steps and 64 presentations.

No validation result is used for training or checkpoint selection.  Each frozen
checkpoint is evaluated once on val40 with greedy generation,
`max_new_tokens=4096`, batch size 8, and one repair turn.  Test280 is not read.

## Task-level result

| Pair overlap | Method | Success | Cell-level | Sheet-level |
|---:|---|---:|---:|---:|
| 0 | PRCB-v3 | 16/40 (40.0%) | 12/29 | 4/11 |
| 1 | PRCB-v4 | 16/40 (40.0%) | 13/29 | 3/11 |

The paired comparison has six v4-only successes and six v3-only successes.  Its
exact two-sided p-value is 1.0.  Therefore overlap size 1 does not improve mean
task success over overlap size 0 in this experiment.

Failure composition also remains similar:

| Pair overlap | Executable value mismatch | Execution error |
|---:|---:|---:|
| 0 | 16 | 8 |
| 1 | 15 | 9 |

The 12 discordant tasks show substantial output turnover despite identical
aggregate accuracy.  V4 recovers six cell-level tasks that v3 misses, but loses
five cell-level and one sheet-level task that v3 solves.

## Final successful-trajectory locator

| Pair overlap | Final locator mass | Eligible tokens | Decisive tokens | Top-5% mass capture |
|---:|---:|---:|---:|---:|
| 0 | 59,852.90 | 25,039 | 718 | 62.95% |
| 1 | 59,874.85 | 24,986 | 717 | 63.04% |

Relative to overlap 0, overlap 1 has 0.037% higher (worse) locator mass, 53
fewer eligible positions, and one fewer decisive position.  These differences
are negligible and disagree in direction, providing no internal evidence that
the sliding schedule fits the teacher distribution better.

PRCB-v4's decisive count is non-monotonic across stages:
`720, 726, 716, 714, 711, 718, 728`, followed by 717 after the final update.
Overlapping shared-vector updates therefore do not remove causal interference;
they redistribute it across more, smaller stages.

## Conclusion

With total compute and every other mechanism controlled, increasing adjacent
pair overlap from 0 to 1 has no measurable benefit on val40: both methods score
40.0%, with a 6-to-6 paired exchange.  It also does not materially improve the
teacher-forced locator objective.  The evidence does not support overlap size
as the current bottleneck.  The remaining problem is that small local changes
produce different complete programs without reliably increasing executable
task correctness.

Because pair width is fixed at two, overlap sizes larger than one are not
defined.  Testing overlap 2 or greater would require changing the active update
window to at least three vectors, which would be a separate window-width
ablation rather than the controlled two-vector PRCB-v4 experiment.
