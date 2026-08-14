# PRCB-v3 SpreadsheetBench validation report

## Frozen design

PRCB-v3 keeps every locator computation on the fixed GPT-5.5 successful
trajectory.  At target position `t`, all hard-Skill, no-Skill, and current-prefix
distributions therefore condition on the identical gold prefix `y*_<t`.
Free-generated tokens are used only by the final task-level evaluation.

The decision margin is

`m_t(p) = log p(y*_t) - max_{v != y*_t} log p(v)`.

The locator uses two lexicographic tiers:

1. hard Skill ranks gold Top-1 while no-Skill and the current prefix do not;
2. all other positions with both positive original Skill margin gain and
   positive current-prefix margin residual.

Within a tier, positions are ordered by current margin residual, original Skill
margin gain, then hard-Skill/current-prefix Top-64 JS as a tie-breaker.  The top
5% per trajectory are selected.  Thus JS does not multiply or override the
decision criterion.

Training starts from the frozen Combined length-8 checkpoint.  Four rounds
update adjacent pairs in head-to-tail order: `[0,1]`, `[2,3]`, `[4,5]`,
`[6,7]`.  Each round uses 8 optimizer steps, accumulation 2, learning rate
`2e-4`, and shrinkage `0.25`.  The objective is

`CE + 0.5 * KL(Skill Top-64 || prefix) + 0.5 * margin_hinge + KL(no-Skill || prefix anchors)`.

Only the active pair changes in a round.  The training set comprises 61 cached
successful trajectories.  Across the four deterministic schedules, all 61 are
used at least once.  The test split is not read.

## Locator results

| Measurement point | Locator mass | Eligible tokens | Decisive tokens | Top-5% mass capture | Core Jaccard vs prior |
|---|---:|---:|---:|---:|---:|
| Before round 1 | 60,974.14 | 25,238 | 720 | 61.96% | -- |
| Before round 2 | 60,668.25 | 25,166 | 713 | 62.13% | 77.78% |
| Before round 3 | 60,477.86 | 25,194 | 734 | 62.16% | 77.67% |
| Before round 4 | 59,899.06 | 25,054 | 721 | 62.77% | 77.10% |
| After round 4 | 59,852.90 | 25,039 | 718 | 62.95% | 78.07% |

From the initial to final measurement, locator mass drops by 1.84%, eligible
tokens by 0.79%, and decisive tokens by only 0.28% (720 to 718).  The decisive
count is non-monotonic, reaching 734 before round 3.  This demonstrates causal
interference between sequential pair updates and shows that the local
teacher-forced objective is only weakly optimized under the present update
budget.

## One-shot val40 evaluation

The frozen final checkpoint was evaluated once with greedy generation,
`max_new_tokens=4096`, generation batch size 8, one repair turn, and the same
SpreadsheetBench executor used by the matched prior runs.

| Method | Success | Rate | vs PRCB-v3 paired changes |
|---|---:|---:|---:|
| Combined 5% + shared KL | 16/40 | 40.0% | v3-only 5, Combined-only 5 |
| PRCB-v1 | 14/40 | 35.0% | -- |
| PRCB-v2 tail-to-head | 11/40 | 27.5% | v3-only 8, v2-only 3 |
| PRCB-v2 head-to-tail | 15/40 | 37.5% | v3-only 6, v2-only 5 |
| **PRCB-v3** | **16/40** | **40.0%** | -- |

Against Combined and PRCB-v2 head-to-tail, the exact paired two-sided p-value
is 1.0 in both cases.  PRCB-v3's Wilson 95% interval is 26.35% to 55.40%.
Consequently, this run does not establish a statistically reliable improvement.

PRCB-v3 succeeds on 12/29 cell-level tasks and 4/11 sheet-level tasks.  Of its
24 failures, 16 execute but produce a value mismatch, while 8 fail during code
execution.  Relative to Combined, five tasks are gained and five are lost.
This large turnover at unchanged aggregate accuracy confirms that code
generation stability remains a central limitation.

## Conclusion

PRCB-v3 fixes the methodological error of comparing diverged free-generation
positions and restores PRCB-v2 head-to-tail's average validation loss in task
success: 40.0% versus 37.5%.  It does not exceed the 40.0% Combined starting
checkpoint.  The likely bottleneck is no longer the absence of an automatically
located decision signal.  It is the weak and interfering translation from
token-local margin constraints into a globally coherent executable program:
each pair is updated once, later pairs can perturb behavior learned by earlier
pairs, and no execution- or structure-level objective constrains the generated
code.

No test280 evaluation was performed.
