# SpreadsheetBench Dynamic G+C Locator Experiment

## Question

Does replacing the full-vocabulary JS component in the dynamic Additive
locator with strongest-wrong-competitor suppression improve soft-prefix
distillation?

The baseline locator is

\[
u_t^{G+JS}=0.5\widetilde G_t+0.5\widetilde{JS}_t.
\]

The treatment locator is

\[
u_t^{G+C}=0.75\widetilde G_t+0.25\widetilde C_t,
\]

where, on the same successful teacher-forced prefix,

\[
C_t=\left[
\max_{v\ne y_t}\log p_t^\theta(v)
-\max_{v\ne y_t}\log q_t^H(v)
\right]_+.
\]

Here (q^H) is Qwen with the full Hard Skill and (p^\theta) is Qwen with
the current soft prefix.  Both components are independently min-max normalized
inside each trajectory.

## Fairness contract

The comparison changes only the locator.  Both runs use the same
Qwen3.6-35B-A3B snapshot, frozen backbone, shared length-8 soft prefix, Train61
successful trajectories, Top-10% token budget, equal-weight full-vocabulary
forward Skill-KL, EOS CE, fixed preservation KL, four relocation stages, 32
optimizer steps per stage, seed, initialization, and Val40 execution protocol.
Test280 was not accessed.

## Result

| Locator | Core loss | Selected / round | Steps | Val40 | Rate |
|---|---|---:|---:|---:|---:|
| (0.5G+0.5JS) | Equal full-vocab KL | 5,522 | 128 | 16/40 | 40.0% |
| (0.75G+0.25C) | Equal full-vocab KL | 5,522 | 128 | 11/40 | 27.5% |

The paired difference is -5/40, or -12.5 percentage points.  A paired
bootstrap 95% interval is [-27.5, +2.5] percentage points.  The interval still
includes zero because Val40 is small, but the observed result does not support
replacing JS with (C) in this form.

## Initial locator behavior

The two initial selected sets have global Jaccard 0.392 and differ by 2,410
tokens, confirming that the new metric materially changes localization.

| Top-10% selector | Positive-G mass captured | Competitor-suppression mass captured | Full-vocab KL mass captured | Mean selected (G) | Mean selected (C) |
|---|---:|---:|---:|---:|---:|
| (G+JS) | 89.11% | 13.83% | 80.90% | 0.749 | 0.672 |
| (G+C) | 84.84% | 36.48% | 53.21% | 0.713 | 1.773 |

Thus (G+C) succeeds at its local goal: selected positions contain 2.64 times
the mean competitor suppression and retain most gold-token gain.  However, it
drops full-distribution coverage by 27.69 percentage points.

## Training dynamics

| Round | (G+JS) global KL | (G+C) global KL | (G+JS) preservation KL | (G+C) preservation KL |
|---:|---:|---:|---:|---:|
| 0 | 2852.302 | 2852.302 | 0.002802 | 0.002802 |
| 1 | 2599.974 | 2666.146 | 0.004937 | 0.004351 |
| 2 | 2334.093 | 2404.458 | 0.007049 | 0.010193 |
| 3 | 2126.991 | 2313.747 | 0.009116 | 0.008288 |
| 4 | 2021.805 | 2034.110 | 0.009405 | 0.012192 |

By round 4, (G+C) nearly catches up in global KL (0.61% higher), but its
preservation KL is 29.64% higher.

## Paired task changes

| Direction | Task IDs |
|---|---|
| (G+JS) fail -> (G+C) pass | 56563, 6698 |
| (G+JS) pass -> (G+C) fail | 12864, 48921, 50768, 53383, 55049, 59595, 9569 |
| Pass in both | 142-12, 37378, 382-10, 463-17, 47798, 48588, 55979, 8942, 9726 |

Execution errors rise from 7 to 12, while evaluation mismatches remain 17.
Mean generated response length falls from 2,881 to 2,210 characters (-23.3%).
This is consistent with a locator that learns decisive local alternatives but
underrepresents broad token-distribution changes needed to assemble robust,
complete executable programs.

## Conclusion

The new locator is active and does what it was designed to do, but that design
does not improve downstream execution.  Strongest-competitor suppression is a
useful non-redundant diagnostic signal, not a sufficient replacement for JS.
The next defensible locator should retain a distribution-coverage term and add
(C) as a small third component, rather than replacing JS entirely; such an
experiment is a new hypothesis and is not inferred as successful here.

## Artifacts

- Baseline: `outputs/SpreadsheetBench_dynamic_additive_skill_no_guard_full_vocab_len8_seed1`
- G+C run: `outputs/SpreadsheetBench_dynamic_gain_competitor_top10_equal_skillkl_len8_seed1`
- G+C checkpoint SHA-256: `75eaf571625e38f9896c65134784d2c31e20a4cc05e4a9e1a25cda6ae1929602`
- Config: `configs/spreadsheetbench/dynamic_gain_competitor_top10_equal_skillkl.yaml`
