# Entropy-Aware Skill Localization on SpreadsheetBench

## Controlled protocol

- Frozen model: Qwen3.6-35B-A3B.
- Data: 61 cached successful GPT-5.5 training trajectories; Val40 for model
  selection; Test280 remains untouched.
- Teacher forcing: the same successful target suffix under no Skill and the
  full hard Skill.
- Soft prefix: length 8, text initialization, seed 1, one epoch.
- Localization: per-trajectory Top-5%, excluding the same frozen Top-5%
  preservation positions.
- Training: selected-token one-hot CE, EOS CE, and no-Skill Top-64 plus
  residual-bucket preservation KL with weight 1.
- Evaluation: local greedy free generation, batch size 8, 4096 new-token cap,
  followed by SpreadsheetBench workbook execution and scoring.

All methods use 2,777 localized target tokens, 61 EOS targets, and 2,777
identical preservation positions. The only intended independent variable is
the localization score.

## Scores

For token position \(t\), no-Skill distribution \(p_{0,t}\), hard-Skill
distribution \(p_{S,t}\), and successful target \(y_t\):

\[
H_t=-\sum_v p_{0,t}(v)\log p_{0,t}(v),\qquad
G_t=[\log p_{S,t}(y_t)-\log p_{0,t}(y_t)]_+,
\]

\[
D_t=D_{JS}(p_{S,t},p_{0,t}),\qquad
\Delta H_t=H(p_{0,t})-H(p_{S,t}).
\]

The frozen existing Combined baseline is \(G_tD_t\). For entropy-amplified
localization, scores are min-max normalized within each trajectory over the
eligible positions:

\[
S_t=0.5\widetilde G_t+0.5\widetilde D_t,\qquad
u_t^{EAC}=S_t(1+\lambda\widetilde H_t).
\]

The \(\lambda=0\) additive Skill control is included so an apparent EAC gain
cannot be attributed to changing the old product-form Combined score.

## Offline results (54,929 target tokens)

| Pair | Pooled Pearson | Pooled Spearman | Mean per-trajectory Pearson | Mean per-trajectory Spearman |
|---|---:|---:|---:|---:|
| Base entropy vs Positive Gain | 0.3460 | 0.3651 | 0.3529 | 0.3474 |
| Base entropy vs full-vocabulary JS | 0.5036 | 0.9349 | 0.5091 | 0.9297 |

| Top-5% selector | Mean base entropy | Median base entropy | Remaining mean | Mean \(\Delta H\) selected | Mean \(\Delta H\) remaining |
|---|---:|---:|---:|---:|---:|
| Positive Gain | 1.1979 | 0.9515 | 0.2242 | 0.1172 | 0.0150 |
| JS | 1.4559 | 1.2054 | 0.2105 | 0.2261 | 0.0092 |
| Frozen Combined | 1.3295 | 1.0981 | 0.2172 | 0.1942 | 0.0109 |
| Entropy | 2.0531 | 1.9684 | 0.1787 | 0.2321 | 0.0089 |

| Top-5% pair | Pooled Jaccard |
|---|---:|
| Entropy / Positive Gain | 0.2166 |
| Entropy / JS | 0.3074 |
| Entropy / Frozen Combined | 0.2594 |
| Positive Gain / JS | 0.3361 |
| Positive Gain / Frozen Combined | 0.6408 |
| JS / Frozen Combined | 0.4252 |

The full Skill has lower entropy on 57.93% of all tokens. At frozen Combined
Top-5% positions, mean entropy reduction is 0.1942 nats versus 0.0109 on the
remainder. Using the per-trajectory entropy median and Combined Top-5% as the
quadrant thresholds, 2,766 tokens are uncertainty-resolution cases and 11 are
confident-error-correction cases. Thus confident-error corrections exist but
are rare under this threshold. Entropy Top-5% nevertheless misses about 74.1%
of Combined Top-5% positions because moderate/high uncertainty is not the same
as the most uncertain tail.

## Val40 free-generation results (completed subset)

| Locator | Success | Cell | Sheet | Execution failures | Semantic failures |
|---|---:|---:|---:|---:|---:|
| Random Top-5% | 13/40 (32.5%) | 10/29 | 3/11 | 8 | 19 |
| Entropy Top-5% | 14/40 (35.0%) | 9/29 | 5/11 | 5 | 21 |
| Positive Gain Top-5% | 0/40 (0.0%) | 0/29 | 0/11 | 39 | 1 |
| JS Top-5% | 14/40 (35.0%) | 11/29 | 3/11 | 7 | 19 |
| Frozen Legacy Combined Top-5% | 16/40 (40.0%) | 11/29 | 5/11 | 9 | 15 |

Entropy gains one task over Random, but the paired exact test is \(p=1.0\),
so H3 is not established. Entropy and JS each trail Legacy Combined by two
tasks and neither difference is significant. The new Positive Gain run has 39
execution failures: outputs predominantly restate the natural-language prompt
instead of emitting executable Python (15 generic syntax errors, six empty
code blocks, six em-dash syntax errors, and related parse failures). Its
preservation KL is also 0.01517, versus 0.00496--0.00553 for Entropy, JS, and
Random.

Additive Skill and EAC \(\lambda\in\{0.25,0.5,1.0\}\) are pending. No H5 or
Test280 claim is valid until those Val40 runs finish and one \(\lambda\) is
frozen without test access.

## Artifacts

- Offline statistics: `outputs/SpreadsheetBench_entropy_localization/offline_summary.json`
- Derived token scores: `outputs/SpreadsheetBench_entropy_localization/token_scores/`
- Fixed manifests: `outputs/SpreadsheetBench_entropy_localization/manifests/`
- Distribution plot: `outputs/SpreadsheetBench_entropy_localization/entropy_vs_skill_effect.svg`
- Partial paired comparison: `outputs/SpreadsheetBench_entropy_localization/val40_partial_comparison.json`
- Reproducible runner: `scripts/run_spreadsheetbench_entropy_localization.sh`
