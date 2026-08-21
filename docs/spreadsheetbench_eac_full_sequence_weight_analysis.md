# EAC Full-Sequence Weight Analysis

## Scope

- 61 successful SpreadsheetBench training trajectories.
- 54,929 EOS-excluded target tokens.
- For this analysis, `G`, `D`, and `H` are recomputed with per-trajectory
  min-max normalization over every target position. This differs from the
  earlier Top-5% locator cache, whose normalization excluded the frozen
  preservation positions and wrote zeros outside that eligible set.
- No Val40 or Test280 data are used.

Let

\[
S_t=0.5\widetilde G_t+0.5\widetilde D_t,
\qquad
E_t^{\lambda}=S_t(1+\lambda\widetilde H_t).
\]

## Marginal distributions

| Signal | Mean | Median | P90 | P95 | P99 | Exact zero | `<=0.01` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `G~` | 0.01332 | 0.0000011 | 0.01397 | 0.06119 | 0.32041 | 42.68% | 88.79% |
| `D~` | 0.01944 | 0.0001488 | 0.04057 | 0.09939 | 0.36556 | 0.11% | 80.17% |
| `H~` | 0.07094 | 0.0036325 | 0.23783 | 0.36909 | 0.69659 | 0.11% | 58.13% |
| `S` | 0.01638 | 0.0001028 | 0.03390 | 0.08478 | 0.30714 | 0.08% | 81.81% |

All signals are strongly right-skewed. In particular, a nominal all-token
weighted loss would assign negligible mass to most positions unless a uniform
floor is added.

## Correlations

| Pair | Pearson | Spearman |
|---|---:|---:|
| `G~ / D~` | 0.5840 | 0.3377 |
| `G~ / H~` | 0.3498 | 0.3623 |
| `D~ / H~` | 0.5113 | **0.9302** |
| `S / H~` | 0.4890 | **0.9356** |

Entropy is therefore largely redundant with the rank ordering already supplied
by full-vocabulary JS. Multiplying by entropy tends to count the JS/uncertainty
axis twice rather than introduce an independent Skill signal.

## Weight concentration

Weights below are normalized by their total mass. ESS is
`(sum w)^2 / sum(w^2) / N`.

| Weight | ESS/N | Top-1% mass | Top-5% mass | Top-10% mass | Top-20% mass |
|---|---:|---:|---:|---:|---:|
| Additive `S` | 6.27% | 31.33% | 69.35% | 86.08% | 96.75% |
| EAC `lambda=.25` | 6.05% | 31.85% | 69.94% | 86.57% | 96.94% |
| EAC `lambda=.5` | 5.83% | 32.35% | 70.50% | 87.00% | 97.10% |
| EAC `lambda=1` | 5.44% | 33.27% | 71.49% | 87.72% | 97.36% |

The mean per-trajectory Top-5% mass changes from 68.00% (`S`) to 69.92%
(`lambda=1`). EAC is thus a soft sparse selector. Increasing lambda makes the
effective support smaller rather than providing broad full-sequence learning.

The mean per-trajectory Top-5% Jaccard between Additive and EAC is 0.954,
0.927, and 0.879 for lambda 0.25, 0.5, and 1 respectively. Most high-weight
positions remain the same; EAC mainly reorders boundary tokens.

## Ranking diagnostics beyond Jaccard

Full-list correlation is nearly saturated and therefore misleading: mean
Spearman is 0.999981/0.999952/0.999875 and mean Kendall tau is
0.998318/0.996897/0.994489 for lambda 0.25/0.5/1. The long low-score tail
dominates both statistics.

Head-sensitive metrics reveal the actual change:

| Lambda | Top-1% overlap | Top-5% overlap | AO@5% | Top-5 rank displacement / K |
|---:|---:|---:|---:|---:|
| 0.25 | 95.82% | 97.60% | 96.38% | 2.91% |
| 0.5 | 92.80% | 96.13% | 93.74% | 5.19% |
| 1 | 85.73% | 93.47% | 89.33% | 8.65% |

`AO@5%` is the average prefix overlap
`mean_{d=1..K} |A_{1:d} intersect B_{1:d}|/d`; it emphasizes changes near the
very top while remaining parameter-free.

The quality of the new ranking should be measured with an external relevance
target rather than overlap alone. At Top-5%, Additive captures 74.29% of raw
positive-gain mass, 61.50% of JS mass, and 95.91% of legacy product mass.
EAC-1 captures 72.94%, 61.88%, and 95.79% respectively. Entropy therefore
trades away some gold-token gain for a very small increase in JS, while leaving
the old Combined mass essentially unchanged.

## Joint regions

| Region | Token fraction | Additive mass | EAC-1 mass |
|---|---:|---:|---:|
| `G~=0` | 42.68% | 20.74% | 20.32% |
| `G~=0, D~>=.5` | 0.12% | 2.88% | 2.77% |
| `H~>=.5, S<=.05` | 0.86% | 1.51% | 1.87% |
| `H~>=.5, S>=.5` | 0.15% | 7.00% | 9.32% |
| `H~<.5, S>=.5` | 0.27% | 10.52% | 9.59% |

The entropy multiplier does not give much mass to high-entropy/low-Skill
positions because the multiplicative form remains zero when Skill relevance is
zero. Its main effect is to move mass between already-high-Skill positions.

## Suitability as a full-sequence loss

Direct EAC-weighted one-hot CE is not recommended. It would continue to fit the
GPT-5.5 trajectory token and use entropy to emphasize difficult tokens, which
is not the same as matching the Hard-Skill Qwen behavior. Existing Val40 EAC
localization also failed to improve over Additive.

EAC can be tested as a continuous weight on full-vocabulary Hard-Skill KL, but
the current statistics do not support entropy as the main factor. A safer first
design is Additive Skill weighting with a uniform floor:

\[
w_t^{S}=\epsilon+(1-\epsilon)S_t,
\qquad
\bar w_t=\frac{w_t^S}{\frac1T\sum_j w_j^S},
\]

\[
L_{\mathrm{skill}}=
\frac1T\sum_t \bar w_t
D_{\mathrm{KL}}(q_t^{\mathrm{HardSkill}}\Vert p_t^{P}).
\]

The floor prevents the bottom 80% of tokens from becoming numerically absent;
mean-one normalization prevents lambda or trajectory length from changing the
effective learning rate. Entropy should only be introduced later as a bounded
ablation, for example

\[
w_t^{EAC}=\epsilon+(1-\epsilon)S_t(1+\lambda\widetilde H_t),
\quad \lambda\in\{0.1,0.25\},
\]

with the same mean-one normalization. Without that normalization, EAC raises
total weight by 8.50%, 17.01%, and 34.01% for lambda 0.25, 0.5, and 1, creating
an effective-learning-rate confound.

The recommended controlled comparison is uniform full-sequence Skill KL,
Additive-weighted Skill KL, and EAC-weighted Skill KL under identical floors,
normalization, preservation, initialization, optimizer, and step budget.
