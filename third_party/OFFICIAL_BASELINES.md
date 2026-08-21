# Official baseline sources

SkillForge keeps third-party repositories outside Git history and pins every
source used by an `Official-Adapted` baseline. Run
`bash scripts/setup_official_distillation_baselines.sh` to materialize them in
`third_party/official/`.

| Baseline | Upstream | Pinned commit | License | Reused implementation |
|---|---|---|---|---|
| SE-KD-Prefix | https://github.com/almogtavor/SE-KD3x | `08b276383a31fe5c07eb6685f9c4557b78e42880` | Apache-2.0 | exact entropy utility, per-sequence ceil Top-k selection rule, forward-KL implementation |
| OPCD-Prefix | https://github.com/microsoft/LMOps | `4f2a9deb5f08e459fd44c2e4792344d78ca89fc3` | MIT | Top-256 student-class support and non-renormalized full reverse-KL density |

The adapters change the trainable parameterization from a full student model
to an eight-token soft prefix and replace upstream datasets/evaluators with
SpreadsheetBench. They do not change the direction of KL, the native position
or class-selection rules, or the on/off-policy state distribution. Results
must therefore be labeled **Official-Adapted**, not exact reproductions of the
papers' original model-scale experiments.
