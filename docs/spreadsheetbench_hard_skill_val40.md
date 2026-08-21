# SpreadsheetBench Hard-Skill Val40 control

## Protocol

This evaluates the frozen Qwen3.6-35B-A3B backbone with the complete 2,773-token
text Skill and no soft prefix on the registered Val40 split.

The protocol is matched to the Dynamic Additive validation run:

- split: `valid_seen`, all 40 tasks;
- `max_prompt_tokens=16384`;
- `max_new_tokens=4096`;
- `generation_batch_size=8`;
- greedy decoding (`temperature=0.0`);
- single-shot generation and the same SpreadsheetBench executor;
- no trained or loaded prefix and zero trainable parameters.

The prompt audit recorded 40 sent prompts, 40 generated tasks, no skipped tasks,
and confirmed that every sent prompt carried the full Skill. The Skill SHA-256
is `82249a54bba13550aa87dcc84087728ebb176d753c95bc09e6b9c49f30942892`.

## Result

| Condition | Success | Cell-level | Sheet-level | Execution failures | Semantic failures |
|---|---:|---:|---:|---:|---:|
| Hard Skill | **21/40 (52.5%)** | 17/29 | 4/11 | 4 | 15 |
| Dynamic Additive, no guard | 16/40 (40.0%) | 13/29 | 3/11 | 7 | 17 |
| Static Additive | 17/40 (42.5%) | 14/29 | 3/11 | 7 | 16 |
| Static Combined 10% full-vocab KL | 15/40 (37.5%) | 11/29 | 4/11 | 7 | 18 |

Hard Skill versus Dynamic Additive has 14 common successes, seven Hard-only
successes, two Dynamic-only successes, and 17 common failures. The net gap is
five tasks; the exact two-sided paired p-value is 0.1796875. The 40-task sample
is too small for this difference to be statistically decisive, but the point
estimate shows that Dynamic Additive is not at the empirical teacher ceiling.

Hard Skill versus Static Additive has 15 common successes, six Hard-only, two
Static-only, and 17 common failures (net +4; exact p=0.2890625).

The previously evaluated Plain Val40 also scored 21/40, but used the Test280-
matched generation settings (`max_new_tokens=8192`, batch size 2), rather than
this run's 4096/batch-8 protocol. Even ignoring that mismatch, equal aggregate
scores conceal major task turnover: Hard Skill and Plain share 15 successes,
and each uniquely solves six tasks. Therefore the Skill changes behavior but
does not provide a positive net gain on this particular small validation
split.

## Interpretation

Qwen plus the full Skill reaches 52.5%, so the Dynamic Additive result of 40.0%
is not explained by a Qwen ceiling at 40%. The compression gap is five tasks
relative to its exact teacher condition, concentrated mostly in cell-level
tasks (17/29 versus 13/29). At the same time, the Hard Skill teacher itself
fails 19/40 tasks, so the backbone/prompt condition still limits the absolute
attainable score.

The Hard Skill score is an empirical teacher reference, not a mathematical
upper bound: the soft prefix uniquely solves two tasks that the teacher misses.

## Artifacts

- Summary: `outputs/SpreadsheetBench_qwen36_hard_skill_val40_dynamic_additive_matched/summary.json`
- Results: `outputs/SpreadsheetBench_qwen36_hard_skill_val40_dynamic_additive_matched/eval/hard_skill/valid_seen/results.jsonl`
- Prompt audit: `outputs/SpreadsheetBench_qwen36_hard_skill_val40_dynamic_additive_matched/prompt_audit.json`
- Log: `outputs/SpreadsheetBench_qwen36_hard_skill_val40_dynamic_additive_matched/eval.log`
- Test280 accessed by this run: no.
