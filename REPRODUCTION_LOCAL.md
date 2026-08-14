# SoftSkill local reproduction

This checkout reproduces the public SearchQA SoftSkill experiment from upstream
commit `4fc53008da110f354746bf36966dc0a2f44d3b92`.

## Verified environment

- Host architecture: `aarch64`
- GPU: NVIDIA GH200 120GB (95 GiB visible to PyTorch)
- Python: 3.13.13
- PyTorch: 2.12.1+cu126
- Transformers: 5.14.1
- Model: `Qwen/Qwen3.5-4B`, snapshot
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`

The project `.venv` uses `--system-site-packages` so that it can reuse the
host's working ARM64 CUDA build of PyTorch. Installing the unpinned
`softprefix` extra directly selected a CUDA 13 build on 2026-08-03, which was
incompatible with the installed 565.57.01 driver.

## Re-run

Activate the existing environment and run the local launcher:

```bash
source .venv/bin/activate
bash scripts/reproduce_searchqa_gh200.sh
```

The launcher materializes SearchQA if needed and runs the upstream training
entry point with the released configuration. The released `batch_size=8`
exhausted the 95 GiB visible GPU memory, so the launcher uses micro-batches of
2 with 4-step gradient accumulation. This preserves the effective batch size
of 8 while leaving the model, data, prefix length, optimizer, epoch count, and
evaluation settings unchanged.

Generated data is under `data/searchqa_split`. Training artifacts are under
`outputs/SoftSkill_searchqa_example`.

## Verified result

The complete three-epoch run and all 1,400 test examples finished successfully:

| Epoch | Loss | Validation hard | Validation soft | Gate action |
|---:|---:|---:|---:|---|
| 1 | 0.310322 | 0.734375 | 0.800521 | accept |
| 2 | 0.228640 | 0.734375 | 0.798958 | reject |
| 3 | 0.195127 | 0.765625 | 0.824479 | accept |

Final test scores from the validation-selected epoch-3 prefix:

- hard: `0.7714285714285715`
- soft: `0.8433367346938776`
- checkpoint shape: `[32, 2560]`, BF16

The machine-readable records are in
`outputs/SoftSkill_searchqa_example/history.json` and
`outputs/SoftSkill_searchqa_example/summary.json`.

## Tests

Run the suite with:

```bash
.venv/bin/pytest -q
```

At the pinned upstream commit, 184 tests pass and 4 tests fail because their
monkeypatched trajectory-builder fakes still use a three-argument signature,
while the implementation now passes the optional `skill_content` keyword.
This upstream test-stub mismatch is independent of the SearchQA training and
evaluation path reproduced above.

## SpreadsheetBench reproduction

The paper-aligned SpreadsheetBench run uses the released 80/40/280 split,
`Qwen/Qwen3.6-35B-A3B`, an 8-token soft prefix, seed 1, three epochs, and
GPT-5.5-generated successful trajectories. The local credential file is
`configs/local/spreadsheetbench_paper_gpt55.local.yaml`; it is ignored by Git,
has mode `0600`, and is redacted from generated `config.json` files.

Teacher collection produced 80 unique rollouts and 61 hard successes, exactly
matching the paper's reported 61/80 training-trajectory yield. The exported SFT
cache contains 61 unique examples with model thoughts removed:

- `rollouts/teacher_gpt55_spreadsheetbench_rollouts/results.jsonl`
- `outputs/SpreadsheetBench_teacher_gpt55_collection/trajectory_sft/examples.jsonl`

The GH200 has enough memory for one copy of the 35B model, but not simultaneous
training and vLLM copies. The single-GPU profile therefore uses micro-batch 1
with two-step accumulation, gradient checkpointing, and local-HF evaluation in
batches of two. This preserves two examples per optimizer update, but it is not
gradient-identical to a physical batch of two: causal-LM loss is averaged over
the non-masked tokens of each micro-batch, and the 61 trajectory targets range
from 172 to 3,646 tokens. The paper settings otherwise remain unchanged. Re-run
from the cached teacher trajectories with:

```bash
MODEL_NAME=Qwen/Qwen3.6-35B-A3B \
  bash scripts/train_spreadsheetbench_paper_single_gpu_from_cache.sh
```

Training selected epoch 1 on the 16-task validation subset:

| Epoch | Loss | Validation hard | Validation soft | Gate action |
|---:|---:|---:|---:|---|
| 1 | 0.390968 | 0.2500 | 0.2500 | accept |
| 2 | 0.364141 | 0.1875 | 0.1875 | reject |
| 3 | 0.347069 | 0.1875 | 0.1875 | reject |

The validation-selected checkpoint scored 85/280 on the test split:

- hard: `0.30357142857142855`
- soft: `0.30357142857142855`
- checkpoint shape: `[8, 2048]`, BF16

The paper reports 28.2% (approximately 79/280) for the corresponding
validation-selected length-8 SoftSkill, so this run is 2.16 percentage points
higher on test. It is not an exact numerical match: the paper reports 50.0% on
the 16-task validation subset, while this run obtained 25.0%. Regenerated
GPT-5.5 trajectories and the necessary single-GPU execution changes prevent a
bit-for-bit reproduction even though the dataset, model, prefix length,
effective batch size, seed, epochs, and trajectory-success count match.

Machine-readable artifacts are under
`outputs/SoftSkill_spreadsheetbench_paper_len8_seed1_single_gpu`. The final
test records are in `eval/checkpoint/valid_unseen/results.jsonl`; three failed
responses reached the configured 8,192-token generation ceiling.
