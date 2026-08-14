from __future__ import annotations

import numpy as np
import torch

from scripts.analyze_selective_distillation_tokens import (
    _curve,
    _expand_windows,
    _score_logits,
    _select_core,
)
from scripts.prepare_spreadsheetbench_selective_stage2 import (
    _shared_preservation_indices,
)


def test_concentration_ratio_counts_all_target_tokens() -> None:
    values = np.asarray([9.0, 1.0] + [0.0] * 18)
    curve = _curve(values, selectable_count=20, ratios=[0.05, 0.10])
    assert curve["0.0500"] == 0.9
    assert curve["0.1000"] == 1.0


def test_selection_and_window_expansion_are_target_relative() -> None:
    scores = np.asarray([0.0, 4.0, 3.0, 0.0, 2.0, 0.0])
    core = _select_core(scores, selectable_count=6, ratio=0.34)
    assert core == [1, 2, 4]
    indices, windows = _expand_windows(core, selectable_count=6, left=1, right=1)
    assert windows == [[0, 5]]
    assert indices == list(range(6))


def test_score_logits_matches_target_log_probability_gain() -> None:
    logits = torch.tensor(
        [
            [[4.0, 0.0, -1.0], [0.0, 3.0, -1.0]],
            [[0.0, 4.0, -1.0], [3.0, 0.0, -1.0]],
        ]
    )
    result = _score_logits(logits, [0, 1], exact_js=True, top_k=2, chunk_size=1)
    assert np.all(result["gain"] > 0)
    assert np.all(result["positive_gain"] == result["gain"])
    assert np.all(result["js"] > 0)
    assert result["skill_topk_ids"].shape == (2, 2)
    assert result["clean_topk_ids"].shape == (2, 2)
    assert result["clean_topk_logp"].shape == (2, 2)
    assert result["clean_residual_log_mass"].shape == (2,)
    assert all(np.isfinite(value).all() for value in result.values())


def test_shared_preservation_is_matched_and_disjoint() -> None:
    first = [1, 3, 5]
    second = [3, 6, 8]
    selected = _shared_preservation_indices(
        first,
        second,
        selectable=12,
        count=3,
        seed=7,
    )
    assert len(selected) == 3
    assert not (set(selected) & set(first))
    assert not (set(selected) & set(second))
    assert selected == _shared_preservation_indices(
        first,
        second,
        selectable=12,
        count=3,
        seed=7,
    )
