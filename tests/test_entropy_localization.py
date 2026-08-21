import numpy as np
import torch

from scripts.analyze_selective_distillation_tokens import _score_logits
from skillopt.softprefix.entropy_localization import (
    entropy_augmented_scores,
    jaccard,
    minmax_normalize,
    select_top_fraction,
)


def test_score_logits_reports_exact_entropies_and_reduction():
    logits = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [2.0, 0.0, -1.0]]]
    )
    result = _score_logits(logits, [0, 1], exact_js=True, top_k=2, chunk_size=1)
    expected = []
    for row in logits:
        logp = torch.log_softmax(row.float(), dim=-1)
        expected.append((-(logp.exp() * logp).sum(dim=-1)).numpy())
    np.testing.assert_allclose(result["skill_entropy"], expected[0], rtol=1e-6)
    np.testing.assert_allclose(result["clean_entropy"], expected[1], rtol=1e-6)
    np.testing.assert_allclose(
        result["entropy_reduction"], expected[1] - expected[0], rtol=1e-6
    )


def test_entropy_amplifies_only_nonzero_skill_relevance():
    result = entropy_augmented_scores(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 2.0, 1.0]),
        np.array([3.0, 2.0, 1.0]),
        alpha=0.5,
        entropy_lambda=1.0,
    )
    assert result["skill_relevance"][0] == 0.0
    assert result["entropy_augmented"][0] == 0.0
    assert np.all(result["entropy_augmented"] >= result["skill_relevance"])


def test_minmax_selection_and_jaccard_are_stable():
    np.testing.assert_allclose(minmax_normalize(np.array([2.0, 4.0, 6.0])), [0.0, 0.5, 1.0])
    assert select_top_fraction(np.array([1.0, 3.0, 3.0, 2.0]), ratio=0.5, forbidden=[1]) == [2, 3]
    assert jaccard([1, 2], [2, 3]) == 1 / 3
