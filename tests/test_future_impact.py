from __future__ import annotations

import numpy as np
import pytest
import torch

from skillopt.softprefix.future_impact import (
    adam_diagonal_direction,
    central_difference_scores,
    chunked_eager_attention,
    deterministic_task_partition,
    jaccard,
    per_token_forward_kl,
    residual_projection,
    select_fraction,
    spearman_correlation,
)


class _AttentionConfig:
    _fil_jvp_attention_query_chunk_size = 2


class _AttentionModule:
    num_key_value_groups = 2
    training = False
    config = _AttentionConfig()


def test_task_partition_is_disjoint_complete_and_reproducible() -> None:
    ids = [f"task-{index}" for index in range(12)]
    first = deterministic_task_partition(
        ids, seed=17, source_count=7, outer_count=2, calibration_count=3
    )
    second = deterministic_task_partition(
        ids, seed=17, source_count=7, outer_count=2, calibration_count=3
    )
    assert first == second
    assert len(first.source) == 7
    assert len(first.outer) == 2
    assert len(first.calibration) == 3
    assert set(first.source) | set(first.outer) | set(first.calibration) == set(ids)
    assert not (set(first.source) & set(first.outer))
    assert not (set(first.source) & set(first.calibration))


def test_per_token_forward_kl_matches_torch() -> None:
    teacher = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
    student = torch.tensor([[0.5, 0.0, 1.0], [0.0, 2.0, 0.5]])
    actual = per_token_forward_kl(
        teacher_logits=teacher, student_logits=student, chunk_size=1
    )
    tq = torch.log_softmax(teacher, dim=-1)
    sp = torch.log_softmax(student, dim=-1)
    expected = (tq.exp() * (tq - sp)).sum(dim=-1).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_central_difference_matches_quadratic_directional_derivative() -> None:
    theta = np.array([1.5, -2.0])
    direction = np.array([0.25, 0.5])
    epsilon = 1e-4
    plus = (theta + epsilon * direction) ** 2
    minus = (theta - epsilon * direction) ** 2
    actual = central_difference_scores(plus, minus, epsilon=epsilon)
    np.testing.assert_allclose(actual, 2.0 * theta * direction, rtol=1e-5, atol=1e-5)


def test_selection_is_stable_excludes_preservation_and_can_require_edge() -> None:
    scores = np.array([0.5, 0.9, 0.9, -0.1, np.nan, 0.2])
    # ceil(6 * .5) = 3; index 1 is forbidden and ties retain lower index.
    assert select_fraction(scores, ratio=0.5, forbidden=[1]) == [0, 2, 5]
    assert select_fraction(
        scores, ratio=0.5, forbidden=[1], positive_only=True, largest=False
    ) == [0, 2, 5]
    assert 3 in select_fraction(scores, ratio=0.5, largest=False)


def test_rank_and_overlap_helpers() -> None:
    assert spearman_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)


def test_adam_direction_and_residual_projection() -> None:
    gradient = torch.tensor([2.0, 1.0])
    moment = torch.tensor([4.0, 1.0])
    direction, stats = adam_diagonal_direction(gradient, moment)
    assert torch.linalg.vector_norm(direction) == pytest.approx(1.0)
    assert stats["gradient_l2"] == pytest.approx(float(torch.linalg.vector_norm(gradient)))

    preconditioner = 1.0 / torch.sqrt(moment)
    residual = torch.tensor([1.0, 1.0])
    block = torch.tensor([1.0, 0.0])
    updated, alpha, closure = residual_projection(residual, block, preconditioner)
    assert alpha == pytest.approx(1.0)
    torch.testing.assert_close(updated, torch.tensor([0.0, 1.0]))
    assert closure > 0.0


def test_chunked_eager_attention_matches_full_and_supports_jvp() -> None:
    torch.manual_seed(7)
    query = torch.randn(1, 4, 5, 3, dtype=torch.float64)
    key = torch.randn(1, 2, 5, 3, dtype=torch.float64)
    value = torch.randn(1, 2, 5, 3, dtype=torch.float64)
    mask = torch.full((1, 1, 5, 5), float("-inf"), dtype=torch.float64)
    mask = torch.triu(mask, diagonal=1)
    scaling = 3**-0.5

    repeated_key = key[:, :, None].expand(1, 2, 2, 5, 3).reshape(1, 4, 5, 3)
    repeated_value = value[:, :, None].expand(1, 2, 2, 5, 3).reshape(1, 4, 5, 3)

    def full(current_query):
        weights = torch.matmul(current_query, repeated_key.transpose(2, 3)) * scaling
        weights = torch.softmax(weights + mask, dim=-1, dtype=torch.float32).to(
            current_query.dtype
        )
        return torch.matmul(weights, repeated_value).transpose(1, 2).contiguous()

    def chunked(current_query):
        return chunked_eager_attention(
            _AttentionModule(),
            current_query,
            key,
            value,
            mask,
            scaling,
        )[0]

    torch.testing.assert_close(chunked(query), full(query), rtol=1e-12, atol=1e-12)
    tangent = torch.randn_like(query)
    _, chunked_jvp = torch.func.jvp(chunked, (query,), (tangent,))
    _, full_jvp = torch.func.jvp(full, (query,), (tangent,))
    torch.testing.assert_close(chunked_jvp, full_jvp, rtol=1e-10, atol=1e-10)
