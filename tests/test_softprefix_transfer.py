"""Tests for cross-model soft-prefix transfer helpers."""
from __future__ import annotations


def test_project_prefix_via_vocab_maps_to_target_hidden_size() -> None:
    import torch

    from skillopt.softprefix.transfer import project_prefix_via_vocab

    source_prefix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    source_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    target_embeddings = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [-10.0, 0.0, 30.0],
        ]
    )

    projected = project_prefix_via_vocab(
        source_prefix,
        source_embeddings,
        target_embeddings,
        temperature=0.01,
    )

    assert projected.shape == (2, 3)
    assert torch.allclose(projected[0], target_embeddings[0], atol=1e-3)
    assert torch.allclose(projected[1], target_embeddings[1], atol=1e-3)


def test_project_prefix_via_vocab_supports_top_k() -> None:
    import torch

    from skillopt.softprefix.transfer import project_prefix_via_vocab

    source_prefix = torch.tensor([[1.0, 0.0]])
    source_embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    target_embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 100.0],
        ]
    )

    projected = project_prefix_via_vocab(
        source_prefix,
        source_embeddings,
        target_embeddings,
        temperature=1.0,
        top_k=1,
    )

    assert torch.allclose(projected, target_embeddings[:1])
