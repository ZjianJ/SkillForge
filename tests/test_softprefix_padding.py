from __future__ import annotations

import torch

from skillopt.softprefix.model import SoftPrefixCausalLM


class _EmbeddingModel:
    def __init__(self) -> None:
        self.embedding = torch.nn.Embedding(16, 1)
        with torch.no_grad():
            self.embedding.weight[:, 0] = torch.arange(16)

    def get_input_embeddings(self):
        return self.embedding


def _wrapper() -> SoftPrefixCausalLM:
    wrapper = object.__new__(SoftPrefixCausalLM)
    wrapper.torch = torch
    wrapper.model = _EmbeddingModel()
    wrapper.device = torch.device("cpu")
    wrapper.prefix_length = 1
    wrapper.prefix_embeddings = torch.nn.Parameter(torch.tensor([[99.0]]))
    return wrapper


def test_default_prefix_is_inserted_after_left_padding_per_row() -> None:
    wrapper = _wrapper()
    input_ids = torch.tensor([[0, 0, 1, 2], [3, 4, 5, 6]])
    attention_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])

    embeddings, full_mask, _ = wrapper._with_prefix(input_ids, attention_mask)

    assert embeddings[:, :, 0].tolist() == [
        [0.0, 0.0, 99.0, 1.0, 2.0],
        [99.0, 3.0, 4.0, 5.0, 6.0],
    ]
    assert full_mask.tolist() == [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]


def test_explicit_prefix_index_is_shifted_across_left_padding() -> None:
    wrapper = _wrapper()
    input_ids = torch.tensor([[0, 0, 1, 2], [3, 4, 5, 6]])
    attention_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])

    embeddings, _, _ = wrapper._with_prefix(
        input_ids,
        attention_mask,
        prefix_insert_idx=torch.tensor([1, 1]),
    )

    assert embeddings[:, :, 0].tolist() == [
        [0.0, 0.0, 1.0, 99.0, 2.0],
        [3.0, 99.0, 4.0, 5.0, 6.0],
    ]
