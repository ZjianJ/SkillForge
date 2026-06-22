"""Tests for soft-prefix training loss memory behavior."""
from __future__ import annotations

from types import SimpleNamespace


def test_causal_forward_computes_masked_loss_without_backbone_labels() -> None:
    import torch

    from skillopt.softprefix.model import SoftPrefixCausalLM

    class FakeBackbone(torch.nn.Module):
        dtype = torch.float32

        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.lm_head = torch.nn.Linear(4, 16, bias=False)
            self.labels_seen = "unset"

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, *, inputs_embeds, attention_mask, labels=None, **kwargs):
            del attention_mask, kwargs
            self.labels_seen = labels
            if labels is not None:
                raise AssertionError("wrapper should not ask HF to compute full-sequence loss")
            return SimpleNamespace(logits=self.lm_head(inputs_embeds))

    backbone = FakeBackbone()
    wrapper = SoftPrefixCausalLM.__new__(SoftPrefixCausalLM)
    wrapper.torch = torch
    wrapper.model = backbone
    wrapper.device = torch.device("cpu")
    wrapper.prefix_length = 2
    wrapper.prefix_embeddings = torch.nn.Parameter(torch.randn(2, 4))

    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "labels": torch.tensor([[-100, -100, 6, 7]]),
    }

    outputs = wrapper.forward(batch)

    assert backbone.labels_seen is None
    assert outputs.loss.requires_grad
    outputs.loss.backward()
    assert wrapper.prefix_embeddings.grad is not None


def test_causal_prefix_can_be_inserted_at_batch_indices() -> None:
    import torch

    from skillopt.softprefix.model import SoftPrefixCausalLM

    class FakeBackbone(torch.nn.Module):
        dtype = torch.float32

        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 1)
            with torch.no_grad():
                self.embedding.weight.copy_(torch.arange(16, dtype=torch.float32).view(16, 1))

        def get_input_embeddings(self):
            return self.embedding

    wrapper = SoftPrefixCausalLM.__new__(SoftPrefixCausalLM)
    wrapper.torch = torch
    wrapper.model = FakeBackbone()
    wrapper.device = torch.device("cpu")
    wrapper.prefix_length = 2
    wrapper.prefix_embeddings = torch.nn.Parameter(torch.tensor([[100.0], [101.0]]))

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    labels = torch.tensor([[-100, 20, 21], [-100, -100, 22]])

    inputs_embeds, full_attention_mask, full_labels = wrapper._with_prefix(
        input_ids,
        attention_mask,
        labels,
        prefix_insert_idx=torch.tensor([1, 2]),
    )

    assert inputs_embeds.squeeze(-1).tolist() == [
        [1.0, 100.0, 101.0, 2.0, 3.0],
        [4.0, 5.0, 100.0, 101.0, 6.0],
    ]
    assert full_attention_mask.tolist() == [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
    ]
    assert full_labels.tolist() == [
        [-100, -100, -100, 20, 21],
        [-100, -100, -100, -100, 22],
    ]
