from __future__ import annotations

import pytest

from skillopt.softprefix.distillation_losses import (
    chunked_full_vocab_forward_kl,
    chunked_full_vocab_forward_kl_vector,
    chunked_weighted_full_vocab_forward_kl,
    topk_residual_forward_kl,
)


def test_full_vocab_forward_kl_matches_manual_and_backpropagates():
    torch = pytest.importorskip("torch")
    teacher = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
    student = torch.tensor([[1.0, 0.5, -0.5], [0.5, 1.5, 0.0]], requires_grad=True)
    loss = chunked_full_vocab_forward_kl(
        teacher_logits=teacher,
        student_logits=student,
        chunk_size=1,
    )
    teacher_logp = torch.log_softmax(teacher, dim=-1)
    manual = (
        torch.softmax(teacher, dim=-1)
        * (teacher_logp - torch.log_softmax(student, dim=-1))
    ).sum(-1).mean()
    assert torch.allclose(loss, manual, atol=1e-6)
    loss.backward()
    assert student.grad is not None


def test_full_vocab_forward_kl_vector_matches_scalar_mean_and_backpropagates():
    torch = pytest.importorskip("torch")
    teacher = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
    student = torch.tensor(
        [[1.0, 0.5, -0.5], [0.5, 1.5, 0.0]], requires_grad=True
    )
    vector = chunked_full_vocab_forward_kl_vector(
        teacher_logits=teacher,
        student_logits=student,
        chunk_size=1,
    )
    scalar = chunked_full_vocab_forward_kl(
        teacher_logits=teacher,
        student_logits=student,
        chunk_size=1,
    )
    assert vector.shape == (2,)
    assert torch.allclose(vector.mean(), scalar, atol=1e-6)
    vector.sum().backward()
    assert student.grad is not None
    assert bool(torch.isfinite(student.grad).all())


def test_topk_residual_forward_kl_matches_aggregated_manual_distribution():
    torch = pytest.importorskip("torch")
    student = torch.tensor([[2.0, 1.0, 0.0, -1.0]], requires_grad=True)
    reference = torch.tensor([[0.5, 0.2, 0.2, 0.1]])
    topk_ids = torch.tensor([[0, 1]])
    reference_topk_logp = reference.gather(1, topk_ids).log()
    reference_residual_log_mass = reference[:, 2:].sum(-1).log()
    loss = topk_residual_forward_kl(
        student_logits=student,
        reference_topk_ids=topk_ids,
        reference_topk_logp=reference_topk_logp,
        reference_residual_log_mass=reference_residual_log_mass,
    )
    student_prob = torch.softmax(student, dim=-1)
    student_aggregate = torch.cat(
        [student_prob.gather(1, topk_ids), student_prob[:, 2:].sum(-1, keepdim=True)],
        dim=-1,
    )
    reference_aggregate = torch.tensor([[0.5, 0.2, 0.3]])
    manual = (
        reference_aggregate
        * (reference_aggregate.log() - student_aggregate.log())
    ).sum(-1).mean()
    assert torch.allclose(loss, manual, atol=1e-6)
    loss.backward()
    assert student.grad is not None


def test_full_vocab_forward_kl_rejects_shape_mismatch():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="shape mismatch"):
        chunked_full_vocab_forward_kl(
            teacher_logits=torch.zeros(2, 3),
            student_logits=torch.zeros(2, 4),
        )


def test_weighted_full_vocab_kl_matches_manual_and_detaches_weights():
    torch = pytest.importorskip("torch")
    teacher = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, -1.0]])
    student = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]], requires_grad=True
    )
    weights = torch.tensor([2.0, 1.0, 0.5], requires_grad=True)
    loss = chunked_weighted_full_vocab_forward_kl(
        teacher_logits=teacher,
        student_logits=student,
        token_weights=weights,
        chunk_size=2,
    )
    teacher_logp = torch.log_softmax(teacher, dim=-1)
    per_token = (
        teacher_logp.exp()
        * (teacher_logp - torch.log_softmax(student, dim=-1))
    ).sum(-1)
    manual = (weights.detach() * per_token).sum() / weights.detach().sum()
    assert torch.allclose(loss, manual, atol=1e-6)
    loss.backward()
    assert student.grad is not None
    assert weights.grad is None
