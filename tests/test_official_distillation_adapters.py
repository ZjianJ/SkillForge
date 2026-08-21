from __future__ import annotations

import math
from pathlib import Path

import pytest

from skillopt.softprefix.official_distillation import (
    chunked_forward_kl,
    chunked_opcd_reverse_kl,
    expected_topk_count,
    load_official_opcd_kl,
    load_official_sekd,
    official_sekd_select,
    official_sekd_select_hidden,
    student_topk_support,
)

OFFICIAL_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "official"


@pytest.mark.skipif(not (OFFICIAL_ROOT / "SE-KD3x" / ".git").exists(), reason="official checkout absent")
def test_sekd_official_selector_uses_ceil_per_sequence_topk():
    torch = pytest.importorskip("torch")
    official = load_official_sekd(OFFICIAL_ROOT)
    # Rows 0 and 4 are nearly uniform (high entropy); rows 1--3 are sharp.
    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [0.0, 9.0, 0.0],
            [0.0, 0.0, 9.0],
            [0.1, 0.0, -0.1],
        ]
    )
    selected, entropy = official_sekd_select(
        logits=logits,
        compute_student_entropy_and_select=official["compute_student_entropy_and_select"],
        k_percent=20.0,
        chunk_size=2,
    )
    assert expected_topk_count(5, 20.0) == 1
    assert selected.tolist() == [0]
    assert entropy[0] > entropy[1]


@pytest.mark.skipif(not (OFFICIAL_ROOT / "SE-KD3x" / ".git").exists(), reason="official checkout absent")
def test_sekd_hidden_selector_matches_logits_adapter_with_identity_head():
    torch = pytest.importorskip("torch")
    official = load_official_sekd(OFFICIAL_ROOT)
    logits = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [0.1, 0.0, -0.1],
            [0.0, 9.0, 0.0],
        ]
    )
    selected_logits, entropy_logits = official_sekd_select(
        logits=logits,
        compute_student_entropy_and_select=official["compute_student_entropy_and_select"],
        k_percent=50.0,
        chunk_size=2,
    )
    selected_hidden, entropy_hidden = official_sekd_select_hidden(
        hidden_states=logits,
        lm_head=torch.nn.Identity(),
        compute_student_entropy_and_select=official["compute_student_entropy_and_select"],
        k_percent=50.0,
        chunk_size=2,
    )
    assert selected_hidden.tolist() == selected_logits.tolist()
    assert torch.allclose(entropy_hidden, entropy_logits)


@pytest.mark.skipif(not (OFFICIAL_ROOT / "SE-KD3x" / ".git").exists(), reason="official checkout absent")
def test_sekd_forward_kl_matches_manual_full_vocab():
    torch = pytest.importorskip("torch")
    official = load_official_sekd(OFFICIAL_ROOT)
    teacher = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]])
    student = torch.tensor([[1.0, 0.5, -0.5], [0.5, 1.5, 0.0]], requires_grad=True)
    loss = chunked_forward_kl(
        teacher_logits=teacher,
        student_logits=student,
        official_forward_kl=official["forward_kl"],
        chunk_size=1,
    )
    q = torch.softmax(teacher, dim=-1)
    manual = (q * (torch.log_softmax(teacher, -1) - torch.log_softmax(student, -1))).sum(-1).mean()
    assert torch.allclose(loss, manual, atol=1e-6)
    loss.backward()
    assert student.grad is not None


@pytest.mark.skipif(not (OFFICIAL_ROOT / "LMOps" / ".git").exists(), reason="official checkout absent")
def test_opcd_official_topk_reverse_kl_is_not_renormalized():
    torch = pytest.importorskip("torch")
    official = load_official_opcd_kl(OFFICIAL_ROOT)
    student = torch.tensor([[3.0, 2.0, 1.0, 0.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 3.0, 0.0, 2.0]])
    indices, _, mass = student_topk_support(student.detach(), k=2)
    student_lp = torch.log_softmax(student, dim=-1).gather(-1, indices)
    teacher_lp = torch.log_softmax(teacher, dim=-1).gather(-1, indices)
    expected = (student_lp.exp() * (student_lp - teacher_lp)).sum(-1).mean()
    loss = chunked_opcd_reverse_kl(
        student_logits=student,
        teacher_topk_logp=teacher_lp,
        topk_indices=indices,
        official_kl_penalty=official["kl_penalty"],
        chunk_size=1,
        renorm_topk=False,
    )
    assert mass.item() < 1.0
    assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert student.grad is not None


def test_expected_topk_count_matches_official_boundary_rule():
    assert expected_topk_count(436, 20.0) == math.ceil(436 * 0.2)
    assert expected_topk_count(1, 0.0) == 1
