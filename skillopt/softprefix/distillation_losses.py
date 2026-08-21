"""Distribution-level losses shared by soft-prefix distillation experiments."""
from __future__ import annotations

from typing import Any


def chunked_full_vocab_forward_kl(
    *,
    teacher_logits: Any,
    student_logits: Any,
    chunk_size: int = 8,
):
    """Return mean ``KL(teacher || student)`` over the complete vocabulary."""
    import torch

    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"Full-vocabulary KL shape mismatch: {tuple(teacher_logits.shape)} != "
            f"{tuple(student_logits.shape)}"
        )
    if teacher_logits.ndim != 2:
        raise ValueError(f"Full-vocabulary KL expects [tokens, vocab], got {tuple(teacher_logits.shape)}")
    count = int(student_logits.shape[0])
    if count == 0:
        return student_logits.sum() * 0.0

    total = student_logits.sum() * 0.0
    width = max(1, int(chunk_size))
    for start in range(0, count, width):
        end = min(count, start + width)
        with torch.no_grad():
            teacher_logp = torch.log_softmax(teacher_logits[start:end].float(), dim=-1)
            teacher_prob = teacher_logp.exp()
        student_logp = torch.log_softmax(student_logits[start:end].float(), dim=-1)
        total = total + (teacher_prob * (teacher_logp - student_logp)).sum()
    return total / count


def chunked_full_vocab_forward_kl_vector(
    *,
    teacher_logits: Any,
    student_logits: Any,
    chunk_size: int = 8,
):
    """Return differentiable per-token ``KL(teacher || student)`` values.

    Unlike :func:`chunked_full_vocab_forward_kl`, this retains the token axis.
    It is used by FIL's forward-mode JVP so one model pass yields every
    token's directional derivative without one backward pass per token.
    """
    import torch

    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"Full-vocabulary KL shape mismatch: {tuple(teacher_logits.shape)} != "
            f"{tuple(student_logits.shape)}"
        )
    if teacher_logits.ndim != 2:
        raise ValueError(f"Full-vocabulary KL expects [tokens, vocab], got {tuple(teacher_logits.shape)}")
    count = int(student_logits.shape[0])
    if count == 0:
        return student_logits.sum(dim=-1) * 0.0
    pieces = []
    width = max(1, int(chunk_size))
    for start in range(0, count, width):
        end = min(count, start + width)
        with torch.no_grad():
            teacher_logp = torch.log_softmax(teacher_logits[start:end].float(), dim=-1)
            teacher_prob = teacher_logp.exp()
        student_logp = torch.log_softmax(student_logits[start:end].float(), dim=-1)
        pieces.append((teacher_prob * (teacher_logp - student_logp)).sum(dim=-1))
    return torch.cat(pieces, dim=0)


def chunked_weighted_full_vocab_forward_kl(
    *,
    teacher_logits: Any,
    student_logits: Any,
    token_weights: Any,
    chunk_size: int = 8,
):
    """Return a stop-gradient token-weighted ``KL(teacher || student)``.

    ``token_weights`` has one non-negative scalar per prediction position.  It
    is detached internally so a state-dependent locator cannot lower the
    objective by changing its own weights instead of matching the teacher.
    """
    import torch

    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"Full-vocabulary KL shape mismatch: {tuple(teacher_logits.shape)} != "
            f"{tuple(student_logits.shape)}"
        )
    if teacher_logits.ndim != 2:
        raise ValueError(f"Full-vocabulary KL expects [tokens, vocab], got {tuple(teacher_logits.shape)}")
    count = int(student_logits.shape[0])
    weights = torch.as_tensor(
        token_weights, dtype=torch.float32, device=student_logits.device
    ).reshape(-1).detach()
    if int(weights.numel()) != count:
        raise ValueError(
            f"Token-weight mismatch: logits={count} weights={int(weights.numel())}"
        )
    if count == 0:
        return student_logits.sum() * 0.0
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("Token weights must be finite and non-negative")
    denominator = weights.sum()
    if float(denominator.detach().cpu()) <= 0.0:
        raise ValueError("Token weights must have positive total mass")

    total = student_logits.sum() * 0.0
    width = max(1, int(chunk_size))
    for start in range(0, count, width):
        end = min(count, start + width)
        with torch.no_grad():
            teacher_logp = torch.log_softmax(teacher_logits[start:end].float(), dim=-1)
            teacher_prob = teacher_logp.exp()
        student_logp = torch.log_softmax(student_logits[start:end].float(), dim=-1)
        per_token = (teacher_prob * (teacher_logp - student_logp)).sum(dim=-1)
        total = total + (weights[start:end] * per_token).sum()
    return total / denominator


def topk_residual_forward_kl(
    *,
    student_logits: Any,
    reference_topk_ids: Any,
    reference_topk_logp: Any,
    reference_residual_log_mass: Any,
):
    """KL(reference || student) over reference Top-k plus one residual bucket.

    This exactly matches the existing preservation objective while accepting
    already-gathered student logits with shape ``[tokens, vocab]``.
    """
    import torch

    if student_logits.ndim != 2:
        raise ValueError(f"Residual KL expects [tokens, vocab], got {tuple(student_logits.shape)}")
    device = student_logits.device
    topk_ids = torch.as_tensor(reference_topk_ids, dtype=torch.long, device=device)
    reference_logp = torch.as_tensor(reference_topk_logp, device=device).float()
    reference_residual = torch.as_tensor(reference_residual_log_mass, device=device).float()
    expected = tuple(topk_ids.shape)
    if tuple(reference_logp.shape) != expected:
        raise ValueError(f"Top-k ID/log-prob shape mismatch: {expected} != {tuple(reference_logp.shape)}")
    if topk_ids.ndim != 2 or int(topk_ids.shape[0]) != int(student_logits.shape[0]):
        raise ValueError(
            f"Residual KL token mismatch: logits={tuple(student_logits.shape)} topk={tuple(topk_ids.shape)}"
        )
    if tuple(reference_residual.shape) != (int(student_logits.shape[0]),):
        raise ValueError(
            f"Residual log-mass shape mismatch: {tuple(reference_residual.shape)} != "
            f"{(int(student_logits.shape[0]),)}"
        )
    if int(student_logits.shape[0]) == 0:
        return student_logits.sum() * 0.0

    student_float = student_logits.float()
    student_logz = torch.logsumexp(student_float, dim=-1, keepdim=True)
    student_topk_logp = student_float.gather(1, topk_ids) - student_logz
    student_topk_mass = student_topk_logp.exp().sum(dim=-1).clamp(max=1.0 - 1e-7)
    student_residual = torch.log1p(-student_topk_mass)

    topk_kl = (reference_logp.exp() * (reference_logp - student_topk_logp)).sum(dim=-1)
    residual_kl = reference_residual.exp() * (reference_residual - student_residual)
    return (topk_kl + residual_kl).mean()
