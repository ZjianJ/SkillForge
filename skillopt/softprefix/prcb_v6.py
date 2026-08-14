"""Function-space helpers for PRCB-v6 residual soft-prompt boosting."""
from __future__ import annotations

from typing import Any


def centered_logit_delta(torch: Any, learner_logits: Any, reference_logits: Any) -> Any:
    """Return a shift-invariant learner logit correction."""
    delta = learner_logits.float() - reference_logits.float()
    return delta - delta.mean(dim=-1, keepdim=True)


def combine_boosted_logits(
    torch: Any,
    base_logits: Any,
    learner_logits: list[Any],
    alphas: list[float],
) -> Any:
    """Combine independent prefixes additively in centered-logit space."""
    if len(learner_logits) != len(alphas):
        raise ValueError("learner_logits and alphas must have equal length")
    combined = base_logits.float()
    for logits, alpha in zip(learner_logits, alphas, strict=True):
        combined = combined + float(alpha) * centered_logit_delta(
            torch, logits, base_logits
        )
    return combined


def topk_residual_kl_from_logits(
    torch: Any,
    candidate_logits: Any,
    *,
    reference_topk_ids: Any,
    reference_topk_logp: Any,
    reference_residual_log_mass: Any,
) -> Any:
    """KL(reference || candidate) on reference Top-k plus a residual bucket."""
    logits = candidate_logits.float()
    ids = reference_topk_ids.to(device=logits.device, dtype=torch.long)
    reference_top = reference_topk_logp.to(logits.device).float()
    reference_residual = reference_residual_log_mass.to(logits.device).float()
    # Cached Top-k values may be FP16.  Renormalize the K+1 bucket
    # approximation so round-off cannot produce a slightly negative KL.
    reference_normalizer = torch.logsumexp(
        torch.cat([reference_top, reference_residual.unsqueeze(-1)], dim=-1),
        dim=-1,
        keepdim=True,
    )
    reference_top = reference_top - reference_normalizer
    reference_residual = reference_residual - reference_normalizer.squeeze(-1)
    normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
    candidate_top = logits.gather(-1, ids) - normalizer
    candidate_mass = candidate_top.exp().sum(dim=-1).clamp(max=1.0 - 1e-7)
    candidate_residual = torch.log1p(-candidate_mass)
    top_kl = (reference_top.exp() * (reference_top - candidate_top)).sum(dim=-1)
    residual_kl = reference_residual.exp() * (
        reference_residual - candidate_residual
    )
    return (top_kl + residual_kl).clamp_min(0.0)


def topk_reference_from_logits(torch: Any, logits: Any, *, top_k: int = 64):
    """Compress a full logit tensor into Top-k log-probabilities and residual mass."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    values, ids = torch.topk(logp, k=min(int(top_k), int(logp.shape[-1])), dim=-1)
    mass = values.exp().sum(dim=-1).clamp(max=1.0 - 1e-7)
    residual = torch.log1p(-mass)
    return ids, values, residual


def choose_stage_alpha(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the lowest safe objective, preferring the smaller alpha on ties."""
    if not evaluations:
        raise ValueError("At least one alpha evaluation is required")
    safe = [row for row in evaluations if bool(row.get("safe", True))]
    if not safe:
        raise ValueError("No safe alpha candidate")
    return min(
        safe,
        key=lambda row: (float(row["global_loss"]), float(row["alpha"])),
    )
