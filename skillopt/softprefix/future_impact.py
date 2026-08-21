"""Pure utilities for the Future-Impact Locator (FIL-v2).

The helpers in this module deliberately do not know about SpreadsheetBench or
the Qwen model.  Keeping the ranking, splitting, finite-difference and residual
matching logic pure makes the expensive experiment auditable with CPU tests.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class TaskPartition:
    source: tuple[str, ...]
    outer: tuple[str, ...]
    calibration: tuple[str, ...]


def deterministic_task_partition(
    task_ids: Iterable[str],
    *,
    seed: int,
    source_count: int,
    outer_count: int,
    calibration_count: int,
) -> TaskPartition:
    """Create a reproducible, task-level three-way split."""
    ids = [str(task_id) for task_id in task_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("FIL task IDs must be unique")
    requested = int(source_count) + int(outer_count) + int(calibration_count)
    if min(source_count, outer_count, calibration_count) < 1:
        raise ValueError("Every FIL partition must be non-empty")
    if requested != len(ids):
        raise ValueError(f"FIL partition sizes sum to {requested}, but there are {len(ids)} tasks")
    shuffled = list(ids)
    random.Random(int(seed)).shuffle(shuffled)
    source_end = int(source_count)
    outer_end = source_end + int(outer_count)
    return TaskPartition(
        source=tuple(shuffled[:source_end]),
        outer=tuple(shuffled[source_end:outer_end]),
        calibration=tuple(shuffled[outer_end:]),
    )


def per_token_forward_kl(
    *,
    teacher_logits: Any,
    student_logits: Any,
    chunk_size: int = 8,
) -> np.ndarray:
    """Return one FP32 ``KL(teacher || student)`` value per position."""
    import torch

    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"FIL logit shape mismatch: {tuple(teacher_logits.shape)} != "
            f"{tuple(student_logits.shape)}"
        )
    if teacher_logits.ndim != 2:
        raise ValueError(f"FIL expects [tokens, vocab] logits, got {tuple(teacher_logits.shape)}")
    count = int(teacher_logits.shape[0])
    result = np.empty(count, dtype=np.float32)
    width = max(1, int(chunk_size))
    with torch.no_grad():
        for start in range(0, count, width):
            end = min(count, start + width)
            teacher_logp = torch.log_softmax(teacher_logits[start:end].float(), dim=-1)
            student_logp = torch.log_softmax(student_logits[start:end].float(), dim=-1)
            teacher_prob = teacher_logp.exp()
            values = (teacher_prob * (teacher_logp - student_logp)).sum(dim=-1)
            result[start:end] = values.cpu().numpy()
    return result


def chunked_eager_attention(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    scaling: float,
    dropout: float = 0.0,
    **_kwargs: Any,
):
    """Exact eager attention with a bounded query-axis working set.

    The function matches Transformers' eager attention algebra but never
    materializes the complete query-by-key matrix.  It uses only ordinary
    PyTorch operators, so forward-mode AD remains available for FIL on Qwen's
    full-attention layers.
    """
    import torch

    repetitions = int(module.num_key_value_groups)
    if repetitions != 1:
        batch, kv_heads, key_length, head_dim = key.shape
        key = (
            key[:, :, None, :, :]
            .expand(batch, kv_heads, repetitions, key_length, head_dim)
            .reshape(batch, kv_heads * repetitions, key_length, head_dim)
        )
        value = (
            value[:, :, None, :, :]
            .expand(batch, kv_heads, repetitions, key_length, head_dim)
            .reshape(batch, kv_heads * repetitions, key_length, head_dim)
        )
    query_length = int(query.shape[-2])
    chunk_size = max(
        1,
        int(getattr(module.config, "_fil_jvp_attention_query_chunk_size", 256)),
    )
    outputs = []
    for start in range(0, query_length, chunk_size):
        end = min(query_length, start + chunk_size)
        weights = torch.matmul(query[:, :, start:end, :], key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            if int(attention_mask.shape[-2]) == query_length:
                mask = attention_mask[..., start:end, :]
            else:
                mask = attention_mask
            weights = weights + mask
        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        weights = torch.nn.functional.dropout(
            weights, p=dropout, training=module.training
        )
        outputs.append(torch.matmul(weights, value))
    output = torch.cat(outputs, dim=2).transpose(1, 2).contiguous()
    return output, None


def central_difference_scores(
    loss_plus: Any,
    loss_minus: Any,
    *,
    epsilon: float,
) -> np.ndarray:
    """Compute directional derivatives from paired loss vectors."""
    plus = np.asarray(loss_plus, dtype=np.float64).reshape(-1)
    minus = np.asarray(loss_minus, dtype=np.float64).reshape(-1)
    if plus.shape != minus.shape:
        raise ValueError(f"FIL finite-difference shape mismatch: {plus.shape} != {minus.shape}")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("FIL epsilon must be finite and positive")
    return ((plus - minus) / (2.0 * float(epsilon))).astype(np.float32)


def select_fraction(
    scores: Any,
    *,
    ratio: float,
    forbidden: Iterable[int] = (),
    largest: bool = True,
    positive_only: bool = False,
) -> list[int]:
    """Select a deterministic per-trajectory fraction with stable tie breaks.

    The budget follows the existing SkillForge convention: ``ceil(T*ratio)``
    where ``T`` is the number of non-EOS trajectory positions before applying
    the preservation exclusion.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not 0.0 < float(ratio) < 1.0:
        raise ValueError("FIL selection ratio must be strictly between zero and one")
    banned = {int(index) for index in forbidden}
    candidates = [
        index
        for index, value in enumerate(values)
        if index not in banned
        and np.isfinite(value)
        and (not positive_only or value > 0.0)
    ]
    budget = min(len(candidates), max(1, math.ceil(len(values) * float(ratio))))
    if largest:
        ordered = sorted(candidates, key=lambda index: (-values[index], index))
    else:
        ordered = sorted(candidates, key=lambda index: (values[index], index))
    return sorted(ordered[:budget])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_correlation(left: Any, right: Any) -> float:
    """Dependency-free Spearman correlation with average ranks for ties."""
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"FIL rank vectors differ: {x.shape} != {y.shape}")
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return float("nan")
    rx = _average_ranks(x[finite])
    ry = _average_ranks(y[finite])
    if float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(map(int, left)), set(map(int, right))
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)


def adam_diagonal_direction(
    gradient: Any,
    second_moment: Any,
    *,
    adam_epsilon: float = 1e-8,
    floor_fraction: float = 1e-3,
):
    """Return a normalized Adam-metric direction ``P g``.

    A small data-derived floor prevents never-observed coordinates from
    receiving unbounded leverage.  The returned direction is unit L2 norm;
    finite-difference epsilon therefore has the interpretable unit of prefix
    parameter L2 displacement.
    """
    import torch

    grad = gradient.detach().float().reshape(-1)
    moment = second_moment.detach().float().reshape(-1)
    if grad.shape != moment.shape:
        raise ValueError(f"FIL gradient/moment mismatch: {tuple(grad.shape)} != {tuple(moment.shape)}")
    positive = moment[moment > 0]
    reference = torch.median(positive) if int(positive.numel()) else moment.new_tensor(1.0)
    floor = torch.clamp(reference * float(floor_fraction), min=float(adam_epsilon) ** 2)
    direction = grad / (torch.sqrt(moment + floor) + float(adam_epsilon))
    norm = torch.linalg.vector_norm(direction)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise ValueError("FIL outer direction is zero or non-finite")
    return direction / norm, {
        "gradient_l2": float(torch.linalg.vector_norm(grad).cpu()),
        "second_moment_mean": float(moment.mean().cpu()),
        "second_moment_floor": float(floor.cpu()),
        "preconditioned_l2_before_normalization": float(norm.cpu()),
    }


def residual_projection(
    residual: Any,
    block_gradient: Any,
    preconditioner: Any,
    *,
    ridge: float = 1e-12,
):
    """Project a target gradient residual onto one non-negative block edge."""
    import torch

    r = residual.detach().float().reshape(-1)
    g = block_gradient.detach().float().reshape(-1)
    p = preconditioner.detach().float().reshape(-1)
    if r.shape != g.shape or r.shape != p.shape:
        raise ValueError("FIL residual, block gradient and preconditioner must match")
    numerator = torch.dot(r, p * g)
    denominator = torch.dot(g, p * g) + float(ridge)
    alpha = torch.clamp(numerator / denominator, min=0.0)
    updated = r - alpha * g
    before = torch.dot(r, p * r).clamp_min(1e-30)
    after = torch.dot(updated, p * updated)
    closure = 1.0 - after / before
    return updated, float(alpha.cpu()), float(closure.cpu())
