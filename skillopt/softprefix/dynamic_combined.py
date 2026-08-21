"""Pure helpers for dynamic full-vocabulary Combined localization."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def full_vocab_dynamic_metrics(
    *,
    teacher_logits: Any,
    student_logits: Any,
    target_ids: Any,
    original_beneficial: Any,
    chunk_size: int = 8,
) -> dict[str, np.ndarray]:
    """Measure the current hard-Skill/student residual over the full vocabulary.

    Returned arrays are detached float32 CPU arrays with one value per target
    position.  ``combined`` is non-zero only where the original hard Skill was
    beneficial relative to the no-Skill model and the current prefix still
    assigns lower probability to the successful target token than the teacher.
    """
    import torch

    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"Dynamic locator logit shape mismatch: {tuple(teacher_logits.shape)} != "
            f"{tuple(student_logits.shape)}"
        )
    if teacher_logits.ndim != 2:
        raise ValueError(
            f"Dynamic locator expects [tokens, vocab], got {tuple(teacher_logits.shape)}"
        )
    count = int(teacher_logits.shape[0])
    targets = torch.as_tensor(target_ids, dtype=torch.long, device=teacher_logits.device).reshape(-1)
    beneficial = np.asarray(original_beneficial, dtype=bool).reshape(-1)
    if int(targets.numel()) != count or len(beneficial) != count:
        raise ValueError(
            "Dynamic locator target/benefit length mismatch: "
            f"logits={count} targets={int(targets.numel())} beneficial={len(beneficial)}"
        )

    teacher_target_logp = np.empty(count, dtype=np.float32)
    student_target_logp = np.empty(count, dtype=np.float32)
    residual_gain = np.empty(count, dtype=np.float32)
    forward_kl = np.empty(count, dtype=np.float32)
    js = np.empty(count, dtype=np.float32)
    competitor_suppression = np.empty(count, dtype=np.float32)
    teacher_entropy = np.empty(count, dtype=np.float32)
    student_entropy = np.empty(count, dtype=np.float32)
    resolved_uncertainty = np.empty(count, dtype=np.float32)
    width = max(1, int(chunk_size))
    log_two = math.log(2.0)

    with torch.no_grad():
        for start in range(0, count, width):
            end = min(count, start + width)
            teacher_logp = torch.log_softmax(teacher_logits[start:end].float(), dim=-1)
            student_logp = torch.log_softmax(student_logits[start:end].float(), dim=-1)
            teacher_prob = teacher_logp.exp()
            student_prob = student_logp.exp()
            ids = targets[start:end, None]
            teacher_target = teacher_logp.gather(1, ids).squeeze(1)
            student_target = student_logp.gather(1, ids).squeeze(1)
            log_mix = torch.logaddexp(teacher_logp, student_logp) - log_two
            teacher_top2_logp, teacher_top2_ids = torch.topk(teacher_logp, k=2, dim=-1)
            student_top2_logp, student_top2_ids = torch.topk(student_logp, k=2, dim=-1)
            teacher_wrong = torch.where(
                teacher_top2_ids[:, 0].eq(targets[start:end]),
                teacher_top2_logp[:, 1],
                teacher_top2_logp[:, 0],
            )
            student_wrong = torch.where(
                student_top2_ids[:, 0].eq(targets[start:end]),
                student_top2_logp[:, 1],
                student_top2_logp[:, 0],
            )

            teacher_target_logp[start:end] = teacher_target.cpu().numpy()
            student_target_logp[start:end] = student_target.cpu().numpy()
            residual_gain[start:end] = torch.clamp(
                teacher_target - student_target, min=0.0
            ).cpu().numpy()
            forward_kl[start:end] = (
                teacher_prob * (teacher_logp - student_logp)
            ).sum(dim=-1).cpu().numpy()
            js[start:end] = (
                0.5
                * (
                    (teacher_prob * (teacher_logp - log_mix)).sum(dim=-1)
                    + (student_prob * (student_logp - log_mix)).sum(dim=-1)
                )
            ).cpu().numpy()
            competitor_suppression[start:end] = torch.clamp(
                student_wrong - teacher_wrong, min=0.0
            ).cpu().numpy()
            current_teacher_entropy = -(teacher_prob * teacher_logp).sum(dim=-1)
            current_student_entropy = -(student_prob * student_logp).sum(dim=-1)
            teacher_confidence = torch.clamp(
                1.0 - current_teacher_entropy / math.log(teacher_logits.shape[-1]),
                min=0.0,
                max=1.0,
            )
            teacher_entropy[start:end] = current_teacher_entropy.cpu().numpy()
            student_entropy[start:end] = current_student_entropy.cpu().numpy()
            resolved_uncertainty[start:end] = (
                torch.clamp(current_student_entropy - current_teacher_entropy, min=0.0)
                * teacher_confidence
            ).cpu().numpy()

    combined = beneficial.astype(np.float32) * residual_gain * np.maximum(js, 0.0)
    return {
        "teacher_target_logp": teacher_target_logp,
        "student_target_logp": student_target_logp,
        "residual_gain": residual_gain,
        "forward_kl": forward_kl,
        "js": js,
        "competitor_suppression": competitor_suppression,
        "teacher_entropy": teacher_entropy,
        "student_entropy": student_entropy,
        "resolved_uncertainty": resolved_uncertainty,
        "combined": combined.astype(np.float32),
    }


def minmax_normalize_scoped(
    values: np.ndarray,
    eligible_indices: Iterable[int] | None = None,
) -> np.ndarray:
    """Min-max normalize one trajectory over its eligible positions."""
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    indices = (
        np.arange(len(source), dtype=np.int64)
        if eligible_indices is None
        else np.asarray(list(eligible_indices), dtype=np.int64)
    )
    result = np.zeros_like(source, dtype=np.float64)
    if len(indices) == 0:
        return result.astype(np.float32)
    scoped = source[indices]
    finite = np.isfinite(scoped)
    if not finite.any():
        return result.astype(np.float32)
    low = float(scoped[finite].min())
    high = float(scoped[finite].max())
    if high > low:
        normalized = (scoped - low) / (high - low)
        normalized[~finite] = 0.0
        result[indices] = normalized
    return result.astype(np.float32)


def dynamic_skill_effect_scores(
    residual_gain: np.ndarray,
    js: np.ndarray,
    competitor_suppression: np.ndarray,
    resolved_uncertainty: np.ndarray,
    *,
    weights: Iterable[float] = (0.45, 0.45, 0.10, 0.0),
    mode: str = "additive",
    eligible_indices: Iterable[int] | None = None,
) -> dict[str, np.ndarray]:
    """Combine G, JS, C and Skill-resolved uncertainty for localization.

    ``additive`` uses ``a*G + b*JS + c*C + d*R`` with four simplex weights.
    ``multiplicative`` uses ``(a*G + b*JS + c*C) * (1 + d*R)``; in that
    mode the first three weights must sum to one and ``d`` is in ``[0, 1]``.
    All signals are independently normalized within the current trajectory.
    """
    arrays = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (residual_gain, js, competitor_suppression, resolved_uncertainty)
    ]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("Four-signal locator arrays must have identical shapes")
    coefficients = np.asarray(list(weights), dtype=np.float64).reshape(-1)
    if len(coefficients) != 4 or not np.isfinite(coefficients).all():
        raise ValueError("Four-signal locator requires four finite weights")
    if np.any(coefficients < 0.0):
        raise ValueError("Four-signal locator weights must be non-negative")
    normalized = [minmax_normalize_scoped(value, eligible_indices) for value in arrays]
    normalized_gain, normalized_js, normalized_competitor, normalized_resolved = normalized
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "additive":
        if not np.isclose(float(coefficients.sum()), 1.0, atol=1e-6):
            raise ValueError("Additive four-signal weights must sum to one")
        score = sum(weight * signal for weight, signal in zip(coefficients, normalized))
    elif normalized_mode == "multiplicative":
        if not np.isclose(float(coefficients[:3].sum()), 1.0, atol=1e-6):
            raise ValueError("Multiplicative G/JS/C weights must sum to one")
        if float(coefficients[3]) > 1.0:
            raise ValueError("Multiplicative uncertainty strength must be in [0, 1]")
        effect = (
            coefficients[0] * normalized_gain
            + coefficients[1] * normalized_js
            + coefficients[2] * normalized_competitor
        )
        score = effect * (1.0 + coefficients[3] * normalized_resolved)
    else:
        raise ValueError(f"Unsupported four-signal mode: {mode!r}")
    return {
        "normalized_gain": normalized_gain,
        "normalized_js": normalized_js,
        "normalized_competitor": normalized_competitor,
        "normalized_resolved_uncertainty": normalized_resolved,
        "skill_effect": np.asarray(score, dtype=np.float32),
    }


def dynamic_additive_scores(
    residual_gain: np.ndarray,
    js: np.ndarray,
    *,
    alpha: float = 0.5,
    eligible_indices: Iterable[int] | None = None,
) -> dict[str, np.ndarray]:
    """Build the state-dependent Additive Skill locator.

    The static Additive Skill locator is ``alpha * G~ + (1-alpha) * JS~``.
    Here ``G`` is the remaining positive target-token log-probability gap
    between the hard-Skill teacher and the current soft prefix, while ``JS``
    is their full-vocabulary Jensen-Shannon divergence.  Min-max
    normalization is performed independently inside each trajectory and only
    over positions eligible for localization.
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    gain = np.asarray(residual_gain, dtype=np.float64).reshape(-1)
    divergence = np.asarray(js, dtype=np.float64).reshape(-1)
    if gain.shape != divergence.shape:
        raise ValueError(f"Additive score shape mismatch: {gain.shape} != {divergence.shape}")
    indices = (
        np.arange(len(gain), dtype=np.int64)
        if eligible_indices is None
        else np.asarray(list(eligible_indices), dtype=np.int64)
    )

    def normalize(values: np.ndarray) -> np.ndarray:
        result = np.zeros_like(values, dtype=np.float64)
        if len(indices) == 0:
            return result
        scoped = values[indices]
        finite = np.isfinite(scoped)
        if not finite.any():
            return result
        low = float(scoped[finite].min())
        high = float(scoped[finite].max())
        if high > low:
            normalized = (scoped - low) / (high - low)
            normalized[~finite] = 0.0
            result[indices] = normalized
        return result

    normalized_gain = normalize(gain)
    normalized_js = normalize(divergence)
    additive = float(alpha) * normalized_gain + (1.0 - float(alpha)) * normalized_js
    return {
        "normalized_gain": normalized_gain.astype(np.float32),
        "normalized_js": normalized_js.astype(np.float32),
        "additive": additive.astype(np.float32),
    }


def dynamic_gain_competitor_scores(
    residual_gain: np.ndarray,
    competitor_suppression: np.ndarray,
    *,
    gain_weight: float = 0.75,
    eligible_indices: Iterable[int] | None = None,
) -> dict[str, np.ndarray]:
    """Build ``gain_weight * G~ + (1-gain_weight) * C~`` locator scores.

    ``C`` is the positive reduction, relative to the current soft prefix, in
    the highest-probability non-gold competitor under the hard-Skill teacher.
    Both components are min-max normalized per trajectory over eligible
    positions, matching the existing dynamic Additive normalization policy.
    """
    if not 0.0 <= float(gain_weight) <= 1.0:
        raise ValueError("gain_weight must be in [0, 1]")
    gain = np.asarray(residual_gain, dtype=np.float64).reshape(-1)
    competitor = np.asarray(competitor_suppression, dtype=np.float64).reshape(-1)
    if gain.shape != competitor.shape:
        raise ValueError(f"G+C score shape mismatch: {gain.shape} != {competitor.shape}")
    indices = (
        np.arange(len(gain), dtype=np.int64)
        if eligible_indices is None
        else np.asarray(list(eligible_indices), dtype=np.int64)
    )

    def normalize(values: np.ndarray) -> np.ndarray:
        result = np.zeros_like(values, dtype=np.float64)
        if len(indices) == 0:
            return result
        scoped = values[indices]
        finite = np.isfinite(scoped)
        if not finite.any():
            return result
        low = float(scoped[finite].min())
        high = float(scoped[finite].max())
        if high > low:
            normalized = (scoped - low) / (high - low)
            normalized[~finite] = 0.0
            result[indices] = normalized
        return result

    normalized_gain = normalize(gain)
    normalized_competitor = normalize(competitor)
    score = (
        float(gain_weight) * normalized_gain
        + (1.0 - float(gain_weight)) * normalized_competitor
    )
    return {
        "normalized_gain": normalized_gain.astype(np.float32),
        "normalized_competitor": normalized_competitor.astype(np.float32),
        "gain_competitor": score.astype(np.float32),
    }


def select_dynamic_top_fraction(
    scores: np.ndarray,
    *,
    ratio: float,
    forbidden_indices: Iterable[int] = (),
) -> list[int]:
    """Select a stable per-trajectory Top-ratio core from positive scores."""
    if not 0.0 < float(ratio) < 1.0:
        raise ValueError("ratio must be strictly between zero and one")
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    requested = max(1, math.ceil(len(values) * float(ratio)))
    forbidden = {int(index) for index in forbidden_indices}
    candidates = [
        index
        for index, value in enumerate(values)
        if index not in forbidden and np.isfinite(value) and float(value) > 0.0
    ]
    candidates.sort(key=lambda index: (-float(values[index]), index))
    return sorted(candidates[:requested])


def locator_loss_weights(
    scores: np.ndarray,
    selected_indices: Iterable[int],
    *,
    effective_weight: float,
) -> np.ndarray:
    """Normalize detached locator scores to a fixed effective token mass."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    indices = np.asarray(list(selected_indices), dtype=np.int64)
    if len(indices) == 0:
        return np.empty(0, dtype=np.float32)
    if float(effective_weight) <= 0.0:
        raise ValueError("effective_weight must be positive")
    raw = np.maximum(values[indices], 0.0)
    if not np.isfinite(raw).all():
        raise ValueError("Selected locator scores must be finite")
    total = float(raw.sum())
    if total <= 0.0:
        raw = np.ones(len(indices), dtype=np.float64)
        total = float(len(indices))
    return (raw * (float(effective_weight) / total)).astype(np.float32)


def jaccard_indices(first: Iterable[int], second: Iterable[int]) -> float:
    """Return set Jaccard, defining two empty sets as perfectly stable."""
    left, right = set(map(int, first)), set(map(int, second))
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


@dataclass(frozen=True)
class DynamicStopDecision:
    stop: bool
    reason: str
    residual_mass_ratio: float
    relative_mass_improvement: float | None
    stagnant_rounds: int


def dynamic_stop_decision(
    *,
    current_mass: float,
    initial_mass: float,
    previous_mass: float | None,
    mean_eligible_kl: float,
    completed_relocations: int,
    max_relocations: int,
    stagnant_rounds: int,
    residual_mass_ratio_threshold: float,
    mean_eligible_kl_threshold: float,
    min_relative_mass_improvement: float,
    global_patience: int,
) -> DynamicStopDecision:
    """Apply auditable stopping rules after a fresh dynamic localization."""
    eps = 1e-12
    ratio = float(current_mass) / max(float(initial_mass), eps)
    improvement = None
    next_stagnant = int(stagnant_rounds)
    if previous_mass is not None:
        improvement = (float(previous_mass) - float(current_mass)) / max(
            abs(float(previous_mass)), eps
        )
        next_stagnant = 0 if improvement >= float(min_relative_mass_improvement) else next_stagnant + 1

    if float(current_mass) <= eps:
        return DynamicStopDecision(True, "no-positive-dynamic-residual", ratio, improvement, next_stagnant)
    if ratio <= float(residual_mass_ratio_threshold):
        return DynamicStopDecision(True, "residual-mass-small", ratio, improvement, next_stagnant)
    if float(mean_eligible_kl) <= float(mean_eligible_kl_threshold):
        return DynamicStopDecision(True, "eligible-full-vocab-kl-small", ratio, improvement, next_stagnant)
    if previous_mass is not None and next_stagnant >= int(global_patience):
        return DynamicStopDecision(True, "global-residual-stagnant", ratio, improvement, next_stagnant)
    if int(completed_relocations) >= int(max_relocations):
        return DynamicStopDecision(True, "relocation-budget-exhausted", ratio, improvement, next_stagnant)
    return DynamicStopDecision(False, "continue", ratio, improvement, next_stagnant)
