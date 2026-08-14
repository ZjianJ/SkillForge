"""Pure helpers for pairwise residual Combined boosting of soft prefixes."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def topk_residual_js(
    torch: Any,
    student_logits: Any,
    *,
    reference_topk_ids: Any,
    reference_topk_logp: Any,
    reference_residual_log_mass: Any,
) -> Any:
    """JS(reference || student) over reference Top-k IDs plus a residual bucket."""
    student_logp = torch.log_softmax(student_logits.float(), dim=-1)
    ids = reference_topk_ids.to(device=student_logits.device, dtype=torch.long)
    reference_top = reference_topk_logp.to(student_logits.device).float()
    reference_residual = reference_residual_log_mass.to(student_logits.device).float()
    student_top = student_logp.gather(-1, ids)
    student_top_mass = student_top.exp().sum(dim=-1).clamp(max=1.0 - 1e-7)
    student_residual = torch.log1p(-student_top_mass)

    log_two = math.log(2.0)
    top_mix = torch.logaddexp(reference_top, student_top) - log_two
    residual_mix = torch.logaddexp(reference_residual, student_residual) - log_two
    reference_kl = (
        reference_top.exp() * (reference_top - top_mix)
    ).sum(dim=-1) + reference_residual.exp() * (reference_residual - residual_mix)
    student_kl = (
        student_top.exp() * (student_top - top_mix)
    ).sum(dim=-1) + student_residual.exp() * (student_residual - residual_mix)
    return 0.5 * (reference_kl + student_kl)


def topk_residual_forward_kl(
    torch: Any,
    student_logits: Any,
    *,
    reference_topk_ids: Any,
    reference_topk_logp: Any,
    reference_residual_log_mass: Any,
) -> Any:
    """KL(reference || student) over reference Top-k IDs plus residual."""
    student_logp = torch.log_softmax(student_logits.float(), dim=-1)
    ids = reference_topk_ids.to(device=student_logits.device, dtype=torch.long)
    reference_top = reference_topk_logp.to(student_logits.device).float()
    reference_residual = reference_residual_log_mass.to(student_logits.device).float()
    student_top = student_logp.gather(-1, ids)
    student_top_mass = student_top.exp().sum(dim=-1).clamp(max=1.0 - 1e-7)
    student_residual = torch.log1p(-student_top_mass)
    return (
        reference_top.exp() * (reference_top - student_top)
    ).sum(dim=-1) + reference_residual.exp() * (
        reference_residual - student_residual
    )


def residual_combined_scores(
    *,
    teacher_target_logp: np.ndarray,
    student_target_logp: np.ndarray,
    teacher_beneficial: np.ndarray,
    teacher_student_js: np.ndarray,
) -> np.ndarray:
    """Score Skill-beneficial positions not yet matched by the current prefix."""
    residual_gain = np.maximum(
        np.asarray(teacher_target_logp, dtype=np.float64)
        - np.asarray(student_target_logp, dtype=np.float64),
        0.0,
    )
    scores = (
        np.asarray(teacher_beneficial, dtype=bool)
        * residual_gain
        * np.maximum(np.asarray(teacher_student_js, dtype=np.float64), 0.0)
    )
    return scores.astype(np.float32)


def reference_decision_margin(
    *,
    target_ids: np.ndarray,
    target_logp: np.ndarray,
    topk_ids: np.ndarray,
    topk_logp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gold-vs-best-alternative margins and whether gold is Top-1.

    The target log-probability is taken from the exact cached target score.  The
    strongest non-target alternative is exact as well because the maximum-logit
    token is necessarily present in the cached Top-k list (k=64 in PRCB).
    """
    targets = np.asarray(target_ids, dtype=np.int64)
    ids = np.asarray(topk_ids, dtype=np.int64)
    logp = np.asarray(topk_logp, dtype=np.float64)
    if ids.ndim != 2 or logp.shape != ids.shape or len(targets) != len(ids):
        raise ValueError("Incompatible target and Top-k reference shapes")
    alternatives = np.where(ids != targets[:, None], logp, -np.inf)
    competitor = alternatives.max(axis=1)
    if not np.isfinite(competitor).all():
        raise ValueError("Every reference row must contain a non-target alternative")
    margin = np.asarray(target_logp, dtype=np.float64) - competitor
    return margin.astype(np.float32), (ids[:, 0] == targets)


def margin_decision_locator(
    *,
    skill_margin: np.ndarray,
    clean_margin: np.ndarray,
    current_margin: np.ndarray,
    skill_top1_gold: np.ndarray,
    clean_top1_gold: np.ndarray,
    current_top1_gold: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the PRCB-v3 hierarchy on one fixed successful trajectory.

    Tier 2 is an unresolved decision flip: hard Skill makes gold Top-1 while
    both no-Skill and the current prefix do not.  Tier 1 contains the remaining
    positions with positive original Skill margin gain and positive current
    margin residual.  Tier 0 is not selectable.
    """
    skill = np.asarray(skill_margin, dtype=np.float64)
    clean = np.asarray(clean_margin, dtype=np.float64)
    current = np.asarray(current_margin, dtype=np.float64)
    original_gain = np.maximum(skill - clean, 0.0)
    residual = np.maximum(skill - current, 0.0)
    eligible = (original_gain > 0) & (residual > 0)
    decisive = (
        np.asarray(skill_top1_gold, dtype=bool)
        & ~np.asarray(clean_top1_gold, dtype=bool)
        & ~np.asarray(current_top1_gold, dtype=bool)
        & eligible
    )
    priority = eligible.astype(np.int8)
    priority[decisive] = 2
    return {
        "margin_priority": priority,
        "original_margin_gain": original_gain.astype(np.float32),
        "margin_residual": residual.astype(np.float32),
        "margin_residual_mass": (original_gain * residual).astype(np.float32),
    }


def select_margin_decision_top_fraction(
    *,
    priority: np.ndarray,
    residual: np.ndarray,
    original_gain: np.ndarray,
    teacher_student_js: np.ndarray,
    ratio: float,
) -> list[int]:
    """Select PRCB-v3 positions lexicographically; JS is only a tie-breaker."""
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between zero and one")
    tiers = np.asarray(priority, dtype=np.int8)
    residual_values = np.asarray(residual, dtype=np.float64)
    gain_values = np.asarray(original_gain, dtype=np.float64)
    js_values = np.asarray(teacher_student_js, dtype=np.float64)
    if not (len(tiers) == len(residual_values) == len(gain_values) == len(js_values)):
        raise ValueError("PRCB-v3 locator arrays must have equal length")
    count = max(1, math.ceil(len(tiers) * ratio))
    candidates = [index for index in range(len(tiers)) if int(tiers[index]) > 0]
    candidates.sort(
        key=lambda index: (
            -int(tiers[index]),
            -float(residual_values[index]),
            -float(gain_values[index]),
            -float(js_values[index]),
            index,
        )
    )
    return sorted(candidates[:count])


def select_positive_top_fraction(scores: np.ndarray, ratio: float) -> list[int]:
    """Select a stable Top-ratio set, excluding non-positive score entries."""
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between zero and one")
    values = np.asarray(scores)
    count = max(1, math.ceil(len(values) * ratio))
    order = np.argsort(-values, kind="stable")
    return sorted(int(index) for index in order[:count] if values[index] > 0)


def select_harmful_anchor_positions(
    harm_scores: np.ndarray,
    *,
    count: int,
    excluded: set[int],
    teacher_beneficial: np.ndarray,
) -> list[int]:
    """Choose the most base-disruptive non-beneficial positions automatically."""
    values = np.asarray(harm_scores)
    beneficial = np.asarray(teacher_beneficial, dtype=bool)
    preferred = [
        index
        for index in range(len(values))
        if index not in excluded and not beneficial[index]
    ]
    fallback = [
        index
        for index in range(len(values))
        if index not in excluded and beneficial[index]
    ]
    preferred.sort(key=lambda index: (-float(values[index]), index))
    fallback.sort(key=lambda index: (-float(values[index]), index))
    return sorted((preferred + fallback)[:count])


def choose_prefix_pair(gradient: Any, prefix: Any) -> tuple[list[int], list[float]]:
    """Choose two rows with largest normalized gradient norm."""
    if gradient.ndim != 2 or prefix.ndim != 2 or gradient.shape != prefix.shape:
        raise ValueError("gradient and prefix must have the same rank-2 shape")
    scores = gradient.float().norm(dim=1) / prefix.detach().float().norm(dim=1).clamp_min(1e-12)
    pair = scores.argsort(descending=True, stable=True)[:2].tolist()
    return sorted(int(index) for index in pair), [float(value) for value in scores.detach().cpu().tolist()]


def causal_prefix_pair(
    *,
    prefix_length: int,
    round_index: int,
    direction: str,
) -> list[int]:
    """Return a fixed adjacent pair in head-to-tail or tail-to-head order."""
    if prefix_length <= 0 or prefix_length % 2:
        raise ValueError("prefix_length must be a positive even integer")
    pair_count = prefix_length // 2
    if not 1 <= round_index <= pair_count:
        raise ValueError(
            f"round_index must be in [1, {pair_count}], got {round_index}"
        )
    normalized = str(direction).strip().lower()
    if normalized == "head_to_tail":
        start = 2 * (round_index - 1)
    elif normalized == "tail_to_head":
        start = prefix_length - 2 * round_index
    else:
        raise ValueError("direction must be head_to_tail or tail_to_head")
    return [start, start + 1]


def sliding_prefix_pair(
    *,
    prefix_length: int,
    round_index: int,
    window_size: int = 2,
) -> list[int]:
    """Return rows for a stride-one head-to-tail sliding window.

    ``window_size=2`` preserves the original overlapping-pair behavior.  A
    larger value trains the contiguous prefix rows beginning at
    ``round_index - 1`` while leaving every other row bit-identical.
    """
    if not 2 <= window_size <= prefix_length:
        raise ValueError("window_size must be between two and prefix_length")
    window_count = prefix_length - window_size + 1
    if not 1 <= round_index <= window_count:
        raise ValueError(
            f"round_index must be in [1, {window_count}], got {round_index}"
        )
    start = round_index - 1
    return list(range(start, start + window_size))


def mask_prefix_gradient_(gradient: Any, active_rows: list[int]) -> None:
    """Zero every prefix-row gradient except the active window in place."""
    active = set(int(index) for index in active_rows)
    for index in range(int(gradient.shape[0])):
        if index not in active:
            gradient[index].zero_()


def shrink_pair_update_(
    torch: Any,
    prefix: Any,
    *,
    round_start: Any,
    active_rows: list[int],
    shrinkage: float,
) -> None:
    """Merge a shrunken window update and restore all frozen rows exactly."""
    active = set(int(index) for index in active_rows)
    with torch.no_grad():
        for index in range(int(prefix.shape[0])):
            if index in active:
                delta = prefix[index] - round_start[index]
                prefix[index].copy_(round_start[index] + float(shrinkage) * delta)
            else:
                prefix[index].copy_(round_start[index])
