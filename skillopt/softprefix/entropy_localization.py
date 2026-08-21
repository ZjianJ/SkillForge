"""Pure helpers for entropy-aware Skill-effect localization."""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def minmax_normalize(values: np.ndarray, eligible: Iterable[int] | None = None) -> np.ndarray:
    """Min-max normalize a score in its localization scope to ``[0, 1]``."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    result = np.zeros_like(array)
    indices = np.arange(len(array)) if eligible is None else np.asarray(list(eligible), dtype=np.int64)
    if len(indices) == 0:
        return result.astype(np.float32)
    scoped = array[indices]
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


def entropy_augmented_scores(
    positive_gain: np.ndarray,
    js: np.ndarray,
    base_entropy: np.ndarray,
    *,
    alpha: float = 0.5,
    entropy_lambda: float = 0.5,
    eligible: Iterable[int] | None = None,
) -> dict[str, np.ndarray]:
    """Return normalized Skill relevance and entropy-amplified relevance.

    ``S = alpha * G~ + (1-alpha) * D~`` and
    ``EAC = S * (1 + lambda * H~)``. Normalization is local to the same
    trajectory/eligible scope used for Top-k localization.
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if float(entropy_lambda) < 0.0:
        raise ValueError("entropy_lambda must be non-negative")
    gain = minmax_normalize(positive_gain, eligible)
    divergence = minmax_normalize(js, eligible)
    entropy = minmax_normalize(base_entropy, eligible)
    skill = float(alpha) * gain + (1.0 - float(alpha)) * divergence
    augmented = skill * (1.0 + float(entropy_lambda) * entropy)
    return {
        "normalized_gain": gain,
        "normalized_js": divergence,
        "normalized_entropy": entropy,
        "skill_relevance": skill.astype(np.float32),
        "entropy_augmented": augmented.astype(np.float32),
    }


def select_top_fraction(
    scores: np.ndarray,
    *,
    ratio: float,
    forbidden: Iterable[int] = (),
) -> list[int]:
    """Select a stable fixed-budget Top-ratio set, excluding forbidden indices."""
    if not 0.0 < float(ratio) < 1.0:
        raise ValueError("ratio must be strictly between zero and one")
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    requested = max(1, math.ceil(len(values) * float(ratio)))
    blocked = {int(index) for index in forbidden}
    candidates = [
        index for index, value in enumerate(values)
        if index not in blocked and np.isfinite(value)
    ]
    if len(candidates) < requested:
        raise ValueError(f"Only {len(candidates)} eligible tokens for a budget of {requested}")
    candidates.sort(key=lambda index: (-float(values[index]), index))
    return sorted(candidates[:requested])


def jaccard(first: Iterable[int], second: Iterable[int]) -> float:
    left, right = set(map(int, first)), set(map(int, second))
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)
