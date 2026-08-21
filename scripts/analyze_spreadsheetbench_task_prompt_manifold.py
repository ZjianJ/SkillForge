#!/usr/bin/env python3
"""Stage-A analysis for the SpreadsheetBench task-specific prompt manifold.

This script is deliberately CPU-only.  It audits the 61 task-specific
Combined-10% checkpoints, measures the intrinsic rank of their prompt
residuals, checks optimization-path consistency, and runs a leakage-safe
task-level cross-validation baseline that predicts prompt residuals from the
task instruction with a small TF-IDF ridge regressor.

The text baseline is a feasibility probe, not the task encoder proposed for
the final method.  Every outer fold fits its vocabulary, PCA basis, centering,
and ridge model using training tasks only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_ROOT = PROJECT_ROOT / "outputs/SpreadsheetBench_task_specific_combined_core10_len8_seed1_coverage_ablation"
DEFAULT_SHARED = PROJECT_ROOT / "outputs/SpreadsheetBench_combined_core10_coverage_shared_preserve_len8_seed1/best_prefix.pt"
DEFAULT_ITEMS = PROJECT_ROOT / "data/spreadsheetbench_split/train/items.json"
DEFAULT_OUT = PROJECT_ROOT / "outputs/SpreadsheetBench_task_prompt_manifold_stage_a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=DEFAULT_TASK_ROOT)
    parser.add_argument("--shared-checkpoint", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=24018)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=512)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--cv-components", type=int, default=16)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prefix(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        tensor = payload.get("prefix_embeddings")
        meta = {k: v for k, v in payload.items() if not torch.is_tensor(v)}
    elif torch.is_tensor(payload):
        tensor = payload
        meta = {}
    else:
        raise TypeError(f"Unsupported checkpoint payload at {path}: {type(payload)!r}")
    if not torch.is_tensor(tensor):
        raise KeyError(f"Checkpoint has no prefix_embeddings tensor: {path}")
    return tensor.detach().float().cpu().numpy(), meta


def cosine_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return numerator / np.maximum(denominator, eps)


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(rankdata(np.asarray(a)), rankdata(np.asarray(b)))


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def tokens(text: str) -> list[str]:
    words = TOKEN_PATTERN.findall(text.lower())
    bigrams = [f"{a}__{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def fit_tfidf(texts: list[str], max_features: int, min_df: int) -> tuple[list[str], np.ndarray]:
    tokenized = [tokens(text) for text in texts]
    document_frequency: Counter[str] = Counter()
    term_frequency: Counter[str] = Counter()
    for row in tokenized:
        document_frequency.update(set(row))
        term_frequency.update(row)
    candidates = [term for term, df in document_frequency.items() if df >= min_df]
    candidates.sort(key=lambda term: (-document_frequency[term], -term_frequency[term], term))
    vocabulary = candidates[:max_features]
    return vocabulary, transform_tfidf(texts, vocabulary, document_frequency, len(texts))


def transform_tfidf(
    texts: list[str],
    vocabulary: list[str],
    train_df: Counter[str] | dict[str, int],
    train_n: int,
) -> np.ndarray:
    index = {term: idx for idx, term in enumerate(vocabulary)}
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=np.float64)
    for row_idx, text in enumerate(texts):
        counts = Counter(tokens(text))
        for term, count in counts.items():
            col_idx = index.get(term)
            if col_idx is not None:
                tf = 1.0 + math.log(float(count))
                idf = math.log((1.0 + train_n) / (1.0 + train_df.get(term, 0))) + 1.0
                matrix[row_idx, col_idx] = tf * idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.maximum(norms, 1e-12)
    return matrix


def tfidf_train_test(
    train_texts: list[str],
    test_texts: list[str],
    max_features: int,
    min_df: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    tokenized = [tokens(text) for text in train_texts]
    df: Counter[str] = Counter()
    tf: Counter[str] = Counter()
    for row in tokenized:
        df.update(set(row))
        tf.update(row)
    candidates = [term for term, count in df.items() if count >= min_df]
    candidates.sort(key=lambda term: (-df[term], -tf[term], term))
    vocabulary = candidates[:max_features]
    return (
        transform_tfidf(train_texts, vocabulary, df, len(train_texts)),
        transform_tfidf(test_texts, vocabulary, df, len(train_texts)),
        vocabulary,
    )


def add_bias(features: np.ndarray) -> np.ndarray:
    return np.concatenate([features, np.ones((len(features), 1), dtype=features.dtype)], axis=1)


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    """Dual-form multi-output ridge; the final feature is an unregularized bias."""
    x_mean = np.mean(x_train[:, :-1], axis=0, keepdims=True)
    y_mean = np.mean(y_train, axis=0, keepdims=True)
    x_train_centered = x_train[:, :-1] - x_mean
    x_test_centered = x_test[:, :-1] - x_mean
    y_centered = y_train - y_mean
    kernel = x_train_centered @ x_train_centered.T
    dual = np.linalg.solve(kernel + alpha * np.eye(len(kernel)), y_centered)
    return x_test_centered @ x_train_centered.T @ dual + y_mean


def stratified_folds(strata: list[str], folds: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, stratum in enumerate(strata):
        groups[stratum].append(idx)
    result = [[] for _ in range(folds)]
    offset = 0
    for stratum in sorted(groups):
        indices = groups[stratum][:]
        rng.shuffle(indices)
        for local_idx, item_idx in enumerate(indices):
            result[(offset + local_idx) % folds].append(item_idx)
        offset = (offset + len(indices)) % folds
    for fold in result:
        fold.sort()
    return result


def choose_ridge_alpha(x: np.ndarray, y: np.ndarray, strata: list[str], seed: int) -> float:
    candidates = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    inner_folds = stratified_folds(strata, folds=4, seed=seed)
    losses: dict[float, list[float]] = {alpha: [] for alpha in candidates}
    all_indices = set(range(len(x)))
    for held_out in inner_folds:
        train = sorted(all_indices - set(held_out))
        if not train or not held_out:
            continue
        for alpha in candidates:
            prediction = ridge_predict(add_bias(x[train]), y[train], add_bias(x[held_out]), alpha)
            losses[alpha].append(float(np.mean((prediction - y[held_out]) ** 2)))
    return min(candidates, key=lambda alpha: (np.mean(losses[alpha]), alpha))


def checkpoint_matrix(paths: Iterable[Path]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    arrays: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for path in paths:
        array, meta = load_prefix(path)
        arrays.append(array.reshape(-1))
        metadata.append(meta)
    return np.stack(arrays), metadata


def main() -> None:
    args = parse_args()
    task_root = args.task_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    training_root = task_root / "training"
    checkpoint_paths = sorted(training_root.glob("*/final_prefix.pt"), key=lambda path: path.parent.name)
    if len(checkpoint_paths) != 61:
        raise RuntimeError(f"Expected 61 final checkpoints, found {len(checkpoint_paths)} under {training_root}")

    task_ids = [path.parent.name for path in checkpoint_paths]
    prompts, checkpoint_meta = checkpoint_matrix(checkpoint_paths)
    shape, _ = load_prefix(checkpoint_paths[0])
    prompt_shape = list(shape.shape)
    if any(meta.get("task_id", task_id) != task_id for task_id, meta in zip(task_ids, checkpoint_meta)):
        raise RuntimeError("At least one checkpoint task_id does not match its directory name")
    if len({tuple(load_prefix(path)[0].shape) for path in checkpoint_paths}) != 1:
        raise RuntimeError("Task-specific checkpoints do not share one shape")

    shared_prompt, _ = load_prefix(args.shared_checkpoint.resolve())
    shared = shared_prompt.reshape(-1).astype(np.float64)
    prompts = prompts.astype(np.float64)
    if prompts.shape[1] != shared.shape[0]:
        raise RuntimeError(f"Task/shared shape mismatch: {prompts.shape} vs {shared.shape}")

    items = {str(row["id"]): row for row in read_json(args.items.resolve())}
    missing_items = sorted(set(task_ids) - set(items))
    if missing_items:
        raise RuntimeError(f"Training items missing task IDs: {missing_items}")

    evaluation_rows = read_jsonl(task_root / "evaluation_results.jsonl")
    evaluations = {str(row["id"]): row for row in evaluation_rows if row.get("condition") == "soft"}
    if set(evaluations) != set(task_ids):
        raise RuntimeError("Soft evaluation IDs do not exactly match the 61 checkpoint IDs")

    summaries = {task_id: read_json(training_root / task_id / "training_summary.json") for task_id in task_ids}
    aggregate_summary = read_json(task_root / "summary.json")

    # Intrinsic prompt geometry.  PCA is mean-centered, so the selected shared
    # anchor does not affect the singular values.
    task_mean = np.mean(prompts, axis=0, keepdims=True)
    centered = prompts - task_mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values**2 / max(len(prompts) - 1, 1)
    explained = eigenvalues / np.maximum(np.sum(eigenvalues), 1e-30)
    cumulative = np.cumsum(explained)
    nonzero = explained[explained > 1e-15]
    effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
    participation_ratio = float(np.sum(eigenvalues) ** 2 / np.maximum(np.sum(eigenvalues**2), 1e-30))

    dims = [1, 2, 4, 8, 16, 32, 48, 60]
    pca_table = []
    for dim in dims:
        actual = min(dim, len(cumulative))
        pca_table.append(
            {
                "components": dim,
                "available_components": actual,
                "cumulative_explained_variance": float(cumulative[actual - 1]),
                "remaining_variance": float(1.0 - cumulative[actual - 1]),
            }
        )

    residuals = prompts - shared[None, :]
    residual_norms = np.linalg.norm(residuals, axis=1)
    centered_norms = np.linalg.norm(centered, axis=1)
    mean_residual = np.mean(residuals, axis=0)
    shared_component_ratio = float(np.linalg.norm(mean_residual) / np.mean(residual_norms))
    task_specific_ratio = float(np.sqrt(np.mean(centered_norms**2)) / np.sqrt(np.mean(residual_norms**2)))

    # Pairwise geometry versus the behavioral proxies already produced during
    # the original experiment.  This is not called a full functional-distance
    # analysis because it does not re-run Qwen logits.
    pair_prompt_distance: list[float] = []
    pair_closure_distance: list[float] = []
    pair_preservation_distance: list[float] = []
    pair_success_mismatch: list[float] = []
    closures = np.asarray([float(summaries[task_id]["core_skill_kl_closure"]) for task_id in task_ids])
    preservation = np.asarray(
        [float(summaries[task_id]["final"]["preservation_clean_kl"]) for task_id in task_ids]
    )
    successes = np.asarray([int(bool(evaluations[task_id].get("hard"))) for task_id in task_ids])
    for left in range(len(task_ids)):
        for right in range(left + 1, len(task_ids)):
            pair_prompt_distance.append(float(np.linalg.norm(centered[left] - centered[right])))
            pair_closure_distance.append(float(abs(closures[left] - closures[right])))
            pair_preservation_distance.append(float(abs(preservation[left] - preservation[right])))
            pair_success_mismatch.append(float(successes[left] != successes[right]))
    pair_prompt_distance_array = np.asarray(pair_prompt_distance)

    # Optimization-path consistency from the available 1/4/8/16/32-step
    # checkpoints.  Step 1 is used as a common per-task path origin because
    # exact step-0 tensors were not saved.
    step_values = [1, 4, 8, 16, 32]
    step_matrices: dict[int, np.ndarray] = {}
    for step in step_values:
        paths = [training_root / task_id / f"prefix_step_{step:03d}.pt" for task_id in task_ids]
        if not all(path.is_file() for path in paths):
            raise RuntimeError(f"Missing one or more step-{step} checkpoints")
        step_matrices[step], _ = checkpoint_matrix(paths)
        step_matrices[step] = step_matrices[step].astype(np.float64)
    final_move = step_matrices[32] - step_matrices[1]
    final_move_norm = np.linalg.norm(final_move, axis=1)
    path_table = []
    for step in [4, 8, 16, 32]:
        move = step_matrices[step] - step_matrices[1]
        path_table.append(
            {
                "step": step,
                "median_direction_cosine_to_step32": float(np.median(cosine_rows(move, final_move))),
                "mean_direction_cosine_to_step32": float(np.mean(cosine_rows(move, final_move))),
                "median_relative_move_norm": float(
                    np.median(np.linalg.norm(move, axis=1) / np.maximum(final_move_norm, 1e-12))
                ),
            }
        )

    # Leakage-safe outer CV.  Each fold independently fits instruction
    # vocabulary, prompt mean, PCA basis, and ridge hyperparameter.
    descriptions = [
        f"instruction_type {items[task_id].get('instruction_type', '')} "
        f"{items[task_id].get('instruction', '')}"
        for task_id in task_ids
    ]
    strata = [f"{items[task_id].get('instruction_type', '')}|success={successes[idx]}" for idx, task_id in enumerate(task_ids)]
    folds = stratified_folds(strata, folds=args.folds, seed=args.seed)
    cv_predictions: list[dict[str, Any]] = []
    predicted_residuals = np.zeros_like(prompts)
    fold_rows = []
    all_indices = set(range(len(task_ids)))
    for fold_idx, test_indices in enumerate(folds):
        train_indices = sorted(all_indices - set(test_indices))
        train_texts = [descriptions[idx] for idx in train_indices]
        test_texts = [descriptions[idx] for idx in test_indices]
        x_train, x_test, vocabulary = tfidf_train_test(
            train_texts, test_texts, max_features=args.max_features, min_df=args.min_df
        )

        prompt_train = prompts[train_indices]
        fold_mean = np.mean(prompt_train, axis=0, keepdims=True)
        fold_centered = prompt_train - fold_mean
        _, _, fold_vt = np.linalg.svd(fold_centered, full_matrices=False)
        component_count = min(args.cv_components, len(train_indices) - 1)
        components = fold_vt[:component_count]
        y_train = fold_centered @ components.T
        alpha = choose_ridge_alpha(
            x_train,
            y_train,
            [strata[idx] for idx in train_indices],
            seed=args.seed + fold_idx + 1,
        )
        y_prediction = ridge_predict(add_bias(x_train), y_train, add_bias(x_test), alpha)
        prompt_prediction = fold_mean + y_prediction @ components
        predicted_residuals[test_indices] = prompt_prediction

        true_residual = prompts[test_indices] - fold_mean
        predicted_from_mean = prompt_prediction - fold_mean
        fold_cosines = cosine_rows(predicted_from_mean, true_residual)
        fold_sse = float(np.sum((prompt_prediction - prompts[test_indices]) ** 2))
        fold_sst = float(np.sum(true_residual**2))
        fold_rows.append(
            {
                "fold": fold_idx,
                "train_tasks": len(train_indices),
                "test_tasks": len(test_indices),
                "features": len(vocabulary),
                "components": component_count,
                "ridge_alpha": alpha,
                "residual_reconstruction_r2": 1.0 - fold_sse / max(fold_sst, 1e-30),
                "mean_residual_cosine": float(np.mean(fold_cosines)),
                "median_residual_cosine": float(np.median(fold_cosines)),
            }
        )
        for local_idx, task_idx in enumerate(test_indices):
            cv_predictions.append(
                {
                    "id": task_ids[task_idx],
                    "fold": fold_idx,
                    "success": int(successes[task_idx]),
                    "residual_cosine": float(fold_cosines[local_idx]),
                    "squared_error": float(np.sum((prompt_prediction[local_idx] - prompts[task_idx]) ** 2)),
                    "mean_baseline_squared_error": float(np.sum(true_residual[local_idx] ** 2)),
                }
            )

    cv_sse = float(sum(row["squared_error"] for row in cv_predictions))
    cv_sst = float(sum(row["mean_baseline_squared_error"] for row in cv_predictions))
    cv_cosines = np.asarray([row["residual_cosine"] for row in cv_predictions])
    cv_summary = {
        "protocol": f"{args.folds}-fold task-level CV; fold-local TF-IDF, PCA, centering and nested ridge alpha",
        "components": args.cv_components,
        "residual_reconstruction_r2": 1.0 - cv_sse / max(cv_sst, 1e-30),
        "mean_residual_cosine": float(np.mean(cv_cosines)),
        "median_residual_cosine": float(np.median(cv_cosines)),
        "positive_cosine_tasks": int(np.sum(cv_cosines > 0.0)),
        "folds": fold_rows,
    }

    task_rows = []
    for idx, task_id in enumerate(task_ids):
        task_rows.append(
            {
                "id": task_id,
                "instruction_type": items[task_id].get("instruction_type", ""),
                "success": int(successes[idx]),
                "core_skill_kl_closure": float(closures[idx]),
                "preservation_clean_kl": float(preservation[idx]),
                "shared_residual_norm": float(residual_norms[idx]),
                "centered_prompt_norm": float(centered_norms[idx]),
            }
        )

    pca16 = float(cumulative[min(16, len(cumulative)) - 1])
    # A merely positive floating-point R2 is not meaningful.  Require both a
    # non-trivial reconstruction gain and directional agreement before calling
    # the representation predictive.  Raw metrics remain reported regardless.
    text_predictive = bool(
        cv_summary["residual_reconstruction_r2"] >= 0.01
        and cv_summary["mean_residual_cosine"] >= 0.05
    )
    decision = {
        "low_dimensional_structure_supported": bool(pca16 >= 0.75),
        "text_only_predictability_supported": text_predictive,
        "text_predictability_gate": "residual R2 >= 0.01 and mean residual cosine >= 0.05",
        "proceed_to_richer_task_representations": True,
        "reason": (
            "PCA-16 captures at least 75% variance; screen richer representations next."
            if pca16 >= 0.75
            else "PCA-16 captures less than 75%; retain retrieval/full-rank controls while screening richer representations."
        ),
    }

    summary = {
        "stage": "A_task_prompt_manifold",
        "scope": "61 successful Train80 trajectories only; no Val40/Test280 access",
        "task_root": str(task_root),
        "tasks": len(task_ids),
        "prompt_shape": prompt_shape,
        "prompt_parameters": int(prompts.shape[1]),
        "successes": int(np.sum(successes)),
        "shared_checkpoint": str(args.shared_checkpoint.resolve()),
        "shared_checkpoint_sha256": sha256_file(args.shared_checkpoint.resolve()),
        "aggregate_summary_sha256": sha256_file(task_root / "summary.json"),
        "geometry": {
            "effective_rank": effective_rank,
            "participation_ratio": participation_ratio,
            "shared_component_ratio": shared_component_ratio,
            "task_specific_rms_ratio": task_specific_ratio,
            "mean_shared_residual_norm": float(np.mean(residual_norms)),
            "median_shared_residual_norm": float(np.median(residual_norms)),
            "pca": pca_table,
            "per_prompt_token_variance_fraction": (
                np.sum(centered.reshape(len(task_ids), prompt_shape[0], prompt_shape[1]) ** 2, axis=(0, 2))
                / np.sum(centered**2)
            ).tolist(),
        },
        "optimization_path": path_table,
        "behavior_proxy_alignment": {
            "warning": "Uses scalar training closure/preservation and binary execution, not full Qwen distribution distances.",
            "prompt_distance_vs_abs_closure_difference_spearman": spearman(
                pair_prompt_distance_array, np.asarray(pair_closure_distance)
            ),
            "prompt_distance_vs_abs_preservation_difference_spearman": spearman(
                pair_prompt_distance_array, np.asarray(pair_preservation_distance)
            ),
            "prompt_distance_vs_success_mismatch_point_biserial": pearson(
                pair_prompt_distance_array, np.asarray(pair_success_mismatch)
            ),
            "residual_norm_vs_closure_spearman": spearman(residual_norms, closures),
            "residual_norm_vs_success_point_biserial": pearson(residual_norms, successes),
        },
        "text_only_cv": cv_summary,
        "decision": decision,
        "source_summary": aggregate_summary,
    }

    with (out_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (out_root / "tasks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)
    with (out_root / "pca.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pca_table[0]))
        writer.writeheader()
        writer.writerows(pca_table)
    with (out_root / "cv_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(cv_predictions, key=lambda value: value["id"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    np.savez_compressed(
        out_root / "prompt_manifold.npz",
        task_ids=np.asarray(task_ids),
        task_mean=task_mean.astype(np.float32),
        singular_values=singular_values.astype(np.float32),
        components=vt[: min(32, len(vt))].astype(np.float32),
        explained_variance_ratio=explained.astype(np.float32),
    )

    report = [
        "# SpreadsheetBench task-prompt manifold: Stage A",
        "",
        f"- Scope: {len(task_ids)} Train80 successful-trajectory tasks; Val40/Test280 untouched.",
        f"- Prompt: shape {prompt_shape}, {prompts.shape[1]:,} parameters, shared length-8 Combined-10% anchor.",
        f"- Existing same-task free-generation success: {int(np.sum(successes))}/{len(successes)} ({np.mean(successes):.2%}).",
        f"- Effective rank: {effective_rank:.2f}; participation ratio: {participation_ratio:.2f}.",
        f"- PCA-8/PCA-16/PCA-32 cumulative variance: {cumulative[7]:.2%} / {cumulative[15]:.2%} / {cumulative[31]:.2%}.",
        f"- Task-specific RMS / shared-residual RMS: {task_specific_ratio:.2%}.",
        f"- Text-only {args.folds}-fold CV reconstruction R2: {cv_summary['residual_reconstruction_r2']:.4f}.",
        f"- Text-only CV residual cosine mean/median: {cv_summary['mean_residual_cosine']:.4f} / {cv_summary['median_residual_cosine']:.4f}.",
        f"- Decision: {decision['reason']}",
        "",
        "The behavior-alignment correlations use existing scalar diagnostics only. A later GPU experiment is required for full-distribution functional distances.",
    ]
    (out_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({"out_root": str(out_root), "decision": decision, "pca16": pca16, "text_cv": cv_summary}, indent=2))


if __name__ == "__main__":
    main()
