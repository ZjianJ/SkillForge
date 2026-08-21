#!/usr/bin/env python3
"""Analyze entropy/Skill effects and prepare controlled Top-5% manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skillopt.softprefix.entropy_localization import (
    entropy_augmented_scores,
    jaccard,
    select_top_fraction,
)


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _seed_for(identifier: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _describe(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()) if len(array) else float("nan"),
        "median": float(np.median(array)) if len(array) else float("nan"),
    }


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "pearson_r": float(pearsonr(x, y).statistic),
        "pearson_p": float(pearsonr(x, y).pvalue),
        "spearman_rho": float(spearmanr(x, y).statistic),
        "spearman_p": float(spearmanr(x, y).pvalue),
    }


def _correlation_distribution(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    result = {}
    for key in ("pearson_r", "spearman_rho"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        result[key] = {
            "trajectories": int(len(values)),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        }
    return result


def _write_plot(out_root: Path, entropy: np.ndarray, gain: np.ndarray, js: np.ndarray) -> None:
    """Write a dependency-free sampled scatter plot as SVG."""
    width, height = 1200, 500
    margin, panel_width, panel_height = 65, 500, 370
    rng = np.random.default_rng(1)
    count = min(12000, len(entropy))
    chosen = np.sort(rng.choice(len(entropy), size=count, replace=False))
    x_values = entropy[chosen]
    x_low, x_high = float(entropy.min()), float(entropy.max())

    def scale(values: np.ndarray, low: float, high: float, start: float, span: float) -> np.ndarray:
        return start + (values - low) / max(high - low, 1e-12) * span

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="600" y="28" text-anchor="middle" font-size="18">Entropy versus Skill effect (12k-token deterministic sample)</text>',
    ]
    for panel, (values, label) in enumerate(((gain, "Positive Skill gain"), (js, "Full-vocabulary JS"))):
        left = margin + panel * 585
        transformed_all = np.log1p(np.maximum(values, 0.0))
        transformed = transformed_all[chosen]
        y_low, y_high = 0.0, float(transformed_all.max())
        xs = scale(x_values, x_low, x_high, left, panel_width)
        ys = height - 65 - scale(transformed, y_low, y_high, 0, panel_height)
        parts += [
            f'<line x1="{left}" y1="{height-65}" x2="{left+panel_width}" y2="{height-65}" stroke="black"/>',
            f'<line x1="{left}" y1="{height-65-panel_height}" x2="{left}" y2="{height-65}" stroke="black"/>',
            f'<text x="{left+panel_width/2}" y="{height-25}" text-anchor="middle">Base entropy (nats)</text>',
            f'<text x="{left+8}" y="{height-65-panel_height-12}">log(1 + {label})</text>',
        ]
        parts.extend(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.1" fill="#246b8e" fill-opacity="0.16"/>'
            for x, y in zip(xs, ys, strict=True)
        )
    parts.append("</svg>")
    (out_root / "entropy_vs_skill_effect.svg").write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", default="outputs/SpreadsheetBench_entropy_localization_stage1")
    parser.add_argument(
        "--source-manifest",
        default=(
            "outputs/SpreadsheetBench_selective_stage2_manifests/"
            "combined_top0.05_core_shared_preserve.jsonl"
        ),
    )
    parser.add_argument("--out-root", default="outputs/SpreadsheetBench_entropy_localization")
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage1_root = _resolve(args.stage1_root)
    out_root = _resolve(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    source_rows = _read_jsonl(_resolve(args.source_manifest))
    fragments = {}
    for path in sorted((stage1_root / "fragments").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        fragments[str(row["id"])] = row
    if len(source_rows) != len(fragments):
        raise ValueError(f"Source/cache trajectory mismatch: {len(source_rows)} != {len(fragments)}")

    method_names = ["random", "entropy", "positive_gain", "js", "combined_legacy", "skill_additive"]
    method_names += [f"eac_lambda{value:g}" for value in args.lambdas]
    manifests: dict[str, list[dict[str, Any]]] = {name: [] for name in method_names}
    selected_by_method: dict[str, list[set[int]]] = {name: [] for name in method_names}
    entropy_parts: list[np.ndarray] = []
    gain_parts: list[np.ndarray] = []
    js_parts: list[np.ndarray] = []
    delta_parts: list[np.ndarray] = []
    selection_stats: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"entropy_selected": [], "entropy_remaining": [], "delta_selected": [], "delta_remaining": []}
        for name in ("positive_gain", "js", "combined_legacy", "entropy")
    }
    quadrants = {
        "low_entropy_low_skill": 0,
        "low_entropy_high_skill": 0,
        "high_entropy_low_skill": 0,
        "high_entropy_high_skill": 0,
    }
    total_tokens = 0
    per_trajectory: list[dict[str, Any]] = []
    fresh_legacy_jaccards: list[float] = []
    per_trajectory_correlations = {"base_entropy_vs_positive_gain": [], "base_entropy_vs_js": []}

    for row in source_rows:
        identifier = str(row["id"])
        fragment = fragments.get(identifier)
        if fragment is None:
            raise ValueError(f"Missing entropy cache for {identifier}")
        cache_path = Path(fragment["cache_path"])
        selectable = int(fragment["selectable_count"])
        preserve = sorted({int(index) for index in row.get("preserve_indices", [])})
        eligible = sorted(set(range(selectable)) - set(preserve))
        requested = max(1, math.ceil(selectable * args.ratio))
        if len(preserve) != requested:
            raise ValueError(f"{identifier}: preservation budget {len(preserve)} != core budget {requested}")
        with np.load(cache_path) as cached:
            required = {"target_ids", "positive_gain", "js", "combined", "clean_entropy", "skill_entropy", "entropy_reduction"}
            missing = required - set(cached.files)
            if missing:
                raise ValueError(f"{cache_path} lacks {sorted(missing)}; rerun Stage 1 with --force")
            entropy = cached["clean_entropy"][:selectable].astype(np.float64)
            gain = cached["positive_gain"][:selectable].astype(np.float64)
            divergence = cached["js"][:selectable].astype(np.float64)
            legacy = cached["combined"][:selectable].astype(np.float64)
            delta = cached["entropy_reduction"][:selectable].astype(np.float64)
            target_ids = cached["target_ids"][:selectable].astype(np.int32)

        normalized = entropy_augmented_scores(
            gain,
            divergence,
            entropy,
            alpha=args.alpha,
            entropy_lambda=0.0,
            eligible=eligible,
        )
        scores: dict[str, np.ndarray] = {
            "entropy": entropy,
            "positive_gain": gain,
            "js": divergence,
            "combined_legacy": legacy,
            "skill_additive": normalized["skill_relevance"],
        }
        for value in args.lambdas:
            scores[f"eac_lambda{value:g}"] = entropy_augmented_scores(
                gain,
                divergence,
                entropy,
                alpha=args.alpha,
                entropy_lambda=value,
                eligible=eligible,
            )["entropy_augmented"]
        rng = np.random.default_rng(_seed_for(identifier, args.seed))
        random_scores = np.zeros(selectable, dtype=np.float64)
        random_scores[np.asarray(eligible, dtype=np.int64)] = rng.random(len(eligible))
        scores["random"] = random_scores

        selections = {
            name: select_top_fraction(score, ratio=args.ratio, forbidden=preserve)
            for name, score in scores.items()
        }
        source_selected = sorted(map(int, row.get("selected_indices", [])))
        fresh_legacy = selections["combined_legacy"]
        fresh_legacy_jaccards.append(jaccard(fresh_legacy, source_selected))
        # Keep the already evaluated current Combined baseline frozen. BF16
        # recomputation can reorder nearly tied scores; replacing the source
        # indices would turn this into a subtly different baseline.
        selections["combined_legacy"] = source_selected
        legacy_matches_source = fresh_legacy == source_selected

        derived_path = out_root / "token_scores" / f"{identifier}.npz"
        _atomic_npz(
            derived_path,
            target_ids=target_ids,
            base_entropy=entropy.astype(np.float32),
            skill_entropy=(entropy - delta).astype(np.float32),
            entropy_reduction=delta.astype(np.float32),
            positive_gain=gain.astype(np.float32),
            full_vocab_js=divergence.astype(np.float32),
            combined_legacy=legacy.astype(np.float32),
            normalized_gain=normalized["normalized_gain"],
            normalized_js=normalized["normalized_js"],
            normalized_entropy=normalized["normalized_entropy"],
            skill_additive=normalized["skill_relevance"],
            **{name: scores[name].astype(np.float32) for name in scores if name.startswith("eac_")},
        )

        for name in method_names:
            selected = selections[name]
            selected_by_method[name].append(set(selected))
            manifests[name].append(
                {
                    "id": identifier,
                    "messages": row["messages"],
                    "target": row["target"],
                    # Keep the frozen no-Skill Top-64 preservation teacher used
                    # by the original Combined run; entropy scalars live in a
                    # separate cache and must not perturb the loss reference.
                    "score_cache": str(Path(row["score_cache"]).resolve()),
                    "stage2_selector": f"entropy_experiment_{name}_top{args.ratio:g}",
                    "selected_indices": selected,
                    "preserve_indices": preserve,
                    "token_score_cache": str(derived_path.resolve()),
                }
            )

        all_indices = np.arange(selectable)
        for name in selection_stats:
            chosen = np.asarray(selections[name], dtype=np.int64)
            remainder = np.setdiff1d(all_indices, chosen, assume_unique=True)
            selection_stats[name]["entropy_selected"].append(entropy[chosen])
            selection_stats[name]["entropy_remaining"].append(entropy[remainder])
            selection_stats[name]["delta_selected"].append(delta[chosen])
            selection_stats[name]["delta_remaining"].append(delta[remainder])

        high_skill = set(selections["combined_legacy"])
        entropy_median = float(np.median(entropy))
        for index in range(selectable):
            entropy_side = "high_entropy" if entropy[index] >= entropy_median else "low_entropy"
            skill_side = "high_skill" if index in high_skill else "low_skill"
            quadrants[f"{entropy_side}_{skill_side}"] += 1
        total_tokens += selectable
        per_trajectory_correlations["base_entropy_vs_positive_gain"].append(
            _correlation(entropy, gain)
        )
        per_trajectory_correlations["base_entropy_vs_js"].append(
            _correlation(entropy, divergence)
        )
        entropy_parts.append(entropy)
        gain_parts.append(gain)
        js_parts.append(divergence)
        delta_parts.append(delta)
        per_trajectory.append(
            {
                "id": identifier,
                "selectable_tokens": selectable,
                "core_tokens": requested,
                "preservation_tokens": len(preserve),
                "base_entropy_mean": float(entropy.mean()),
                "entropy_reduction_mean": float(delta.mean()),
                "legacy_source_reproduced": legacy_matches_source,
            }
        )

    manifest_dir = out_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in manifests.items():
        path = manifest_dir / f"{name}_top{args.ratio:g}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    entropy = np.concatenate(entropy_parts)
    gain = np.concatenate(gain_parts)
    divergence = np.concatenate(js_parts)
    delta = np.concatenate(delta_parts)
    methods_for_overlap = ["entropy", "positive_gain", "js", "combined_legacy"]
    overlap: dict[str, dict[str, float]] = {}
    for left_index, left in enumerate(methods_for_overlap):
        for right in methods_for_overlap[left_index + 1 :]:
            per_jaccard = [
                jaccard(a, b)
                for a, b in zip(selected_by_method[left], selected_by_method[right], strict=True)
            ]
            intersection = sum(
                len(a & b)
                for a, b in zip(selected_by_method[left], selected_by_method[right], strict=True)
            )
            union = sum(
                len(a | b)
                for a, b in zip(selected_by_method[left], selected_by_method[right], strict=True)
            )
            overlap[f"{left}__{right}"] = {
                "mean_trajectory_jaccard": float(np.mean(per_jaccard)),
                "median_trajectory_jaccard": float(np.median(per_jaccard)),
                "pooled_jaccard": intersection / union,
            }

    distributions = {}
    for name, fields in selection_stats.items():
        distributions[name] = {
            key: _describe(np.concatenate(parts)) for key, parts in fields.items()
        }
    quadrant_report = {
        key: {"tokens": value, "fraction": value / total_tokens}
        for key, value in quadrants.items()
    }
    summary = {
        "scope": "61 successful GPT-5.5 trajectories; Qwen teacher forcing; EOS excluded",
        "trajectories": len(source_rows),
        "selectable_tokens": total_tokens,
        "ratio": args.ratio,
        "alpha": args.alpha,
        "lambdas": args.lambdas,
        "normalization": "per-trajectory min-max over non-preservation eligible positions",
        "formulas": {
            "base_entropy": "-sum_v p0(v) log p0(v)",
            "positive_gain": "max(log pS(y)-log p0(y), 0)",
            "skill_shift": "JS(pS,p0), exact full vocabulary",
            "entropy_reduction": "H(p0)-H(pS)",
            "legacy_combined": "positive_gain * JS",
            "skill_additive": "alpha*G_tilde + (1-alpha)*JS_tilde",
            "EAC": "skill_additive * (1 + lambda*H_tilde)",
        },
        "correlations": {
            "pooled_tokens": {
                "base_entropy_vs_positive_gain": _correlation(entropy, gain),
                "base_entropy_vs_js": _correlation(entropy, divergence),
            },
            "per_trajectory_distribution": {
                name: _correlation_distribution(rows)
                for name, rows in per_trajectory_correlations.items()
            },
        },
        "selection_distributions": distributions,
        "overlap": overlap,
        "quadrants": quadrant_report,
        "quadrant_thresholds": {
            "high_skill_effect": "per-trajectory Legacy Combined Top-5%",
            "high_entropy": "at or above the per-trajectory base-entropy median",
        },
        "global": {
            "base_entropy": _describe(entropy),
            "skill_entropy": _describe(entropy - delta),
            "entropy_reduction": _describe(delta),
            "fraction_skill_reduces_entropy": float(np.mean(delta > 0)),
        },
        "legacy_combined_source_exact_matches": sum(row["legacy_source_reproduced"] for row in per_trajectory),
        "fresh_vs_frozen_legacy_combined_mean_jaccard": float(np.mean(fresh_legacy_jaccards)),
        "legacy_combined_baseline_policy": "use frozen original selected_indices; do not replace with BF16 recomputation",
        "test_split_accessed": False,
    }
    _atomic_json(out_root / "offline_summary.json", summary)
    _atomic_json(out_root / "per_trajectory.json", per_trajectory)
    with (out_root / "overlap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair", "mean_trajectory_jaccard", "median_trajectory_jaccard", "pooled_jaccard"])
        writer.writeheader()
        for pair, values in overlap.items():
            writer.writerow({"pair": pair, **values})
    _write_plot(out_root, entropy, gain, divergence)

    report = [
        "# SpreadsheetBench Entropy Localization — Offline Analysis",
        "",
        f"- Trajectories: {len(source_rows)}; selectable tokens: {total_tokens}",
        f"- Pooled H(base) vs Positive Gain: Pearson {summary['correlations']['pooled_tokens']['base_entropy_vs_positive_gain']['pearson_r']:.4f}; Spearman {summary['correlations']['pooled_tokens']['base_entropy_vs_positive_gain']['spearman_rho']:.4f}",
        f"- Pooled H(base) vs JS: Pearson {summary['correlations']['pooled_tokens']['base_entropy_vs_js']['pearson_r']:.4f}; Spearman {summary['correlations']['pooled_tokens']['base_entropy_vs_js']['spearman_rho']:.4f}",
        f"- Skill reduces entropy on {summary['global']['fraction_skill_reduces_entropy']:.2%} of tokens",
        f"- High entropy + high Skill effect: {quadrant_report['high_entropy_high_skill']['fraction']:.2%}",
        f"- Low entropy + high Skill effect: {quadrant_report['low_entropy_high_skill']['fraction']:.2%}",
        "",
        "Free-generation claims are intentionally deferred until matched Val40 evaluation.",
    ]
    (out_root / "OFFLINE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
