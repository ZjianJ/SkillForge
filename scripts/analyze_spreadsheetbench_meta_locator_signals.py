#!/usr/bin/env python3
"""Audit G/JS/C/resolved-entropy signals from one initial dynamic locator pass."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skillopt.softprefix.dynamic_combined import dynamic_skill_effect_scores, jaccard_indices
from skillopt.softprefix.entropy_localization import select_top_fraction


CANDIDATES = {
    "F0_G_JS": ("additive", [0.50, 0.50, 0.00, 0.00]),
    "F1_G_JS_C": ("additive", [0.45, 0.45, 0.10, 0.00]),
    "F2_G_JS_x_R": ("multiplicative", [0.50, 0.50, 0.00, 0.25]),
    "F3_four_additive": ("additive", [0.40, 0.40, 0.10, 0.10]),
    "F4_four_multiplicative": ("multiplicative", [0.45, 0.45, 0.10, 0.25]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locator-root", required=True)
    parser.add_argument("--trajectory-manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--ratio", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    locator_root = Path(args.locator_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    source_rows = {
        str(row.get("id", row.get("task_id"))): row
        for row in (
            json.loads(line)
            for line in Path(args.trajectory_manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    locator_rows = [
        json.loads(line)
        for line in (locator_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    signal_parts = {name: [] for name in ("G", "JS", "C", "R", "KL")}
    selected_parts = {
        name: {signal: 0.0 for signal in signal_parts} for name in CANDIDATES
    }
    total_mass = {signal: 0.0 for signal in signal_parts}
    selected_counts = {name: 0 for name in CANDIDATES}
    overlaps = {name: [] for name in CANDIDATES if name != "F0_G_JS"}
    per_trajectory = []

    for locator_row in locator_rows:
        task_id = str(locator_row["id"])
        source = source_rows[task_id]
        preserve = sorted({int(value) for value in source.get("preserve_indices", [])})
        with np.load(locator_row["array_path"]) as cached:
            arrays = {
                "G": cached["residual_gain"].astype(np.float64),
                "JS": cached["full_vocab_js"].astype(np.float64),
                "C": cached["competitor_suppression"].astype(np.float64),
                "R": cached["resolved_uncertainty"].astype(np.float64),
                "KL": cached["full_vocab_kl"].astype(np.float64),
            }
        eligible = np.ones(len(arrays["G"]), dtype=bool)
        if preserve:
            eligible[np.asarray(preserve, dtype=np.int64)] = False
        eligible_indices = np.flatnonzero(eligible)
        selected_by_method = {}
        for name, (mode, weights) in CANDIDATES.items():
            scored = dynamic_skill_effect_scores(
                arrays["G"], arrays["JS"], arrays["C"], arrays["R"],
                weights=weights, mode=mode, eligible_indices=eligible_indices,
            )["skill_effect"]
            selected = select_top_fraction(scored, ratio=args.ratio, forbidden=preserve)
            selected_by_method[name] = selected
            chosen = np.asarray(selected, dtype=np.int64)
            selected_counts[name] += len(selected)
            for signal, values in arrays.items():
                selected_parts[name][signal] += float(values[chosen].sum()) if len(chosen) else 0.0
        baseline = selected_by_method["F0_G_JS"]
        for name in overlaps:
            overlaps[name].append(jaccard_indices(baseline, selected_by_method[name]))
        for signal, values in arrays.items():
            total_mass[signal] += float(values[eligible].sum())
            signal_parts[signal].append(values[eligible])
        per_trajectory.append(
            {
                "id": task_id,
                "eligible_tokens": int(eligible.sum()),
                "selected": {name: len(values) for name, values in selected_by_method.items()},
                "jaccard_vs_f0": {
                    name: jaccard_indices(baseline, selected_by_method[name]) for name in overlaps
                },
            }
        )

    pooled = {name: np.concatenate(parts) for name, parts in signal_parts.items()}
    correlation = {}
    names = ["G", "JS", "C", "R", "KL"]
    for left in names:
        correlation[left] = {}
        for right in names:
            value = spearmanr(pooled[left], pooled[right]).statistic
            correlation[left][right] = None if not np.isfinite(value) else float(value)
    methods = {}
    for name in CANDIDATES:
        methods[name] = {
            "mode": CANDIDATES[name][0],
            "weights": CANDIDATES[name][1],
            "selected_tokens": selected_counts[name],
            "mean_jaccard_vs_f0": 1.0 if name == "F0_G_JS" else float(np.mean(overlaps[name])),
            "mass_capture": {
                signal: selected_parts[name][signal] / max(total_mass[signal], 1e-12)
                for signal in names
            },
        }
    summary = {
        "tokens": int(len(pooled["G"])),
        "trajectories": len(locator_rows),
        "spearman": correlation,
        "methods": methods,
        "per_trajectory": per_trajectory,
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# G/JS/C/Resolved-Entropy Initial Locator Audit",
        "",
        "| Method | Selected | Jaccard vs F0 | G mass | JS mass | C mass | R mass | KL mass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in methods.items():
        mass = item["mass_capture"]
        lines.append(
            f"| {name} | {item['selected_tokens']:,} | {item['mean_jaccard_vs_f0']:.3f} "
            f"| {mass['G']:.2%} | {mass['JS']:.2%} | {mass['C']:.2%} "
            f"| {mass['R']:.2%} | {mass['KL']:.2%} |"
        )
    (out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"tokens": summary["tokens"], "methods": methods}, indent=2))


if __name__ == "__main__":
    main()
