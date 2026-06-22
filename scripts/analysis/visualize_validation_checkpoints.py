#!/usr/bin/env python3
"""Summarize which training epoch produced the best validation checkpoint.

The script scans ``outputs/**/summary.json`` files produced by soft-prefix
training runs, extracts per-epoch validation scores, reports how often epoch
1/2/3 was selected as the best checkpoint, and writes simple visualizations.
The heatmap values are deltas from each run's best validation score, so rows
from easier and harder benchmarks remain comparable.

Examples:
    python scripts/analysis/visualize_validation_checkpoints.py

    python scripts/analysis/visualize_validation_checkpoints.py \
        --outputs-dir outputs/agentic \
        --out-dir outputs/validation_checkpoint_viz_agentic

    python scripts/analysis/visualize_validation_checkpoints.py \
        --include 'main_|SoftSkill_qwen36' \
        --metric valid_seen_soft
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any, NamedTuple

METRICS = ("valid_seen_score", "valid_seen_soft", "valid_seen_hard")
TASK_ALIASES = {
    "alfworld": "alfworld",
    "doc_vqa": "docvqa",
    "docvqa": "docvqa",
    "livemathematicianbench": "livemath",
    "live_math": "livemath",
    "livemath": "livemath",
    "officeqa": "officeqa",
    "searchqa": "searchqa",
    "spreadsheetbench": "spreadsheetbench",
}
MAIN_RE = re.compile(r"^main_(?P<task>.+)_seed(?P<seed>\d+)$")


class ValidationRecord(NamedTuple):
    summary_path: Path
    run_dir: Path
    setting: str
    run: str
    task: str
    seed: str
    family: str
    best_epoch: int
    best_score: float
    epoch_scores: dict[int, float]
    epoch_losses: dict[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/validation_checkpoint_viz"))
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="valid_seen_score",
        help="Validation metric used for the heatmap and fallback best-epoch selection.",
    )
    parser.add_argument(
        "--include",
        default="",
        help="Optional regex that the summary path or run directory name must match.",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Optional regex for summary paths or run directory names to skip.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="DPI for saved PNG files.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def normalize_task(task: str) -> str:
    key = task.strip().lower().replace("-", "_")
    return TASK_ALIASES.get(key, key)


def infer_task(run_name: str) -> str:
    lowered = run_name.lower().replace("-", "_")
    for alias, canonical in sorted(TASK_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in lowered:
            return canonical
    if m := MAIN_RE.match(run_name):
        return normalize_task(m["task"])
    return "unknown"


def infer_seed(run_name: str) -> str:
    match = re.search(r"(?:^|_)seed(?P<seed>\d+)(?:_|$)", run_name)
    return match["seed"] if match else ""


def infer_family(run_name: str) -> str:
    if run_name.startswith("main_"):
        return "main"
    if run_name.startswith("len"):
        return "length"
    if run_name.startswith("model_"):
        return "model"
    if run_name.startswith("lora_"):
        return "lora"
    if run_name.startswith("random_init_prefix_"):
        return "random_init"
    if run_name.startswith("skillopt_prefix_") or run_name.startswith("skillopt_"):
        return "skillopt"
    if run_name.startswith("soft_prefix_"):
        return "soft_prefix"
    return "unknown"


def setting_for_summary(outputs_dir: Path, run_dir: Path) -> str:
    try:
        rel_parent = run_dir.parent.relative_to(outputs_dir)
    except ValueError:
        return run_dir.parent.name or "default"
    return "default" if str(rel_parent) == "." else rel_parent.as_posix()


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def epoch_number(row: dict[str, Any]) -> int | None:
    epoch = row.get("epoch")
    if isinstance(epoch, bool):
        return None
    if isinstance(epoch, int):
        return epoch
    if isinstance(epoch, float) and epoch.is_integer():
        return int(epoch)
    return None


def select_best_epoch(history: list[dict[str, Any]], metric: str) -> tuple[int, float] | None:
    accepted: list[tuple[int, float]] = []
    scored: list[tuple[int, float]] = []
    for row in history:
        epoch = epoch_number(row)
        score = numeric(row.get(metric))
        if epoch is None or score is None:
            continue
        scored.append((epoch, score))
        if row.get("action") == "accept_new_best":
            accepted.append((epoch, score))

    if accepted:
        return accepted[-1]
    if not scored:
        return None

    # Training uses strict improvements for best checkpoint updates, so ties
    # keep the earlier checkpoint when action metadata is unavailable.
    return max(scored, key=lambda item: item[1])


def collect_records(
    outputs_dir: Path,
    metric: str = "valid_seen_score",
    include: str = "",
    exclude: str = "",
) -> list[ValidationRecord]:
    include_re = re.compile(include) if include else None
    exclude_re = re.compile(exclude) if exclude else None
    records: list[ValidationRecord] = []

    for summary_path in sorted(outputs_dir.rglob("summary.json")):
        run_dir = summary_path.parent
        match_text = f"{summary_path.as_posix()} {run_dir.name}"
        if include_re and not include_re.search(match_text):
            continue
        if exclude_re and exclude_re.search(match_text):
            continue

        summary = load_json(summary_path)
        raw_history = summary.get("history", [])
        if not isinstance(raw_history, list):
            continue
        history = [row for row in raw_history if isinstance(row, dict)]
        selected = select_best_epoch(history, metric)
        if selected is None:
            continue

        epoch_scores: dict[int, float] = {}
        epoch_losses: dict[int, float] = {}
        for row in history:
            epoch = epoch_number(row)
            score = numeric(row.get(metric))
            if epoch is not None and score is not None:
                epoch_scores[epoch] = score
            loss = numeric(row.get("loss"))
            if epoch is not None and loss is not None:
                epoch_losses[epoch] = loss
        if not epoch_scores:
            continue

        best_epoch, best_score = selected
        records.append(
            ValidationRecord(
                summary_path=summary_path,
                run_dir=run_dir,
                setting=setting_for_summary(outputs_dir, run_dir),
                run=run_dir.name,
                task=infer_task(run_dir.name),
                seed=infer_seed(run_dir.name),
                family=infer_family(run_dir.name),
                best_epoch=best_epoch,
                best_score=best_score,
                epoch_scores=epoch_scores,
                epoch_losses=epoch_losses,
            )
        )

    return records


def summarize_best_epochs(records: list[ValidationRecord]) -> list[dict[str, float | int]]:
    if not records:
        return []
    max_epoch = max(max(record.epoch_scores) for record in records)
    total = len(records)
    rows: list[dict[str, float | int]] = []
    for epoch in range(1, max_epoch + 1):
        count = sum(1 for record in records if record.best_epoch == epoch)
        rows.append({"epoch": epoch, "count": count, "frequency": count / total})
    return rows


def group_records_by_task(records: list[ValidationRecord]) -> dict[str, list[ValidationRecord]]:
    grouped: dict[str, list[ValidationRecord]] = defaultdict(list)
    for record in records:
        grouped[record.task].append(record)
    return {task: grouped[task] for task in sorted(grouped)}


def build_heatmap_matrix(records: list[ValidationRecord]) -> tuple[list[int], list[ValidationRecord], list[list[float]]]:
    sorted_records = sorted(records, key=lambda record: (record.setting, record.task, record.run))
    if not sorted_records:
        return [], [], []
    epochs = list(range(1, max(max(record.epoch_scores) for record in sorted_records) + 1))
    matrix = [
        [
            record.epoch_scores[epoch] - record.best_score
            if epoch in record.epoch_scores
            else float("nan")
            for epoch in epochs
        ]
        for record in sorted_records
    ]
    return epochs, sorted_records, matrix


def build_loss_accuracy_points(records: list[ValidationRecord]) -> list[tuple[ValidationRecord, int, float, float]]:
    points: list[tuple[ValidationRecord, int, float, float]] = []
    for record in sorted(records, key=lambda item: (item.setting, item.task, item.run)):
        for epoch in sorted(record.epoch_scores):
            loss = record.epoch_losses.get(epoch)
            score = record.epoch_scores.get(epoch)
            if loss is not None and score is not None:
                points.append((record, epoch, loss, score))
    return points


def write_records_csv(records: list[ValidationRecord], path: Path) -> None:
    max_epoch = max((max(record.epoch_scores) for record in records), default=0)
    fieldnames = [
        "summary_path",
        "setting",
        "run",
        "family",
        "task",
        "seed",
        "best_epoch",
        "best_score",
        *[f"epoch_{epoch}" for epoch in range(1, max_epoch + 1)],
        *[f"loss_epoch_{epoch}" for epoch in range(1, max_epoch + 1)],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row: dict[str, str | int | float] = {
                "summary_path": record.summary_path.as_posix(),
                "setting": record.setting,
                "run": record.run,
                "family": record.family,
                "task": record.task,
                "seed": record.seed,
                "best_epoch": record.best_epoch,
                "best_score": record.best_score,
            }
            for epoch in range(1, max_epoch + 1):
                row[f"epoch_{epoch}"] = record.epoch_scores.get(epoch, "")
                row[f"loss_epoch_{epoch}"] = record.epoch_losses.get(epoch, "")
            writer.writerow(row)


def write_counts_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "count", "frequency"])
        writer.writeheader()
        writer.writerows(rows)


def import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to write PNG visualizations") from exc
    return plt


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def heatmap_color(value: float, min_delta: float) -> str:
    if value != value:
        return "#F2F2F2"
    if min_delta >= 0.0:
        t = 1.0
    else:
        t = max(0.0, min(1.0, value / min_delta))
    # Interpolate from light yellow at best (0.0) to dark purple at worst.
    best = (255, 247, 188)
    worst = (94, 60, 153)
    r = round(best[0] + (worst[0] - best[0]) * t)
    g = round(best[1] + (worst[1] - best[1]) * t)
    b = round(best[2] + (worst[2] - best[2]) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def write_best_epoch_frequency_svg(rows: list[dict[str, float | int]], path: Path) -> None:
    counts = [int(row["count"]) for row in rows]
    total = sum(counts)
    max_count = max(counts, default=0)
    bar_width = 70
    gap = 28
    left = 70
    top = 45
    chart_height = 190
    width = left + len(rows) * (bar_width + gap) + 30
    height = top + chart_height + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="25" font-family="sans-serif" font-size="18" font-weight="700">Best Checkpoint Epoch Frequency</text>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{width - 20}" y2="{top + chart_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#333"/>',
    ]
    for idx, row in enumerate(rows):
        epoch = int(row["epoch"])
        count = int(row["count"])
        frequency = float(row["frequency"])
        bar_height = 0 if max_count == 0 else chart_height * count / max_count
        x = left + idx * (bar_width + gap) + gap
        y = top + chart_height - bar_height
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="#4C78A8"/>',
                f'<text x="{x + bar_width / 2:.1f}" y="{max(38, y - 8):.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{count} ({frequency:.1%})</text>',
                f'<text x="{x + bar_width / 2:.1f}" y="{top + chart_height + 22}" text-anchor="middle" font-family="sans-serif" font-size="13">ep{epoch}</text>',
            ]
        )
    parts.append(f'<text x="20" y="{height - 14}" font-family="sans-serif" font-size="12">n={total} runs</text>')
    parts.append("</svg>")
    write_text(path, "\n".join(parts))


def write_validation_heatmap_svg(records: list[ValidationRecord], metric: str, path: Path) -> None:
    epochs, sorted_records, matrix = build_heatmap_matrix(records)
    if not sorted_records:
        return

    cell_w = 76
    cell_h = 22
    left = 360
    top = 62
    right = 40
    bottom = 40
    width = left + cell_w * len(epochs) + right
    height = top + cell_h * len(sorted_records) + bottom
    values = [value for row in matrix for value in row if value == value]
    min_delta = min(values, default=0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="28" font-family="sans-serif" font-size="18" font-weight="700">Validation Delta from Best by Epoch ({escape(metric)})</text>',
        '<text x="20" y="48" font-family="sans-serif" font-size="12" fill="#555">Cell values are validation accuracy minus that run&apos;s best validation accuracy; best cells are 0.0.</text>',
    ]
    for col_idx, epoch in enumerate(epochs):
        x = left + col_idx * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x:.1f}" y="{top - 12}" text-anchor="middle" font-family="sans-serif" font-size="12">ep{epoch}</text>'
        )
    for row_idx, record in enumerate(sorted_records):
        y = top + row_idx * cell_h
        label = escape(f"{record.setting}/{record.run}")
        parts.append(
            f'<text x="{left - 8}" y="{y + cell_h * 0.68:.1f}" text-anchor="end" font-family="sans-serif" font-size="10">{label}</text>'
        )
        for col_idx, epoch in enumerate(epochs):
            x = left + col_idx * cell_w
            value = matrix[row_idx][col_idx]
            color = heatmap_color(value, min_delta)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" stroke="white" stroke-width="1"/>'
            )
            if value == value:
                parts.append(
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h * 0.68:.1f}" text-anchor="middle" font-family="sans-serif" font-size="9">{value:.3f}</text>'
                )
            if record.best_epoch == epoch:
                parts.append(
                    f'<rect x="{x + 3}" y="{y + 3}" width="{cell_w - 6}" height="{cell_h - 6}" fill="none" stroke="#FFFFFF" stroke-width="2"/>'
                )
    parts.append("</svg>")
    write_text(path, "\n".join(parts))


def scaled(value: float, min_value: float, max_value: float, size: float) -> float:
    if max_value == min_value:
        return size / 2
    return (value - min_value) / (max_value - min_value) * size


def write_loss_accuracy_svg(records: list[ValidationRecord], metric: str, path: Path) -> None:
    points = build_loss_accuracy_points(records)
    if not points:
        return

    losses = [point[2] for point in points]
    scores = [point[3] for point in points]
    min_loss, max_loss = min(losses), max(losses)
    min_score, max_score = min(scores), max(scores)
    left = 72
    top = 54
    chart_w = 520
    chart_h = 320
    width = left + chart_w + 36
    height = top + chart_h + 72
    other_radius = 6.2
    best_radius = 7.5
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="30" font-family="sans-serif" font-size="24" font-weight="700">Loss vs Validation Accuracy ({escape(metric)})</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#333"/>',
        f'<text x="{left + chart_w / 2:.1f}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="16">training loss</text>',
        f'<text x="18" y="{top + chart_h / 2:.1f}" text-anchor="middle" font-family="sans-serif" font-size="16" transform="rotate(-90 18 {top + chart_h / 2:.1f})">{escape(metric)}</text>',
        f'<text x="{left}" y="{top + chart_h + 20}" text-anchor="middle" font-family="sans-serif" font-size="13">{min_loss:.3g}</text>',
        f'<text x="{left + chart_w}" y="{top + chart_h + 20}" text-anchor="middle" font-family="sans-serif" font-size="13">{max_loss:.3g}</text>',
        f'<text x="{left - 8}" y="{top + chart_h}" text-anchor="end" font-family="sans-serif" font-size="13">{min_score:.3g}</text>',
        f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="sans-serif" font-size="13">{max_score:.3g}</text>',
    ]
    for record, epoch, loss, score in points:
        x = left + scaled(loss, min_loss, max_loss, chart_w)
        y = top + chart_h - scaled(score, min_score, max_score, chart_h)
        color = "#F58518" if epoch == record.best_epoch else "#4C78A8"
        label = escape(f"{record.setting}/{record.run} epoch {epoch}: loss={loss:.4g}, {metric}={score:.4g}")
        radius = best_radius if epoch == record.best_epoch else other_radius
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" opacity="0.82">')
        parts.append(f"<title>{label}</title>")
        parts.append("</circle>")
    parts.extend(
        [
            f'<circle cx="{left + chart_w - 150}" cy="42" r="{best_radius}" fill="#F58518" opacity="0.82"/>',
            f'<text x="{left + chart_w - 134}" y="47" font-family="sans-serif" font-size="14">best checkpoint epoch</text>',
            f'<circle cx="{left + chart_w - 150}" cy="24" r="{other_radius}" fill="#4C78A8" opacity="0.82"/>',
            f'<text x="{left + chart_w - 134}" y="29" font-family="sans-serif" font-size="14">other epoch</text>',
            f'<text x="20" y="{height - 10}" font-family="sans-serif" font-size="14">n={len(points)} epoch points from {len(records)} runs</text>',
            "</svg>",
        ]
    )
    write_text(path, "\n".join(parts))


def plot_best_epoch_frequency(rows: list[dict[str, float | int]], path: Path, dpi: int) -> None:
    plt = import_pyplot()
    epochs = [int(row["epoch"]) for row in rows]
    counts = [int(row["count"]) for row in rows]
    labels = [f"ep{epoch}" for epoch in epochs]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color="#4C78A8")
    total = sum(counts)
    ax.set_title("Best Checkpoint Epoch Frequency")
    ax.set_xlabel("Best validation checkpoint epoch")
    ax.set_ylabel("Run count")
    ax.bar_label(
        bars,
        labels=[f"{count}\n({count / total:.1%})" if total else "0" for count in counts],
        padding=3,
        fontsize=9,
    )
    ax.set_ylim(0, max(counts, default=0) * 1.25 + 0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_artifacts(
    records: list[ValidationRecord],
    out_dir: Path,
    metric: str,
    dpi: int,
    plot_png: bool = True,
) -> list[dict[str, float | int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = summarize_best_epochs(records)
    write_records_csv(records, out_dir / "records.csv")
    write_counts_csv(counts, out_dir / "best_epoch_counts.csv")
    write_best_epoch_frequency_svg(counts, out_dir / "best_epoch_frequency.svg")
    write_validation_heatmap_svg(records, metric, out_dir / "validation_accuracy_heatmap.svg")
    write_loss_accuracy_svg(records, metric, out_dir / "loss_vs_validation_accuracy.svg")
    if plot_png:
        plot_best_epoch_frequency(counts, out_dir / "best_epoch_frequency.png", dpi)
        plot_validation_heatmap(records, metric, out_dir / "validation_accuracy_heatmap.png", dpi)
        plot_loss_accuracy(records, metric, out_dir / "loss_vs_validation_accuracy.png", dpi)
    return counts


def plot_validation_heatmap(records: list[ValidationRecord], metric: str, path: Path, dpi: int) -> None:
    plt = import_pyplot()
    if not records:
        return

    epochs, sorted_records, matrix = build_heatmap_matrix(records)
    height = min(max(4.0, 0.28 * len(sorted_records) + 1.5), 24.0)
    width = max(5.0, 1.15 * len(epochs) + 3.5)

    fig, ax = plt.subplots(figsize=(width, height))
    min_delta = min((value for row in matrix for value in row if value == value), default=0.0)
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=min_delta, vmax=0.0)
    ax.set_title(f"Validation Delta from Best by Epoch ({metric})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Run")
    ax.set_xticks(range(len(epochs)), [str(epoch) for epoch in epochs])
    labels = [f"{record.setting}/{record.run}" for record in sorted_records]
    ax.set_yticks(range(len(labels)), labels, fontsize=7 if len(labels) <= 60 else 5)

    for row_idx, record in enumerate(sorted_records):
        if record.best_epoch in epochs:
            col_idx = epochs.index(record.best_epoch)
            ax.scatter(col_idx, row_idx, marker="s", facecolors="none", edgecolors="white", linewidths=1.2)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{metric} - run best")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_loss_accuracy(records: list[ValidationRecord], metric: str, path: Path, dpi: int) -> None:
    points = build_loss_accuracy_points(records)
    if not points:
        return

    plt = import_pyplot()
    fig, ax = plt.subplots(figsize=(7, 5))
    other = [point for point in points if point[1] != point[0].best_epoch]
    best = [point for point in points if point[1] == point[0].best_epoch]
    if other:
        ax.scatter(
            [point[2] for point in other],
            [point[3] for point in other],
            color="#4C78A8",
            alpha=0.75,
            label="Other epochs",
            s=78,
        )
    if best:
        ax.scatter(
            [point[2] for point in best],
            [point[3] for point in best],
            color="#F58518",
            alpha=0.9,
            label="Best checkpoint epochs",
            s=110,
            edgecolors="white",
            linewidths=0.9,
        )
    ax.set_title(f"Loss vs Validation Accuracy ({metric})", fontsize=16)
    ax.set_xlabel("Training loss", fontsize=13)
    ax.set_ylabel(metric, fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def print_summary(records: list[ValidationRecord], counts: list[dict[str, float | int]], out_dir: Path) -> None:
    print(f"Collected {len(records)} training summaries with validation history.")
    for row in counts:
        print(f"  ep{row['epoch']}: {row['count']} runs ({row['frequency']:.1%})")
    grouped = group_records_by_task(records)
    if grouped:
        print("Per-benchmark run counts:")
        for task, task_records in grouped.items():
            print(f"  {task}: {len(task_records)}")
    print(f"Wrote artifacts to {out_dir}")


def main() -> int:
    args = parse_args()
    records = collect_records(args.outputs_dir, metric=args.metric, include=args.include, exclude=args.exclude)
    if not records:
        print(f"No training summaries with metric {args.metric!r} found under {args.outputs_dir}", file=sys.stderr)
        return 1

    counts = summarize_best_epochs(records)
    try:
        write_artifacts(records, args.out_dir, args.metric, args.dpi)
        for task, task_records in group_records_by_task(records).items():
            write_artifacts(task_records, args.out_dir / task, args.metric, args.dpi)
    except RuntimeError as exc:
        write_artifacts(records, args.out_dir, args.metric, args.dpi, plot_png=False)
        for task, task_records in group_records_by_task(records).items():
            write_artifacts(task_records, args.out_dir / task, args.metric, args.dpi, plot_png=False)
        print(f"Warning: {exc}; CSV and SVG files were still written.", file=sys.stderr)

    print_summary(records, counts, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
