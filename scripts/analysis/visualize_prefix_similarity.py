#!/usr/bin/env python3
"""Visualize soft-prefix similarity and drift across experiment runs.

Examples:
    python scripts/analysis/visualize_prefix_similarity.py

    python scripts/analysis/visualize_prefix_similarity.py \
        --outputs-dir outputs \
        --settings prompt_start skill_section \
        --tasks livemath searchqa docvqa \
        --position-metric cos_to_init

The script uses existing ``interpret/nearest_tokens.json`` files when present.
If checkpoint files are available, it can also compute pairwise prefix cosine
similarities and prefix-vs-initial drift directly from ``best_prefix.pt`` or
``latest_prefix.pt``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


TASK_ALIASES = {
    "searchqa": "searchqa",
    "livemath": "livemath",
    "livemathematicianbench": "livemath",
    "live_math": "livemath",
    "docvqa": "docvqa",
    "doc_vqa": "docvqa",
}
TASK_LABELS = {
    "searchqa": "SearchQA",
    "livemath": "LiveMath",
    "docvqa": "DocVQA",
}
DEFAULT_SETTINGS = ("prompt_start", "skill_section")
DEFAULT_TASKS = ("livemath", "searchqa", "docvqa")
CHECKPOINT_CANDIDATES = ("best_prefix.pt", "latest_prefix.pt")


@dataclass
class RunRecord:
    run_dir: Path
    setting: str
    task: str
    model: str
    label: str
    summary: dict[str, Any]
    history: list[dict[str, Any]]
    config: dict[str, Any]
    prefix: np.ndarray | None = None
    position_stats: list[dict[str, float]] = field(default_factory=list)
    trajectory: list[dict[str, float | int | str | None]] = field(default_factory=list)
    checkpoint_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def group_key(self) -> tuple[str, str]:
        return self.model, self.task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/prefix_similarity_viz"))
    parser.add_argument("--settings", nargs="+", default=list(DEFAULT_SETTINGS))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument(
        "--checkpoint",
        choices=("best", "latest"),
        default="best",
        help="Checkpoint preference when both best_prefix.pt and latest_prefix.pt exist.",
    )
    parser.add_argument(
        "--position-metric",
        choices=("delta_l2", "cos_to_init"),
        default="cos_to_init",
        help="Metric for the prefix-position heatmap.",
    )
    parser.add_argument(
        "--accuracy-metric",
        choices=("test_soft", "test_hard", "valid_seen_soft", "valid_seen_hard", "best_score"),
        default="test_soft",
        help="Y-axis metric for final accuracy/drift points.",
    )
    parser.add_argument(
        "--drift-metric",
        choices=("mean_delta_l2", "one_minus_mean_cos"),
        default="one_minus_mean_cos",
        help="X-axis drift metric for accuracy vs drift plots.",
    )
    parser.add_argument(
        "--include-pattern",
        default="",
        help="Optional regex that run directory names must match.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through to Hugging Face model/tokenizer loading when checkpoint drift must be computed.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for model embedding lookup when computing init drift from checkpoints.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="DPI for saved PNG files.",
    )
    return parser.parse_args()


def normalize_task(task: str) -> str:
    key = task.strip().lower().replace("-", "_")
    return TASK_ALIASES.get(key, key)


def sanitize_filename(text: str) -> str:
    text = text.replace("/", "-")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "plot"


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def infer_task(run_dir: Path, config: dict[str, Any]) -> str:
    runtime = config.get("runtime", {}) if isinstance(config, dict) else {}
    if runtime.get("env"):
        return normalize_task(str(runtime["env"]))
    name = run_dir.name.lower()
    for alias, canonical in TASK_ALIASES.items():
        if alias in name:
            return canonical
    return "unknown"


def infer_model(config: dict[str, Any]) -> str:
    soft_cfg = config.get("soft_prefix", {}) if isinstance(config, dict) else {}
    runtime = config.get("runtime", {}) if isinstance(config, dict) else {}
    model = soft_cfg.get("model_name") or runtime.get("model_name") or runtime.get("target_model")
    return str(model or "unknown")


def discover_runs(
    outputs_dir: Path,
    settings: list[str],
    tasks: set[str],
    include_pattern: str,
) -> list[RunRecord]:
    include_re = re.compile(include_pattern) if include_pattern else None
    records: list[RunRecord] = []
    for setting in settings:
        setting_dir = outputs_dir / setting
        if not setting_dir.is_dir():
            continue
        for run_dir in sorted(path for path in setting_dir.iterdir() if path.is_dir()):
            if include_re and not include_re.search(run_dir.name):
                continue
            summary = load_json(run_dir / "summary.json", {})
            config = load_json(run_dir / "config.json", {})
            history = load_json(run_dir / "history.json", summary.get("history", []))
            if not summary and not config:
                continue
            task = infer_task(run_dir, config)
            if task not in tasks:
                continue
            model = infer_model(config)
            records.append(
                RunRecord(
                    run_dir=run_dir,
                    setting=setting,
                    task=task,
                    model=model,
                    label=f"{setting}/{run_dir.name}",
                    summary=summary,
                    history=history if isinstance(history, list) else [],
                    config=config,
                )
            )
    return records


def import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("This script requires matplotlib: pip install matplotlib") from exc
    return plt


def maybe_import_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def load_prefix_checkpoint(path: Path) -> np.ndarray | None:
    torch = maybe_import_torch()
    if torch is None or not path.is_file():
        return None
    try:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(state, dict) or "prefix_embeddings" not in state:
        return None
    tensor = state["prefix_embeddings"]
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().float().cpu().numpy()
    return np.asarray(tensor, dtype=np.float32)


def resolve_checkpoint(record: RunRecord, preference: str) -> Path | None:
    keys = (
        ("best_prefix_path", "latest_prefix_path")
        if preference == "best"
        else ("latest_prefix_path", "best_prefix_path")
    )
    for key in keys:
        raw = str(record.summary.get(key) or "").strip()
        if raw:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = record.run_dir / path
            if path.is_file():
                return path

    names = CHECKPOINT_CANDIDATES if preference == "best" else tuple(reversed(CHECKPOINT_CANDIDATES))
    for name in names:
        path = record.run_dir / name
        if path.is_file():
            return path
    return None


def resolve_epoch_checkpoint(record: RunRecord, epoch: int) -> Path | None:
    names = (
        f"epoch_{epoch:02d}_prefix.pt",
        f"epoch_{epoch}_prefix.pt",
        f"prefix_epoch_{epoch:02d}.pt",
        f"prefix_epoch_{epoch}.pt",
        f"checkpoint_epoch_{epoch:02d}.pt",
        f"checkpoint_epoch_{epoch}.pt",
        f"epoch_{epoch:02d}.pt",
        f"epoch_{epoch}.pt",
    )
    dirs = (
        record.run_dir,
        record.run_dir / "checkpoints",
        record.run_dir / "prefix_checkpoints",
    )
    for directory in dirs:
        for name in names:
            path = directory / name
            if path.is_file():
                return path
    return None


def load_interpret_stats(record: RunRecord) -> list[dict[str, float]]:
    for path in (
        record.run_dir / "interpret" / "nearest_tokens.json",
        record.run_dir / "nearest_tokens.json",
    ):
        data = load_json(path, None)
        if not isinstance(data, dict):
            continue
        positions = data.get("positions")
        if not isinstance(positions, list):
            continue
        stats: list[dict[str, float]] = []
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            row: dict[str, float] = {}
            for key in ("delta_l2", "cos_to_init", "norm", "hard_decode_cosine"):
                if isinstance(pos.get(key), (int, float)):
                    row[key] = float(pos[key])
            stats.append(row)
        if stats:
            return stats
    return []


def init_text_path(record: RunRecord) -> Path | None:
    soft_cfg = record.config.get("soft_prefix", {}) if isinstance(record.config, dict) else {}
    runtime = record.config.get("runtime", {}) if isinstance(record.config, dict) else {}
    raw = soft_cfg.get("init_text_path") or runtime.get("skill_init")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path if path.is_file() else None


def compute_init_prefix(
    *,
    record: RunRecord,
    prefix_length: int,
    device: str,
    trust_remote_code: bool,
    cache: dict[tuple[str, str, int], np.ndarray | None],
) -> np.ndarray | None:
    path = init_text_path(record)
    if path is None:
        return None
    cache_key = (record.model, str(path), prefix_length)
    if cache_key in cache:
        return cache[cache_key]

    torch = maybe_import_torch()
    if torch is None:
        cache[cache_key] = None
        return None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        cache[cache_key] = None
        return None

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        cache[cache_key] = None
        return None

    tokenizer = AutoTokenizer.from_pretrained(record.model, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        record.model,
        torch_dtype=torch.float32,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    with torch.no_grad():
        token_embeds = model.get_input_embeddings()(input_ids)[0].float().cpu().numpy()
    repeats = math.ceil(prefix_length / max(token_embeds.shape[0], 1))
    init = np.tile(token_embeds, (repeats, 1))[:prefix_length].astype(np.float32)
    cache[cache_key] = init
    return init


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.maximum(denom, 1e-8)
    return np.sum(a * b, axis=1) / denom


def compute_position_stats(prefix: np.ndarray, init: np.ndarray | None) -> list[dict[str, float]]:
    stats: list[dict[str, float]] = []
    for idx in range(prefix.shape[0]):
        row = {"norm": float(np.linalg.norm(prefix[idx]))}
        if init is not None and init.shape == prefix.shape:
            delta = prefix[idx] - init[idx]
            row["delta_l2"] = float(np.linalg.norm(delta))
            row["cos_to_init"] = float(cosine_rows(prefix[idx : idx + 1], init[idx : idx + 1])[0])
        stats.append(row)
    return stats


def stats_drift(stats: list[dict[str, float]], metric: str) -> float | None:
    if metric == "mean_delta_l2":
        values = [pos.get("delta_l2") for pos in stats]
        values = [float(v) for v in values if isinstance(v, (int, float))]
        return float(np.mean(values)) if values else None
    values = [pos.get("cos_to_init") for pos in stats]
    values = [float(v) for v in values if isinstance(v, (int, float))]
    return float(1.0 - np.mean(values)) if values else None


def hydrate_prefix_data(records: list[RunRecord], args: argparse.Namespace) -> None:
    init_cache: dict[tuple[str, str, int], np.ndarray | None] = {}
    for record in records:
        record.position_stats = load_interpret_stats(record)
        checkpoint = resolve_checkpoint(record, args.checkpoint)
        if checkpoint is None:
            if not record.position_stats:
                record.warnings.append("no checkpoint or interpret JSON")
            continue
        prefix = load_prefix_checkpoint(checkpoint)
        if prefix is None:
            record.warnings.append(f"could not load checkpoint {checkpoint}")
            continue
        record.prefix = prefix
        record.checkpoint_path = checkpoint
        if not record.position_stats or args.position_metric not in record.position_stats[0]:
            init = compute_init_prefix(
                record=record,
                prefix_length=prefix.shape[0],
                device=args.device,
                trust_remote_code=args.trust_remote_code,
                cache=init_cache,
            )
            record.position_stats = compute_position_stats(prefix, init)
        for row in record.history:
            epoch = row.get("epoch")
            if not isinstance(epoch, int):
                continue
            epoch_checkpoint = resolve_epoch_checkpoint(record, epoch)
            if epoch_checkpoint is None:
                continue
            epoch_prefix = load_prefix_checkpoint(epoch_checkpoint)
            if epoch_prefix is None:
                continue
            init = compute_init_prefix(
                record=record,
                prefix_length=epoch_prefix.shape[0],
                device=args.device,
                trust_remote_code=args.trust_remote_code,
                cache=init_cache,
            )
            stats = compute_position_stats(epoch_prefix, init)
            point = {
                "epoch": epoch,
                "mean_delta_l2": stats_drift(stats, "mean_delta_l2"),
                "one_minus_mean_cos": stats_drift(stats, "one_minus_mean_cos"),
                "valid_seen_soft": row.get("valid_seen_soft"),
                "valid_seen_hard": row.get("valid_seen_hard"),
                "loss": row.get("loss"),
                "source": str(epoch_checkpoint),
            }
            record.trajectory.append(point)


def pairwise_prefix_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.ndim != 2 or b.ndim != 2:
        return float("nan")
    if a.shape[1] != b.shape[1]:
        return float("nan")
    rows = min(a.shape[0], b.shape[0])
    if rows <= 0:
        return float("nan")
    return float(np.mean(cosine_rows(a[:rows], b[:rows])))


def draw_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    out_path: Path,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    dpi: int = 180,
) -> None:
    plt = import_matplotlib()
    height = max(3.5, 0.35 * len(row_labels) + 1.8)
    width = max(5.5, 0.35 * len(col_labels) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_facecolor("#f2f2f2")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_similarity_heatmaps(records: list[RunRecord], out_dir: Path, dpi: int) -> list[Path]:
    paths: list[Path] = []
    for (model, task), group in sorted(group_records(records).items()):
        group = [record for record in group if record.prefix is not None]
        if len(group) < 2:
            continue
        labels = [short_label(record) for record in group]
        matrix = np.full((len(group), len(group)), np.nan, dtype=np.float32)
        for i, left in enumerate(group):
            for j, right in enumerate(group):
                matrix[i, j] = pairwise_prefix_similarity(left.prefix, right.prefix)
        out_path = out_dir / f"prefix_similarity_{sanitize_filename(model)}_{task}.png"
        draw_heatmap(
            matrix,
            labels,
            labels,
            f"Prefix-Level Similarity: {model} / {TASK_LABELS.get(task, task)}",
            out_path,
            cmap="viridis",
            vmin=-1.0,
            vmax=1.0,
            dpi=dpi,
        )
        paths.append(out_path)
    return paths


def plot_position_heatmaps(records: list[RunRecord], out_dir: Path, metric: str, dpi: int) -> list[Path]:
    paths: list[Path] = []
    for (model, task), group in sorted(group_records(records).items()):
        rows: list[np.ndarray] = []
        labels: list[str] = []
        for record in group:
            values = [
                float(pos[metric])
                for pos in record.position_stats
                if isinstance(pos.get(metric), (int, float))
            ]
            if values:
                rows.append(np.asarray(values, dtype=np.float32))
                labels.append(short_label(record))
        if not rows:
            continue
        max_len = max(len(row) for row in rows)
        plot_len = min(max_len, 32)
        matrix = np.full((len(rows), plot_len), np.nan, dtype=np.float32)
        for idx, row in enumerate(rows):
            matrix[idx, : min(len(row), plot_len)] = row[:plot_len]
        out_path = out_dir / f"prefix_position_{metric}_{sanitize_filename(model)}_{task}.png"
        draw_heatmap(
            matrix,
            labels,
            [str(i) for i in range(plot_len)],
            f"Prefix-Position {metric}: {model} / {TASK_LABELS.get(task, task)}",
            out_path,
            cmap="magma" if metric == "delta_l2" else "viridis",
            vmin=-1.0 if metric == "cos_to_init" else None,
            vmax=1.0 if metric == "cos_to_init" else None,
            dpi=dpi,
        )
        paths.append(out_path)
    return paths


def group_records(records: list[RunRecord]) -> dict[tuple[str, str], list[RunRecord]]:
    groups: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        groups.setdefault(record.group_key, []).append(record)
    return groups


def short_label(record: RunRecord) -> str:
    name = record.run_dir.name
    name = name.replace("datasize_", "data_")
    name = name.replace("_soft_prefix_", "_")
    return f"{record.setting}/{name}"


def metric_value(record: RunRecord, metric: str) -> float | None:
    if metric in {"valid_seen_soft", "valid_seen_hard"}:
        accepted = [row for row in record.history if row.get("action") == "accept_new_best"]
        row = accepted[-1] if accepted else (record.history[-1] if record.history else {})
        value = row.get(metric)
    else:
        value = record.summary.get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def drift_value(record: RunRecord, metric: str) -> float | None:
    return stats_drift(record.position_stats, metric)


def plot_accuracy_vs_drift_group(
    ax: Any,
    *,
    model: str,
    task: str,
    group_points: list[tuple[RunRecord, float, float, int | None]],
    args: argparse.Namespace,
    colors: dict[str, str],
) -> None:
    by_series: dict[tuple[str, str], list[tuple[RunRecord, float, float, int | None]]] = {}
    for item in group_points:
        record = item[0]
        series_name = record.run_dir.name if item[3] is not None else record.setting
        by_series.setdefault((record.setting, series_name), []).append(item)
    for (setting, series_name), setting_points in sorted(by_series.items()):
        setting_points.sort(key=lambda item: (item[0].run_dir.name, item[3] if item[3] is not None else 10**9, item[1]))
        xs = [item[1] for item in setting_points]
        ys = [item[2] for item in setting_points]
        label = setting if series_name == setting else f"{setting}/{series_name}"
        ax.scatter(
            xs,
            ys,
            label=label,
            color=colors.get(setting),
            s=60,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )
        for idx, (record, x, y, epoch) in enumerate(setting_points):
            point_label = f"e{epoch}" if epoch is not None else record.run_dir.name
            dx, dy = (4, 4) if idx % 2 == 0 else (4, -12)
            ax.annotate(
                point_label,
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8,
            )
    ax.set_title(f"{model}\n{TASK_LABELS.get(task, task)}")
    ax.set_xlabel(args.drift_metric.replace("_", " "))
    ax.set_ylabel(args.accuracy_metric.replace("_", " "))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def plot_accuracy_vs_drift(records: list[RunRecord], out_dir: Path, args: argparse.Namespace) -> list[Path]:
    points: list[tuple[RunRecord, float, float, int | None]] = []
    for record in records:
        trajectory_points = []
        if args.accuracy_metric in {"valid_seen_soft", "valid_seen_hard"}:
            for point in record.trajectory:
                drift = point.get(args.drift_metric)
                acc = point.get(args.accuracy_metric)
                epoch = point.get("epoch")
                if isinstance(drift, (int, float)) and isinstance(acc, (int, float)) and isinstance(epoch, int):
                    trajectory_points.append((record, float(drift), float(acc), epoch))
        if trajectory_points:
            points.extend(trajectory_points)
            continue
        drift = drift_value(record, args.drift_metric)
        acc = metric_value(record, args.accuracy_metric)
        if drift is not None and acc is not None:
            points.append((record, drift, acc, None))
    if not points:
        return []

    groups = sorted({record.group_key for record, _, _, _ in points})
    ncols = min(3, len(groups))
    first_row_groups = groups[:ncols]
    plt = import_matplotlib()
    colors = {"prompt_start": "#1f77b4", "skill_section": "#ff7f0e"}
    paths: list[Path] = []
    for model, task in first_row_groups:
        fig, ax = plt.subplots(figsize=(10, 8))
        group_points = [(r, x, y, e) for r, x, y, e in points if r.group_key == (model, task)]
        plot_accuracy_vs_drift_group(
            ax,
            model=model,
            task=task,
            group_points=group_points,
            args=args,
            colors=colors,
        )
        out_path = (
            out_dir
            / f"accuracy_vs_prefix_drift_{args.drift_metric}_{args.accuracy_metric}_{sanitize_filename(model)}_{task}.png"
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        paths.append(out_path)
    return paths


def write_manifest(out_dir: Path, records: list[RunRecord], plot_paths: list[Path]) -> None:
    rows = []
    for record in records:
        rows.append(
            {
                "run_dir": str(record.run_dir),
                "setting": record.setting,
                "task": record.task,
                "model": record.model,
                "checkpoint_path": str(record.checkpoint_path or ""),
                "has_prefix": record.prefix is not None,
                "position_stats": len(record.position_stats),
                "trajectory_points": len(record.trajectory),
                "warnings": record.warnings,
            }
        )
    manifest = {
        "plots": [str(path) for path in plot_paths],
        "runs": rows,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    outputs_dir = args.outputs_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = {normalize_task(task) for task in args.tasks}
    records = discover_runs(outputs_dir, args.settings, tasks, args.include_pattern)
    if not records:
        print("No matching runs found.", file=sys.stderr)
        return

    hydrate_prefix_data(records, args)
    plot_paths: list[Path] = []
    plot_paths.extend(plot_similarity_heatmaps(records, out_dir, args.dpi))
    plot_paths.extend(plot_position_heatmaps(records, out_dir, args.position_metric, args.dpi))
    plot_paths.extend(plot_accuracy_vs_drift(records, out_dir, args))
    write_manifest(out_dir, records, plot_paths)

    print(f"Discovered {len(records)} runs.")
    print(f"Wrote {len(plot_paths)} plots to {out_dir}")
    if not plot_paths:
        print("No plots were created; provide checkpoints or interpret/nearest_tokens.json files.")
    skipped = [record for record in records if record.warnings]
    if skipped:
        print(f"{len(skipped)} runs had missing prefix data; see manifest.json for details.")


if __name__ == "__main__":
    main()
