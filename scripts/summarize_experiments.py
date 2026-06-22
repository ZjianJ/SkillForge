#!/usr/bin/env python3
"""Summarize experiment results from outputs/<setting>/*/summary.json.

Covers all 7 experiment families from run_experiments.sh:

  EXP 1 – Main setting (3 seeds × 3 datasets)
  EXP 2 – Vary prefix length (8 / 256; len=32 from EXP 1 seed=1)
  EXP 3 – Vary model size
  EXP 4 – LoRA tuning (injection-position independent; reported once)
  EXP 5 – Initial skill prefix eval, no training (position independent; once)
  EXP 6 – Random init (vocab_mean soft prefix)
  EXP 7 – Init from final SkillOpt artifact

Experiments 1–3 and 6–7 are reported per injection-position setting
(prompt_start, skill_section). Experiments 4–5 are reported once globally.

Usage:
    python scripts/summarize_experiments.py
    python scripts/summarize_experiments.py --outputs_dir outputs
    python scripts/summarize_experiments.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

PREFIXES = (
    "len",
    "main",
    "model",
    "lora",
    "init_prefix",
    "random_init_prefix",
    "skillopt_prefix",
)
EXPERIMENT_SETTINGS = ("prompt_start", "skill_section")
POSITION_INDEPENDENT_FAMILIES = frozenset({"lora", "init_prefix"})

MAIN_RE = re.compile(r"^main_(?P<dataset>.+)_seed(?P<seed>\d+)$")
LEN_RE = re.compile(r"^len(?P<length>\d+)_(?P<dataset>.+)$")
MODEL_RE = re.compile(r"^model_(?P<model>.+)_(?P<dataset>[^_]+(?:_[^_]+)*)$")
LORA_RE = re.compile(r"^lora_(?P<task>.+)$")
INIT_PREFIX_RE = re.compile(r"^init_prefix_(?P<task>.+)$")
RANDOM_INIT_RE = re.compile(r"^random_init_prefix_(?P<task>.+)$")
SKILLOPT_RE = re.compile(r"^skillopt_prefix_(?P<task>.+)$")

TASK_ALIASES = {"livemathematicianbench": "livemath"}

PRIMARY_METRIC = "test_hard"
AUX_METRICS = ("test_soft", "valid_seen_soft", "valid_seen_hard")
METRICS = (PRIMARY_METRIC, *AUX_METRICS)
METRIC_LABELS = {
    "test_hard": "test hard",
    "test_soft": "test soft",
    "valid_seen_soft": "val soft",
    "valid_seen_hard": "val hard",
}

EXP_TITLES = {
    "main": "EXP 1 – Main setting (len=32, init=skill md)",
    "len": "EXP 2 – Vary prefix length (seed=1)",
    "model": "EXP 3 – Vary model size (seed=1)",
    "lora": "EXP 4 – LoRA tuning (seed=1; position-independent)",
    "init_prefix": (
        "EXP 5 – Initial skill prefix eval, no training (position-independent)"
    ),
    "random_init": "EXP 6 – Random init soft prefix (vocab_mean, seed=1)",
    "skillopt": "EXP 7 – Init from final SkillOpt artifact (seed=1)",
}


def normalize_task(task: str) -> str:
    return TASK_ALIASES.get(task, task)


def load_summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "summary.json"
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def aggregate_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    if len(values) == 1:
        return {"best": values[0], "avg": values[0], "std": 0.0}
    return {
        "best": max(values),
        "avg": statistics.mean(values),
        "std": statistics.stdev(values),
    }


def fmt(value: float | None, width: int = 8) -> str:
    if value is None:
        return " " * width
    return f"{value:>{width}.4f}"


def best_validation_metrics(summary: dict[str, Any]) -> dict[str, float | None]:
    """Return validation metrics for the epoch that produced the best prefix."""
    history = summary.get("history", [])
    accepted = [h for h in history if h.get("action") == "accept_new_best"]
    best_epoch = accepted[-1] if accepted else None
    if best_epoch is None and history:
        best_score = summary.get("best_score")
        candidates = [
            h for h in history if h.get("valid_seen_score") == best_score
        ]
        best_epoch = candidates[-1] if candidates else history[-1]

    if best_epoch is None:
        return {
            "valid_seen_soft": summary.get("best_score"),
            "valid_seen_hard": None,
        }

    return {
        "valid_seen_soft": best_epoch.get("valid_seen_soft"),
        "valid_seen_hard": best_epoch.get("valid_seen_hard"),
    }


def metrics_from_summary(summary: dict[str, Any], family: str) -> dict[str, float | None]:
    if family == "init_prefix":
        return {
            "test_hard": summary.get("init_test_hard"),
            "test_soft": summary.get("init_test_soft"),
            "valid_seen_hard": summary.get("init_valid_seen_hard"),
            "valid_seen_soft": summary.get("init_valid_seen_soft"),
        }
    row = {m: summary.get(m) for m in METRICS}
    row.update(best_validation_metrics(summary))
    return row


def discover_settings(outputs_dir: Path) -> list[tuple[str, Path]]:
    """Return (setting_name, runs_dir) pairs for each experiment setting."""
    if not outputs_dir.is_dir():
        return []
    settings: list[tuple[str, Path]] = []
    for name in EXPERIMENT_SETTINGS:
        setting_dir = outputs_dir / name
        if setting_dir.is_dir():
            settings.append((name, setting_dir))
    if settings:
        return settings
    return [("default", outputs_dir)]


def discover_position_independent_dir(outputs_dir: Path) -> Path | None:
    path = outputs_dir / "position_independent"
    return path if path.is_dir() else None


def discover_runs(runs_dir: Path) -> list[Path]:
    runs: list[Path] = []
    if not runs_dir.is_dir():
        return runs
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        if any(child.name.startswith(p) for p in PREFIXES):
            runs.append(child)
    return runs


def classify_run(name: str) -> dict[str, Any]:
    if m := MAIN_RE.match(name):
        return {"family": "main", "dataset": m["dataset"], "seed": int(m["seed"])}
    if m := LEN_RE.match(name):
        return {"family": "len", "dataset": m["dataset"], "length": int(m["length"])}
    if m := MODEL_RE.match(name):
        return {"family": "model", "dataset": m["dataset"], "model": m["model"]}
    if m := LORA_RE.match(name):
        return {"family": "lora", "task": normalize_task(m["task"])}
    if m := INIT_PREFIX_RE.match(name):
        return {"family": "init_prefix", "task": normalize_task(m["task"])}
    if m := RANDOM_INIT_RE.match(name):
        return {"family": "random_init", "task": normalize_task(m["task"])}
    if m := SKILLOPT_RE.match(name):
        return {"family": "skillopt", "task": normalize_task(m["task"])}
    return {"family": "unknown", "raw": name}


def empty_results() -> dict[str, Any]:
    return {
        "main": {},
        "len": {},
        "model": {},
        "lora": {},
        "init_prefix": {},
        "random_init": {},
        "skillopt": {},
        "skipped": [],
    }


def collect_results(runs_dir: Path) -> dict[str, Any]:
    main: dict[str, list[dict[str, Any]]] = {}
    length: dict[tuple[int, str], dict[str, Any]] = {}
    model: dict[tuple[str, str], dict[str, Any]] = {}
    lora: dict[str, dict[str, Any]] = {}
    init_prefix: dict[str, dict[str, Any]] = {}
    random_init: dict[str, dict[str, Any]] = {}
    skillopt: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []

    for run_dir in discover_runs(runs_dir):
        summary = load_summary(run_dir)
        if summary is None:
            skipped.append(run_dir.name)
            continue

        info = classify_run(run_dir.name)
        family = info["family"]
        if family == "unknown":
            skipped.append(run_dir.name)
            continue

        row = {
            "run": run_dir.name,
            **metrics_from_summary(summary, family),
            "epochs": len(summary.get("history", [])),
        }

        if family == "main":
            main.setdefault(info["dataset"], []).append(
                {**row, "seed": info["seed"]}
            )
        elif family == "len":
            length[(info["length"], info["dataset"])] = row
        elif family == "model":
            model[(info["model"], info["dataset"])] = row
        elif family == "lora":
            lora[info["task"]] = row
        elif family == "init_prefix":
            init_prefix[info["task"]] = row
        elif family == "random_init":
            random_init[info["task"]] = row
        elif family == "skillopt":
            skillopt[info["task"]] = row

    for dataset in main:
        main[dataset].sort(key=lambda r: r["seed"])

    return {
        "main": main,
        "len": length,
        "model": model,
        "lora": lora,
        "init_prefix": init_prefix,
        "random_init": random_init,
        "skillopt": skillopt,
        "skipped": skipped,
    }


def merge_position_independent(
    by_setting: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Keep the first copy of lora / init_prefix runs across settings."""
    merged = empty_results()
    for data in by_setting.values():
        for family in POSITION_INDEPENDENT_FAMILIES:
            for key, row in data[family].items():
                merged[family].setdefault(key, row)
        for name in data["skipped"]:
            if name not in merged["skipped"]:
                merged["skipped"].append(name)
    return merged


def strip_position_independent(data: dict[str, Any]) -> dict[str, Any]:
    out = {**data}
    for family in POSITION_INDEPENDENT_FAMILIES:
        out[family] = {}
    return out


def summarize_main(main: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset, runs in sorted(main.items()):
        out[dataset] = {
            "n_seeds": len(runs),
            "runs": runs,
            "metrics": {
                metric: aggregate_stats(
                    [r[metric] for r in runs if r.get(metric) is not None]
                )
                for metric in METRICS
            },
        }
    return out


def print_main_metric(
    main_summary: dict[str, Any],
    metric: str,
    title: str,
) -> None:
    print(f"\n## {title}\n")
    header = f"{'dataset':<12} {'seeds':>5}  {'best':>8}  {'avg':>8}  {'std':>8}"
    print(header)
    print("-" * len(header))
    for dataset, info in main_summary.items():
        seeds = ",".join(str(r["seed"]) for r in info["runs"])
        stats = info["metrics"].get(metric, {})
        if not stats:
            continue
        print(
            f"{dataset:<12} {seeds:>5}  "
            f"{fmt(stats.get('best'))}  {fmt(stats.get('avg'))}  {fmt(stats.get('std'))}"
        )


def print_len_metric(
    rows: dict[tuple[int, str], dict[str, Any]],
    metric: str,
    title: str,
    include_len32_note: bool = False,
) -> None:
    print(f"\n## {title}\n")
    header = f"{'length':>6}  {'dataset':<12}  {METRIC_LABELS[metric]:>10}"
    print(header)
    print("-" * len(header))
    for (length, dataset), row in sorted(rows.items()):
        print(f"{length:>6}  {dataset:<12}  {fmt(row.get(metric), 10)}")
    if include_len32_note:
        print("\n  Note: len=32 numbers come from EXP 1 seed=1 (not duplicated here).")


def print_model_metric(
    rows: dict[tuple[str, str], dict[str, Any]],
    metric: str,
    title: str,
) -> None:
    print(f"\n## {title}\n")
    header = f"{'model':<22}  {'dataset':<12}  {METRIC_LABELS[metric]:>10}"
    print(header)
    print("-" * len(header))
    for (model_name, dataset), row in sorted(rows.items()):
        print(f"{model_name:<22}  {dataset:<12}  {fmt(row.get(metric), 10)}")


def print_task_metric(
    rows: dict[str, dict[str, Any]],
    metric: str,
    title: str,
) -> None:
    print(f"\n## {title}\n")
    header = f"{'task':<12}  {METRIC_LABELS[metric]:>10}"
    print(header)
    print("-" * len(header))
    for task, row in sorted(rows.items()):
        print(f"{task:<12}  {fmt(row.get(metric), 10)}")


def print_position_independent_report(data: dict[str, Any]) -> None:
    print("=" * 72)
    print("Position-independent experiments (EXP 4–5)")
    print("=" * 72)

    if data["lora"]:
        print_task_metric(
            data["lora"],
            PRIMARY_METRIC,
            f"{EXP_TITLES['lora']} – test hard",
        )

    if data["init_prefix"]:
        print_task_metric(
            data["init_prefix"],
            PRIMARY_METRIC,
            f"{EXP_TITLES['init_prefix']} – test hard",
        )
        print(
            "\n  Note: EXP 5 evaluates the skill-md-initialized soft prefix "
            "before any training (train.num_epochs=0)."
        )

    for metric in AUX_METRICS:
        if not data["lora"] and not data["init_prefix"]:
            break
        print(f"\n## Auxiliary metrics - {METRIC_LABELS[metric]} (EXP 4–5)")
        if data["lora"]:
            print_task_metric(
                data["lora"],
                metric,
                f"{EXP_TITLES['lora']} – {METRIC_LABELS[metric]}",
            )
        if data["init_prefix"]:
            print_task_metric(
                data["init_prefix"],
                metric,
                f"{EXP_TITLES['init_prefix']} – {METRIC_LABELS[metric]}",
            )

    if data["skipped"]:
        print("\n## Skipped (no summary.json)\n")
        for name in data["skipped"]:
            print(f"  - {name}")
    print()


def print_report(data: dict[str, Any], setting: str | None = None) -> None:
    print("=" * 72)
    title = "Experiments 1–3, 6–7 (per injection position)"
    if setting:
        title = f"{title} — {setting}"
    print(title)
    print("=" * 72)

    main_summary = summarize_main(data["main"])
    if main_summary:
        print_main_metric(
            main_summary,
            PRIMARY_METRIC,
            f"{EXP_TITLES['main']} – test hard aggregate over seeds",
        )

        print("\n  Per-seed test hard:")
        for dataset, info in main_summary.items():
            print(f"  [{dataset}]")
            sub_header = f"    {'seed':>4}  {'test hard':>10}"
            print(sub_header)
            for r in info["runs"]:
                print(f"    {r['seed']:>4}  {fmt(r.get(PRIMARY_METRIC), 10)}")
            print()

    if data["len"]:
        print_len_metric(
            data["len"],
            PRIMARY_METRIC,
            f"{EXP_TITLES['len']} – test hard",
            include_len32_note=True,
        )

    if data["model"]:
        print_model_metric(
            data["model"],
            PRIMARY_METRIC,
            f"{EXP_TITLES['model']} – test hard",
        )

    if data["random_init"]:
        print_task_metric(
            data["random_init"],
            PRIMARY_METRIC,
            f"{EXP_TITLES['random_init']} – test hard",
        )

    if data["skillopt"]:
        print_task_metric(
            data["skillopt"],
            PRIMARY_METRIC,
            f"{EXP_TITLES['skillopt']} – test hard",
        )

    has_aux = (
        main_summary
        or data["len"]
        or data["model"]
        or data["random_init"]
        or data["skillopt"]
    )
    if has_aux:
        for metric in AUX_METRICS:
            print(f"\n## Auxiliary metrics - {METRIC_LABELS[metric]}")
            if main_summary:
                print_main_metric(
                    main_summary,
                    metric,
                    f"{EXP_TITLES['main']} – {METRIC_LABELS[metric]} aggregate over seeds",
                )
            if data["len"]:
                print_len_metric(
                    data["len"],
                    metric,
                    f"{EXP_TITLES['len']} – {METRIC_LABELS[metric]}",
                )
            if data["model"]:
                print_model_metric(
                    data["model"],
                    metric,
                    f"{EXP_TITLES['model']} – {METRIC_LABELS[metric]}",
                )
            if data["random_init"]:
                print_task_metric(
                    data["random_init"],
                    metric,
                    f"{EXP_TITLES['random_init']} – {METRIC_LABELS[metric]}",
                )
            if data["skillopt"]:
                print_task_metric(
                    data["skillopt"],
                    metric,
                    f"{EXP_TITLES['skillopt']} – {METRIC_LABELS[metric]}",
                )

    if data["skipped"]:
        print("\n## Skipped (no summary.json)\n")
        for name in data["skipped"]:
            print(f"  - {name}")

    print()


def collect_all_results(outputs_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    by_setting_raw = {
        setting: collect_results(runs_dir)
        for setting, runs_dir in discover_settings(outputs_dir)
    }
    global_data = merge_position_independent(by_setting_raw)
    pos_indep_dir = discover_position_independent_dir(outputs_dir)
    if pos_indep_dir is not None:
        dedicated = collect_results(pos_indep_dir)
        global_data = merge_position_independent(
            {"position_independent": dedicated, **by_setting_raw}
        )
    by_setting = {
        setting: strip_position_independent(data)
        for setting, data in by_setting_raw.items()
    }
    return global_data, by_setting


def build_json_payload(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "main": summarize_main(data["main"]),
        "len": {
            f"len{k[0]}_{k[1]}": v for k, v in sorted(data["len"].items())
        },
        "model": {
            f"model_{k[0]}_{k[1]}": v for k, v in sorted(data["model"].items())
        },
        "lora": {
            f"lora_{k}": v for k, v in sorted(data["lora"].items())
        },
        "init_prefix": {
            f"init_prefix_{k}": v for k, v in sorted(data["init_prefix"].items())
        },
        "random_init": {
            f"random_init_prefix_{k}": v
            for k, v in sorted(data["random_init"].items())
        },
        "skillopt": {
            f"skillopt_prefix_{k}": v for k, v in sorted(data["skillopt"].items())
        },
        "skipped": data["skipped"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs_dir",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing experiment run folders",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON instead of a text report",
    )
    args = parser.parse_args()
    outputs_dir = args.outputs_dir.resolve()

    global_data, by_setting = collect_all_results(outputs_dir)
    if args.json:
        payload = {
            "position_independent": build_json_payload(global_data),
            "by_setting": {
                setting: build_json_payload(data)
                for setting, data in by_setting.items()
            },
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        has_global = global_data["lora"] or global_data["init_prefix"]
        if has_global:
            print_position_independent_report(global_data)
        for i, (setting, data) in enumerate(by_setting.items()):
            has_setting = any(
                data[key]
                for key in ("main", "len", "model", "random_init", "skillopt")
            )
            if not has_setting and not data["skipped"]:
                continue
            if has_global or i > 0:
                print()
            print_report(data, setting=None if setting == "default" else setting)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
