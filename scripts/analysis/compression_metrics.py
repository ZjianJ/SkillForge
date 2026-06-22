#!/usr/bin/env python3
"""Summarize hard-skill-vs-soft-prefix compression diagnostics.

This script reads existing evaluation artifacts and reports the stable columns
for the planned single-round QA compression table:

  - hard-skill markdown token count
  - soft-prefix length
  - mean generated output tokens

Latency is intentionally not reported here; it should be measured in a separate
controlled benchmark if needed.

Examples:
    python scripts/analysis/compression_metrics.py

    python scripts/analysis/compression_metrics.py \
        --soft-result searchqa=outputs/skill_section/main_searchqa_seed1 \
        --hard-skill-result searchqa=outputs/hard_skill_searchqa_gpt-5.5_.../eval

    python scripts/analysis/compression_metrics.py --format csv --out table12.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


TASKS = ("searchqa", "livemath", "docvqa")
TASK_LABELS = {
    "searchqa": "SearchQA",
    "livemath": "LiveMath",
    "docvqa": "DocVQA",
}
TASK_ALIASES = {
    "searchqa": "searchqa",
    "search": "searchqa",
    "livemath": "livemath",
    "livemathematicianbench": "livemath",
    "live_math": "livemath",
    "docvqa": "docvqa",
    "doc_vqa": "docvqa",
}
SOFT_RUN_NAMES = {
    "searchqa": "main_searchqa_seed1",
    "livemath": "main_livemath_seed1",
    "docvqa": "main_docvqa_seed1",
}
HARD_SKILL_RUN_PATTERNS = {
    "searchqa": ("hard_skill_searchqa_*", "skillopt_searchqa_*"),
    "livemath": (
        "hard_skill_livemath_*",
        "hard_skill_livemathematicianbench_*",
        "skillopt_livemath_*",
        "skillopt_livemathematicianbench_*",
    ),
    "docvqa": ("hard_skill_docvqa_*", "skillopt_docvqa_*"),
}
SKILL_PATHS = {
    "searchqa": Path("ckpt/searchqa/gpt5.5_skill.md"),
    "livemath": Path("ckpt/livemath/gpt5.5_skill.md"),
    "docvqa": Path("ckpt/docvqa/gpt5.5_skill.md"),
}
RESULT_CANDIDATES = (
    "results.jsonl",
    "eval/best/valid_unseen/results.jsonl",
    "eval/init/valid_unseen/results.jsonl",
    "eval/plain/valid_unseen/results.jsonl",
    "eval/valid_unseen/results.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root. Defaults to the repository containing scripts/analysis/.",
    )
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen3.5-4B",
        help="Tokenizer used for token counts.",
    )
    parser.add_argument(
        "--tokenizer-mode",
        choices=("auto", "hf", "simple"),
        default="auto",
        help="Token counter to use. Use hf for paper numbers; auto falls back to simple if transformers is unavailable.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS),
        help="Tasks to report. Aliases: searchqa, livemath/livemathematicianbench, docvqa.",
    )
    parser.add_argument(
        "--soft-root",
        type=Path,
        default=Path("outputs/skill_section"),
        help="Root containing default soft-prefix runs.",
    )
    parser.add_argument(
        "--soft-result",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="Override soft-prefix result path for one task. PATH may be a run dir or results.jsonl.",
    )
    parser.add_argument(
        "--hard-skill-result",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="SkillOpt hard-skill result path for one task. PATH may be a run dir or results.jsonl.",
    )
    parser.add_argument(
        "--skill-path",
        action="append",
        default=[],
        metavar="TASK=PATH",
        help="Override skill markdown path for one task.",
    )
    parser.add_argument(
        "--method-name",
        default="SoftSkill-NTP",
        help="Label for the soft-prefix method row.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional output file.")
    return parser.parse_args()


def normalize_task(task: str) -> str:
    key = task.strip().lower().replace("-", "_")
    if key not in TASK_ALIASES:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASK_ALIASES)}")
    return TASK_ALIASES[key]


def parse_task_paths(entries: list[str], *, repo_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"expected TASK=PATH, got {entry!r}")
        task, raw_path = entry.split("=", 1)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        paths[normalize_task(task)] = path
    return paths


def resolve_result_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_file():
        return path
    if not path.is_dir():
        return None
    for rel in RESULT_CANDIDATES:
        candidate = path / rel
        if candidate.is_file():
            return candidate
    matches = sorted(path.glob("**/results.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def default_soft_run(repo_root: Path, soft_root: Path, task: str) -> Path | None:
    root = soft_root if soft_root.is_absolute() else repo_root / soft_root
    return root / SOFT_RUN_NAMES[task]


def default_hard_skill_result(repo_root: Path, task: str) -> Path | None:
    outputs_dir = repo_root / "outputs"
    candidates: list[Path] = []
    for pattern in HARD_SKILL_RUN_PATTERNS[task]:
        candidates.extend(path for path in outputs_dir.glob(pattern) if path.is_dir())
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        result_path = resolve_result_path(candidate)
        if result_path is not None:
            return result_path
    return None


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num}: invalid JSONL row") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_config(run_or_result_path: Path | None) -> dict[str, Any]:
    if run_or_result_path is None:
        return {}
    start = run_or_result_path if run_or_result_path.is_dir() else run_or_result_path.parent
    for directory in (start, *start.parents):
        cfg_path = directory / "config.json"
        if cfg_path.is_file():
            with cfg_path.open(encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    return {}


def response_text(row: dict[str, Any]) -> str:
    for key in ("response", "final_response", "output", "text"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    value = row.get("predicted_answer")
    return value if isinstance(value, str) else ""


def count_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded.get("input_ids", [])
    if hasattr(ids, "ndim") and int(ids.ndim) == 2:
        return int(ids.shape[1])
    return len(ids)


def mean_output_tokens(tokenizer: Any, rows: list[dict[str, Any]]) -> float | None:
    token_counts = [count_tokens(tokenizer, response_text(row)) for row in rows]
    return statistics.mean(token_counts) if token_counts else None


def final_answer_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    hits = 0
    for row in rows:
        text = response_text(row).lower()
        predicted = row.get("predicted_answer")
        if "<answer>" in text or (isinstance(predicted, str) and predicted.strip()):
            hits += 1
    return hits / len(rows)


def soft_prefix_length(config: dict[str, Any]) -> int | str | None:
    soft_cfg = config.get("soft_prefix")
    if isinstance(soft_cfg, dict) and "prefix_length" in soft_cfg:
        return soft_cfg["prefix_length"]
    runtime = config.get("runtime")
    if isinstance(runtime, dict) and "prefix_length" in runtime:
        return runtime["prefix_length"]
    return None


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}" if abs(value) >= 10 else f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


class SimpleTokenizer:
    """Dependency-free rough token counter used only when HF tokenizers are unavailable."""

    _TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": list(range(len(self._TOKEN_RE.findall(text))))}


def load_tokenizer(model_name: str, *, mode: str) -> Any:
    if mode == "simple":
        return SimpleTokenizer()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        if mode == "hf":
            raise SystemExit(
                "transformers is required for --tokenizer-mode hf. "
                "Install it or run inside the project/vLLM environment."
            ) from exc
        print(
            "warning: transformers is unavailable; using rough simple token counts. "
            "Run with --tokenizer-mode hf in the project/vLLM environment for paper numbers.",
            file=sys.stderr,
        )
        return SimpleTokenizer()
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    repo_root = args.repo_root.resolve()
    tokenizer = load_tokenizer(args.model_name, mode=args.tokenizer_mode)
    tasks = [normalize_task(task) for task in args.tasks]
    soft_overrides = parse_task_paths(args.soft_result, repo_root=repo_root)
    hard_skill_overrides = parse_task_paths(args.hard_skill_result, repo_root=repo_root)
    skill_path_overrides = parse_task_paths(args.skill_path, repo_root=repo_root)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        skill_path = skill_path_overrides.get(task, repo_root / SKILL_PATHS[task])
        skill_text = skill_path.read_text(encoding="utf-8") if skill_path.is_file() else ""
        skill_tokens = count_tokens(tokenizer, skill_text) if skill_text else None

        hard_skill_result = resolve_result_path(hard_skill_overrides.get(task)) or default_hard_skill_result(repo_root, task)
        hard_skill_rows = load_jsonl(hard_skill_result)
        rows.append(
            {
                "Task": TASK_LABELS[task],
                "Method": "SkillOpt",
                "Hard-skill tokens": skill_tokens,
                "Soft-prefix length": None,
                "Output tokens": mean_output_tokens(tokenizer, hard_skill_rows),
                "N": len(hard_skill_rows) or None,
                "Result path": str(hard_skill_result.relative_to(repo_root)) if hard_skill_result else "",
            }
        )

        soft_run = soft_overrides.get(task) or default_soft_run(repo_root, args.soft_root, task)
        soft_result = resolve_result_path(soft_run)
        soft_rows = load_jsonl(soft_result)
        config = load_config(soft_run if soft_run and soft_run.exists() else soft_result)
        rows.append(
            {
                "Task": TASK_LABELS[task],
                "Method": args.method_name,
                "Hard-skill tokens": None,
                "Soft-prefix length": soft_prefix_length(config),
                "Output tokens": mean_output_tokens(tokenizer, soft_rows),
                "N": len(soft_rows) or None,
                "Result path": str(soft_result.relative_to(repo_root)) if soft_result else "",
            }
        )
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    columns = ["Task", "Method", "Hard-skill tokens", "Soft-prefix length", "Output tokens", "N"]
    rendered = ["| " + " | ".join(columns) + " |"]
    rendered.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        rendered.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(rendered)


def render_csv(rows: list[dict[str, Any]]) -> str:
    columns = ["Task", "Method", "Hard-skill tokens", "Soft-prefix length", "Output tokens", "N", "Result path"]
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    if args.format == "json":
        text = json.dumps(rows, ensure_ascii=False, indent=2)
    elif args.format == "csv":
        text = render_csv(rows)
    else:
        text = render_markdown(rows)

    if args.out:
        out_path = args.out if args.out.is_absolute() else args.repo_root / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
