#!/usr/bin/env python3
"""Reconstruct rollout cache index files from existing prediction artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _normalize_rollout_backend(value: str) -> str:
    raw = value.strip().lower()
    if raw in {"openai", "openai_chat", "openai_compatible", "openai-compatible", "chat"}:
        return "openai"
    return raw


def _skill_fingerprint(path: str) -> str:
    if not path:
        return ""
    skill_path = Path(path)
    if not skill_path.exists():
        return ""
    content = skill_path.read_text(encoding="utf-8").strip()
    if not content:
        return ""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_metadata(args: argparse.Namespace) -> None:
    common: dict[str, Any] = {
        "env": args.env,
        "rollout_backend": _normalize_rollout_backend(args.rollout_backend),
        "target_backend": args.target_backend,
        "target_model": args.target_model,
        "max_completion_tokens": args.max_completion_tokens,
        "trajectory_use_skill": args.trajectory_use_skill,
        "trajectory_skill_fingerprint": _skill_fingerprint(args.skill_path),
        "trajectory_rollouts_per_task": args.trajectory_rollouts_per_task,
        "split_dir": args.split_dir,
        "split_seed": args.split_seed,
    }
    if args.env == "officeqa":
        common.update(
            {
                "max_tool_turns": args.max_tool_turns,
            }
        )
    elif args.env == "spreadsheetbench":
        common.update(
            {
                "mode": args.mode,
                "max_turns": args.max_turns,
                "task_timeout": args.task_timeout,
                "workers": args.workers,
                "use_eval_feedback": args.use_eval_feedback,
                "data_root": args.data_root,
                "train_size": args.train_size,
            }
        )
    _write_json(Path(args.rollout_dir) / "rollout_meta.json", common)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _extract_task_description(user_prompt: str) -> str:
    match = re.search(r"# Instruction\s*\n(?P<body>.*?)(?:\n\nInstruction type:|\Z)", user_prompt, re.S)
    return match.group("body").strip() if match else ""


def _extract_instruction_type(user_prompt: str) -> str:
    match = re.search(r"^Instruction type:\s*(.+)$", user_prompt, re.M)
    return match.group(1).strip() if match else ""


def _task_type(instruction_type: str) -> str:
    lowered = instruction_type.lower()
    if "cell" in lowered:
        return "cell_level"
    if "sheet" in lowered:
        return "sheet_level"
    return "other"


def _verification_counts(conversation: list[Any]) -> tuple[int, int, str]:
    blocks = [
        str(message.get("content") or "")
        for message in conversation
        if isinstance(message, dict)
        and str(message.get("role", "")).lower() == "system"
        and "[POST-EXECUTION VERIFICATION]" in str(message.get("content") or "")
    ]
    if not blocks:
        return 0, 0, "missing-verification"
    text = "\n".join(blocks)
    pass_count = len(re.findall(r"Eval Result \(case \d+\): PASS", text))
    fail_count = len(re.findall(r"Eval Result \(case \d+\): FAIL", text))
    n_cases = pass_count + fail_count
    if n_cases == 0:
        n_cases = 1 if ("PASS" in text or "FAIL" in text) else 0
        pass_count = 1 if "PASS" in text and "FAIL" not in text else 0
    fail_reason = "" if n_cases > 0 and pass_count == n_cases else "eval-mismatch"
    return n_cases, pass_count, fail_reason


def _last_assistant_response(conversation: list[Any]) -> str:
    response = ""
    for message in conversation:
        if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant":
            response = str(message.get("content") or "").strip()
    return response


def _spreadsheet_row(pred_dir: Path) -> dict[str, Any] | None:
    conv_path = pred_dir / "conversation.json"
    if not conv_path.exists():
        return None
    conversation = _read_json(conv_path)
    if not isinstance(conversation, list):
        return None

    user_prompt = _read_text(pred_dir / "target_user_prompt.txt")
    instruction_type = _extract_instruction_type(user_prompt)
    task_description = _extract_task_description(user_prompt)
    n_cases, n_pass, fail_reason = _verification_counts(conversation)
    response = _last_assistant_response(conversation) or _read_text(pred_dir / "raw.txt").strip()
    hard = 1 if n_cases > 0 and n_pass == n_cases else 0
    soft = (n_pass / n_cases) if n_cases else 0.0
    return {
        "id": pred_dir.name,
        "ok": bool(hard),
        "instruction_type": instruction_type,
        "task_type": _task_type(instruction_type),
        "task_description": task_description,
        "n_cases": n_cases,
        "n_exec_pass": n_cases,
        "n_pass": n_pass,
        "soft": soft,
        "hard": hard,
        "n_turns": sum(
            1
            for message in conversation
            if isinstance(message, dict) and str(message.get("role", "")).lower() == "assistant"
        ),
        "cases": [{"stage": "eval", "ok": bool(hard)} for _ in range(max(n_cases, 1))],
        "response": response,
        "fail_reason": fail_reason,
        "cache_source": "predictions",
    }


def _write_spreadsheet_results(rollout_dir: Path) -> tuple[int, int]:
    pred_root = rollout_dir / "predictions"
    rows: list[dict[str, Any]] = []
    if not pred_root.exists():
        raise FileNotFoundError(f"missing predictions directory: {pred_root}")
    for pred_dir in sorted(path for path in pred_root.iterdir() if path.is_dir()):
        row = _spreadsheet_row(pred_dir)
        if row is not None:
            rows.append(row)
    results_path = rollout_dir / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows), sum(1 for row in rows if row.get("hard"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=["officeqa", "spreadsheetbench"], required=True)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--target-model", default="gpt-5.5")
    parser.add_argument("--target-backend", default="openai_chat")
    parser.add_argument("--rollout-backend", default="openai_compatible")
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--trajectory-use-skill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-rollouts-per-task", type=int, default=1)
    parser.add_argument("--split-dir", default="")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--skill-path", default="")
    parser.add_argument("--max-tool-turns", type=int, default=12)
    parser.add_argument("--mode", default="multi")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--task-timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--use-eval-feedback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--train-size", type=int, default=0)
    parser.add_argument("--results", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_dir = Path(args.rollout_dir)
    _write_metadata(args)
    print(f"wrote {rollout_dir / 'rollout_meta.json'}")
    if args.env == "spreadsheetbench" and args.results:
        total, passed = _write_spreadsheet_results(rollout_dir)
        print(f"wrote {rollout_dir / 'results.jsonl'} ({total} rows, {passed} hard=1)")


if __name__ == "__main__":
    main()
