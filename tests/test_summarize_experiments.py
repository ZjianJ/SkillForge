from __future__ import annotations

import importlib.util
import re
from pathlib import Path


def load_summarizer():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_experiments.py"
    spec = importlib.util.spec_from_file_location("summarize_experiments", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_print_report_uses_test_hard_as_primary_metric(capsys) -> None:
    summarizer = load_summarizer()
    data = {
        "main": {
            "docvqa": [
                {
                    "run": "main_docvqa_seed1",
                    "seed": 1,
                    "best_score": 0.91,
                    "test_hard": 0.81,
                    "test_soft": 0.86,
                    "valid_seen_hard": 0.89,
                    "valid_seen_soft": 0.91,
                    "epochs": 1,
                },
                {
                    "run": "main_docvqa_seed2",
                    "seed": 2,
                    "best_score": 0.95,
                    "test_hard": 0.83,
                    "test_soft": 0.88,
                    "valid_seen_hard": 0.93,
                    "valid_seen_soft": 0.95,
                    "epochs": 1,
                },
            ]
        },
        "len": {
            (8, "docvqa"): {
                "run": "len8_docvqa",
                "best_score": 0.94,
                "test_hard": 0.82,
                "test_soft": 0.87,
                "valid_seen_hard": 0.92,
                "valid_seen_soft": 0.94,
                "epochs": 1,
            }
        },
        "model": {
            ("Qwen3.5-9B", "docvqa"): {
                "run": "model_Qwen3.5-9B_docvqa",
                "best_score": 0.97,
                "test_hard": 0.85,
                "test_soft": 0.90,
                "valid_seen_hard": 0.96,
                "valid_seen_soft": 0.97,
                "epochs": 1,
            }
        },
        "lora": {},
        "init_prefix": {},
        "random_init": {},
        "skillopt": {},
        "skipped": [],
    }

    summarizer.print_report(data)

    output = capsys.readouterr().out
    assert "## EXP 1" in output
    assert "test hard aggregate over seeds" in output
    assert "docvqa" in output
    assert "0.8300    0.8200    0.0141" in output
    assert "## EXP 2" in output
    assert re.search(r"\b8\s+docvqa\s+0\.8200\b", output)
    assert "## EXP 3" in output
    assert re.search(r"Qwen3\.5-9B\s+docvqa\s+0\.8500\b", output)
    assert "## Auxiliary metrics - test soft" in output
    assert "## Auxiliary metrics - val soft" in output
    assert "## Auxiliary metrics - val hard" in output


def test_init_prefix_uses_init_test_metrics(capsys) -> None:
    summarizer = load_summarizer()
    data = {
        "main": {},
        "len": {},
        "model": {},
        "lora": {},
        "init_prefix": {
            "docvqa": {
                "run": "init_prefix_docvqa",
                "test_hard": 0.42,
                "test_soft": 0.55,
                "valid_seen_hard": 0.48,
                "valid_seen_soft": 0.51,
                "epochs": 0,
            }
        },
        "random_init": {},
        "skillopt": {},
        "skipped": [],
    }

    summarizer.print_position_independent_report(data)
    output = capsys.readouterr().out
    assert "## EXP 5" in output
    assert re.search(r"docvqa\s+0\.4200\b", output)
    assert "before any training" in output


def test_classify_new_experiment_families() -> None:
    summarizer = load_summarizer()
    assert summarizer.classify_run("init_prefix_docvqa") == {
        "family": "init_prefix",
        "task": "docvqa",
    }
    assert summarizer.classify_run("random_init_prefix_livemathematicianbench") == {
        "family": "random_init",
        "task": "livemath",
    }
    assert summarizer.classify_run("skillopt_prefix_searchqa") == {
        "family": "skillopt",
        "task": "searchqa",
    }
