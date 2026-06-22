"""Tests for shared soft-prefix trajectory rollout behavior."""
from __future__ import annotations

import json
from pathlib import Path

from skillopt.softprefix.trainer import (
    SoftPrefixSettings,
    _build_officeqa_trajectory_examples,
    _build_spreadsheet_trajectory_examples,
    _collect_officeqa_trajectory_examples,
    _collect_spreadsheet_trajectory_examples,
)


def test_officeqa_openai_trajectory_rollout_uses_chat_backend(tmp_path: Path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    captured: dict = {}

    def fake_run_officeqa_vllm_rollout(**kwargs):
        captured.update(kwargs)
        return [{"id": item["id"], "hard": 1, "soft": 1.0} for item in kwargs["items"]]

    def fake_build_officeqa_trajectory_examples(rollout_dir, results, settings):
        del rollout_dir, settings
        return [{"id": row["id"], "prompt": "prompt", "target": "target", "hard": row["hard"], "soft": row["soft"]} for row in results]

    monkeypatch.setattr(trainer, "_run_officeqa_vllm_rollout", fake_run_officeqa_vllm_rollout)
    monkeypatch.setattr(trainer, "_run_officeqa_local_hf_rollout", lambda **kwargs: (_ for _ in ()).throw(AssertionError("local_hf should not run")))
    monkeypatch.setattr(trainer, "_build_officeqa_trajectory_examples", fake_build_officeqa_trajectory_examples)

    examples = _collect_officeqa_trajectory_examples(
        prefix_model=object(),
        train_items=[{"id": "office_0000", "question": "q0"}],
        cfg={},
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "trajectory_rollout_backend": "openai_compatible",
            }
        ),
        init_text="",
        out_root=str(tmp_path),
    )

    assert captured["items"] == [{"id": "office_0000", "question": "q0"}]
    assert examples == [{"id": "office_0000", "prompt": "prompt", "target": "target", "hard": 1, "soft": 1.0}]


def test_officeqa_trajectory_rollout_expands_items_for_repeated_attempts(tmp_path: Path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    captured: dict = {}

    def fake_run_officeqa_local_hf_rollout(**kwargs):
        captured["items"] = kwargs["items"]
        return [
            {
                "id": item["id"],
                "hard": 1,
                "soft": 1.0,
            }
            for item in kwargs["items"]
        ]

    def fake_build_officeqa_trajectory_examples(rollout_dir, results, settings):
        del rollout_dir, settings
        return [
            {
                "id": row["id"],
                "prompt": "prompt",
                "target": "target",
                "hard": row["hard"],
                "soft": row["soft"],
            }
            for row in results
        ]

    monkeypatch.setattr(trainer, "_run_officeqa_local_hf_rollout", fake_run_officeqa_local_hf_rollout)
    monkeypatch.setattr(trainer, "_build_officeqa_trajectory_examples", fake_build_officeqa_trajectory_examples)

    examples = _collect_officeqa_trajectory_examples(
        prefix_model=object(),
        train_items=[
            {"id": "office_0000", "question": "q0"},
            {"id": "office_0001", "question": "q1"},
        ],
        cfg={},
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "trajectory_rollout_backend": "local_hf",
                "trajectory_rollouts_per_task": 2,
            }
        ),
        init_text="",
        out_root=str(tmp_path),
    )

    assert [item["id"] for item in captured["items"]] == [
        "office_0000__sample_00",
        "office_0000__sample_01",
        "office_0001__sample_00",
        "office_0001__sample_01",
    ]
    assert [example["id"] for example in examples] == [
        "office_0000__sample_00",
        "office_0000__sample_01",
        "office_0001__sample_00",
        "office_0001__sample_01",
    ]


def test_officeqa_trajectory_examples_supervise_tool_calls_per_turn(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions" / "office_0000"
    pred_dir.mkdir(parents=True)
    (pred_dir / "conversation.json").write_text(
        json.dumps(
            [
                {"role": "user", "content": "initial user prompt"},
                {"type": "message", "content": ""},
                {"type": "tool_call", "cmd": "grep(pattern='debt', path='/tmp/doc.txt')", "obs": "10: debt row"},
                {"type": "tool_call", "cmd": "read(path='/tmp/doc.txt', start=8, limit=5)", "obs": "excerpt"},
                {"type": "message", "content": "<answer>42</answer>"},
            ]
        ),
        encoding="utf-8",
    )

    examples = _build_officeqa_trajectory_examples(
        str(tmp_path),
        [
            {
                "id": "office_0000",
                "hard": 1,
                "soft": 1.0,
                "response": "<answer>42</answer>",
                "target_system_prompt": "system prompt",
                "target_user_prompt": "initial user prompt",
            }
        ],
        SoftPrefixSettings.from_dict(
            {"model_name": "Qwen/Qwen3.5-4B", "train_on_final_only": False}
        ),
    )

    assert len(examples) == 2
    tool_example = examples[0]
    assert tool_example["id"] == "office_0000__turn_01"
    assert tool_example["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "initial user prompt"},
    ]
    assert tool_example["target_message"] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "officeqa_tool_1_1",
                "type": "function",
                "function": {
                    "name": "grep",
                    "arguments": json.dumps({"pattern": "debt", "path": "/tmp/doc.txt"}, ensure_ascii=False),
                },
            },
            {
                "id": "officeqa_tool_1_2",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": json.dumps({"path": "/tmp/doc.txt", "start": 8, "limit": 5}, ensure_ascii=False),
                },
            },
        ],
    }

    answer_example = examples[1]
    assert answer_example["id"] == "office_0000__turn_02"
    assert answer_example["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "initial user prompt"},
        tool_example["target_message"],
        {"role": "tool", "tool_call_id": "officeqa_tool_1_1", "content": "10: debt row"},
        {"role": "tool", "tool_call_id": "officeqa_tool_1_2", "content": "excerpt"},
    ]
    assert answer_example["target_message"] == {"role": "assistant", "content": "<answer>42</answer>"}


def test_officeqa_trajectory_examples_final_only_by_default(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions" / "office_0000"
    pred_dir.mkdir(parents=True)
    (pred_dir / "conversation.json").write_text(
        json.dumps(
            [
                {"role": "user", "content": "initial user prompt"},
                {"type": "message", "content": ""},
                {"type": "tool_call", "cmd": "grep(pattern='debt', path='/tmp/doc.txt')", "obs": "10: debt row"},
                {"type": "message", "content": "<answer>42</answer>"},
            ]
        ),
        encoding="utf-8",
    )

    examples = _build_officeqa_trajectory_examples(
        str(tmp_path),
        [
            {
                "id": "office_0000",
                "hard": 1,
                "soft": 1.0,
                "response": "<answer>42</answer>",
                "target_system_prompt": "system prompt",
                "target_user_prompt": "initial user prompt",
            }
        ],
        SoftPrefixSettings.from_dict({"model_name": "Qwen/Qwen3.5-4B"}),
    )

    assert len(examples) == 1
    assert examples[0]["target_message"] == {"role": "assistant", "content": "<answer>42</answer>"}


def test_spreadsheet_trajectory_examples_final_only_by_default(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions" / "sheet_0000"
    pred_dir.mkdir(parents=True)
    (pred_dir / "target_system_prompt.txt").write_text("system prompt", encoding="utf-8")
    (pred_dir / "target_user_prompt.txt").write_text("user prompt", encoding="utf-8")
    (pred_dir / "conversation.json").write_text(
        """[
  {"role": "assistant", "content": "bad code"},
  {"role": "user", "content": "execution feedback"},
  {"role": "assistant", "content": "```python\\nprint('fixed')\\n```"},
  {"role": "system", "content": "[POST-EXECUTION VERIFICATION] pass"}
]""",
        encoding="utf-8",
    )

    examples = _build_spreadsheet_trajectory_examples(
        str(tmp_path),
        [{"id": "sheet_0000", "hard": 1, "soft": 1.0, "task_type": "cell_level"}],
        SoftPrefixSettings.from_dict({"model_name": "Qwen/Qwen3.5-4B"}),
    )

    assert examples == [
        {
            "id": "sheet_0000",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
                {"role": "assistant", "content": "bad code"},
                {"role": "user", "content": "execution feedback"},
            ],
            "target": "```python\nprint('fixed')\n```",
            "hard": 1.0,
            "soft": 1.0,
            "task_type": "cell_level",
            "task_description": "",
            "rollout_dir": str(tmp_path),
        }
    ]


def test_spreadsheet_trajectory_examples_supervise_each_assistant_turn(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions" / "sheet_0000"
    pred_dir.mkdir(parents=True)
    (pred_dir / "target_system_prompt.txt").write_text("system prompt", encoding="utf-8")
    (pred_dir / "target_user_prompt.txt").write_text("user prompt", encoding="utf-8")
    (pred_dir / "conversation.json").write_text(
        """[
  {"role": "assistant", "content": "bad code"},
  {"role": "user", "content": "execution feedback"},
  {"role": "assistant", "content": "```python\\nprint('fixed')\\n```"},
  {"role": "system", "content": "[POST-EXECUTION VERIFICATION] pass"}
]""",
        encoding="utf-8",
    )

    examples = _build_spreadsheet_trajectory_examples(
        str(tmp_path),
        [{"id": "sheet_0000", "hard": 1, "soft": 1.0, "task_type": "cell_level"}],
        SoftPrefixSettings.from_dict(
            {"model_name": "Qwen/Qwen3.5-4B", "train_on_final_only": False}
        ),
    )

    assert examples == [
        {
            "id": "sheet_0000__turn_01",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
            ],
            "target": "bad code",
            "hard": 1.0,
            "soft": 1.0,
            "task_type": "cell_level",
            "task_description": "",
            "rollout_dir": str(tmp_path),
        },
        {
            "id": "sheet_0000__turn_02",
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
                {"role": "assistant", "content": "bad code"},
                {"role": "user", "content": "execution feedback"},
            ],
            "target": "```python\nprint('fixed')\n```",
            "hard": 1.0,
            "soft": 1.0,
            "task_type": "cell_level",
            "task_description": "",
            "rollout_dir": str(tmp_path),
        }
    ]


def test_spreadsheet_trajectory_rollout_expands_items_for_repeated_attempts(tmp_path: Path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    captured: dict = {}

    def fake_run_spreadsheet_batch_codegen(**kwargs):
        captured.update(kwargs)
        return [
            {
                "id": item["id"],
                "hard": 1,
                "soft": 1.0,
                "task_type": "cell_level",
            }
            for item in kwargs["items"]
        ]

    def fake_build_spreadsheet_trajectory_examples(rollout_dir, results, settings):
        del rollout_dir, settings
        return [
            {
                "id": row["id"],
                "messages": [{"role": "user", "content": "prompt"}],
                "target": "target",
                "hard": row["hard"],
                "soft": row["soft"],
            }
            for row in results
        ]

    monkeypatch.setattr(trainer, "run_spreadsheet_batch_codegen", fake_run_spreadsheet_batch_codegen)
    monkeypatch.setattr(trainer, "_build_spreadsheet_trajectory_examples", fake_build_spreadsheet_trajectory_examples)

    examples = _collect_spreadsheet_trajectory_examples(
        items=[
            {"id": "sheet_0000", "instruction": "do a"},
            {"id": "sheet_0001", "instruction": "do b"},
        ],
        cfg={"data_root": "data/spreadsheetbench", "mode": "multi", "max_turns": 3, "workers": 2},
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "trajectory_rollouts_per_task": 2,
            }
        ),
        out_root=str(tmp_path),
        init_text="",
    )

    assert [item["id"] for item in captured["items"]] == [
        "sheet_0000__sample_00",
        "sheet_0000__sample_01",
        "sheet_0001__sample_00",
        "sheet_0001__sample_01",
    ]
    assert captured["data_root"] == "data/spreadsheetbench"
    assert captured["mode"] == "multi"
    assert captured["max_turns"] == 3
    assert captured["max_api_workers"] == 2
    assert [example["id"] for example in examples] == [
        "sheet_0000__sample_00",
        "sheet_0000__sample_01",
        "sheet_0001__sample_00",
        "sheet_0001__sample_01",
    ]


def test_spreadsheet_trajectory_rollout_reconstructs_results_from_prediction_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from skillopt.softprefix import trainer

    rollout_dir = tmp_path / "rollouts"
    pred_dir = rollout_dir / "predictions" / "sheet_0000"
    pred_dir.mkdir(parents=True)
    (pred_dir / "target_system_prompt.txt").write_text("system prompt", encoding="utf-8")
    (pred_dir / "target_user_prompt.txt").write_text("user prompt", encoding="utf-8")
    (pred_dir / "raw.txt").write_text("```python\nprint('ok')\n```", encoding="utf-8")
    (pred_dir / "conversation.json").write_text(
        json.dumps(
            [
                {"role": "assistant", "content": "```python\nprint('ok')\n```"},
                {
                    "role": "system",
                    "content": "[POST-EXECUTION VERIFICATION]\n\n## Eval Result (case 1): PASS",
                },
            ]
        ),
        encoding="utf-8",
    )

    def fake_run_spreadsheet_batch_codegen(**kwargs):
        results_path = Path(kwargs["out_root"]) / "results.jsonl"
        assert results_path.exists()
        return [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]

    monkeypatch.setattr(trainer, "run_spreadsheet_batch_codegen", fake_run_spreadsheet_batch_codegen)

    examples = _collect_spreadsheet_trajectory_examples(
        items=[{"id": "sheet_0000", "instruction": "do it", "instruction_type": "cell"}],
        cfg={"data_root": "data/spreadsheetbench"},
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "trajectory_rollout_dir": str(rollout_dir),
            }
        ),
        out_root=str(tmp_path),
        init_text="",
    )

    assert [example["id"] for example in examples] == ["sheet_0000"]
    result_rows = [
        json.loads(line)
        for line in (rollout_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert result_rows == [
        {
            "id": "sheet_0000",
            "ok": True,
            "task_type": "cell_level",
            "task_description": "do it",
            "n_cases": 1,
            "n_pass": 1,
            "soft": 1.0,
            "hard": 1,
            "n_turns": 1,
            "cases": [{"stage": "eval", "ok": True}],
            "response": "```python\nprint('ok')\n```",
            "fail_reason": "",
            "cache_source": "predictions",
        }
    ]


def test_officeqa_and_spreadsheet_rollouts_write_metadata(tmp_path: Path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    def fake_run_officeqa_vllm_rollout(**kwargs):
        return [{"id": item["id"], "hard": 1, "soft": 1.0, "response": "answer"} for item in kwargs["items"]]

    def fake_build_officeqa_trajectory_examples(rollout_dir, results, settings):
        del rollout_dir, settings
        return [{"id": row["id"], "messages": [], "target": row["response"], "hard": 1, "soft": 1.0} for row in results]

    def fake_run_spreadsheet_batch_codegen(**kwargs):
        return [{"id": item["id"], "hard": 1, "soft": 1.0, "task_type": "cell_level"} for item in kwargs["items"]]

    def fake_build_spreadsheet_trajectory_examples(rollout_dir, results, settings):
        del rollout_dir, settings
        return [{"id": row["id"], "messages": [], "target": "code", "hard": 1, "soft": 1.0} for row in results]

    monkeypatch.setattr(trainer, "_run_officeqa_vllm_rollout", fake_run_officeqa_vllm_rollout)
    monkeypatch.setattr(trainer, "_build_officeqa_trajectory_examples", fake_build_officeqa_trajectory_examples)
    monkeypatch.setattr(trainer, "run_spreadsheet_batch_codegen", fake_run_spreadsheet_batch_codegen)
    monkeypatch.setattr(trainer, "_build_spreadsheet_trajectory_examples", fake_build_spreadsheet_trajectory_examples)

    office_rollout_dir = tmp_path / "office_rollouts"
    spreadsheet_rollout_dir = tmp_path / "spreadsheet_rollouts"
    settings = {
        "model_name": "Qwen/Qwen3.5-4B",
        "trajectory_rollout_backend": "openai_compatible",
        "trajectory_use_skill": True,
    }

    _collect_officeqa_trajectory_examples(
        prefix_model=object(),
        train_items=[{"id": "office_0000"}],
        cfg={"target_backend": "openai_chat", "target_model": "gpt-5.5", "max_tool_turns": 8},
        settings=SoftPrefixSettings.from_dict({**settings, "trajectory_rollout_dir": str(office_rollout_dir)}),
        init_text="skill",
        out_root=str(tmp_path / "office_out"),
    )
    _collect_spreadsheet_trajectory_examples(
        items=[{"id": "sheet_0000", "instruction": "do it"}],
        cfg={"target_backend": "openai_chat", "target_model": "gpt-5.5", "data_root": "data/spreadsheetbench"},
        settings=SoftPrefixSettings.from_dict({**settings, "trajectory_rollout_dir": str(spreadsheet_rollout_dir)}),
        out_root=str(tmp_path / "sheet_out"),
        init_text="skill",
    )

    office_meta = json.loads((office_rollout_dir / "rollout_meta.json").read_text(encoding="utf-8"))
    spreadsheet_meta = json.loads((spreadsheet_rollout_dir / "rollout_meta.json").read_text(encoding="utf-8"))
    assert office_meta["env"] == "officeqa"
    assert office_meta["target_model"] == "gpt-5.5"
    assert office_meta["max_tool_turns"] == 8
    assert spreadsheet_meta["env"] == "spreadsheetbench"
    assert spreadsheet_meta["target_backend"] == "openai_chat"
    assert spreadsheet_meta["trajectory_use_skill"] is True
