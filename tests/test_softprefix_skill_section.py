"""Tests for inserting soft-prefix embeddings at rollout skill sections."""
from __future__ import annotations


def test_livemath_prompt_builder_returns_skill_section_insert_index() -> None:
    from skillopt.softprefix.data import build_livemath_prompt_and_insert_idx

    class CharTokenizer:
        chat_template = "fake"

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ):
            assert tokenize is False
            assert add_generation_prompt is True
            assert enable_thinking is False
            system = messages[0]["content"]
            user = messages[1]["content"]
            return f"<system>{system}</system><user>{user}</user><assistant>"

        def __call__(self, text, **kwargs):
            del kwargs
            return {"input_ids": [ord(ch) for ch in text]}

    item = {
        "id": "math0",
        "question": "What is 1+1?",
        "choices": [
            {"label": "A", "text": "1"},
            {"label": "B", "text": "2"},
        ],
        "correct_choice": {"label": "B", "text": "2"},
    }

    prompt, insert_idx = build_livemath_prompt_and_insert_idx(
        CharTokenizer(),
        item,
        injection_position="skill_section",
        enable_thinking=False,
    )

    assert "## Skill" not in prompt
    assert "{skill_section}" not in prompt
    assert insert_idx is not None
    assert prompt[:insert_idx].endswith("multiple-choice questions.\n\n")
    assert prompt[insert_idx:].startswith("## Task Format")


def test_livemath_dataset_sets_prefix_insert_idx_for_skill_section() -> None:
    from skillopt.softprefix.data import LiveMathPrefixDataset

    class CharTokenizer:
        chat_template = "fake"
        eos_token = "<eos>"

        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            return f"<system>{messages[0]['content']}</system><user>{messages[1]['content']}</user><assistant>"

        def __call__(self, text, **kwargs):
            del kwargs
            return {"input_ids": [ord(ch) for ch in text]}

    dataset = LiveMathPrefixDataset(
        [
            {
                "id": "math0",
                "question": "What is 1+1?",
                "choices": [
                    {"label": "A", "text": "1"},
                    {"label": "B", "text": "2"},
                ],
                "correct_choice": {"label": "B", "text": "2"},
            }
        ],
        CharTokenizer(),
        max_prompt_tokens=4096,
        max_target_tokens=32,
        injection_position="skill_section",
    )

    example = dataset[0]

    assert example.prefix_insert_idx is not None
    assert example.input_ids[example.prefix_insert_idx] == ord("#")


def test_searchqa_dataset_sets_prefix_insert_idx_for_skill_section() -> None:
    from skillopt.softprefix.data import SearchQAPrefixDataset

    class CharTokenizer:
        chat_template = "fake"
        eos_token = "<eos>"

        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            return f"<system>{messages[0]['content']}</system><user>{messages[1]['content']}</user><assistant>"

        def __call__(self, text, **kwargs):
            del kwargs
            return {"input_ids": [ord(ch) for ch in text]}

    dataset = SearchQAPrefixDataset(
        [
            {
                "id": "qa0",
                "question": "Who wrote Hamlet?",
                "context": "[DOC] Hamlet was written by William Shakespeare.",
                "answers": ["William Shakespeare"],
            }
        ],
        CharTokenizer(),
        max_prompt_tokens=4096,
        max_target_tokens=32,
        injection_position="skill_section",
    )

    example = dataset[0]

    assert example.prefix_insert_idx is not None
    assert 0 < example.prefix_insert_idx < len(example.input_ids)
    assert "<|skillopt_soft_prefix_insert|>" not in "".join(map(chr, example.input_ids))


def test_searchqa_eval_passes_skill_section_insert_index(tmp_path) -> None:
    from skillopt.softprefix import trainer

    class CharTokenizer:
        chat_template = "fake"

        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            return f"<system>{messages[0]['content']}</system><user>{messages[1]['content']}</user><assistant>"

        def __call__(self, text, **kwargs):
            del kwargs
            return {"input_ids": [ord(ch) for ch in text]}

    class FakePrefixModel:
        tokenizer = CharTokenizer()

    captured: dict[str, object] = {}

    class FakeGenerator:
        def generate_from_prompts(
            self,
            prompts,
            *,
            max_prompt_tokens,
            max_new_tokens,
            temperature,
            prefix_insert_indices=None,
        ):
            del max_prompt_tokens, max_new_tokens, temperature
            captured["prompts"] = list(prompts)
            captured["prefix_insert_indices"] = list(prefix_insert_indices or [])
            return ["<answer>William Shakespeare</answer>"]

    hard, soft, _ = trainer.evaluate_searchqa_prefix(
        FakePrefixModel(),
        [
            {
                "id": "qa0",
                "question": "Who wrote Hamlet?",
                "context": "[DOC] Hamlet was written by William Shakespeare.",
                "answers": ["William Shakespeare"],
            }
        ],
        out_dir=str(tmp_path),
        max_prompt_tokens=4096,
        max_new_tokens=32,
        temperature=0.0,
        generator=FakeGenerator(),
        injection_position="skill_section",
    )

    assert (hard, soft) == (1.0, 1.0)
    assert captured["prefix_insert_indices"]
    assert "<|skillopt_soft_prefix_insert|>" not in captured["prompts"][0]


def test_spreadsheet_codegen_prompt_builder_returns_skill_section_insert_index() -> None:
    from skillopt.softprefix.data import build_spreadsheet_codegen_prompt_and_insert_idx

    class CharTokenizer:
        chat_template = "fake"

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ):
            del enable_thinking
            assert tokenize is False
            assert add_generation_prompt is True
            system = messages[0]["content"]
            user = messages[1]["content"]
            return f"<system>{system}</system><user>{user}</user><assistant>"

        def __call__(self, text, **kwargs):
            del kwargs
            return {"input_ids": [ord(ch) for ch in text]}

    prompt, insert_idx = build_spreadsheet_codegen_prompt_and_insert_idx(
        CharTokenizer(),
        user="# Instruction\nFill column A",
        injection_position="skill_section",
        enable_thinking=False,
    )

    assert "## Skill" in prompt
    assert "<|skillopt_soft_prefix_insert|>" not in prompt
    assert "{skill_section}" not in prompt
    assert insert_idx is not None
    assert prompt[:insert_idx].endswith("## Skill\n")
    assert prompt[insert_idx:].startswith("</system>")


def test_spreadsheet_repair_eval_messages_include_skill_section_marker(monkeypatch, tmp_path) -> None:
    from skillopt.softprefix import trainer
    from skillopt.softprefix.data import _SOFT_PREFIX_INSERT_MARKER

    class CharTokenizer:
        chat_template = "fake"

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ):
            del enable_thinking
            assert tokenize is False
            assert add_generation_prompt is True
            return (
                f"<system>{messages[0]['content']}</system>"
                f"<user>{messages[1]['content']}</user><assistant>"
            )

        def __call__(self, text, **kwargs):
            del kwargs
            return {"input_ids": [ord(ch) for ch in text]}

    class FakePrefixModel:
        tokenizer = CharTokenizer()

    captured: dict[str, object] = {}

    class FakeGenerator:
        def generate_from_messages(self, messages, **kwargs):
            del kwargs
            captured["messages"] = messages
            return "```python\nprint('ok')\n```"

    monkeypatch.setattr(
        trainer,
        "find_spreadsheet_test_cases",
        lambda task_dir: [("0", f"{task_dir}/input.xlsx", f"{task_dir}/gold.xlsx")],
    )
    monkeypatch.setattr(
        trainer,
        "run_spreadsheet_generated_code",
        lambda code, input_path, pred_path, timeout: (True, ""),
    )
    monkeypatch.setattr(trainer, "evaluate_spreadsheet", lambda *args, **kwargs: {"ok": True, "reason": ""})
    monkeypatch.setattr(trainer, "auto_verify_spreadsheet_output", lambda *args, **kwargs: "verified")

    trainer.evaluate_spreadsheet_prefix(
        FakePrefixModel(),
        [
            {
                "id": "sheet0",
                "instruction": "Fill column A",
                "spreadsheet_path": "sheet0",
            }
        ],
        out_dir=str(tmp_path / "out"),
        data_root=str(tmp_path / "data"),
        max_prompt_tokens=4096,
        max_new_tokens=128,
        temperature=0.0,
        generator=FakeGenerator(),
        injection_position="skill_section",
        repair_turns=2,
    )

    messages = captured["messages"]
    system = messages[0]["content"]
    assert system.count(_SOFT_PREFIX_INSERT_MARKER) == 1


def test_spreadsheet_repair_eval_messages_include_skill_section_marker(monkeypatch, tmp_path) -> None:
    from skillopt.softprefix import trainer
    from skillopt.softprefix.data import _SOFT_PREFIX_INSERT_MARKER

    class CharTokenizer:
        chat_template = "fake"

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ):
            del enable_thinking
            assert tokenize is False
            assert add_generation_prompt is True
            return (
                f"<system>{messages[0]['content']}</system>"
                f"<user>{messages[1]['content']}</user><assistant>"
            )

        def __call__(self, text, **kwargs):
            del kwargs
            return {"input_ids": [ord(ch) for ch in text]}

    class FakePrefixModel:
        tokenizer = CharTokenizer()

    captured: dict[str, object] = {}

    class FakeGenerator:
        def generate_from_messages(self, messages, **kwargs):
            del kwargs
            captured["messages"] = messages
            return "```python\nprint('ok')\n```"

    monkeypatch.setattr(
        trainer,
        "find_spreadsheet_test_cases",
        lambda task_dir: [("0", f"{task_dir}/input.xlsx", f"{task_dir}/gold.xlsx")],
    )
    monkeypatch.setattr(
        trainer,
        "run_spreadsheet_generated_code",
        lambda code, input_path, pred_path, timeout: (True, ""),
    )
    monkeypatch.setattr(trainer, "evaluate_spreadsheet", lambda *args, **kwargs: {"ok": True, "reason": ""})
    monkeypatch.setattr(trainer, "auto_verify_spreadsheet_output", lambda *args, **kwargs: "verified")

    trainer.evaluate_spreadsheet_prefix(
        FakePrefixModel(),
        [
            {
                "id": "sheet0",
                "instruction": "Fill column A",
                "spreadsheet_path": "sheet0",
            }
        ],
        out_dir=str(tmp_path / "out"),
        data_root=str(tmp_path / "data"),
        max_prompt_tokens=4096,
        max_new_tokens=128,
        temperature=0.0,
        generator=FakeGenerator(),
        injection_position="skill_section",
        repair_turns=2,
    )

    messages = captured["messages"]
    system = messages[0]["content"]
    assert system.count(_SOFT_PREFIX_INSERT_MARKER) == 1
