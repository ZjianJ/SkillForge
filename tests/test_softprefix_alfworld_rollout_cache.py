"""Tests for ALFWorld trajectory rollout cache selection."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillopt.softprefix.trainer import (
    SoftPrefixSettings,
    _build_alfworld_trajectory_examples,
    _expand_trajectory_rollout_items,
    _normalize_trajectory_rollout_backend,
    _run_alfworld_chat_rollout,
    _train_adapter,
    _should_extend_alfworld_cached_rollout,
    _resolve_alfworld_trajectory_rollout_dir,
)


def test_softprefix_rollout_backend_defaults_to_openai() -> None:
    settings = SoftPrefixSettings.from_dict({"model_name": "Qwen/Qwen3.5-4B"})

    assert settings.trajectory_rollout_backend == "openai"


def test_softprefix_init_eval_toggle_defaults_off() -> None:
    default_settings = SoftPrefixSettings.from_dict({"model_name": "Qwen/Qwen3.5-4B"})
    enabled_settings = SoftPrefixSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "eval_init_prefix": True,
        }
    )

    assert default_settings.eval_init_prefix is False
    assert enabled_settings.eval_init_prefix is True
    assert default_settings.eval_init_val is True
    assert SoftPrefixSettings.from_dict(
        {"model_name": "Qwen/Qwen3.5-4B", "eval_init_val": "0"}
    ).eval_init_val is False


def test_softprefix_checkpoint_path_defaults_empty() -> None:
    default_settings = SoftPrefixSettings.from_dict({"model_name": "Qwen/Qwen3.5-4B"})
    checkpoint_settings = SoftPrefixSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "checkpoint_path": "outputs/run/best_prefix.pt",
        }
    )

    assert default_settings.checkpoint_path == ""
    assert checkpoint_settings.checkpoint_path == "outputs/run/best_prefix.pt"


def test_softprefix_rewind_ckpt_defaults_off() -> None:
    default_settings = SoftPrefixSettings.from_dict({"model_name": "Qwen/Qwen3.5-4B"})
    rewind_settings = SoftPrefixSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "rewind_ckpt": True,
        }
    )

    assert default_settings.rewind_ckpt is False
    assert rewind_settings.rewind_ckpt is True


def test_softprefix_init_strategy_defaults_to_text() -> None:
    default_settings = SoftPrefixSettings.from_dict({"model_name": "Qwen/Qwen3.5-4B"})
    mean_settings = SoftPrefixSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "init_strategy": "vocab_mean",
        }
    )

    assert default_settings.init_strategy == "text"
    assert mean_settings.init_strategy == "vocab_mean"


def test_vocab_mean_initializer_copies_embedding_mean_to_each_prefix_row() -> None:
    torch = pytest.importorskip("torch")
    from skillopt.softprefix.model import _initialize_prefix_from_vocab_mean

    class FakeEmbedding:
        weight = torch.tensor(
            [
                [1.0, 3.0],
                [5.0, 7.0],
                [9.0, 11.0],
            ]
        )

    class FakeModel:
        def get_input_embeddings(self):
            return FakeEmbedding()

    prefix = torch.nn.Parameter(torch.empty(2, 2))

    _initialize_prefix_from_vocab_mean(torch, FakeModel(), prefix)

    assert torch.equal(prefix, torch.tensor([[5.0, 7.0], [5.0, 7.0]]))


def test_softprefix_plain_baseline_toggle_defaults_off() -> None:
    default_settings = SoftPrefixSettings.from_dict({"model_name": "Qwen/Qwen3.5-4B"})
    enabled_settings = SoftPrefixSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "eval_plain_baseline": True,
        }
    )

    assert default_settings.eval_plain_baseline is False
    assert enabled_settings.eval_plain_baseline is True


def test_softprefix_strip_trajectory_thoughts_toggle_defaults_off() -> None:
    default_settings = SoftPrefixSettings.from_dict({"model_name": "Qwen/Qwen3.5-4B"})
    enabled_settings = SoftPrefixSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "strip_trajectory_thoughts": True,
        }
    )

    assert default_settings.strip_trajectory_thoughts is False
    assert enabled_settings.strip_trajectory_thoughts is True


def test_trajectory_rollouts_per_task_defaults_to_one_and_rejects_invalid() -> None:
    default_settings = SoftPrefixSettings.from_dict({"model_name": "Qwen/Qwen3.5-4B"})
    repeated_settings = SoftPrefixSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "trajectory_rollouts_per_task": 3,
        }
    )

    assert default_settings.trajectory_rollouts_per_task == 1
    assert repeated_settings.trajectory_rollouts_per_task == 3
    with pytest.raises(ValueError, match="trajectory_rollouts_per_task"):
        SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "trajectory_rollouts_per_task": 0,
            }
        )


def test_expand_trajectory_rollout_items_suffixes_repeated_attempt_ids() -> None:
    items = [
        {"id": "train_0000", "question": "q0"},
        {"id": "train_0001", "question": "q1"},
    ]

    expanded = _expand_trajectory_rollout_items(items, rollouts_per_task=2)

    assert [item["id"] for item in expanded] == [
        "train_0000__sample_00",
        "train_0000__sample_01",
        "train_0001__sample_00",
        "train_0001__sample_01",
    ]
    assert [item["trajectory_source_id"] for item in expanded] == [
        "train_0000",
        "train_0000",
        "train_0001",
        "train_0001",
    ]
    assert _expand_trajectory_rollout_items(items, rollouts_per_task=1) == items


@pytest.mark.parametrize(
    ("raw_backend", "expected"),
    [
        ("", "openai"),
        ("openai", "openai"),
        ("openai_chat", "openai"),
        ("openai_compatible", "openai"),
        ("vllm", "vllm"),
    ],
)
def test_trajectory_rollout_backend_aliases(raw_backend: str, expected: str) -> None:
    assert _normalize_trajectory_rollout_backend(raw_backend) == expected


def test_explicit_alfworld_rollout_dir_is_shared_across_student_outputs(tmp_path: Path) -> None:
    shared_cache = tmp_path / "teacher_gpt55_alfworld"
    settings = SoftPrefixSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "trajectory_rollout_backend": "openai_compatible",
            "trajectory_rollout_dir": str(shared_cache),
        }
    )

    first_student_dir = _resolve_alfworld_trajectory_rollout_dir(
        settings,
        out_root=str(tmp_path / "student_qwen_4b"),
    )
    second_student_dir = _resolve_alfworld_trajectory_rollout_dir(
        settings,
        out_root=str(tmp_path / "student_qwen_9b"),
    )

    assert first_student_dir == str(shared_cache.resolve())
    assert second_student_dir == str(shared_cache.resolve())


def test_alfworld_chat_rollout_chunks_pending_items_by_env_workers(tmp_path: Path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    items = [
        {
            "id": f"train_{idx:04d}",
            "gamefile": f"/dummy/train/pick_and_place/{idx}/game.tw-pddl",
        }
        for idx in range(5)
    ]
    env_calls: list[dict] = []
    rollout_calls: list[dict] = []

    def fake_build_alfworld_env(**kwargs):
        env_calls.append(kwargs)
        return {"env_num": kwargs["env_num"]}

    def fake_run_alfworld_batch(env_manager, **kwargs):
        rollout_calls.append({"env": env_manager, **kwargs})
        return [
            {
                "id": result_id,
                "hard": 1,
                "soft": 1.0,
                "n_turns": 1,
                "fail_reason": "",
            }
            for result_id in kwargs["result_ids"]
        ]

    monkeypatch.setattr(trainer, "build_alfworld_env", fake_build_alfworld_env)
    monkeypatch.setattr(trainer, "run_alfworld_batch", fake_run_alfworld_batch)

    results = _run_alfworld_chat_rollout(
        items=items,
        out_root=str(tmp_path),
        cfg={"seed": 7, "workers": 2, "max_api_workers": 4, "max_steps": 3},
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "trajectory_rollout_backend": "openai_compatible",
            }
        ),
    )

    assert [call["env_num"] for call in env_calls] == [2, 2, 1]
    assert [call["specific_gamefiles"] for call in env_calls] == [
        [items[0]["gamefile"], items[1]["gamefile"]],
        [items[2]["gamefile"], items[3]["gamefile"]],
        [items[4]["gamefile"]],
    ]
    assert [call["max_api_workers"] for call in rollout_calls] == [2, 2, 1]
    assert [call["result_ids"] for call in rollout_calls] == [
        ["train_0000", "train_0001"],
        ["train_0002", "train_0003"],
        ["train_0004"],
    ]
    assert [row["id"] for row in results] == [item["id"] for item in items]


def test_alfworld_chat_rollout_uses_vllm_inference_backend(tmp_path: Path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, trust_remote_code=False):
            del model_name, trust_remote_code
            return "fake-tokenizer"

    class FakeVllmClient:
        def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
            self.base_url = base_url
            self.timeout_seconds = timeout_seconds
            self.prefix = None

        def set_prefix(self, prefix_embeddings) -> None:
            self.prefix = prefix_embeddings

    captured: dict = {}

    def fake_run_alfworld_batch(env_manager, **kwargs):
        del env_manager
        captured.update(kwargs)
        prompt = kwargs["prompt_renderer"]("system", "user")
        return [
            {
                "id": kwargs["result_ids"][0],
                "hard": 1,
                "soft": 1.0,
                "n_turns": 1,
                "fail_reason": "",
            }
        ]

    def fake_template(tokenizer, messages, *, enable_thinking=False):
        captured["template_args"] = (tokenizer, messages, enable_thinking)
        return "rendered prompt"

    monkeypatch.setattr(trainer, "build_alfworld_env", lambda **kwargs: object())
    monkeypatch.setattr(trainer, "run_alfworld_batch", fake_run_alfworld_batch)
    monkeypatch.setattr(trainer, "_import_torch_and_transformers", lambda: (None, None, FakeAutoTokenizer))
    monkeypatch.setattr(trainer, "_apply_text_chat_template", fake_template)
    monkeypatch.setattr("skillopt.softprefix.vllm_prompt_embeds.SoftPrefixVllmClient", FakeVllmClient)

    _run_alfworld_chat_rollout(
        items=[{"id": "train_0000", "gamefile": "/dummy/train/pick_and_place/0/game.tw-pddl"}],
        out_root=str(tmp_path),
        cfg={"seed": 7, "workers": 1, "max_steps": 1, "target_backend": "qwen_chat"},
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "trust_remote_code": True,
                "inference_backend": "vllm_prompt_embeds",
                "inference_base_url": "http://127.0.0.1:8010",
                "inference_timeout_seconds": 12.5,
                "max_prompt_tokens": 123,
            }
        ),
    )

    assert isinstance(captured["generator"], FakeVllmClient)
    assert captured["generator"].base_url == "http://127.0.0.1:8010"
    assert captured["generator"].timeout_seconds == 12.5
    assert captured["generator"].prefix == []
    assert captured["max_prompt_tokens"] == 123
    assert captured["template_args"] == (
        "fake-tokenizer",
        [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
        False,
    )


def test_alfworld_chat_rollout_caps_action_generation_with_target_qwen_max_tokens(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from skillopt.softprefix import trainer

    captured: dict = {}

    def fake_run_alfworld_batch(env_manager, **kwargs):
        del env_manager
        captured.update(kwargs)
        return [
            {
                "id": kwargs["result_ids"][0],
                "hard": 1,
                "soft": 1.0,
                "n_turns": 1,
                "fail_reason": "",
            }
        ]

    monkeypatch.setattr(trainer, "build_alfworld_env", lambda **kwargs: object())
    monkeypatch.setattr(trainer, "run_alfworld_batch", fake_run_alfworld_batch)

    _run_alfworld_chat_rollout(
        items=[{"id": "train_0000", "gamefile": "/dummy/train/pick_and_place/0/game.tw-pddl"}],
        out_root=str(tmp_path),
        cfg={
            "seed": 7,
            "workers": 1,
            "max_steps": 1,
            "target_qwen_chat_max_tokens": 512,
        },
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "trajectory_max_new_tokens": 32768,
            }
        ),
    )

    assert captured["max_completion_tokens"] == 512


def test_alfworld_chat_rollout_passes_skill_content_when_enabled(tmp_path: Path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    captured: dict = {}

    def fake_run_alfworld_batch(env_manager, **kwargs):
        del env_manager
        captured.update(kwargs)
        return [
            {
                "id": kwargs["result_ids"][0],
                "hard": 1,
                "soft": 1.0,
                "n_turns": 1,
                "fail_reason": "",
            }
        ]

    monkeypatch.setattr(trainer, "build_alfworld_env", lambda **kwargs: object())
    monkeypatch.setattr(trainer, "run_alfworld_batch", fake_run_alfworld_batch)

    _run_alfworld_chat_rollout(
        items=[{"id": "train_0000", "gamefile": "/dummy/train/pick_and_place/0/game.tw-pddl"}],
        out_root=str(tmp_path),
        cfg={"seed": 7, "workers": 1, "max_steps": 1},
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "trajectory_use_skill": True,
            }
        ),
        skill_content="# Best Skill\nUse containers before sinks.",
    )

    assert captured["skill_content"] == "# Best Skill\nUse containers before sinks."
    assert captured["fallback_on_invalid_response"] is False


def test_alfworld_chat_rollout_uses_repeated_attempt_result_ids(tmp_path: Path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    rollout_calls: list[dict] = []

    def fake_run_alfworld_batch(env_manager, **kwargs):
        del env_manager
        rollout_calls.append(kwargs)
        return [
            {
                "id": result_id,
                "hard": 1,
                "soft": 1.0,
                "n_turns": 1,
                "fail_reason": "",
            }
            for result_id in kwargs["result_ids"]
        ]

    monkeypatch.setattr(trainer, "build_alfworld_env", lambda **kwargs: object())
    monkeypatch.setattr(trainer, "run_alfworld_batch", fake_run_alfworld_batch)

    expanded_items = _expand_trajectory_rollout_items(
        [{"id": "train_0000", "gamefile": "/dummy/train/pick_and_place/0/game.tw-pddl"}],
        rollouts_per_task=2,
    )
    results = _run_alfworld_chat_rollout(
        items=expanded_items,
        out_root=str(tmp_path),
        cfg={"seed": 7, "workers": 4, "max_steps": 1},
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "trajectory_rollouts_per_task": 2,
            }
        ),
    )

    assert [call["result_ids"] for call in rollout_calls] == [["train_0000__sample_00", "train_0000__sample_01"]]
    assert [row["id"] for row in results] == ["train_0000__sample_00", "train_0000__sample_01"]


def test_init_prefix_eval_only_skips_training_setup(tmp_path: Path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    class FakeModel:
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
        device = "cpu"

    class FakeDataLoader:
        def setup(self, cfg):
            pass

        def build_eval_batch(self, env_num: int, split: str, seed: int):
            return type(
                "Batch",
                (),
                {
                    "payload": [
                        {
                            "id": f"{split}_{idx}",
                            "gamefile": f"/dummy/{split}/{idx}/game.tw-pddl",
                        }
                        for idx in range(env_num)
                    ]
                },
            )()

    calls: list[tuple[str, str, int]] = []

    def fake_evaluate(env, prefix_model, items, *, cfg, settings, out_dir, desc):
        calls.append((desc, Path(out_dir).relative_to(tmp_path).as_posix(), len(items)))
        return 0.25, 0.5, []

    def fail_collect_trajectories(**kwargs):
        raise AssertionError("eval-only init prefix should not collect training trajectories")

    def fail_save_checkpoint(*args, **kwargs):
        raise AssertionError("eval-only init prefix should not save checkpoints")

    monkeypatch.setattr(trainer, "_set_seed", lambda seed: None)
    monkeypatch.setattr(trainer, "_build_dataloader", lambda env, cfg, seed: FakeDataLoader())
    monkeypatch.setattr(trainer, "_evaluate_prefix", fake_evaluate)
    monkeypatch.setattr(trainer, "_collect_alfworld_trajectory_examples", fail_collect_trajectories)

    summary = _train_adapter(
        cfg={
            "env": "alfworld",
            "out_root": str(tmp_path),
            "seed": 42,
            "num_epochs": 0,
            "sel_env_num": 2,
            "test_env_num": 3,
            "eval_test": True,
        },
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "eval_init_prefix": True,
                "training_data": "trajectory_sft",
            }
        ),
        adapter_model=FakeModel(),
        best_path=str(tmp_path / "best_prefix.pt"),
        latest_path=str(tmp_path / "latest_prefix.pt"),
        best_summary_key="best_prefix_path",
        latest_summary_key="latest_prefix_path",
        save_checkpoint=fail_save_checkpoint,
        load_checkpoint=lambda *args, **kwargs: None,
    )

    assert calls == [
        ("  Init Val", "eval/init/valid_seen", 2),
        ("  Init Test", "eval/init/valid_unseen", 3),
    ]
    assert summary["init_valid_seen_hard"] == 0.25
    assert summary["init_valid_seen_soft"] == 0.5
    assert summary["init_test_hard"] == 0.25
    assert summary["init_test_soft"] == 0.5
    assert summary["history"] == []


def test_init_prefix_eval_can_skip_validation_eval(tmp_path: Path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    class FakeModel:
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
        device = "cpu"

    class FakeDataLoader:
        def setup(self, cfg):
            pass

        def build_eval_batch(self, env_num: int, split: str, seed: int):
            return type(
                "Batch",
                (),
                {
                    "payload": [
                        {
                            "id": f"{split}_{idx}",
                            "gamefile": f"/dummy/{split}/{idx}/game.tw-pddl",
                        }
                        for idx in range(env_num)
                    ]
                },
            )()

    calls: list[tuple[str, str, int]] = []

    def fake_evaluate(env, prefix_model, items, *, cfg, settings, out_dir, desc):
        calls.append((desc, Path(out_dir).relative_to(tmp_path).as_posix(), len(items)))
        return 0.25, 0.5, []

    def fail_collect_trajectories(**kwargs):
        raise AssertionError("eval-only init prefix should not collect training trajectories")

    def fail_save_checkpoint(*args, **kwargs):
        raise AssertionError("eval-only init prefix should not save checkpoints")

    monkeypatch.setattr(trainer, "_set_seed", lambda seed: None)
    monkeypatch.setattr(trainer, "_build_dataloader", lambda env, cfg, seed: FakeDataLoader())
    monkeypatch.setattr(trainer, "_evaluate_prefix", fake_evaluate)
    monkeypatch.setattr(trainer, "_collect_alfworld_trajectory_examples", fail_collect_trajectories)

    summary = _train_adapter(
        cfg={
            "env": "alfworld",
            "out_root": str(tmp_path),
            "seed": 42,
            "num_epochs": 0,
            "sel_env_num": 2,
            "test_env_num": 3,
            "eval_test": True,
        },
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "eval_init_prefix": True,
                "eval_init_val": False,
                "training_data": "trajectory_sft",
            }
        ),
        adapter_model=FakeModel(),
        best_path=str(tmp_path / "best_prefix.pt"),
        latest_path=str(tmp_path / "latest_prefix.pt"),
        best_summary_key="best_prefix_path",
        latest_summary_key="latest_prefix_path",
        save_checkpoint=fail_save_checkpoint,
        load_checkpoint=lambda *args, **kwargs: None,
    )

    assert calls == [("  Init Test", "eval/init/valid_unseen", 3)]
    assert "init_valid_seen_hard" not in summary
    assert "init_valid_seen_soft" not in summary
    assert summary["init_test_hard"] == 0.25
    assert summary["init_test_soft"] == 0.5
    assert summary["history"] == []


def test_checkpoint_eval_only_loads_prefix_and_skips_training_setup(tmp_path: Path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    checkpoint_path = tmp_path / "trained" / "best_prefix.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    class FakeModel:
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
        device = "cpu"

    class FakeDataLoader:
        def setup(self, cfg):
            pass

        def build_eval_batch(self, env_num: int, split: str, seed: int):
            return type(
                "Batch",
                (),
                {
                    "payload": [
                        {
                            "id": f"{split}_{idx}",
                            "gamefile": f"/dummy/{split}/{idx}/game.tw-pddl",
                        }
                        for idx in range(env_num)
                    ]
                },
            )()

    calls: list[tuple[str, str, int]] = []
    loaded_paths: list[str] = []

    def fake_evaluate(env, prefix_model, items, *, cfg, settings, out_dir, desc):
        calls.append((desc, Path(out_dir).relative_to(tmp_path).as_posix(), len(items)))
        return 0.75, 0.875, []

    def fail_collect_trajectories(**kwargs):
        raise AssertionError("checkpoint eval-only should not collect training trajectories")

    def fail_save_checkpoint(*args, **kwargs):
        raise AssertionError("checkpoint eval-only should not save checkpoints")

    monkeypatch.setattr(trainer, "_set_seed", lambda seed: None)
    monkeypatch.setattr(trainer, "_build_dataloader", lambda env, cfg, seed: FakeDataLoader())
    monkeypatch.setattr(trainer, "_evaluate_prefix", fake_evaluate)
    monkeypatch.setattr(trainer, "_collect_alfworld_trajectory_examples", fail_collect_trajectories)

    summary = _train_adapter(
        cfg={
            "env": "alfworld",
            "out_root": str(tmp_path),
            "seed": 42,
            "num_epochs": 0,
            "sel_env_num": 2,
            "test_env_num": 3,
            "eval_test": True,
        },
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "checkpoint_path": str(checkpoint_path),
                "training_data": "trajectory_sft",
            }
        ),
        adapter_model=FakeModel(),
        best_path=str(tmp_path / "best_prefix.pt"),
        latest_path=str(tmp_path / "latest_prefix.pt"),
        best_summary_key="best_prefix_path",
        latest_summary_key="latest_prefix_path",
        save_checkpoint=fail_save_checkpoint,
        load_checkpoint=lambda _torch, _model, path: loaded_paths.append(path),
    )

    assert loaded_paths == [str(checkpoint_path)]
    assert calls == [
        ("  Checkpoint Val", "eval/checkpoint/valid_seen", 2),
        ("  Checkpoint Test", "eval/checkpoint/valid_unseen", 3),
    ]
    assert summary["best_prefix_path"] == str(checkpoint_path)
    assert summary["checkpoint_valid_seen_hard"] == 0.75
    assert summary["checkpoint_valid_seen_soft"] == 0.875
    assert summary["checkpoint_test_hard"] == 0.75
    assert summary["checkpoint_test_soft"] == 0.875
    assert summary["history"] == []


def test_train_adapter_rewinds_to_best_checkpoint_after_rejected_epoch(
    tmp_path: Path, monkeypatch
) -> None:
    from skillopt.softprefix import trainer

    class FakeLoss:
        def __init__(self, value: float):
            self.value = value

        def __truediv__(self, _other):
            return self

        def backward(self):
            pass

        def detach(self):
            return self

        def cpu(self):
            return self

        def __float__(self):
            return self.value

    class FakeTorch:
        class Generator:
            def manual_seed(self, seed):
                pass

        class optim:
            class AdamW:
                def __init__(self, params, lr, weight_decay):
                    pass

                def zero_grad(self, set_to_none=True):
                    pass

                def step(self):
                    pass

        class utils:
            class data:
                class DataLoader:
                    def __init__(self, dataset, batch_size, shuffle, collate_fn, generator):
                        self.dataset = list(dataset)
                        self.collate_fn = collate_fn

                    def __iter__(self):
                        for item in self.dataset:
                            yield self.collate_fn([item])

                    def __len__(self):
                        return len(self.dataset)

    class FakeModel:
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
        device = "cpu"

        def trainable_parameters(self):
            return []

        def forward(self, _batch):
            return type("Outputs", (), {"loss": FakeLoss(1.0)})()

    class FakeDataLoader:
        def setup(self, cfg):
            pass

        def build_eval_batch(self, env_num: int, split: str, seed: int):
            size = env_num if env_num > 0 else 1
            return type(
                "Batch",
                (),
                {
                    "payload": [
                        {"id": f"{split}_{idx}", "gamefile": f"/dummy/{split}/{idx}/game.tw-pddl"}
                        for idx in range(size)
                    ]
                },
            )()

    val_scores = iter([(0.0, 0.8), (0.0, 0.7), (0.0, 0.8)])
    loaded_paths: list[str] = []
    saved_paths: list[str] = []
    best_path = str(tmp_path / "best_prefix.pt")
    latest_path = str(tmp_path / "latest_prefix.pt")

    def fake_evaluate(env, prefix_model, items, *, cfg, settings, out_dir, desc):
        hard, soft = next(val_scores)
        return hard, soft, []

    def fake_save_checkpoint(_torch, _model, path):
        saved_paths.append(path)
        Path(path).write_text("checkpoint", encoding="utf-8")

    monkeypatch.setattr(trainer, "_set_seed", lambda seed: None)
    monkeypatch.setattr(trainer, "_import_torch_and_transformers", lambda: (FakeTorch, None, None))
    monkeypatch.setattr(trainer, "_build_dataloader", lambda env, cfg, seed: FakeDataLoader())
    monkeypatch.setattr(trainer, "_build_dataset", lambda env, train_items, adapter_model, cfg, settings: [object()])
    monkeypatch.setattr(trainer, "_batch_to_tensors", lambda torch_mod, batch, device: batch)
    monkeypatch.setattr(trainer, "PrefixBatchCollator", lambda pad_token_id: lambda batch: batch)
    monkeypatch.setattr(trainer, "_evaluate_prefix", fake_evaluate)

    summary = _train_adapter(
        cfg={
            "env": "alfworld",
            "out_root": str(tmp_path),
            "seed": 42,
            "num_epochs": 2,
            "batch_size": 1,
            "accumulation": 1,
            "sel_env_num": 1,
            "test_env_num": 1,
            "gate_metric": "soft",
            "eval_test": True,
        },
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "rewind_ckpt": True,
            }
        ),
        adapter_model=FakeModel(),
        best_path=best_path,
        latest_path=latest_path,
        best_summary_key="best_prefix_path",
        latest_summary_key="latest_prefix_path",
        save_checkpoint=fake_save_checkpoint,
        load_checkpoint=lambda _torch, _model, path: loaded_paths.append(path),
    )

    assert summary["history"][1]["action"] == "reject"
    assert loaded_paths == [best_path, best_path]
    assert saved_paths == [best_path, latest_path, latest_path]


def test_plain_baseline_eval_only_skips_prefix_training_setup(tmp_path: Path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    class FakeModel:
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
        device = "cpu"

    class FakeDataLoader:
        def setup(self, cfg):
            pass

        def build_eval_batch(self, env_num: int, split: str, seed: int):
            return type(
                "Batch",
                (),
                {
                    "payload": [
                        {
                            "id": f"{split}_{idx}",
                            "gamefile": f"/dummy/{split}/{idx}/game.tw-pddl",
                        }
                        for idx in range(env_num)
                    ]
                },
            )()

    calls: list[tuple[str, str, int]] = []

    def fake_plain_baseline(env, prefix_model, items, *, cfg, settings, out_dir, desc):
        calls.append((desc, Path(out_dir).relative_to(tmp_path).as_posix(), len(items)))
        return 0.0, 0.125, []

    def fail_collect_trajectories(**kwargs):
        raise AssertionError("plain baseline eval-only should not collect training trajectories")

    def fail_save_checkpoint(*args, **kwargs):
        raise AssertionError("plain baseline eval-only should not save checkpoints")

    monkeypatch.setattr(trainer, "_set_seed", lambda seed: None)
    monkeypatch.setattr(trainer, "_build_dataloader", lambda env, cfg, seed: FakeDataLoader())
    monkeypatch.setattr(trainer, "_evaluate_plain_baseline", fake_plain_baseline, raising=False)
    monkeypatch.setattr(trainer, "_collect_alfworld_trajectory_examples", fail_collect_trajectories)

    summary = _train_adapter(
        cfg={
            "env": "alfworld",
            "out_root": str(tmp_path),
            "seed": 42,
            "num_epochs": 0,
            "sel_env_num": 2,
            "test_env_num": 3,
            "eval_test": True,
        },
        settings=SoftPrefixSettings.from_dict(
            {
                "model_name": "Qwen/Qwen3.5-4B",
                "eval_plain_baseline": True,
                "training_data": "trajectory_sft",
            }
        ),
        adapter_model=FakeModel(),
        best_path=str(tmp_path / "best_prefix.pt"),
        latest_path=str(tmp_path / "latest_prefix.pt"),
        best_summary_key="best_prefix_path",
        latest_summary_key="latest_prefix_path",
        save_checkpoint=fail_save_checkpoint,
        load_checkpoint=lambda *args, **kwargs: None,
    )

    assert calls == [
        ("  Plain Val", "eval/plain/valid_seen", 2),
        ("  Plain Test", "eval/plain/valid_unseen", 3),
    ]
    assert summary["plain_valid_seen_hard"] == 0.0
    assert summary["plain_valid_seen_soft"] == 0.125
    assert summary["plain_test_hard"] == 0.0
    assert summary["plain_test_soft"] == 0.125
    assert summary["history"] == []


def test_timed_out_alfworld_cache_row_is_extendable_when_max_steps_increases(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    pred_dir = rollout_dir / "predictions" / "train_0000"
    pred_dir.mkdir(parents=True)
    (pred_dir / "conversation.json").write_text(
        json.dumps(
            [
                {"step": -1, "type": "initial_obs", "obs_text": "initial"},
                {
                    "step": 0,
                    "action": "look",
                    "model_response": "<think>x</think><action>look</action>",
                    "done": False,
                },
            ]
        )
    )
    row = {
        "id": "train_0000",
        "hard": 0,
        "soft": 0.0,
        "n_turns": 2,
        "fail_reason": "Timeout after 5 steps",
        "rollout_backend": "openai",
    }

    assert _should_extend_alfworld_cached_rollout(
        row,
        rollout_dir=str(rollout_dir),
        rollout_backend="openai",
        cached_max_steps=5,
        target_max_steps=20,
    )


def test_successful_alfworld_cache_row_is_not_extendable(tmp_path: Path) -> None:
    row = {
        "id": "train_0000",
        "hard": 1,
        "soft": 1.0,
        "n_turns": 2,
        "fail_reason": "",
        "rollout_backend": "openai",
    }

    assert not _should_extend_alfworld_cached_rollout(
        row,
        rollout_dir=str(tmp_path),
        rollout_backend="openai",
        cached_max_steps=5,
        target_max_steps=20,
    )


def test_alfworld_trajectory_examples_strip_think_blocks_when_enabled(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    pred_dir = rollout_dir / "predictions" / "train_0000"
    pred_dir.mkdir(parents=True)
    (pred_dir / "conversation.json").write_text(
        json.dumps(
            [
                {"step": -1, "type": "initial_obs", "obs_text": "initial"},
                {
                    "step": 0,
                    "prompt": "prompt 0",
                    "model_response": "<think>inspect room</think><action>look</action>",
                },
                {
                    "step": 1,
                    "prompt": "prompt 1",
                    "model_response": "<think>open it</think>\n<action>open fridge</action>",
                },
            ]
        )
    )
    settings = SoftPrefixSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "strip_trajectory_thoughts": True,
        }
    )

    examples = _build_alfworld_trajectory_examples(
        str(rollout_dir),
        [{"id": "train_0000", "hard": 1, "soft": 1.0}],
        settings,
    )

    assert [example["target"] for example in examples] == [
        "<action>look</action>",
        "<action>open fridge</action>",
    ]


def test_alfworld_trajectory_examples_extract_action_when_think_tag_is_missing(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    pred_dir = rollout_dir / "predictions" / "train_0000"
    pred_dir.mkdir(parents=True)
    (pred_dir / "conversation.json").write_text(
        json.dumps(
            [
                {"step": -1, "type": "initial_obs", "obs_text": "initial"},
                {
                    "step": 0,
                    "prompt": "prompt 0",
                    "model_response": (
                        "The goal is to cool the pot.\n"
                        "I should remove it from the burner.\n\n"
                        "<action> take pot 1 from stoveburner 3 </action>"
                    ),
                },
            ]
        )
    )
    settings = SoftPrefixSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "strip_trajectory_thoughts": True,
        }
    )

    examples = _build_alfworld_trajectory_examples(
        str(rollout_dir),
        [{"id": "train_0000", "hard": 1, "soft": 1.0}],
        settings,
    )

    assert examples[0]["target"] == "<action>take pot 1 from stoveburner 3</action>"


def test_alfworld_trajectory_examples_include_skill_prompt_when_enabled(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    pred_dir = rollout_dir / "predictions" / "train_0000"
    pred_dir.mkdir(parents=True)
    (pred_dir / "conversation.json").write_text(
        json.dumps(
            [
                {"step": -1, "type": "initial_obs", "obs_text": "initial"},
                {
                    "step": 0,
                    "prompt": "You are in a kitchen.",
                    "model_response": "<think>look</think><action>look</action>",
                },
            ]
        )
    )
    settings = SoftPrefixSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "trajectory_use_skill": True,
        }
    )

    examples = _build_alfworld_trajectory_examples(
        str(rollout_dir),
        [{"id": "train_0000", "hard": 1, "soft": 1.0}],
        settings,
        skill_content="# Best Skill\nOpen receptacles before placing objects.",
    )

    assert "## Skill Knowledge" in examples[0]["messages"][1]["content"]
    assert "Open receptacles before placing objects." in examples[0]["messages"][1]["content"]
    assert examples[0]["messages"][1]["content"].endswith("You are in a kitchen.")


def test_alfworld_skill_section_examples_replace_cached_skill_with_marker(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    pred_dir = rollout_dir / "predictions" / "train_0000"
    pred_dir.mkdir(parents=True)
    cached_prompt = (
        "\n\n## Skill Knowledge\n"
        "Below is a skill document with learned strategies. Use these guidelines to inform your decisions:\n\n"
        "# Best Skill\nOpen receptacles before placing objects.\n\n"
        "You are in a kitchen."
    )
    (pred_dir / "conversation.json").write_text(
        json.dumps(
            [
                {"step": -1, "type": "initial_obs", "obs_text": "initial"},
                {
                    "step": 0,
                    "prompt": cached_prompt,
                    "model_response": "<think>look</think><action>look</action>",
                },
            ]
        )
    )
    settings = SoftPrefixSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "injection_position": "skill_section",
        }
    )

    examples = _build_alfworld_trajectory_examples(
        str(rollout_dir),
        [{"id": "train_0000", "hard": 1, "soft": 1.0}],
        settings,
        skill_content="# Best Skill\nOpen receptacles before placing objects.",
    )

    prompt = examples[0]["messages"][1]["content"]
    assert prompt.count("<|skillopt_soft_prefix_insert|>") == 1
    assert "Open receptacles before placing objects." not in prompt
    assert prompt.endswith("You are in a kitchen.")
