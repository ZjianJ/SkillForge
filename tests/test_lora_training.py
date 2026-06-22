"""Tests for LoRA baseline training glue."""
from __future__ import annotations

from pathlib import Path


def test_officeqa_lora_config_uses_trajectory_sft() -> None:
    from skillopt.config import load_config

    cfg = load_config("configs/officeqa/lora.yaml")

    assert cfg["env"]["name"] == "officeqa"
    assert cfg["lora"]["model_name"] == "Qwen/Qwen3.5-4B"
    assert cfg["lora"]["training_data"] == "trajectory_sft"
    assert cfg["lora"]["trajectory_rollout_dir"] == "rollouts/teacher_gpt55_officeqa_rollouts"
    assert cfg["lora"]["train_on_final_only"] is False


def test_lora_settings_parse_training_knobs() -> None:
    from skillopt.softprefix.lora import LoraSettings

    settings = LoraSettings.from_dict(
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "learning_rate": 2e-4,
            "r": 16,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"],
            "training_data": "trajectory_sft",
            "trajectory_rollouts_per_task": 3,
            "trajectory_use_skill": True,
            "strip_trajectory_thoughts": True,
            "train_on_final_only": False,
            "docvqa_max_image_tokens": 1024,
            "architecture": "vision_lm",
        }
    )

    assert settings.model_name == "Qwen/Qwen3.5-4B"
    assert settings.learning_rate == 2e-4
    assert settings.r == 16
    assert settings.alpha == 32
    assert settings.dropout == 0.1
    assert settings.target_modules == ["q_proj", "v_proj"]
    assert settings.training_data == "trajectory_sft"
    assert settings.trajectory_rollouts_per_task == 3
    assert settings.trajectory_use_skill is True
    assert settings.strip_trajectory_thoughts is True
    assert settings.train_on_final_only is False
    assert settings.docvqa_max_image_tokens == 1024
    assert settings.architecture == "vision_lm"


def test_train_lora_uses_adapter_checkpoint_names(tmp_path: Path, monkeypatch) -> None:
    from skillopt.softprefix import trainer
    from skillopt.softprefix.lora import LoraSettings

    class FakeModel:
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
        device = "cpu"

        def trainable_parameters(self):
            return []

    def fake_train_adapter(**kwargs):
        assert isinstance(kwargs["settings"], LoraSettings)
        assert kwargs["adapter_model"] is fake_model
        assert kwargs["best_summary_key"] == "best_lora_path"
        assert kwargs["latest_summary_key"] == "latest_lora_path"
        assert kwargs["best_path"].endswith("best_lora")
        assert kwargs["latest_path"].endswith("latest_lora")
        return {"best_lora_path": kwargs["best_path"], "latest_lora_path": kwargs["latest_path"]}

    fake_model = FakeModel()
    monkeypatch.setattr(trainer, "_set_seed", lambda seed: None)
    monkeypatch.setattr(trainer, "_build_lora_model", lambda env, settings: fake_model)
    monkeypatch.setattr(trainer, "_train_adapter", fake_train_adapter)

    summary = trainer.train_lora(
        cfg={"env": "alfworld", "out_root": str(tmp_path), "seed": 42},
        lora_cfg={"model_name": "Qwen/Qwen3.5-4B"},
    )

    assert summary == {
        "best_lora_path": str(tmp_path / "best_lora"),
        "latest_lora_path": str(tmp_path / "latest_lora"),
    }


def test_train_lora_accepts_docvqa_vlm(tmp_path: Path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    class FakeVisionModel:
        processor = object()
        tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
        device = "cpu"

        def trainable_parameters(self):
            return []

    built: dict[str, str] = {}
    fake_model = FakeVisionModel()

    def fake_build_lora_model(env, settings):
        built["env"] = env
        built["architecture"] = settings.architecture
        return fake_model

    def fake_train_adapter(**kwargs):
        assert kwargs["adapter_model"] is fake_model
        return {"best_lora_path": kwargs["best_path"], "latest_lora_path": kwargs["latest_path"]}

    monkeypatch.setattr(trainer, "_set_seed", lambda seed: None)
    monkeypatch.setattr(trainer, "_build_lora_model", fake_build_lora_model)
    monkeypatch.setattr(trainer, "_train_adapter", fake_train_adapter)

    summary = trainer.train_lora(
        cfg={"env": "docvqa", "out_root": str(tmp_path), "seed": 42},
        lora_cfg={"model_name": "Qwen/Qwen3.5-4B", "architecture": "vision_lm"},
    )

    assert built == {"env": "docvqa", "architecture": "vision_lm"}
    assert summary["best_lora_path"] == str(tmp_path / "best_lora")


def test_officeqa_lora_eval_uses_local_tool_generator(tmp_path: Path, monkeypatch) -> None:
    from skillopt.softprefix import trainer

    class FakeLoraModel:
        tokenizer = object()

        def generate_chat_completion(self, messages, **kwargs):
            raise AssertionError("helper should be monkeypatched before generation")

    called = {}

    def fake_eval_with_tools(items, **kwargs):
        called["generator"] = kwargs["generator"]
        called["items"] = items
        return 1.0, 1.0, []

    monkeypatch.setattr(trainer, "_evaluate_officeqa_prefix_with_local_tools", fake_eval_with_tools)

    hard, soft, rows = trainer.evaluate_officeqa_prefix(
        FakeLoraModel(),
        [{"id": "UID0001", "question": "q", "ground_truth": "a"}],
        out_dir=str(tmp_path),
        max_prompt_tokens=128,
        max_new_tokens=32,
        temperature=0.0,
        use_local_tools=True,
    )

    assert (hard, soft, rows) == (1.0, 1.0, [])
    assert isinstance(called["generator"], FakeLoraModel)
