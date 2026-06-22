"""Tests for ALFWorld rollout logging."""
from __future__ import annotations

import json

import pytest

from skillopt.envs.alfworld import rollout


class FakeEnvManager:
    def reset(self, payload):
        return (
            {
                "text": ["observation"],
                "anchor": ["Your task is to: test task"],
            },
            [{"extra.gamefile": "/tmp/alfworld/pick_and_place/game.tw-pddl"}],
        )

    def step(self, actions):
        return (
            {
                "text": ["after action"],
                "anchor": ["not done"],
            },
            [0.0],
            [True],
            [{"won": False}],
        )


def test_chat_target_errors_are_logged_during_rollout(tmp_path, monkeypatch, capsys) -> None:
    def raise_api_error(**kwargs):
        raise RuntimeError("rate limit")

    monkeypatch.setattr(rollout, "chat_target", raise_api_error)

    rollout.run_alfworld_batch(
        FakeEnvManager(),
        skill_content="",
        max_steps=1,
        out_root=str(tmp_path),
        max_api_workers=1,
        result_ids=["train_0000"],
    )

    captured = capsys.readouterr()
    assert "[alfworld rollout] target API error" in captured.err
    assert "id=train_0000" in captured.err
    assert "step=0" in captured.err
    assert "RuntimeError: rate limit" in captured.err


def test_chat_target_timeout_is_configurable_during_rollout(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    def return_action(**kwargs):
        captured.update(kwargs)
        return "<think>ok</think><action>look</action>", {}

    monkeypatch.setattr(rollout, "chat_target", return_action)

    rollout.run_alfworld_batch(
        FakeEnvManager(),
        skill_content="",
        max_steps=1,
        out_root=str(tmp_path),
        max_api_workers=1,
        result_ids=["train_0000"],
        api_timeout_seconds=123.0,
    )

    assert captured["timeout"] == 123.0


def test_rollout_uses_generator_backend_when_provided(tmp_path, monkeypatch) -> None:
    class TwoEnvManager:
        def reset(self, payload):
            return (
                {
                    "text": ["obs 0", "obs 1"],
                    "anchor": ["Your task is to: task 0", "Your task is to: task 1"],
                },
                [
                    {"extra.gamefile": "/tmp/alfworld/pick_and_place/0/game.tw-pddl"},
                    {"extra.gamefile": "/tmp/alfworld/pick_and_place/1/game.tw-pddl"},
                ],
            )

        def step(self, actions):
            assert actions == [
                "<think>ok 0</think><action>look</action>",
                "<think>ok 1</think><action>look</action>",
            ]
            return (
                {"anchor": ["done 0", "done 1"]},
                [1.0, 1.0],
                [True, True],
                [{"won": True}, {"won": True}],
            )

    class FakeGenerator:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def generate_from_prompts(self, prompts, *, max_prompt_tokens, max_new_tokens, temperature):
            assert max_prompt_tokens == 128
            assert max_new_tokens == 8
            assert temperature == 0.25
            self.calls.append(list(prompts))
            return [
                f"<think>ok {idx}</think><action>look</action>"
                for idx, _prompt in enumerate(prompts)
            ]

    def fail_chat_target(**kwargs):
        raise AssertionError("chat_target should not be used with a generator backend")

    monkeypatch.setattr(rollout, "chat_target", fail_chat_target)
    generator = FakeGenerator()

    results = rollout.run_alfworld_batch(
        TwoEnvManager(),
        skill_content="",
        max_steps=1,
        out_root=str(tmp_path),
        temperature=0.25,
        max_completion_tokens=8,
        result_ids=["train_0000", "train_0001"],
        generator=generator,
        prompt_renderer=lambda system, user: f"{system}\n{user}",
        max_prompt_tokens=128,
    )

    assert generator.calls == [[
        "You are an expert agent operating in the ALFRED Embodied Environment.\nobs 0",
        "You are an expert agent operating in the ALFRED Embodied Environment.\nobs 1",
    ]]
    assert [row["hard"] for row in results] == [1, 1]


def test_rollout_can_raise_on_generator_response_missing_action_tag(tmp_path) -> None:
    class BadGenerator:
        def __init__(self) -> None:
            self.call_count = 0

        def generate_from_prompts(self, prompts, *, max_prompt_tokens, max_new_tokens, temperature):
            del prompts, max_prompt_tokens, max_new_tokens, temperature
            self.call_count += 1
            return ["I should inspect the room but forgot the action tag."]

    generator = BadGenerator()
    with pytest.raises(ValueError, match="missing <action> tag"):
        rollout.run_alfworld_batch(
            FakeEnvManager(),
            skill_content="",
            max_steps=1,
            out_root=str(tmp_path),
            result_ids=["train_0000"],
            generator=generator,
            fallback_on_invalid_response=False,
            invalid_response_retries=0,
        )

    assert generator.call_count == 1


def test_rollout_can_raise_on_chat_response_missing_action_tag(tmp_path, monkeypatch) -> None:
    call_count = {"n": 0}

    def return_bad_response(**kwargs):
        del kwargs
        call_count["n"] += 1
        return "I should inspect the room but forgot the action tag.", {}

    monkeypatch.setattr(rollout, "chat_target", return_bad_response)

    with pytest.raises(ValueError, match="missing <action> tag"):
        rollout.run_alfworld_batch(
            FakeEnvManager(),
            skill_content="",
            max_steps=1,
            out_root=str(tmp_path),
            max_api_workers=1,
            result_ids=["train_0000"],
            fallback_on_invalid_response=False,
            invalid_response_retries=0,
        )

    assert call_count["n"] == 1


def test_rollout_retries_until_action_is_extracted(tmp_path, monkeypatch) -> None:
    call_count = {"n": 0}

    def return_eventually_valid_response(**kwargs):
        del kwargs
        call_count["n"] += 1
        if call_count["n"] < 3:
            return "still missing the action tag", {}
        return "<think>ok</think><action>look</action>", {}

    monkeypatch.setattr(rollout, "chat_target", return_eventually_valid_response)

    rollout.run_alfworld_batch(
        FakeEnvManager(),
        skill_content="",
        max_steps=1,
        out_root=str(tmp_path),
        max_api_workers=1,
        result_ids=["train_0000"],
        invalid_response_retries=3,
    )

    assert call_count["n"] == 3


def test_invalid_model_response_is_logged_before_raise(tmp_path, monkeypatch, capsys) -> None:
    bad_response = "x" * 600

    def return_bad_response(**kwargs):
        del kwargs
        return bad_response, {}

    monkeypatch.setattr(rollout, "chat_target", return_bad_response)

    with pytest.raises(ValueError, match="missing <action> tag"):
        rollout.run_alfworld_batch(
            FakeEnvManager(),
            skill_content="",
            max_steps=1,
            out_root=str(tmp_path),
            max_api_workers=1,
            result_ids=["train_0000"],
            fallback_on_invalid_response=False,
            invalid_response_retries=0,
        )

    captured = capsys.readouterr()
    assert "[alfworld rollout] invalid model response" in captured.out
    assert "id=train_0000" in captured.out
    assert "step=0" in captured.out
    assert bad_response in captured.out


class CrashingAfterFirstStepEnvManager:
    def __init__(self) -> None:
        self.step_count = 0

    def reset(self, payload):
        return (
            {
                "text": ["observation"],
                "anchor": ["Your task is to: test task"],
            },
            [{"extra.gamefile": "/tmp/alfworld/pick_and_place/game.tw-pddl"}],
        )

    def step(self, actions):
        self.step_count += 1
        if self.step_count == 1:
            return (
                {
                    "text": ["after first action"],
                    "anchor": ["still running"],
                },
                [0.0],
                [False],
                [{"won": False}],
            )
        raise RuntimeError("environment crashed")


def test_conversation_is_saved_incrementally_during_rollout(tmp_path, monkeypatch) -> None:
    def return_action(**kwargs):
        return "<think>try first action</think><action>look</action>", {}

    monkeypatch.setattr(rollout, "chat_target", return_action)

    with pytest.raises(RuntimeError, match="environment crashed"):
        rollout.run_alfworld_batch(
            CrashingAfterFirstStepEnvManager(),
            skill_content="",
            max_steps=2,
            out_root=str(tmp_path),
            max_api_workers=1,
            result_ids=["train_0000"],
        )

    conv_path = tmp_path / "predictions" / "train_0000" / "conversation.json"
    assert conv_path.exists()
    conversation = json.loads(conv_path.read_text())
    assert [record["step"] for record in conversation] == [-1, 0]
    assert conversation[1]["action"] == "look"


class ReplayThenContinueEnvManager:
    def __init__(self) -> None:
        self.actions: list[str] = []

    def reset(self, payload):
        return (
            {
                "text": ["initial prompt"],
                "anchor": ["Your task is to: continue cached task"],
            },
            [{"extra.gamefile": "/tmp/alfworld/pick_and_place/game.tw-pddl"}],
        )

    def step(self, actions):
        action = actions[0]
        self.actions.append(action)
        step_idx = len(self.actions) - 1
        done = step_idx == 2
        return (
            {
                "text": [f"prompt after {action}"],
                "anchor": [f"feedback after {action}"],
            },
            [1.0 if done else 0.0],
            [done],
            [{"won": done}],
        )


def test_cached_conversation_actions_are_replayed_before_continuing(tmp_path, monkeypatch) -> None:
    env_manager = ReplayThenContinueEnvManager()
    api_calls: list[str] = []
    cached_conversation = [
        {"step": -1, "type": "initial_obs", "obs_anchor": "initial", "obs_text": "initial prompt"},
        {
            "step": 0,
            "action": "go to cached room",
            "model_response": "<think>cached 0</think><action>go to cached room</action>",
            "env_feedback": "cached feedback 0",
            "reward": 0.0,
            "done": False,
        },
        {
            "step": 1,
            "action": "open cached object",
            "model_response": "<think>cached 1</think><action>open cached object</action>",
            "env_feedback": "cached feedback 1",
            "reward": 0.0,
            "done": False,
        },
    ]

    def return_live_action(**kwargs):
        api_calls.append(kwargs["user"])
        return "<think>continue after cache</think><action>finish task</action>", {}

    monkeypatch.setattr(rollout, "chat_target", return_live_action)

    results = rollout.run_alfworld_batch(
        env_manager,
        skill_content="",
        max_steps=4,
        out_root=str(tmp_path),
        max_api_workers=1,
        result_ids=["train_0000"],
        cached_conversations={"train_0000": cached_conversation},
    )

    assert api_calls == ["prompt after <think>cached 1</think><action>open cached object</action>"]
    assert env_manager.actions == [
        "<think>cached 0</think><action>go to cached room</action>",
        "<think>cached 1</think><action>open cached object</action>",
        "<think>continue after cache</think><action>finish task</action>",
    ]
    assert results[0]["hard"] == 1
    assert results[0]["n_turns"] == 4

    conv_path = tmp_path / "predictions" / "train_0000" / "conversation.json"
    conversation = json.loads(conv_path.read_text())
    assert [record["step"] for record in conversation] == [-1, 0, 1, 2]
    assert conversation[-1]["action"] == "finish task"
