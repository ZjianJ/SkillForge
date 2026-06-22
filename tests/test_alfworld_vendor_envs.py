"""Tests for the vendored ALFWorld process wrapper."""
from __future__ import annotations

import importlib
import sys
import types


def test_specific_gamefile_skips_full_alfworld_collection(monkeypatch) -> None:
    gamefile = "/tmp/alfworld/game.tw-pddl"

    class FakeAlfredTWEnv:
        def __init__(self, config, train_eval="train"):
            self.config = config
            self.train_eval = train_eval
            self.collect_game_files()

        def collect_game_files(self, verbose=False):
            raise AssertionError("specific-game workers should not scan the full split")

    fake_environment = types.ModuleType("alfworld.agents.environment")
    fake_environment.get_environment = lambda env_type: FakeAlfredTWEnv
    fake_agents = types.ModuleType("alfworld.agents")
    fake_agents.environment = fake_environment
    fake_alfworld = types.ModuleType("alfworld")
    fake_alfworld.agents = fake_agents
    fake_gymnasium = types.ModuleType("gymnasium")
    fake_gymnasium.Env = object

    monkeypatch.setitem(sys.modules, "alfworld", fake_alfworld)
    monkeypatch.setitem(sys.modules, "alfworld.agents", fake_agents)
    monkeypatch.setitem(sys.modules, "alfworld.agents.environment", fake_environment)
    monkeypatch.setitem(sys.modules, "gymnasium", fake_gymnasium)
    sys.modules.pop("skillopt.envs.alfworld.vendor.alfworld_envs", None)

    alfworld_envs = importlib.import_module("skillopt.envs.alfworld.vendor.alfworld_envs")

    base_env = alfworld_envs._build_base_env(
        {"env": {"type": "AlfredTWEnv"}},
        is_train=True,
        eval_dataset="train",
        gamefile=gamefile,
    )

    assert base_env.game_files == [gamefile]
    assert base_env.num_games == 1
    assert base_env.train_eval == "train"
