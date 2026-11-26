import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training.paths  # noqa: E402
from doom_env import make_env  # noqa: E402
from naksha import actions, model  # noqa: E402
from submission.agent import DoomAgent  # noqa: E402


def test_agent_actions_and_latency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "naksha.pt"
    model.save_checkpoint(path, model.NakshaNet(), {})
    monkeypatch.setenv("NAKSHA_WEIGHTS", str(path))
    monkeypatch.chdir(tmp_path)
    env = make_env(training.paths.ROOT / "maps/two_rooms.wad", "MAP01")
    try:
        agent = DoomAgent(env.observation_space, env.action_space, "cuda")
        assert agent.device == "cuda"
        assert next(agent.model.parameters()).device.type == "cpu"
        assert torch.get_num_threads() == 1
        obs, _ = env.reset(seed=0)
        agent.reset()
        timings = []
        for _ in range(50):
            start = perf_counter()
            action = agent.act(obs)
            timings.append((perf_counter() - start) * 1_000)
            assert action.shape == (9,) and action.dtype == np.int8
            assert env.action_space.contains(action)
            np.testing.assert_array_equal(action, actions.to_buttons(agent.prev_action))
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset(seed=0)
                agent.reset()
        agent.act(obs)
        assert agent.last_out["logits"].shape == (actions.NUM_ACTIONS,)
        assert agent.last_out["intention"].sum() == pytest.approx(1)
        assert 0 <= agent.last_out["exit_vis"] <= 1
        assert np.median(timings) < 10
    finally:
        env.close()
