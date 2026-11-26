import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training.paths  # noqa: E402
from doom_env import make_env  # noqa: E402
from naksha import actions, features, model  # noqa: E402
from naksha.memory import MEMORY_VERSION, NakshaMemNet  # noqa: E402
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


def fixed_observations():
    rng = np.random.default_rng(16)
    return [
        {
            "rgb": rng.integers(0, 256, (120, 160, 3), dtype=np.uint8),
            "rays": {"distance": rng.random(64), "hit_type": rng.integers(0, 6, 64)},
            "vitals": rng.uniform(-10, 10, size=17),
            "visible_entities": {
                "mask": rng.integers(0, 2, 32), "entity_type": rng.integers(0, 12, 32),
                "features": rng.random((32, 5)),
            },
        }
        for _ in range(24)
    ]


@pytest.mark.parametrize("temperature", [0.0, 0.7])
@torch.inference_mode()
def test_plain_agent_matches_previous_actions(tmp_path, monkeypatch, temperature):
    monkeypatch.delenv("NAKSHA_TEMPERATURE", raising=False)
    path = tmp_path / "plain.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model.save_checkpoint(path, model.NakshaNet(), {"temperature": temperature})
    agent = DoomAgent(None, None, "cpu", weights=path)
    legacy, _ = model.load_checkpoint(path)
    legacy.eval()
    observations = fixed_observations()

    for episode in (1, 2):
        rng = torch.Generator().manual_seed(1234 + episode)
        hidden = legacy.initial_state(1)
        prev_action = 0
        for obs in observations:
            encoded = features.encode(obs, prev_action)
            inputs = {key: torch.from_numpy(data)[None, None] for key, data in encoded.items()}
            out, hidden = legacy(inputs, hidden)
            logits = out["logits"][0, 0]
            if temperature > 0:
                prev_action = int(torch.multinomial((logits / temperature).softmax(dim=-1), 1, generator=rng))
            else:
                prev_action = int(logits.argmax())
            assert agent.act(obs).tobytes() == actions.to_buttons(prev_action).tobytes()
        agent.reset()
    assert agent.memory_rgb() is None


def test_memory_agent_reset_and_latency(tmp_path):
    path = tmp_path / "grid.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        model.save_checkpoint(path, NakshaMemNet(), {"memory": MEMORY_VERSION})
    agent = DoomAgent(None, None, "cuda", weights=path)
    assert torch.get_num_threads() == 1
    assert next(agent.model.parameters()).device.type == "cpu"
    observations = fixed_observations()
    empty = agent.memory_rgb()
    agent.act(observations[0])
    assert agent.memory.grid.pose() == (0.0, 0.0, 0.0)
    assert not np.array_equal(agent.memory_rgb(), empty)
    agent.act(observations[1])
    assert agent.memory.grid.pose() != (0.0, 0.0, 0.0)
    agent.reset()
    np.testing.assert_array_equal(agent.memory_rgb(), empty)
    agent.act(observations[1])
    assert agent.memory.grid.pose() == (0.0, 0.0, 0.0)

    for obs in observations:
        agent.act(obs)
    timings = []
    for i in range(240):
        obs = observations[i % len(observations)]
        start = perf_counter()
        agent.act(obs)
        timings.append((perf_counter() - start) * 1_000)
    p50, p99 = np.percentile(timings, [50, 99])
    print(f"Memory agent latency (1 CPU thread, {len(timings)} acts) p50 {p50:.3f} ms, p99 {p99:.3f} ms")
    assert p99 < 10
