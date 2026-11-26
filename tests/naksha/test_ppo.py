import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import training.paths  # noqa: E402, F401
from naksha.model import NakshaNet, load_checkpoint, save_checkpoint  # noqa: E402
from training.ppo.ppo import EpisodeReward, gae  # noqa: E402


def test_gae_bootstraps_and_respects_done():
    rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    values = np.array([0.5, 1.0, 2.0, 1.5, 1.0], dtype=np.float32)
    dones = np.array([False, True, False, False])

    advantage, target = gae(rewards, values, dones, gamma=0.5, lam=0.5)

    # Deltas are 1, 1, 1.75, 3. The terminal step stops the trace across episodes.
    np.testing.assert_allclose(advantage, [1.25, 1.0, 2.5, 3.0])
    np.testing.assert_allclose(target, [1.75, 2.0, 4.5, 4.5])


@pytest.mark.parametrize(("steps", "map_args"), [
    (64, ["--families", "arena", "--sizes", "s"]),
    (32, ["--stock-frac", "1.0", "--stock-maps", "MAP06"]),
], ids=("generated", "stock"))
def test_ppo_smoke(tmp_path, steps, map_args):
    initial = tmp_path / "imitation.pt"
    output = tmp_path / "fine_tuned.pt"
    imitation = NakshaNet()
    save_checkpoint(initial, imitation, {})

    run = subprocess.run(
        [sys.executable, "-m", "training.ppo.ppo", "--init", str(initial), "--out", str(output),
         "--minutes", "0", "--workers", "2", "--steps-per-worker", str(steps), "--seq", "16",
         "--device", "cpu", *map_args],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True, timeout=90,
    )
    assert run.returncode == 0, run.stderr
    summary = json.loads(run.stdout.splitlines()[-1])
    assert (summary["iterations"], summary["steps"], summary["worker_restarts"]) == (1, 2 * steps, 0), run.stderr
    trained, meta = load_checkpoint(output)
    assert (meta["init"], meta["iterations"], meta["steps"]) == (str(initial), 1, 2 * steps)
    assert not torch.equal(trained.logits.weight, imitation.logits.weight)
    with (tmp_path / "runs/ppo/fine_tuned/log.csv").open() as log:
        row, = csv.DictReader(log)
    assert row["steps"] == str(2 * steps)
    assert {"stock_exit_rate", "generated_exit_rate"} <= row.keys()
    assert all(math.isfinite(float(row[key])) for key in ("kl", "entropy", "value_loss", "fps"))


def test_stock_door_reward_requires_a_rise_above_spawn_height():
    state = {
        "x": 0, "y": 0, "sector_ceiling": np.array([160, 64]), "keys": np.zeros(6),
        "health": 100, "kills": 0, "exit_reached": False, "dead": False,
    }
    reward = EpisodeReward(state, {0: 0, 1: 1}, stock=True)
    for ceilings, door_bonus in [
        ([160, 64], 0.0), ([32, 96], 0.0), ([160, 97], 0.5),
        ([160, 64], 0.0), ([160, 130], 0.0),
    ]:
        shaped, _, _ = reward.step({**state, "sector_ceiling": np.array(ceilings)})
        assert shaped == pytest.approx(door_bonus - 0.0005)


def _reward_state(**over):
    state = {
        "x": 0, "y": 0, "sector_ceiling": np.array([64]), "keys": np.zeros(6), "health": 100, "kills": 0,
        "exit_reached": False, "dead": False, "tic": 100, "episode_ticks": 2100,
    }
    return {**state, **over}


def _one_step(before, after, **kw):
    return EpisodeReward(before, {}, **kw).step(after)[0]


def test_default_reward_terms_are_unchanged():
    start = _reward_state()
    assert _one_step(start, _reward_state(health=70)) == pytest.approx(-0.0005 - 0.003 * 30)
    assert _one_step(start, _reward_state(dead=True, health=0)) == pytest.approx(-0.0005 - 1.0 - 0.3)
    exit_reward = 1.0 + 0.5 * (2100 - 100) / 2100
    assert _one_step(start, _reward_state(exit_reached=True, health=50), ) == pytest.approx(-0.0005 + exit_reward - 0.15)


def test_survival_pays_for_health_at_exit_and_charges_early_death():
    healthy = _one_step(_reward_state(), _reward_state(exit_reached=True), survival=True)
    hurt = _one_step(_reward_state(health=50), _reward_state(exit_reached=True, health=50), survival=True)
    assert healthy - hurt == pytest.approx(0.25)
    early = _one_step(_reward_state(health=10), _reward_state(dead=True, health=0, tic=100), survival=True)
    late = _one_step(_reward_state(health=10, tic=2000), _reward_state(dead=True, health=0, tic=2000), survival=True)
    assert early < late < 0


def test_survival_healing_earns_less_than_damage_costs():
    heal = _one_step(_reward_state(health=40), _reward_state(health=65), survival=True)
    hurt = _one_step(_reward_state(health=65), _reward_state(health=40), survival=True)
    assert heal == pytest.approx(-0.0005 + 0.004 * 25)
    assert heal + hurt < 0
    over = _one_step(_reward_state(health=90), _reward_state(health=125), survival=True)
    assert over == pytest.approx(-0.0005 + 0.004 * 10)


def test_stall_penalty_starts_after_150_steps_without_a_new_cell():
    reward = EpisodeReward(_reward_state(), {}, stall_penalty=0.002)
    shaped = [reward.step(_reward_state())[0] for _ in range(151)]
    assert shaped[148] == pytest.approx(-0.0005)
    assert shaped[149] == pytest.approx(-0.0005 - 0.002)
    moved = reward.step(_reward_state(x=64))[0]
    assert moved == pytest.approx(-0.0005 + 0.02)
