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
