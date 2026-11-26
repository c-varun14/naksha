import csv
import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.ppo import ppo_mem  # noqa: E402
from naksha.memory import MEMORY_VERSION, NakshaMemNet, load_any  # noqa: E402
from naksha.model import NakshaNet, save_checkpoint  # noqa: E402


def test_ppo_mem_smoke(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    initial = tmp_path / "imitation.pt"
    output = tmp_path / "fine_tuned.pt"
    imitation = NakshaMemNet()
    save_checkpoint(initial, imitation, {"memory": MEMORY_VERSION})

    ppo_mem.main([
        "--init", str(initial), "--out", str(output), "--minutes", "0", "--workers", "2",
        "--steps-per-worker", "64", "--seq", "16", "--minibatch-seqs", "2", "--device", "cpu",
        "--families", "arena", "--sizes", "s",
    ])

    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert (summary["iterations"], summary["steps"], summary["worker_restarts"]) == (1, 128, 0)
    trained, meta = load_any(output)
    assert isinstance(trained, NakshaMemNet)
    assert (meta["init"], meta["iterations"], meta["steps"], meta["memory"]) == (
        str(initial), 1, 128, MEMORY_VERSION,
    )
    assert not torch.equal(trained.logits.weight, imitation.logits.weight)
    assert not torch.equal(trained.memory_encoder[0].weight, imitation.memory_encoder[0].weight)
    for name, weight in trained.odometry.state_dict().items():
        torch.testing.assert_close(weight, imitation.odometry.state_dict()[name], rtol=0, atol=0)
    with (tmp_path / summary["log"]).open() as log:
        row, = csv.DictReader(log)
    assert row["steps"] == "128"
    assert all(math.isfinite(float(row[key])) for key in ("kl", "entropy", "value_loss", "fps"))


def test_ppo_mem_requires_grid_checkpoint(tmp_path, capsys):
    initial = tmp_path / "plain.pt"
    save_checkpoint(initial, NakshaNet(), {})
    output = tmp_path / "out.pt"
    with pytest.raises(SystemExit, match="2"):
        ppo_mem.main(["--init", str(initial), "--out", str(output), "--minutes", "0", "--device", "cpu"])
    assert "--init must be a grid-v1 checkpoint" in capsys.readouterr().err
    assert not output.exists()
