"""The ensemble checkpoint: members vote by averaging action probabilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submission"))

from naksha import features  # noqa: E402
from naksha.memory import load_any  # noqa: E402
from naksha.model import EnsembleNet, NakshaNet, _VERSIONS  # noqa: E402


def _inputs(rng: np.random.Generator) -> dict[str, torch.Tensor]:
    out = {}
    for key, (shape, dtype) in features.SHAPES.items():
        if np.dtype(dtype) == np.uint8:
            data = rng.integers(0, 255, shape).astype(dtype)
        else:
            data = rng.random(shape).astype(dtype)
        out[key] = torch.from_numpy(data)[None, None]
    return out


def _bundle(path: Path, members: list[NakshaNet]) -> None:
    state_dict = {f"members.{i}.{key}": value
                  for i, member in enumerate(members) for key, value in member.state_dict().items()}
    torch.save({"state_dict": state_dict, "meta": {**_VERSIONS, "ensemble": len(members)}}, path)


def test_single_member_ensemble_matches_the_model(tmp_path):
    torch.manual_seed(0)
    model = NakshaNet().eval()
    _bundle(tmp_path / "e1.pt", [model])
    ensemble, meta = load_any(tmp_path / "e1.pt")
    assert isinstance(ensemble, EnsembleNet) and meta["ensemble"] == 1
    rng = np.random.default_rng(0)
    s_model, s_ens = model.initial_state(1), ensemble.initial_state(1)
    with torch.inference_mode():
        for _ in range(10):
            inputs = _inputs(rng)
            o_model, s_model = model(inputs, s_model)
            o_ens, s_ens = ensemble(inputs, s_ens)
            assert torch.allclose(o_model["logits"].log_softmax(-1), o_ens["logits"], atol=1e-5)
            assert torch.allclose(o_model["exit_vis"], o_ens["exit_vis"], atol=1e-4)


def test_ensemble_averages_member_probabilities(tmp_path):
    torch.manual_seed(1)
    members = [NakshaNet().eval() for _ in range(3)]
    _bundle(tmp_path / "e3.pt", members)
    ensemble, _ = load_any(tmp_path / "e3.pt")
    inputs = _inputs(np.random.default_rng(1))
    with torch.inference_mode():
        out, state = ensemble(inputs, ensemble.initial_state(1))
        expected = torch.stack([m(inputs, m.initial_state(1))[0]["logits"].softmax(-1) for m in members]).mean(0)
    assert state.shape == (3, 1, 1, 256)
    assert torch.allclose(out["logits"].exp(), expected, atol=1e-5)
    assert torch.allclose(out["logits"].exp().sum(-1), torch.ones(1, 1), atol=1e-5)
