import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training.paths  # noqa: E402, F401
from naksha import actions, features, model  # noqa: E402


def test_forward_shapes() -> None:
    net = model.NakshaNet()
    inputs = {key: torch.from_numpy(np.zeros((3, 2, *shape), dtype))
              for key, (shape, dtype) in features.SHAPES.items()}
    inputs["ent_mask"][0, 0, 0] = 1
    with torch.inference_mode():
        out, state = net(inputs, net.initial_state(2, "cpu"))
    assert {key: value.shape for key, value in out.items()} == {
        "logits": (3, 2, actions.NUM_ACTIONS), "value": (3, 2), "intention": (3, 2, 5),
        "exit_vis": (3, 2), "exit_dir": (3, 2, 2),
    }
    assert state.shape == (1, 2, model.HIDDEN)
    assert all(torch.isfinite(value).all() for value in out.values())
    assert net.count_params() <= 2_000_000


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    net = model.NakshaNet()
    path = tmp_path / "naksha.pt"
    model.save_checkpoint(path, net, {"updates": 10})
    restored, meta = model.load_checkpoint(path)
    assert meta == {"FEATURES_VERSION": features.FEATURES_VERSION, "ACTIONS_VERSION": actions.ACTIONS_VERSION,
                    "MODEL_VERSION": model.MODEL_VERSION, "updates": 10}
    for key, expected in net.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize("version", ("FEATURES_VERSION", "ACTIONS_VERSION", "MODEL_VERSION"))
def test_checkpoint_rejects_version_mismatch(tmp_path: Path, version: str) -> None:
    path = tmp_path / "naksha.pt"
    model.save_checkpoint(path, model.NakshaNet(), {})
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["meta"][version] += 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match=version):
        model.load_checkpoint(path)
