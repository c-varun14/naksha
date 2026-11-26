import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training.paths  # noqa: E402, F401
from naksha import features, model  # noqa: E402
from training.bc import train  # noqa: E402
from training.bc.wide import WideNet  # noqa: E402
from training.data.shards import write_episode  # noqa: E402


def _episode(length: int) -> dict[str, np.ndarray]:
    steps = {key: np.ones((length, *shape), dtype) for key, (shape, dtype) in features.SHAPES.items()}
    steps.update(
        action=np.zeros(length, dtype=np.int16), label=np.zeros(length, dtype=np.int16),
        intention=np.ones(length, dtype=np.int8), exit_vis=np.ones(length, dtype=np.int8),
        exit_dir=np.ones((length, 2), dtype=np.float16), pose=np.ones((length, 3), dtype=np.float32),
        reward=np.ones(length, dtype=np.float32), done=np.arange(length) == length - 1,
    )
    steps["prev_action"][:] = 0
    steps["prev_action"][:, 0] = 1
    return steps


@pytest.fixture
def one_torch_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original)


def test_wide_outputs_and_parameter_count(one_torch_thread) -> None:
    inputs = {key: torch.from_numpy(np.zeros((3, 2, *shape), dtype))
              for key, (shape, dtype) in features.SHAPES.items()}
    inputs["ent_mask"][0, 0, 0] = 1
    base, wide = model.NakshaNet(), WideNet()
    with torch.inference_mode():
        expected, _ = base(inputs, base.initial_state(2))
        outputs, state = wide(inputs, wide.initial_state(2))
    assert outputs.keys() == expected.keys()
    for key in outputs:
        assert outputs[key].shape == expected[key].shape
        assert torch.isfinite(outputs[key]).all()
    assert state.shape == (1, 2, 512)
    assert wide.count_params() > 2.5 * base.count_params()


def test_train_wide(tmp_path, monkeypatch, capsys, one_torch_thread) -> None:
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "shards"
    for name in ("one", "two"):
        write_episode(data, name, _episode(18), {})
    out = tmp_path / "wide.pt"
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--wide", "--minutes", "0.02", "--device", "cpu",
        "--data", str(data), "--out", str(out), "--batch", "1", "--seq", "2",
    ])
    train.main()
    result = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert result["total_updates"] > 0
    checkpoint = torch.load(out, map_location="cpu", weights_only=True)
    assert checkpoint["meta"]["arch"] == "wide2"
    for key in ("val_loss", "val_acc"):
        assert result[key] == checkpoint["meta"][key]
    with pytest.raises(ValueError, match="FEATURES_VERSION mismatch"):
        model.load_checkpoint(out)


def test_wide_rejects_init(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--wide", "--init", "x", "--data", "unused",
        "--out", "unused.pt", "--minutes", "0.02", "--device", "cpu",
    ])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2
    assert "--wide cannot be combined with --init" in capsys.readouterr().err
