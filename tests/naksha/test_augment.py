import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training.paths  # noqa: E402
from doom_env import make_env  # noqa: E402
from naksha import actions, features  # noqa: E402
from training.data.augment import mirror, mirror_rate_batch  # noqa: E402
from training.data.shards import _FIELDS  # noqa: E402


def test_mirror_round_trip() -> None:
    rng = np.random.default_rng(0)
    batch = {}
    for key, (shape, dtype) in _FIELDS.items():
        size = (3, 2, *shape)
        if key in ("action", "label"):
            batch[key] = rng.integers(-1, actions.NUM_ACTIONS, size=size, dtype=dtype)
        elif np.issubdtype(dtype, np.floating):
            batch[key] = rng.standard_normal(size).astype(dtype)
        else:
            batch[key] = rng.integers(0, 2, size=size).astype(dtype)
    batch["memory"] = rng.integers(0, 2, size=(3, 2, 16, 24, 24), dtype=np.uint8)
    batch["valid"] = np.array([[True, True], [True, True], [False, True]])
    original = {key: array.copy() for key, array in batch.items()}
    mask = np.array([True, False])

    mirrored = mirror(batch, mask)
    restored = mirror(mirrored, mask)
    assert mirrored is not batch
    assert restored.keys() == mirrored.keys() == batch.keys()
    for key, expected in original.items():
        np.testing.assert_array_equal(batch[key], expected, err_msg=key)
        np.testing.assert_array_equal(mirrored[key][:, 1], expected[:, 1], err_msg=key)
        np.testing.assert_array_equal(restored[key], expected, err_msg=key)
        assert mirrored[key].dtype == restored[key].dtype == expected.dtype

    for key in ("ent_mask", "intention", "exit_vis", "valid", "pose", "reward", "done"):
        np.testing.assert_array_equal(mirrored[key], original[key], err_msg=key)
    np.testing.assert_array_equal(mirrored["memory"][:, 0], original["memory"][:, 0, ..., ::-1])


def test_mirror_directions() -> None:
    batch = {key: np.zeros((1, 1, *shape), dtype) for key, (shape, dtype) in features.SHAPES.items()}
    batch["rays"][0, 0, [0, 5, 32]] = [[1] * 7, np.arange(7), [2] * 7]
    batch["ents"][0, 0, 0, :5] = [0.5, 0.25, 0.75, 1, 2]
    batch["vitals"][0, 0, 14:] = [1, 2, 3]
    batch["exit_dir"] = np.array([[[0.25, 0.75]]])
    batch["pixels"][0, 0, :, :, 0] = 123
    mirrored = mirror(batch, np.array([True]))

    np.testing.assert_array_equal(mirrored["rays"][0, 0, [0, 59, 32]], batch["rays"][0, 0, [0, 5, 32]])
    np.testing.assert_array_equal(mirrored["rays"][0, 0, 5], np.zeros(7))
    np.testing.assert_array_equal(mirrored["ents"][0, 0, 0, :5], [0.5, -0.25, 0.75, 1, -2])
    np.testing.assert_array_equal(mirrored["vitals"][0, 0, 14:], [1, -2, 3])
    np.testing.assert_array_equal(mirrored["exit_dir"], [[[-0.25, 0.75]]])
    np.testing.assert_array_equal(mirrored["pixels"][..., 39], batch["pixels"][..., 0])
    assert not mirrored["pixels"][..., :39].any()


def test_mirror_actions() -> None:
    names = actions.NAMES
    reflected = {
        "strafe_left": "strafe_right",
        "turn_left": "turn_right",
        "forward_turn_left": "forward_turn_right",
        "attack_turn_left": "attack_turn_right",
    }
    reflected.update({right: left for left, right in reflected.items()})
    labels = np.array([*(actions.INDEX[name] for name in names), -1], dtype=np.int16)
    expected = np.array([*(actions.INDEX[reflected.get(name, name)] for name in names), -1])
    one_hot = np.eye(actions.NUM_ACTIONS, dtype=np.float32)
    batch = {
        "label": labels[:, None],
        "action": labels[::-1, None],
        "prev_action": np.concatenate((one_hot, np.zeros_like(one_hot[:1])))[:, None],
    }

    mirrored = mirror(batch, np.array([True]))
    np.testing.assert_array_equal(mirrored["label"][:, 0], expected)
    np.testing.assert_array_equal(mirrored["action"][:, 0], expected[::-1])
    np.testing.assert_array_equal(mirrored["prev_action"][:-1, 0], one_hot[expected[:-1]])
    assert not mirrored["prev_action"][-1].any()


@pytest.mark.parametrize("rate", [0, 0.5, 1])
def test_mirror_rate_batch(rate: float) -> None:
    batch = {"rays": np.arange(2 * 8 * 64 * 7).reshape(2, 8, 64, 7)}
    seed = 0
    mask = np.random.default_rng(seed).random(8) < rate
    sampled = mirror_rate_batch(batch, rate, np.random.default_rng(seed))
    np.testing.assert_array_equal(sampled["rays"], mirror(batch, mask)["rays"])


def test_mirror_real_turns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    env = make_env(training.paths.ROOT / "maps/two_rooms.wad", "MAP01", render_buffers=False)
    try:
        observations = {}
        for name in ("turn_left", "turn_right"):
            env.reset(seed=0)
            action = actions.INDEX[name]
            buttons = actions.to_buttons(action)
            for _ in range(4):
                obs, _, terminated, truncated, _ = env.step(buttons)
                assert not (terminated or truncated)
            observations[name] = features.encode(obs, prev_action=action)
    finally:
        env.close()

    def asymmetry(rays: np.ndarray) -> float:
        distances = rays[:, 0]
        # Compare the left and right halves of the forward-facing cone.
        return float((distances[1:16] - distances[:48:-1]).mean())

    left = observations["turn_left"]
    batch = {key: array[None, None] for key, array in left.items()}
    mirrored = mirror(batch, np.array([True]))
    right_asymmetry = asymmetry(observations["turn_right"]["rays"])
    assert asymmetry(left["rays"]) * right_asymmetry < 0
    assert asymmetry(mirrored["rays"][0, 0]) * right_asymmetry > 0
