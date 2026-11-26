import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import training.paths  # noqa: E402, F401
from naksha import features  # noqa: E402
from training.data.shards import SequenceDataset, write_episode  # noqa: E402


def _episode(length: int) -> dict[str, np.ndarray]:
    steps = {key: np.ones((length, *shape), dtype) for key, (shape, dtype) in features.SHAPES.items()}
    steps.update(
        action=np.arange(length, dtype=np.int16), label=np.full(length, -1, dtype=np.int16),
        intention=np.ones(length, dtype=np.int8), exit_vis=np.ones(length, dtype=np.int8),
        exit_dir=np.ones((length, 2), dtype=np.float16), pose=np.ones((length, 3), dtype=np.float32),
        reward=np.ones(length, dtype=np.float32), done=np.arange(length) == length - 1,
    )
    steps["prev_action"][:] = 0
    steps["prev_action"][np.arange(length), np.maximum(0, steps["action"] - 1)] = 1
    return steps


def test_round_trip_and_padding(tmp_path: Path) -> None:
    steps = _episode(3)
    steps["rays"][0, 0, 0] = 0.12345
    write_episode(tmp_path, "episode", steps, {"seed": 4})
    dataset = SequenceDataset([tmp_path])
    assert dataset.lengths == [3]
    assert dataset.metadata == [{"seed": 4}]
    compact_dtypes = {"rays": np.float16, "ents": np.float16, "ent_mask": np.float16,
                      "vitals": np.float16, "prev_action": np.uint8}
    with np.load(tmp_path / "episode.npz", allow_pickle=False) as stored:
        for key, expected in steps.items():
            assert stored[key].dtype == expected.dtype
            np.testing.assert_array_equal(stored[key], expected)
    for key, expected in steps.items():
        array = dataset.episodes[0][key]
        assert array.dtype == compact_dtypes.get(key, expected.dtype)
        np.testing.assert_array_equal(array, expected.astype(array.dtype))
    before = sum(array.nbytes for array in steps.values())
    after = sum(array.nbytes for array in dataset.episodes[0].values())
    assert after < before
    print(f"Episode arrays: {before:,} -> {after:,} bytes ({1 - after / before:.1%} saved)")
    chunk = dataset.sample(4, 5, np.random.default_rng(0))
    assert chunk["valid"].shape == (5, 4)
    for key, expected in steps.items():
        assert chunk[key].shape == (5, 4, *expected.shape[1:])
        assert chunk[key].dtype == expected.dtype
    for column in range(4):
        valid = chunk["valid"][:, column]
        length = int(valid.sum())
        assert 1 <= length <= 3
        np.testing.assert_array_equal(valid, np.arange(5) < length)
        for key, expected in steps.items():
            compact = expected[-length:].astype(compact_dtypes.get(key, expected.dtype))
            np.testing.assert_array_equal(chunk[key][:length, column], compact.astype(expected.dtype))
            assert not chunk[key][length:, column].any()


def test_split_and_episode_weights(tmp_path: Path) -> None:
    write_episode(tmp_path, "one", _episode(1), {})
    write_episode(tmp_path, "two", _episode(2), {})
    dataset = SequenceDataset(tmp_path, weights=[1, 0])
    chunk = dataset.sample(5, 3, np.random.default_rng(0))
    assert (chunk["valid"].sum(axis=0) == 1).all()
    train, val = dataset.split(0.5, seed=0)
    assert sorted(train.lengths + val.lengths) == [1, 2]
    assert train.lengths != val.lengths


def test_rejects_mismatched_dtype(tmp_path: Path) -> None:
    steps = _episode(2)
    steps["label"] = steps["label"].astype(np.int64)
    with pytest.raises(ValueError, match="label"):
        write_episode(tmp_path, "episode", steps, {})
