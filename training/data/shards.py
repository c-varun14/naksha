"""Episode storage and recurrent sequence sampling."""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import copy
from pathlib import Path

import numpy as np

import training.paths  # noqa: F401
from naksha import features

_FIELDS = {
    **features.SHAPES,
    "action": ((), np.int16),
    "label": ((), np.int16),
    "intention": ((), np.int8),
    "exit_vis": ((), np.int8),
    "exit_dir": ((2,), np.float16),
    "pose": ((3,), np.float32),
    "reward": ((), np.float32),
    "done": ((), np.bool_),
}
_RAM_DTYPES = {
    "rays": np.float16,
    "ents": np.float16,
    "ent_mask": np.float16,
    "vitals": np.float16,
    "prev_action": np.uint8,
}


def _validate(steps: dict[str, np.ndarray]) -> int:
    if steps.keys() != _FIELDS.keys():
        missing = sorted(_FIELDS.keys() - steps.keys())
        extra = sorted(steps.keys() - _FIELDS.keys())
        raise ValueError(f"episode fields: missing {missing}, extra {extra}")
    length = steps["action"].size
    if length == 0:
        raise ValueError("episode must have at least one step")
    for key, (shape, dtype) in _FIELDS.items():
        array = steps[key]
        if array.shape != (length, *shape) or array.dtype != dtype:
            raise ValueError(f"{key}: expected shape {(length, *shape)} and dtype {np.dtype(dtype)}")
    return length


def write_episode(out_dir: str | Path, name: str, steps: dict[str, np.ndarray], meta: dict) -> None:
    _validate(steps)
    metadata = json.dumps(meta, indent=2) + "\n"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(metadata, encoding="utf-8")
    np.savez_compressed(out_dir / f"{name}.npz", **steps)


class SequenceDataset:
    def __init__(
        self, dirs: str | Path | Sequence[str | Path], weights: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if isinstance(dirs, (str, Path)):
            dirs = [dirs]
        self.episodes: list[dict[str, np.ndarray]] = []
        self.metadata: list[dict] = []
        self.lengths: list[int] = []
        for directory in dirs:
            paths = sorted(Path(directory).glob("*.npz"))
            if not paths:
                raise ValueError(f"no episodes in {directory}")
            for path in paths:
                with np.load(path, allow_pickle=False) as episode:
                    steps = dict(episode)
                self.lengths.append(_validate(steps))
                for key, dtype in _RAM_DTYPES.items():
                    steps[key] = steps[key].astype(dtype)
                self.episodes.append(steps)
                self.metadata.append(json.loads(path.with_suffix(".json").read_text(encoding="utf-8")))
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        if self.weights is not None and (
            self.weights.shape != (len(self.episodes),)
            or not np.isfinite(self.weights).all() or (self.weights < 0).any()
            or not (self.weights > 0).any()
        ):
            raise ValueError("weights must be finite, nonnegative and supply one value per episode")

    def sample(
        self, batch: int, seq_len: int, rng: np.random.Generator, *, min_steps: int = 1,
    ) -> dict[str, np.ndarray]:
        """Sample chunks with at least min_steps valid steps, padding on the right."""
        if batch < 1 or not 1 <= min_steps <= seq_len:
            raise ValueError("batch and min_steps must be positive, and min_steps cannot exceed seq_len")
        eligible = np.flatnonzero(np.asarray(self.lengths) >= min_steps)
        weights = None if self.weights is None else self.weights[eligible]
        if not len(eligible) or (weights is not None and not weights.any()):
            raise ValueError(f"no sampleable episodes with at least {min_steps} steps")
        if weights is not None:
            weights = weights / weights.max()
            weights /= weights.sum()
        indices = rng.choice(eligible, size=batch, p=weights)
        result = {key: np.zeros((seq_len, batch, *shape), dtype)
                  for key, (shape, dtype) in _FIELDS.items()}
        result["valid"] = np.zeros((seq_len, batch), dtype=bool)
        for column, index in enumerate(indices):
            length = self.lengths[index]
            start = int(rng.integers(length - min_steps + 1))
            size = min(length - start, seq_len)
            for key, array in self.episodes[index].items():
                # Assignment converts the compact arrays into the original field dtypes.
                result[key][:size, column] = array[start:start + size]
            result["valid"][:size, column] = True
        return result

    def _subset(self, indices: np.ndarray) -> SequenceDataset:
        subset = copy(self)
        subset.episodes = [self.episodes[i] for i in indices]
        subset.metadata = [self.metadata[i] for i in indices]
        subset.lengths = [self.lengths[i] for i in indices]
        subset.weights = None if self.weights is None else self.weights[indices]
        return subset

    def split(
        self, val_frac: float, seed: int, *, min_steps: int = 1,
    ) -> tuple[SequenceDataset, SequenceDataset]:
        if not 0 <= val_frac <= 1:
            raise ValueError("val_frac must be between 0 and 1")
        eligible = np.flatnonzero(np.asarray(self.lengths) >= min_steps)
        count = len(eligible)
        val_size = round(count * val_frac)
        if count > 1 and 0 < val_frac < 1:
            val_size = min(count - 1, max(1, val_size))
        order = np.random.default_rng(seed).permutation(eligible)
        return self._subset(order[val_size:]), self._subset(order[:val_size])
