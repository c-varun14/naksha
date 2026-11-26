"""Left-right reflection of recurrent imitation batches."""

from __future__ import annotations

import numpy as np

import training.paths  # noqa: F401
from naksha import actions, features

_RAY_MIRROR = -np.arange(features.NUM_RAYS) % features.NUM_RAYS
_ACTION_MIRROR = np.arange(actions.NUM_ACTIONS)
for _stem in ("strafe", "turn", "forward_turn", "attack_turn"):
    _left = actions.INDEX[f"{_stem}_left"]
    _right = actions.INDEX[f"{_stem}_right"]
    _ACTION_MIRROR[[_left, _right]] = [_right, _left]

# The trailing entry makes the unlabelled sentinel map to itself via index -1.
_LABEL_MIRROR = np.append(_ACTION_MIRROR, -1)


def mirror(batch: dict, mask: np.ndarray) -> dict:
    """Reflect each selected batch column across all time steps without mutating the input."""
    result = batch.copy()
    if not mask.any():
        return result

    for key, axis, order in (
        ("rays", -2, _RAY_MIRROR),
        ("prev_action", -1, _ACTION_MIRROR),
    ):
        if key in batch:
            result[key] = batch[key].copy()
            result[key][:, mask] = np.take(batch[key][:, mask], order, axis=axis)

    for key, columns in (("ents", (1, 4)), ("vitals", (15,)), ("exit_dir", (0,))):
        if key in batch:
            result[key] = batch[key].copy()
            selected = batch[key][:, mask]
            selected[..., columns] *= -1
            result[key][:, mask] = selected

    for key in ("label", "action"):
        if key in batch:
            result[key] = batch[key].copy()
            result[key][:, mask] = _LABEL_MIRROR[batch[key][:, mask]]

    if "pixels" in batch:
        result["pixels"] = batch["pixels"].copy()
        result["pixels"][:, mask] = batch["pixels"][:, mask, ..., ::-1]

    # GridMemory.crops uses column - 12 as the right offset for both crops.
    # MemoryDataset.sample has already unpacked the 24 columns.
    if "memory" in batch and batch["memory"].shape[-1] == 24:
        result["memory"] = batch["memory"].copy()
        result["memory"][:, mask] = batch["memory"][:, mask, ..., ::-1]

    # Pose passes through unchanged. Odometry targets need their own reflection.
    return result


def mirror_rate_batch(batch: dict, rate: float, rng: np.random.Generator) -> dict:
    """Reflect each sequence independently with probability rate."""
    if not 0 <= rate <= 1:
        raise ValueError("mirror rate must be between 0 and 1")
    if rate == 0:
        return batch.copy()
    return mirror(batch, rng.random(batch["rays"].shape[1]) < rate)
