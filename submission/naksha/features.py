"""The only place an official observation becomes network input. Used by training and by
submission/agent.py, so train and eval see identical inputs. Bump FEATURES_VERSION on any change.
"""

from __future__ import annotations

import numpy as np

from .actions import NUM_ACTIONS

FEATURES_VERSION = 1
NUM_RAYS = 64
RAY_TYPES = 6
MAX_ENTITIES = 32
ENTITY_TYPES = 12
PIX_H, PIX_W = 30, 40

_VITALS_SCALE = np.array(
    [100, 100, 7, 50, 50, 50, 50, 100, 1, 1, 1, 1, 1, 1, 20, 20, 20], np.float32
)

SHAPES = {
    "rays": ((NUM_RAYS, 1 + RAY_TYPES), np.float32),
    "ents": ((MAX_ENTITIES, 5 + ENTITY_TYPES), np.float32),
    "ent_mask": ((MAX_ENTITIES,), np.float32),
    "vitals": ((17,), np.float32),
    "prev_action": ((NUM_ACTIONS,), np.float32),
    "pixels": ((3, PIX_H, PIX_W), np.uint8),
}
KEYS = tuple(SHAPES)

_EYE_RAY = np.eye(RAY_TYPES, dtype=np.float32)
_EYE_ENT = np.eye(ENTITY_TYPES, dtype=np.float32)
_EYE_ACT = np.eye(NUM_ACTIONS, dtype=np.float32)


def downsample(rgb: np.ndarray) -> np.ndarray:
    """(120, 160, 3) uint8 -> (3, 30, 40) uint8 by 4x4 area averaging."""
    small = rgb.reshape(PIX_H, 4, PIX_W, 4, 3).astype(np.float32).mean(axis=(1, 3))
    return np.ascontiguousarray(np.clip(small + 0.5, 0, 255).astype(np.uint8).transpose(2, 0, 1))


def encode(obs: dict, prev_action: int = 0) -> dict[str, np.ndarray]:
    rays = obs["rays"]
    dist = np.asarray(rays["distance"], np.float32).reshape(NUM_RAYS, 1)
    hit = np.clip(np.asarray(rays["hit_type"], np.int64), 0, RAY_TYPES - 1)
    out_rays = np.concatenate([dist, _EYE_RAY[hit]], axis=1)

    ve = obs["visible_entities"]
    mask = np.asarray(ve["mask"], np.float32).reshape(MAX_ENTITIES)
    feats = np.asarray(ve["features"], np.float32).reshape(MAX_ENTITIES, 5).copy()
    feats[:, 0] = np.clip(feats[:, 0], 0.0, 4.0)
    feats[:, 3:5] = np.clip(feats[:, 3:5] / 20.0, -3.0, 3.0)
    etype = np.clip(np.asarray(ve["entity_type"], np.int64), 0, ENTITY_TYPES - 1)
    ents = np.concatenate([feats, _EYE_ENT[etype]], axis=1) * mask[:, None]

    vit = np.clip(np.asarray(obs["vitals"], np.float32) / _VITALS_SCALE, -5.0, 5.0)
    vit = np.nan_to_num(vit, nan=0.0, posinf=5.0, neginf=-5.0)

    return {
        "rays": np.nan_to_num(out_rays, nan=1.0).astype(np.float32),
        "ents": np.nan_to_num(ents, nan=0.0).astype(np.float32),
        "ent_mask": mask,
        "vitals": vit.astype(np.float32),
        "prev_action": _EYE_ACT[int(prev_action) % NUM_ACTIONS].copy(),
        "pixels": downsample(np.asarray(obs["rgb"], np.uint8)),
    }


def stack(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([it[k] for it in items]) for k in KEYS}
