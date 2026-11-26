"""The shipped agent returns a valid action, quickly, on extreme but legal observations."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "env"))
sys.path.insert(0, str(ROOT / "submission"))

from doom_env import spaces as S  # noqa: E402

WEIGHTS = ROOT / "submission" / "weights" / "naksha.pt"


def _base_obs() -> dict:
    space = S.observation_space()
    obs = space.sample()
    obs["rays"]["distance"] = np.full(S.NUM_RAYS, 0.5, np.float32)
    obs["rays"]["hit_type"] = np.ones(S.NUM_RAYS, np.int64)
    obs["visible_entities"]["mask"] = np.zeros(S.MAX_ENTITIES, np.int8)
    obs["visible_entities"]["features"] = np.zeros((S.MAX_ENTITIES, 5), np.float32)
    obs["visible_entities"]["entity_type"] = np.zeros(S.MAX_ENTITIES, np.int64)
    obs["vitals"] = np.zeros(len(S.VITALS_FIELDS), np.float32)
    obs["vitals"][S.VITALS_INDEX["health"]] = 100
    obs["vitals"][S.VITALS_INDEX["selected_weapon_ammo"]] = 50
    return obs


def _case(name: str) -> dict:
    o = _base_obs()
    ve, rays, vit = o["visible_entities"], o["rays"], o["vitals"]
    if name == "all_entities":
        ve["mask"][:] = 1
        ve["entity_type"][:] = np.arange(S.MAX_ENTITIES) % (len(S.ENTITY_TYPES) - 1) + 1
        ang = np.linspace(-0.7, 0.7, S.MAX_ENTITIES)
        ve["features"][:] = np.stack([np.linspace(0.05, 3.9, S.MAX_ENTITIES), np.sin(ang), np.cos(ang),
                                      np.full(S.MAX_ENTITIES, 30.0), np.full(S.MAX_ENTITIES, -30.0)], axis=1)
    elif name == "open_area":
        rays["distance"][:] = 1.0
        rays["hit_type"][:] = 0
    elif name == "face_wall":
        rays["distance"][:] = 0.0
    elif name == "health_1":
        vit[S.VITALS_INDEX["health"]] = 1
    elif name == "health_200":
        vit[S.VITALS_INDEX["health"]] = 200
        vit[S.VITALS_INDEX["armor"]] = 200
    elif name == "all_keys":
        for k in ("key_blue_card", "key_yellow_card", "key_red_card", "key_blue_skull", "key_yellow_skull", "key_red_skull"):
            vit[S.VITALS_INDEX[k]] = 1
    elif name == "zero_ammo":
        for k in ("selected_weapon_ammo", "ammo_bullets", "ammo_shells", "ammo_rockets", "ammo_cells"):
            vit[S.VITALS_INDEX[k]] = 0
    elif name == "black_frame":
        o["rgb"] = np.zeros_like(o["rgb"])
        o["depth"] = np.zeros_like(o["depth"])
    elif name == "fast_motion":
        vit[S.VITALS_INDEX["vel_forward"]] = 60.0
        vit[S.VITALS_INDEX["vel_right"]] = -60.0
    return o


CASES = ["no_entities", "all_entities", "open_area", "face_wall", "health_1", "health_200",
         "all_keys", "zero_ammo", "black_frame", "fast_motion"]


@pytest.mark.skipif(not WEIGHTS.exists() or WEIGHTS.stat().st_size < 1000, reason="weights not present (LFS)")
def test_edge_case_observations_give_valid_actions_in_time():
    from agent import DoomAgent

    agent = DoomAgent(S.observation_space(), S.action_space(), "cpu")
    agent.reset()
    for name in CASES:
        obs = _case(name)
        assert S.observation_space().contains(obs), name
        for _ in range(5):
            t0 = time.perf_counter()
            action = np.asarray(agent.act(obs))
            dt = time.perf_counter() - t0
            assert action.shape == (9,), name
            assert set(np.unique(action)).issubset({0, 1}), name
            assert dt < 0.040, f"{name}: {dt * 1000:.1f} ms"
        agent.reset()
