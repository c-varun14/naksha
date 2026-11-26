"""Survival-focused teacher using visible threats and remembered health pickups."""

from __future__ import annotations

import math

import numpy as np

from training.teacher.teacher import ACT, EXPLORE, FIGHT, Teacher

MELEE, HITSCAN, RANGED, PROJECTILE, HEALTH = range(1, 6)
ENGAGE_RANGE = {MELEE: 450.0, HITSCAN: 800.0, RANGED: 700.0}


class Teacher2(Teacher):
    def reset(self, env=None):
        super().reset(env)
        self.health_seen: set[tuple[int, int]] = set()
        self.dodge: tuple[int, int] | None = None

    def _decide(self, obs, x, y, heading, ceil, floor, vit, enemies):
        entities = obs["visible_entities"]
        rays = obs["rays"]["distance"] * 1024.0
        targets = []
        dodge = None
        visible_health = set()
        for i in np.flatnonzero(entities["mask"]):
            kind = int(entities["entity_type"][i])
            dist, sin, cos, forward, right = map(float, entities["features"][i])
            dist *= 1024.0
            rel = math.degrees(math.atan2(sin, cos))
            closing = sin * right - cos * forward
            if kind == HEALTH:
                angle = math.radians(heading + rel)
                rc = self.cell(x + dist * math.cos(angle), y + dist * math.sin(angle))
                self.known[rc] = True
                visible_health.add(rc)
            elif kind in ENGAGE_RANGE and dist < ENGAGE_RANGE[kind]:
                priority = dist
                if kind == HITSCAN:
                    priority -= 100.0
                elif kind == MELEE and dist < 200.0:
                    priority -= 180.0
                targets.append((priority, dist, rel, kind, closing))
            elif kind == PROJECTILE and dist < 300 and closing > 6 and forward < -4:
                # Predict the projectile's lateral offset as it crosses our forward axis.
                t = -dist * cos / forward
                offset = -dist * sin + right * t
                if 0 < t <= 24 and abs(offset) < 32 and (dodge is None or t < dodge[0]):
                    dodge = (t, offset)

        start = self.cell(x, y)
        if start not in visible_health and math.dist((x, y), self.centre(start)) < 20:
            self.health_seen.discard(start)
        self.health_seen.update(visible_health)

        if self.dodge is not None:
            macro, ray = self.dodge
            self.dodge = None
            if rays[ray] > 32:
                return macro, FIGHT, False
        if dodge is not None:
            left = dodge[1] > 0
            for side in (left, not left):
                ray = 16 if side else 48
                if rays[ray] > 40:
                    macro = ACT["strafe_left"] if side else ACT["strafe_right"]
                    self.dodge = (macro, ray)
                    return macro, FIGHT, False

        if targets and vit[3] > 0:
            _, dist, rel, kind, closing = min(targets)
            melee_threat = kind == MELEE and (dist < 96 or (dist < 200 and closing > 1))
            just_fired = self.prev_macro in (ACT["attack"], ACT["attack_turn_left"], ACT["attack_turn_right"])
            if melee_threat and abs(rel) <= 25 and rays[32] > 44 and just_fired:
                return ACT["back"], FIGHT, False
            enemies = [(dist, rel)]
        else:
            enemies = []
        return super()._decide(obs, x, y, heading, ceil, floor, vit, enemies)

    def _choose(self, start, vit, heading, x, y, ceil, floor):
        intent, path = super()._choose(start, vit, heading, x, y, ceil, floor)
        if intent == EXPLORE and vit[0] < 40 and self.health_seen:
            prev, order = self._bfs(start, vit)
            for rc in order:
                if rc in self.health_seen:
                    health_path = self._path(prev, rc)
                    if len(health_path) <= 7:
                        return EXPLORE, health_path
                    break
        return intent, path

    def executed(self, macro: int, label: int, last_use_before: int) -> None:
        super().executed(macro, label, last_use_before)
        if macro != label:
            self.dodge = None
