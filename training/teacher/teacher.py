"""Reference policy used only to label training data (never part of the submission).

It knows its exact pose (privileged state) and the map grid, but it only plans over cells the
agent has actually observed: cells swept by its own 64 rays, keys seen in its field of view, and
the exit only once it was in view with a clear line of sight. That keeps its behaviour
reproducible from the student's observations.

Priorities: fight a close visible enemy > fetch a seen key > open the matching locked door >
go to a seen exit > explore the nearest frontier > visit unvisited known cells.
Doors: press USE once, release, wait for the door to rise, walk through.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from training import paths  # noqa: F401
from naksha.actions import INDEX as ACT
from training.gen.compile import DOORS, _sectors, parse

EXPLORE, GET_KEY, GO_DOOR, GO_EXIT, FIGHT = range(5)
CELL = 64.0
KEY_VITAL = {"r": 10, "b": 8, "y": 9}  # vitals index of the keycard flag
LOCK_KEY = {"R": "r", "B": "b", "Y": "y"}
ENEMY_TYPES = (1, 2, 3)
KEY_TYPE = 9
FIGHT_RANGE = 800.0
EXIT_VIEW_RANGE = 12 * CELL
NEIGH = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


class Teacher:
    def __init__(self, spec: dict, seed: int = 0):
        self.rows = parse(spec["grid"])
        self.H, self.W = len(self.rows), len(self.rows[0])
        self.sid, _, self.kinds = _sectors(self.rows)
        g = np.array([list(r) for r in self.rows])
        self.wall = g == "#"
        self.door = np.isin(g, list(DOORS))
        self.locked = np.isin(g, list("RBY"))
        self.keys = {(r, c): g[r, c] for r, c in zip(*np.nonzero(np.isin(g, list("rby"))))}
        ex = np.argwhere(g == "E")
        self.exit = tuple(ex[0]) if len(ex) else None
        self.g = g
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ geometry
    def cell(self, x: float, y: float) -> tuple[int, int]:
        c = int(math.floor(x / CELL))
        r = self.H - 1 - int(math.floor(y / CELL))
        return min(max(r, 0), self.H - 1), min(max(c, 0), self.W - 1)

    def centre(self, rc) -> tuple[float, float]:
        r, c = rc
        return c * CELL + CELL / 2, (self.H - 1 - r) * CELL + CELL / 2

    def _cells_of(self, xs: np.ndarray, ys: np.ndarray):
        cs = np.clip(np.floor(xs / CELL).astype(int), 0, self.W - 1)
        rs = np.clip(self.H - 1 - np.floor(ys / CELL).astype(int), 0, self.H - 1)
        return rs, cs

    # ------------------------------------------------------------------ episode
    def reset(self, env=None):
        self.env = env
        self.known = np.zeros((self.H, self.W), bool)
        self.visited = np.zeros((self.H, self.W), bool)
        self.key_seen: set = set()
        self.exit_seen = False
        self.t = 0
        self.last_use = -100
        self.prev_macro = ACT["noop"]
        self.prev_ceil = None
        self.trail: deque = deque(maxlen=12)
        self.recover = 0
        self.recover_macro = ACT["back"]

    def door_open(self, rc, ceil, floor) -> bool:
        s = self.sid[rc[0]][rc[1]]
        return bool(ceil[s] - floor[s] >= 60)

    def door_moving(self, rc, ceil) -> bool:
        if self.prev_ceil is None:
            return False
        s = self.sid[rc[0]][rc[1]]
        return bool(abs(ceil[s] - self.prev_ceil[s]) > 0.5)

    def has_key(self, vit, colour: str) -> bool:
        return bool(vit[KEY_VITAL[colour]] > 0.5)

    # ------------------------------------------------------------------ perception
    def _observe(self, obs, x, y, heading, ceil, floor):
        rays = obs["rays"]
        d = np.asarray(rays["distance"], np.float64) * 1024.0
        ang = np.radians(heading + np.arange(64) * 5.625)
        ts = np.arange(0.0, 1024.0, 16.0)
        m = ts[None, :] <= d[:, None]
        px = x + np.cos(ang)[:, None] * ts[None, :]
        py = y + np.sin(ang)[:, None] * ts[None, :]
        rs, cs = self._cells_of(px[m], py[m])
        self.known[rs, cs] = True
        hit = d < 1023.0
        hx = x + np.cos(ang[hit]) * (d[hit] + 8.0)
        hy = y + np.sin(ang[hit]) * (d[hit] + 8.0)
        rs, cs = self._cells_of(hx, hy)
        self.known[rs, cs] = True
        r0, c0 = self.cell(x, y)
        self.known[r0, c0] = True
        self.visited[r0, c0] = True

        ve = obs["visible_entities"]
        mask = np.asarray(ve["mask"]) > 0
        feats = np.asarray(ve["features"], np.float64)
        types = np.asarray(ve["entity_type"])
        enemies = []
        h = math.radians(heading)
        for i in np.nonzero(mask)[0]:
            dist = feats[i, 0] * 1024.0
            rel = math.degrees(math.atan2(feats[i, 1], feats[i, 2]))
            if types[i] == KEY_TYPE:
                a = h + math.radians(rel)
                kc = self.cell(x + dist * math.cos(a), y + dist * math.sin(a))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        cand = (kc[0] + dr, kc[1] + dc)
                        if cand in self.keys:
                            self.key_seen.add(cand)
            elif types[i] in ENEMY_TYPES:
                enemies.append((dist, rel))

        exit_vis, exit_dir = 0, (0.0, 0.0)
        if self.exit is not None:
            ex, ey = self.centre(self.exit)
            dist = math.hypot(ex - x, ey - y)
            rel = _wrap(math.degrees(math.atan2(ey - y, ex - x)) - heading)
            if dist <= EXIT_VIEW_RANGE and abs(rel) <= 46.0 and self._los(x, y, ex, ey, ceil, floor):
                exit_vis = 1
                exit_dir = (math.sin(math.radians(rel)), math.cos(math.radians(rel)))
                self.exit_seen = True
        return enemies, exit_vis, exit_dir

    def _los(self, x0, y0, x1, y1, ceil, floor) -> bool:
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 16.0))
        xs = np.linspace(x0, x1, n)[1:-1]
        ys = np.linspace(y0, y1, n)[1:-1]
        rs, cs = self._cells_of(xs, ys)
        for r, c in zip(rs, cs):
            if self.wall[r, c]:
                return False
            if self.door[r, c] and not self.door_open((r, c), ceil, floor):
                return False
        return True

    # ------------------------------------------------------------------ planning
    def _passable(self, r, c, vit) -> bool:
        if not self.known[r, c] or self.wall[r, c]:
            return False
        ch = self.g[r, c]
        if ch in LOCK_KEY:
            return self.has_key(vit, LOCK_KEY[ch])
        return True

    def _bfs(self, start, vit):
        prev = {start: None}
        q = deque([start])
        order = []
        while q:
            cur = q.popleft()
            order.append(cur)
            for dr, dc in NEIGH:
                n = (cur[0] + dr, cur[1] + dc)
                if 0 <= n[0] < self.H and 0 <= n[1] < self.W and n not in prev and self._passable(*n, vit):
                    prev[n] = cur
                    q.append(n)
        return prev, order

    @staticmethod
    def _path(prev, goal):
        out = []
        while goal is not None:
            out.append(goal)
            goal = prev[goal]
        return out[::-1]

    def _frontier(self, rc) -> bool:
        r, c = rc
        for dr, dc in NEIGH:
            n = (r + dr, c + dc)
            if 0 <= n[0] < self.H and 0 <= n[1] < self.W and not self.known[n]:
                return True
        return False

    def _choose(self, start, vit, heading, x, y, ceil, floor):
        prev, order = self._bfs(start, vit)
        reach = set(order)
        keys = [k for k in self.key_seen if not self.has_key(vit, self.keys[k]) and k in reach]
        if keys:
            return GET_KEY, self._path(prev, min(keys, key=lambda k: len(self._path(prev, k))))
        doors = [d for d in reach if self.g[d] in LOCK_KEY and not self.door_open(d, ceil, floor)]
        if doors:
            return GO_DOOR, self._path(prev, min(doors, key=lambda d: len(self._path(prev, d))))
        if self.exit_seen and self.exit in reach:
            return GO_EXIT, self._path(prev, self.exit)
        front = [rc for rc in order if self._frontier(rc) and rc != start]
        if front:
            def cost(rc):
                cx, cy = self.centre(rc)
                turn = abs(_wrap(math.degrees(math.atan2(cy - y, cx - x)) - heading))
                return len(self._path(prev, rc)) + turn / 90.0 * 1.5
            return EXPLORE, self._path(prev, min(front, key=cost))
        unvisited = [rc for rc in order if not self.visited[rc]]
        if unvisited:
            return EXPLORE, self._path(prev, unvisited[0])
        return EXPLORE, None

    def _steer(self, x, y, heading, tx, ty) -> int:
        delta = _wrap(math.degrees(math.atan2(ty - y, tx - x)) - heading)
        if abs(delta) > 50:
            return ACT["turn_left"] if delta > 0 else ACT["turn_right"]
        if abs(delta) > 12:
            return ACT["forward_turn_left"] if delta > 0 else ACT["forward_turn_right"]
        return ACT["forward"]

    # ------------------------------------------------------------------ main
    def act(self, obs, priv) -> tuple[int, int, dict]:
        x, y, heading = priv["x"], priv["y"], priv["angle"]
        ceil = priv["sector_ceiling"]
        floor = priv["sector_floor"]
        vit = np.asarray(obs["vitals"])
        enemies, exit_vis, exit_dir = self._observe(obs, x, y, heading, ceil, floor)
        macro, intent, waiting = self._decide(obs, x, y, heading, ceil, floor, vit, enemies)

        # stuck recovery: if we keep trying to move but do not, back off / sidestep briefly
        self.trail.append((x, y))
        moving = macro in (ACT["forward"], ACT["forward_turn_left"], ACT["forward_turn_right"])
        if self.recover > 0:
            macro = self.recover_macro
            self.recover -= 1
        elif moving and not waiting and len(self.trail) == self.trail.maxlen:
            (x0, y0) = self.trail[0]
            if math.hypot(x - x0, y - y0) < 10.0:
                self.recover = 4
                self.recover_macro = int(self.rng.choice([ACT["back"], ACT["strafe_left"], ACT["strafe_right"]]))
                macro = self.recover_macro
                self.trail.clear()

        if macro == ACT["use"]:
            self.last_use = self.t
        self.prev_macro = macro
        self.prev_ceil = np.array(ceil, copy=True)
        self.t += 1
        return int(macro), int(intent), {"exit_vis": exit_vis, "exit_dir": exit_dir}

    def executed(self, macro: int, label: int, last_use_before: int) -> None:
        """DAgger: another policy chose the executed macro; keep the USE edge logic truthful."""
        if macro != label:
            self.prev_macro = macro
            self.last_use = self.t - 1 if macro == ACT["use"] else last_use_before
            self.recover = 0

    def _decide(self, obs, x, y, heading, ceil, floor, vit, enemies):
        if enemies and vit[3] > 0:
            dist, rel = min(enemies)
            if dist < FIGHT_RANGE:
                if abs(rel) > 25:
                    return (ACT["turn_left"] if rel > 0 else ACT["turn_right"]), FIGHT, False
                if abs(rel) > 5:
                    return (ACT["attack_turn_left"] if rel > 0 else ACT["attack_turn_right"]), FIGHT, False
                return ACT["attack"], FIGHT, False

        start = self.cell(x, y)
        intent, path = self._choose(start, vit, heading, x, y, ceil, floor)
        if not path:
            return ACT["turn_left"], intent, False
        if len(path) == 1:
            tx, ty = self.centre(path[0])
            if math.hypot(tx - x, ty - y) < 12:
                return ACT["turn_left"], intent, False
            return self._steer(x, y, heading, tx, ty), intent, False

        nxt = path[1]
        if self.door[nxt] and not self.door_open(nxt, ceil, floor):
            tx, ty = self.centre(nxt)
            dist = math.hypot(tx - x, ty - y)
            if dist <= 100:
                delta = _wrap(math.degrees(math.atan2(ty - y, tx - x)) - heading)
                if abs(delta) > 15:
                    return (ACT["turn_left"] if delta > 0 else ACT["turn_right"]), intent, True
                if (not self.door_moving(nxt, ceil) and self.t - self.last_use >= 8
                        and self.prev_macro != ACT["use"]):
                    return ACT["use"], intent, True
                return ACT["noop"], intent, True
            return self._steer(x, y, heading, tx, ty), intent, False

        wp = nxt
        cx, cy = self.centre(nxt)
        if len(path) > 2 and math.hypot(cx - x, cy - y) < 24:
            wp = path[2]
        tx, ty = self.centre(wp)
        return self._steer(x, y, heading, tx, ty), intent, False
