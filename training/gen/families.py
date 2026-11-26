"""Procedural map families on a room lattice, emitted as ASCII grids for compile.py.

arena    (tier 1): one or two large rooms, several monsters, items
rooms    (tier 2): room lattice with doors, openings, dead ends and loops
locks    (tier 3): rooms + 1-3 locked doors on the start->exit path, keys hidden in side branches
gauntlet (tier 4): locks + more monsters and scarcer items

Every map is checked for key-aware solvability before it is returned.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

MONSTERS_EASY = "zzzsi"
MONSTERS_HARD = "zsicd"
ITEMS = "hmakl"
KEY_OF = {"R": "r", "B": "b", "Y": "y"}
DOORS = set("DRBY")


@dataclass
class Layout:
    grid: list[list[str]]
    rooms: dict  # (i, j) -> (r0, c0, r1, c1) interior bounds, inclusive
    edges: list = field(default_factory=list)  # ((i,j),(i2,j2), (r,c) opening cell, kind)


def _lattice(rng: random.Random, nr: int, nc: int, wmin=3, wmax=7, hmin=3, hmax=6) -> Layout:
    cw = [rng.randint(wmin, wmax) for _ in range(nc)]
    rh = [rng.randint(hmin, hmax) for _ in range(nr)]
    W = 1 + sum(w + 1 for w in cw)
    H = 1 + sum(h + 1 for h in rh)
    g = [["#"] * W for _ in range(H)]
    rooms = {}
    y = 1
    for i in range(nr):
        x = 1
        for j in range(nc):
            rooms[(i, j)] = (y, x, y + rh[i] - 1, x + cw[j] - 1)
            for r in range(y, y + rh[i]):
                for c in range(x, x + cw[j]):
                    g[r][c] = "."
            x += cw[j] + 1
        y += rh[i] + 1
    return Layout(g, rooms)


def _neighbours(i, j, nr, nc):
    for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        a, b = i + di, j + dj
        if 0 <= a < nr and 0 <= b < nc:
            yield (a, b)


def _opening(lay: Layout, a, b, rng):
    """A wall cell between rooms a and b with both perpendicular neighbours inside the rooms."""
    (r0, c0, r1, c1), (s0, d0, s1, d1) = lay.rooms[a], lay.rooms[b]
    if a[0] == b[0]:  # side by side: vertical wall column
        col = c1 + 1 if a[1] < b[1] else d1 + 1
        lo, hi = max(r0, s0), min(r1, s1)
        return (rng.randint(lo, hi), col)
    row = r1 + 1 if a[0] < b[0] else s1 + 1
    lo, hi = max(c0, d0), min(c1, d1)
    return (row, rng.randint(lo, hi))


def _room_graph(rng, nr, nc, drop_frac, extra_loops):
    cells = [(i, j) for i in range(nr) for j in range(nc)]
    keep = set(cells)
    for c in rng.sample(cells, int(len(cells) * drop_frac)):
        keep.discard(c)
        # keep connectivity
        if not _connected(keep, nr, nc):
            keep.add(c)
    start = rng.choice(sorted(keep))
    # random spanning tree (randomised DFS)
    tree, seen, stack = [], {start}, [start]
    while stack:
        cur = stack[-1]
        nb = [n for n in _neighbours(*cur, nr, nc) if n in keep and n not in seen]
        if not nb:
            stack.pop()
            continue
        n = rng.choice(nb)
        seen.add(n)
        tree.append((cur, n))
        stack.append(n)
    tree_set = {frozenset(e) for e in tree}
    extras = [(a, b) for a in keep for b in _neighbours(*a, nr, nc)
              if b in keep and a < b and frozenset((a, b)) not in tree_set]
    rng.shuffle(extras)
    return keep, start, tree, extras[:extra_loops]


def _connected(keep, nr, nc):
    if not keep:
        return False
    s = next(iter(keep))
    seen, q = {s}, [s]
    while q:
        cur = q.pop()
        for n in _neighbours(*cur, nr, nc):
            if n in keep and n not in seen:
                seen.add(n)
                q.append(n)
    return len(seen) == len(keep)


def _tree_path(tree, a, b):
    adj = {}
    for x, y in tree:
        adj.setdefault(x, []).append(y)
        adj.setdefault(y, []).append(x)
    prev, q = {a: None}, deque([a])
    while q:
        cur = q.popleft()
        for n in adj.get(cur, []):
            if n not in prev:
                prev[n] = cur
                q.append(n)
    path, cur = [], b
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1], adj


def _free_cells(lay, room, g, avoid=()):
    r0, c0, r1, c1 = lay.rooms[room]
    out = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if g[r][c] == "." and (r, c) not in avoid and not _touches_door(g, r, c):
                out.append((r, c))
    return out


def _touches_door(g, r, c):
    return any(g[r + dr][c + dc] in DOORS for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))


def _exit_cell(lay, room, g, rng):
    """Prefer a cell against the room wall (an alcove look), never next to a door."""
    cand = [(r, c) for r, c in _free_cells(lay, room, g)
            if any(g[r + dr][c + dc] == "#" for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
    return rng.choice(cand or _free_cells(lay, room, g))


def generate(family: str, seed: int, size: str = "m") -> dict:
    """Return a map spec dict (see INTERFACES §5 minus wad path)."""
    for attempt in range(50):
        rng = random.Random(seed * 1000 + attempt)
        try:
            grid, meta = _generate(family, rng, size)
        except _Retry:
            continue
        rows = ["".join(r) for r in grid]
        if solvable(rows):
            return {"family": family, "seed": seed, "size": size, "grid": "\n".join(rows),
                    "player_angle": rng.choice((0, 90, 180, 270)), **meta}
    raise RuntimeError(f"could not generate a solvable {family} map for seed {seed}")


class _Retry(Exception):
    pass


def _generate(family, rng, size):
    dims = {"s": (2, 2), "m": (rng.randint(2, 3), rng.randint(2, 4)), "l": (rng.randint(3, 4), rng.randint(3, 5))}[size]
    if family == "arena":
        nr, nc = (1, rng.randint(1, 2)) if size != "l" else (2, 2)
        lay = _lattice(rng, nr, nc, 6, 11, 5, 9)
        keep, start, tree, extras = _room_graph(rng, nr, nc, 0.0, 0)
        door_p, n_locks, n_mon, n_items, ticks, mons = 0.0, 0, rng.randint(3, 8), rng.randint(2, 5), 2100, MONSTERS_EASY
        merge_p = 0.5
    else:
        nr, nc = dims
        lay = _lattice(rng, nr, nc)
        drop = rng.choice((0.0, 0.15, 0.25)) if nr * nc > 4 else 0.0
        keep, start, tree, extras = _room_graph(rng, nr, nc, drop, rng.randint(0, 2))
        merge_p = 0.15
        if family == "rooms":
            door_p, n_locks, ticks, mons = 0.6, 0, 2100 if nr * nc <= 6 else 3150, MONSTERS_EASY
            n_mon, n_items = rng.randint(0, 4), rng.randint(1, 4)
        elif family == "locks":
            door_p, n_locks, ticks, mons = 0.5, rng.choice((1, 1, 2, 2, 3)), 4200, MONSTERS_EASY
            n_mon, n_items = rng.randint(0, 3), rng.randint(1, 3)
        elif family == "gauntlet":
            door_p, n_locks, ticks, mons = 0.5, rng.choice((1, 2, 3)), 4200, MONSTERS_HARD
            n_mon, n_items = rng.randint(4, 9), rng.randint(0, 2)
        else:
            raise ValueError(family)
    g = lay.grid
    rooms_kept = sorted(keep)
    for room in set(lay.rooms) - keep:  # fill dropped rooms with wall
        r0, c0, r1, c1 = lay.rooms[room]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                g[r][c] = "#"

    path, adj = _tree_path(tree, start, start)
    # goal room: farthest room from start in the tree
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        for n in adj.get(cur, []):
            if n not in dist:
                dist[n] = dist[cur] + 1
                q.append(n)
    goal = max(rooms_kept, key=lambda r: (dist.get(r, -1), rng.random())) if len(rooms_kept) > 1 else start
    path, adj = _tree_path(tree, start, goal)

    # locked edges on the start->goal path (tree edges only, so they cut the graph)
    path_edges = list(zip(path, path[1:]))
    locks = []
    if n_locks and len(path_edges) >= 1:
        n_locks = min(n_locks, len(path_edges))
        idx = sorted(rng.sample(range(len(path_edges)), n_locks))
        colours = rng.sample(["R", "B", "Y"], n_locks)
        locks = [(path_edges[k], colours[n]) for n, k in enumerate(idx)]
    locked = {frozenset(e): col for e, col in locks}

    # zones: components of the tree after cutting locked edges; loops only inside a zone
    def zone_of():
        z, zid = {}, 0
        for room in rooms_kept:
            if room in z:
                continue
            st = [room]
            z[room] = zid
            while st:
                cur = st.pop()
                for n in adj.get(cur, []):
                    if n not in z and frozenset((cur, n)) not in locked:
                        z[n] = zid
                        st.append(n)
            zid += 1
        return z

    zone = zone_of()
    edges = [(a, b) for a, b in tree] + [(a, b) for a, b in extras if zone[a] == zone[b]]
    merged_rooms = set()
    for a, b in edges:
        col = locked.get(frozenset((a, b)))
        if col is None and rng.random() < merge_p and (a, b) not in merged_rooms:
            _merge_wall(lay, a, b, g)
            merged_rooms.add((a, b))
            continue
        r, c = _opening(lay, a, b, rng)
        g[r][c] = col if col else ("D" if rng.random() < door_p else ".")

    # place keys: key for lock k goes in the zone reachable before lock k, preferring a side
    # branch (a room off the start->goal path) so the agent must explore and come back.
    path_set = set(path)
    for (e, col) in locks:
        before = e[0] if path.index(e[0]) < path.index(e[1]) else e[1]
        zid = zone[before]
        zone_rooms = [rm for rm in rooms_kept if zone[rm] == zid]
        side = [rm for rm in zone_rooms if rm not in path_set]
        cand_rooms = side or [rm for rm in zone_rooms if rm != start] or zone_rooms
        room = rng.choice(cand_rooms)
        cells = _free_cells(lay, room, g)
        if not cells:
            raise _Retry
        r, c = rng.choice(cells)
        g[r][c] = KEY_OF[col]

    pr, pc = rng.choice(_free_cells(lay, start, g))
    g[pr][pc] = "P"
    er, ec = _exit_cell(lay, goal, g, rng)
    if (er, ec) == (pr, pc):
        raise _Retry
    g[er][ec] = "E"

    occupied = {(pr, pc), (er, ec)}
    other_rooms = [rm for rm in rooms_kept if rm != start] or rooms_kept
    for _ in range(n_mon):
        rm = rng.choice(other_rooms)
        cells = [x for x in _free_cells(lay, rm, g) if x not in occupied and abs(x[0] - pr) + abs(x[1] - pc) > 4]
        if cells:
            r, c = rng.choice(cells)
            g[r][c] = rng.choice(mons)
            occupied.add((r, c))
    for _ in range(n_items):
        rm = rng.choice(rooms_kept)
        cells = [x for x in _free_cells(lay, rm, g) if x not in occupied]
        if cells:
            r, c = rng.choice(cells)
            g[r][c] = rng.choice(ITEMS)
            occupied.add((r, c))
    return g, {"episode_ticks": ticks, "locks": [col for _, col in locks], "rooms": len(rooms_kept)}


def _merge_wall(lay, a, b, g):
    """Remove the whole wall segment between rooms a and b (bigger, irregular rooms)."""
    (r0, c0, r1, c1), (s0, d0, s1, d1) = lay.rooms[a], lay.rooms[b]
    if a[0] == b[0]:
        col = c1 + 1 if a[1] < b[1] else d1 + 1
        for r in range(max(r0, s0), min(r1, s1) + 1):
            g[r][col] = "."
    else:
        row = r1 + 1 if a[0] < b[0] else s1 + 1
        for c in range(max(c0, d0), min(c1, d1) + 1):
            g[row][c] = "."


def solvable(rows: list[str]) -> bool:
    """BFS over (cell, keys held): can the player reach E? Doors pass, locked doors need the key."""
    H, W = len(rows), len(rows[0])
    start = next((r, c) for r in range(H) for c in range(W) if rows[r][c] == "P")
    keys0 = frozenset()
    seen = {(start, keys0)}
    q = deque([(start, keys0)])
    while q:
        (r, c), keys = q.popleft()
        ch = rows[r][c]
        if ch == "E":
            return True
        if ch in "rby":
            keys = keys | {ch}
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            n = rows[nr][nc]
            if n == "#":
                continue
            if n in "RBY" and n.lower() not in keys:
                continue
            st = ((nr, nc), keys)
            if st not in seen:
                seen.add(st)
                q.append(st)
    return False


if __name__ == "__main__":
    import sys

    fam = sys.argv[1] if len(sys.argv) > 1 else "locks"
    for s in range(3):
        spec = generate(fam, s, "m")
        print(spec["family"], spec["seed"], spec["locks"], spec["episode_ticks"])
        print(spec["grid"])
        print()
