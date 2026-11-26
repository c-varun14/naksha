"""Procedural map families emitted as ASCII grids for compile.py.

arena    (tier 1): one or two large rooms, several monsters, items
rooms    (tier 2): room lattice with doors, openings, dead ends and loops
locks    (tier 3): rooms + 1-3 locked doors on the start->exit path, keys hidden in side branches
gauntlet (tier 4): locks + more monsters and scarcer items
maze: large DFS maze with narrow corridors
irregular: L, U and plus-shaped rooms joined by corridors
comb: a corridor with many individually doored side rooms
locksfirst: reverse-order corridor locks or a hub with key spokes
ambush: tight corridors with monster alcoves and optional close-range demons

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
    rooms: dict  # room id -> (r0, c0, r1, c1) interior bounds, inclusive
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
            if family == "locksfirst" and any(
                solvable([row.replace(KEY_OF[lock], ".") for row in rows]) for lock in meta["locks"]
            ):
                continue
            angle = meta.pop("player_angle", None)
            return {"family": family, "seed": seed, "size": size, "grid": "\n".join(rows),
                    "player_angle": rng.choice((0, 90, 180, 270)) if angle is None else angle, **meta}
    raise RuntimeError(f"could not generate a solvable {family} map for seed {seed}")


class _Retry(Exception):
    pass


def _generate(family, rng, size):
    if family == "comb":
        return _comb(rng, size)
    if family == "locksfirst":
        return _locksfirst(rng, size)
    if family == "ambush":
        return _ambush(rng, size)
    if family == "maze":
        return _maze(rng, {"s": 15, "m": 25, "l": 35}[size])
    if family == "irregular":
        return _irregular(rng, {"s": 3, "m": 4, "l": 5}[size])

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


def _carve(grid, bounds):
    r0, c0, r1, c1 = bounds
    for r in range(r0, r1 + 1):
        grid[r][c0:c1 + 1] = ["."] * (c1 - c0 + 1)


def _trim_walls(grid):
    """Crop unused space while retaining one cell of wall around the map."""
    cells = [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch != "#"]
    rows, cols = zip(*cells)
    left, right = min(cols) - 1, max(cols) + 2
    return [row[left:right] for row in grid[min(rows) - 1:max(rows) + 2]]


def _place(lay, room, rng, symbols):
    r, c = rng.choice(_free_cells(lay, room, lay.grid))
    lay.grid[r][c] = rng.choice(symbols)


def _side_rooms(rng, doors, width=1, bend_at=None):
    """Build a west-to-east corridor with one isolated room per door in walking order."""
    grid = [["#"] * (10 * len(doors) + 10) for _ in range(22)]
    lay = Layout(grid, {})
    row, start, left = 7, 1, rng.randint(4, 6)
    for room, door in enumerate(doors):
        if room == bend_at:
            # Leave the gap between rooms clear for a two-turn offset.
            turn = left - 2
            shift = rng.randint(4, 6)
            _carve(grid, (row, start, row + width - 1, turn + width - 1))
            _carve(grid, (row, turn, row + shift + width - 1, turn + width - 1))
            row, start = row + shift, turn
        right = left + rng.randint(3, 6) - 1
        depth = rng.randint(2, 5)
        if rng.choice((False, True)):
            wall = row - 1
            bounds = (wall - depth, left, wall - 1, right)
        else:
            wall = row + width
            bounds = (wall + 1, left, wall + depth, right)
        _carve(grid, bounds)
        lay.rooms[room] = bounds
        grid[wall][rng.randint(left, right)] = door
        left = right + rng.randint(4, 5)
    _carve(grid, (row, start, row + width - 1, right + rng.randint(1, 3)))
    grid[7][1] = "P"
    return lay


def _comb(rng, size):
    count = rng.randint(*{"s": (4, 6), "m": (6, 9), "l": (9, 12)}[size])
    bend = rng.randrange(1, count) if rng.choice((False, True)) else None
    lay = _side_rooms(rng, "D" * count, rng.randint(1, 2), bend)
    r, c = _exit_cell(lay, rng.randrange(1, count), lay.grid, rng)
    lay.grid[r][c] = "E"
    for symbols, total in (("hk", rng.randint(0, 2)), ("zi", rng.randint(0, 3))):
        for room in rng.sample(list(lay.rooms), total):
            _place(lay, room, rng, symbols)
    return _trim_walls(lay.grid), {"episode_ticks": 3500 if size == "l" else 2800,
                                   "locks": [], "rooms": count, "player_angle": 0}


def _locksfirst(rng, size):
    if rng.choice((False, True)):
        locks = rng.sample(list(KEY_OF), rng.randint(2, 3))
        lay = _locks_hub(rng, locks)
        goal = "exit"
    else:
        locks = rng.sample(list(KEY_OF), rng.randint(1, 3))
        lay = _side_rooms(rng, [*locks, "D"])
        for room, lock in enumerate(locks, start=1):
            _place(lay, room, rng, KEY_OF[lock])
        goal = 0
    r, c = _exit_cell(lay, goal, lay.grid, rng)
    lay.grid[r][c] = "E"
    rooms = list(lay.rooms)
    for _ in range(rng.randint(0, 3)):
        _place(lay, rng.choice(rooms), rng, "zi")
    return _trim_walls(lay.grid), {"episode_ticks": 4200 if size == "l" else 3500,
                                   "locks": locks, "rooms": len(lay.rooms)}


def _locks_hub(rng, locks):
    center = 24
    half_height, half_width = rng.randint(2, 4), rng.randint(2, 4)
    hub = (center - half_height, center - half_width, center + half_height, center + half_width)
    grid = [["#"] * (2 * center + 1) for _ in range(2 * center + 1)]
    lay = Layout(grid, {"hub": hub})
    _carve(grid, hub)
    grid[center][center] = "P"
    directions = rng.sample([(-1, 0), (0, 1), (1, 0), (0, -1)], len(locks) + 1)
    for (dr, dc), room in zip(directions, [*locks, "exit"]):
        door_types = locks if room == "exit" else ["D"]
        end = rng.randint(2, 4)
        doors = {end: door_types[0]}
        for door in door_types[1:]:
            end += rng.randint(2, 4)
            doors[end] = door
        radius = half_height if dr else half_width
        r0, c0 = center + radius * dr, center + radius * dc
        for step in range(1, end + 1):
            grid[r0 + step * dr][c0 + step * dc] = doors.get(step, ".")

        # Define the end room in coordinates along and across the spoke.
        depth = rng.randint(2, 5)
        left, right = -rng.randint(1, 2), rng.randint(1, 3)
        a = (r0 + (end + 1) * dr + left * dc, c0 + (end + 1) * dc - left * dr)
        b = (r0 + (end + depth) * dr + right * dc, c0 + (end + depth) * dc - right * dr)
        bounds = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        _carve(grid, bounds)
        lay.rooms[room] = bounds
        if room != "exit":
            _place(lay, room, rng, KEY_OF[room])
    return lay


def _ambush(rng, size):
    count = rng.randint(*{"s": (4, 6), "m": (5, 7), "l": (6, 8)}[size])
    column = rng.randint(11, 13)
    alcoves = []
    for _ in range(count):
        alcoves.append(column)
        column += rng.randint(2, 4)
    end = column + rng.randint(2, 4)
    row = 7
    grid = [["#"] * (end + 2) for _ in range(15)]
    lay = Layout(grid, {})
    _carve(grid, (row, 1, row, end))
    start_room = rng.choice((False, True))
    if start_room:
        start = (row - 2, 1, row + 2, 5)
        _carve(grid, start)
        lay.rooms["start"] = start
    grid[row][2] = "P"
    if start_room:
        nearby = [(r, c) for r, c in _free_cells(lay, "start", grid) if abs(r - row) + abs(c - 2) <= 4]
        for r, c in rng.sample(nearby, rng.randint(2, 4)):
            grid[r][c] = "d"

    sides = [-1, 1] + rng.choices((-1, 1), k=count - 2)
    rng.shuffle(sides)
    monsters = rng.sample(MONSTERS_HARD, min(count, len(MONSTERS_HARD)))
    monsters += rng.choices(MONSTERS_HARD, k=count - len(monsters))
    rng.shuffle(monsters)
    for c, side, monster in zip(alcoves, sides, monsters):
        depth = rng.randint(1, 3)
        for step in range(1, depth):
            grid[row + step * side][c] = "."
        grid[row + depth * side][c] = monster

    locks = []
    if rng.choice((False, True)):
        lock = rng.choice(list(KEY_OF))
        locks.append(lock)
        side = rng.choice((-1, 1))
        near, far = sorted((row + 2 * side, row + rng.randint(3, 5) * side))
        lay.rooms["exit"] = (near, 7, far, 9)
        _carve(grid, lay.rooms["exit"])
        grid[row + side][8] = lock
        grid[row][end] = KEY_OF[lock]
        r, c = _exit_cell(lay, "exit", grid, rng)
        grid[r][c] = "E"
    else:
        grid[row][end] = "E"
    pickups = [rng.choice("hm"), "k"] + rng.choices("hmk", k=rng.randint(0, 1))
    cells = [c for c in range(6, end) if not _touches_door(grid, row, c)]
    for c, pickup in zip(rng.sample(cells, len(pickups)), pickups):
        grid[row][c] = pickup
    return _trim_walls(grid), {"episode_ticks": 3500, "locks": locks, "rooms": count + len(lay.rooms),
                               "player_angle": 0}


def _maze(rng, side):
    width = rng.randint(1, 2)
    lay = _lattice(rng, side, side, width, width, width, width)
    _, start, tree, extras = _room_graph(rng, side, side, 0.0, rng.randint(0, 3))
    g = lay.grid
    for a, b in tree + extras:
        _merge_wall(lay, a, b, g)
    pr, pc, _, _ = lay.rooms[start]

    # BFS order gives the farthest cell after any loop shortcuts.
    seen, queue = {(pr, pc)}, deque([(pr, pc)])
    while queue:
        er, ec = queue.popleft()
        for r, c in _neighbours(er, ec, len(g), len(g[0])):
            if g[r][c] == "." and (r, c) not in seen:
                seen.add((r, c))
                queue.append((r, c))
    g[pr][pc], g[er][ec] = "P", "E"

    # A one-cell door spans a narrow passage. Wide mazes keep their full width.
    if width == 1:
        candidates = [_opening(lay, a, b, rng) for a, b in tree]
        candidates = [(r, c) for r, c in candidates if abs(r - er) + abs(c - ec) > 1
                      and g[r][c] == "."]
        for r, c in rng.sample(candidates, rng.randint(0, 2)):
            g[r][c] = "D"
    return g, {"episode_ticks": 4200, "locks": [], "rooms": side * side}


def _irregular(rng, side):
    pitch = 16  # Room widths of 9–11 cells leave a gap for corridors.
    extent = pitch * side + 1
    lay = Layout([["#"] * extent for _ in range(extent)], {})
    g = lay.grid
    keep, start, tree, _ = _room_graph(rng, side, side, 0.0, 0)
    centers = {}
    shapes = ["L", "U", "+"]
    for i, j in sorted(keep):
        r, c = centers[i, j] = (pitch * i + pitch // 2, pitch * j + pitch // 2)
        h, w = rng.randint(4, 5), rng.randint(4, 5)
        lay.rooms[i, j] = (r - h, c - w, r + h, c + w)
        shape = shapes.pop(rng.randrange(len(shapes))) if shapes else rng.choice(("L", "U", "+"))
        if shape == "+":
            rects = [(-1, -w, 1, w), (-h, -1, h, 1)]
        else:
            rects = [(-1, -w, h, w), (-h, -w, h, -2)]
            if shape == "U":
                rects.append((-h, 2, h, w))
        flip_r, flip_c = rng.choice((-1, 1)), rng.choice((-1, 1))
        for r0, c0, r1, c1 in rects:
            for dr in range(r0, r1 + 1):
                for dc in range(c0, c1 + 1):
                    g[r + flip_r * dr][c + flip_c * dc] = "."
        pillars = [(y, x) for y in range(r - h + 1, r + h) for x in range(c - w + 1, c + w)
                   if abs(y - r) > 1 and abs(x - c) > 1
                   and all(g[y + dy][x + dx] == "." for dy in (-1, 0, 1) for dx in (-1, 0, 1))]
        if pillars:
            y, x = rng.choice(pillars)
            g[y][x] = "#"

    _, adj = _tree_path(tree, start, start)
    queue, seen = deque([start]), {start}
    while queue:
        goal = queue.popleft()
        for room in adj[goal]:
            if room not in seen:
                seen.add(room)
                queue.append(room)
    path, _ = _tree_path(tree, start, goal)
    on_path = set(path)
    branches = [(a, b, [room for room in adj[a] if room not in on_path])
                for a, b in zip(path, path[1:])]
    lock_sites = [(a, b, rooms) for a, b, rooms in branches if rooms]
    num_locks = rng.randint(0, min(2, len(lock_sites)))
    locks = {}
    for (a, b, rooms), color in zip(rng.sample(lock_sites, num_locks), rng.sample(sorted(KEY_OF), num_locks)):
        locks[frozenset((a, b))] = color
        y, x = rng.choice(_free_cells(lay, rng.choice(rooms), g))
        g[y][x] = KEY_OF[color]

    # Carve from center to center so the corridor meets even recessed room faces.
    # Doors occupy the middle cell of a one-wide connector between room footprints.
    door_edges = {frozenset(edge) for edge in rng.sample(tree, rng.randint(1, min(4, len(tree))))}
    for a, b in tree:
        edge = frozenset((a, b))
        door = locks.get(edge, "D" if edge in door_edges else ".")
        width = rng.randint(1, 3) if door == "." else 1
        r, c = centers[a]
        y, x = centers[b]
        dr, dc = (y - r) // pitch, (x - c) // pitch
        for step in range(1, pitch):
            for offset in range(-(width // 2), width - width // 2):
                rr, cc = r + step * dr + offset * dc, c + step * dc + offset * dr
                if g[rr][cc] == "#":
                    g[rr][cc] = "."
        g[r + pitch // 2 * dr][c + pitch // 2 * dc] = door

    pr, pc = centers[start]
    g[pr][pc] = "P"
    er, ec = _exit_cell(lay, goal, g, rng)
    g[er][ec] = "E"
    for symbols, count in ((MONSTERS_EASY, rng.randint(0, side)), (ITEMS, rng.randint(1, side))):
        rooms = sorted(keep - {start})
        for room in rng.sample(rooms, count):
            r, c = rng.choice(_free_cells(lay, room, g))
            g[r][c] = rng.choice(symbols)
    return g, {"episode_ticks": 4200, "locks": list(locks.values()), "rooms": len(keep)}


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
