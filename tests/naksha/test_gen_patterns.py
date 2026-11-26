from collections import deque

import pytest

from training.gen.compile import compile_grid
from training.gen.families import generate, solvable


def _shortest_distance(rows):
    start = next((r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "P")
    seen = {(start, frozenset())}
    queue = deque([(start, frozenset(), 0)])
    while queue:
        (r, c), keys, distance = queue.popleft()
        if rows[r][c] == "E":
            return distance
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            nr, nc = r + dr, c + dc
            ch = rows[nr][nc]
            if ch == "#" or (ch in "RBY" and ch.lower() not in keys):
                continue
            state = ((nr, nc), keys | {ch} if ch in "rby" else keys)
            if state not in seen:
                seen.add(state)
                queue.append((*state, distance + 1))


@pytest.mark.parametrize("family", ("comb", "locksfirst", "ambush"))
@pytest.mark.parametrize("size", "sml")
@pytest.mark.parametrize("seed", range(15))
def test_pattern_compiles_and_is_solvable(family, size, seed):
    spec = generate(family, seed, size)
    grid = spec["grid"]
    rows = grid.splitlines()
    assert grid.count("P") == 1
    assert grid.count("E") >= 1
    assert solvable(rows)
    assert compile_grid(grid, player_angle=spec["player_angle"])

    if family == "comb":
        minimum, maximum = {"s": (4, 6), "m": (6, 9), "l": (9, 12)}[size]
        assert minimum <= grid.count("D") == spec["rooms"] <= maximum
        assert spec["episode_ticks"] == (3500 if size == "l" else 2800)
        assert spec["player_angle"] == 0
    elif family == "locksfirst":
        assert spec["episode_ticks"] == (4200 if size == "l" else 3500)
        assert 1 <= len(spec["locks"]) <= 3
        assert set(spec["locks"]) == set(grid) & set("RBY")
        for lock in spec["locks"]:
            key = lock.lower()
            assert grid.count(key) == 1
            assert not solvable([row.replace(key, ".") for row in rows])
        unlocked = [row.translate(str.maketrans("RBY", "DDD")) for row in rows]
        assert _shortest_distance(rows) > _shortest_distance(unlocked)
    else:
        assert spec["episode_ticks"] == 3500
        assert sum(grid.count(monster) for monster in "zsicd") >= 4
        assert len(spec["locks"]) <= 1
        for lock in spec["locks"]:
            assert not solvable([row.replace(lock.lower(), ".") for row in rows])
