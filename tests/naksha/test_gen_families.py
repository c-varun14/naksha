import hashlib
import json

import pytest

from training.gen.compile import write_map
from training.gen.families import generate, solvable

# Full map specs captured before adding the new families.
BASELINE = {
    "arena": (
        "7dd7e2fb57df43f85e9a88826bc13d25130cfd1cff2c4eaca6acc4d152d1b7dd",
        "7f4536f3d2b0177103a1a31feb44b5b018808b1d045226167d965c522c363b48",
        "8c7cc47c7fcde69a4a1bb052da35b9638c5271bc32f147c89e92135d4d333143",
    ),
    "rooms": (
        "0e903caaa8e9c2f83e88a937a762a75012ea58fbcb85c09bbdcc04d727842e0b",
        "10b2fc52946ac10aa48dd3f929b13f9835244abb8fa1f6a78a93a65160f4c7ea",
        "9bf0e8bc7bb23245385445d5fb1a6cbf23b5f4d9576ab64aa6c11db576a4eabb",
    ),
    "locks": (
        "7f17c8056545514182eaf8ae9f787c98d83ff55f43d8ced6373358e1b61a591c",
        "10c515113c0aaf87961cd337054360c8a9a1a7761b4c46b8496df6bebb1e4bbd",
        "a3efb791631a81c92e4065d0a9ebed89e9d275fa583a6b915fc3c9a4bd0428c8",
    ),
    "gauntlet": (
        "2a894a5f630484ea4766b1d2042c361a7718873bf6fcafc49ddfdf2b523f52ca",
        "843213a655b4ea61d0d0e028017cc46dfb1fd54c0d9fd162589ec3070e295634",
        "f289a1de7a81101ea61f47f4e85c8606bb3ba3367a13051425348c0235a4b896",
    ),
}


@pytest.mark.parametrize("family", BASELINE)
def test_existing_family_hashes(family):
    for seed, size, expected in zip((0, 23, 409), "sml", BASELINE[family], strict=True):
        spec = json.dumps(generate(family, seed, size), sort_keys=True).encode()
        assert hashlib.sha256(spec).hexdigest() == expected


@pytest.mark.parametrize("family", ("maze", "irregular"))
@pytest.mark.parametrize("size,maze_side,room_side", [("s", 15, 3), ("m", 25, 4), ("l", 35, 5)])
def test_new_family_sizes_compile_and_are_solvable(tmp_path, family, size, maze_side, room_side):
    for seed in (0, 1, 2, 23):
        spec = generate(family, seed, size)
        rows = spec["grid"].splitlines()
        assert spec["episode_ticks"] == 4200
        assert solvable(rows)
        assert all(len(row) == len(rows) for row in rows)
        if family == "maze":
            assert spec["rooms"] == maze_side ** 2
            assert len(rows) in (2 * maze_side + 1, 3 * maze_side + 1)
            width = (len(rows) - 1) // maze_side - 1
            pitch = width + 1
            # Wall openings represent the DFS tree plus up to three optional loop edges.
            openings = sum(
                ch != "#" and ((r % pitch == 0) != (c % pitch == 0))
                for r, row in enumerate(rows) for c, ch in enumerate(row)
            )
            assert maze_side ** 2 - 1 <= openings // width <= maze_side ** 2 + 2
        else:
            assert spec["rooms"] == room_side ** 2
            assert len(rows) == 16 * room_side + 1
            assert len(spec["locks"]) <= 2
            assert set(spec["locks"]) == set(spec["grid"]) & set("RBY")
        assert write_map(spec["grid"], tmp_path / "test.wad", player_angle=spec["player_angle"]).is_file()


def test_maze_sides_override_makes_smaller_mazes(monkeypatch):
    from training.gen import families

    monkeypatch.setattr(families, "MAZE_SIDES", {"s": 7, "m": 9, "l": 11})
    for size, side in (("s", 7), ("m", 9), ("l", 11)):
        spec = families.generate("maze", 62000, size)
        rows = spec["grid"].split("\n")
        assert spec["rooms"] == side ** 2
        assert len(rows) in (2 * side + 1, 3 * side + 1)
