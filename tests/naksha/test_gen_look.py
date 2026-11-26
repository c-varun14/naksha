"""Appearance randomisation changes only the look of a generated map, never how it plays."""
import random
import re

import numpy as np

from doom_env import make_env
from training.gen.compile import compile_grid, write_map
from training.gen.families import generate
from training.gen.look import FLATS, WALLS, restyle

GRID = """
#########
#P..#...#
#...D..E#
#r..#...#
###R#####
#.......#
#########
"""


def _geometry(text):
    return [line for line in text.split("\n") if not line.startswith(("sector", "sidedef"))]


def test_default_output_is_unchanged(tmp_path):
    wad = write_map(GRID, tmp_path / "a.wad")
    assert compile_grid(GRID).encode("ascii") in wad.read_bytes()


def test_restyle_keeps_geometry_and_uses_known_textures():
    text = compile_grid(GRID)
    styled = restyle(text, random.Random(3))
    assert styled != text
    assert _geometry(styled) == _geometry(text)
    assert styled.count("sector {") == text.count("sector {")
    assert 'texturefloor = "GATE3"' in styled and 'texturefloor = "FLAT5_4"' in styled
    for name in re.findall(r'texturemiddle = "([^"]+)"', styled):
        assert name in WALLS or name in ("-", "GRAY1", "DOORTRAK")
    for name in re.findall(r'texture(?:floor|ceiling) = "([^"]+)"', styled):
        assert name in FLATS or name in ("GATE3", "FLAT5_4")


def test_restyled_map_plays_the_same(tmp_path):
    spec = generate("locks", 5, "s")
    plain = make_env(str(write_map(spec["grid"], tmp_path / "p.wad")), "MAP01", episode_ticks=200)
    styled = make_env(str(write_map(spec["grid"], tmp_path / "s.wad", look_seed=11)), "MAP01", episode_ticks=200)
    try:
        a, _ = plain.reset(seed=0)
        b, _ = styled.reset(seed=0)
        assert np.array_equal(a["rays"]["hit_type"], b["rays"]["hit_type"])
        assert np.allclose(a["rays"]["distance"], b["rays"]["distance"])
        assert plain.mapinfo.door_sectors == styled.mapinfo.door_sectors
        assert not np.array_equal(a["rgb"], b["rgb"])
    finally:
        plain.close()
        styled.close()
