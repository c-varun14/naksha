"""Appearance randomisation for generated training maps (domain randomisation).

Each floor region gets its own floor flat, wall texture and light level; the map gets one ceiling
flat and one ceiling height. Layout, doors, keys, monsters, the exit tile (GATE3 floor, GRAY1 walls)
and door textures never change, so the map plays the same and only looks different.
Training-only: never imported by submission/.
"""
from __future__ import annotations

import random
import re

from training.gen.compile import CEIL, DOOR_FLOOR, EXIT_FLOOR, EXIT_WALL, JAMB

WALLS = (
    "BRICK1", "BRICK4", "BRICK5", "BROWN144", "BROWNHUG", "CEMENT2", "CEMENT4", "COMPTALL", "CRACKLE4",
    "GRAY5", "GRAY7", "GRAYBIG", "GSTONE1", "ICKWALL1", "METAL1", "PIPE4", "PIPEWAL2", "ROCK1", "ROCK3",
    "SHAWN2", "SP_HOT1", "STONE2", "STONE4", "STUCCO", "TANROCK4", "TEKWALL4", "WOOD3", "WOOD5", "BLAKWAL1",
    "BRICK12", "BSTONE1", "MARBGRAY", "METAL3", "MODWALL3", "PANEL6", "SILVER2", "SPACEW4", "STARG3",
    "STARGR2", "STARTAN1", "SUPPORT2", "TEKBRON1", "BIGBRIK2", "BRONZE2", "CEMENT7", "STONE7",
    "STARTAN2", "STARTAN3", "BROWN1", "BROWN96",
)
FLATS = (
    "FLOOR0_3", "FLOOR0_6", "FLOOR1_1", "FLOOR4_1", "FLOOR4_6", "FLOOR5_2", "FLOOR5_3", "FLOOR6_1", "FLAT1_1",
    "FLAT1_2", "FLAT2", "FLAT4", "FLAT8", "FLAT9", "FLAT18", "FLAT19", "FLAT23", "CEIL1_3", "CEIL3_1",
    "CEIL3_3", "CEIL4_1", "CEIL5_1", "MFLR8_2", "MFLR8_4", "RROCK03", "RROCK10", "RROCK13", "RROCK18",
    "RROCK19", "TLITE6_1", "CRATOP1", "FLOOR7_2", "GRASS2", "DEM1_6", "RROCK05",
    "FLOOR0_1", "FLAT14", "FLOOR4_8", "FLAT5", "FLOOR5_1", "FLAT1", "CEIL1_1",
)
CEILINGS = (96, 112, 128, 160, 192, 224)
LIGHT = (96, 255)

_SECTOR = re.compile(r'^sector \{ texturefloor = "([^"]+)"; textureceiling = "([^"]+)"; heightfloor = 0; '
                     r'heightceiling = (\d+); lightlevel = (\d+); \}$')
_SIDEDEF = re.compile(r'^sidedef \{ sector = (\d+); texturetop = "-"; texturemiddle = "([^"]+)";')


def restyle(text: str, rng: random.Random) -> str:
    """Return TEXTMAP text with a random look; the geometry and specials are untouched."""
    lines = text.split("\n")
    ceiling_flat, height = rng.choice(FLATS), rng.choice(CEILINGS)
    walls: dict[int, str] = {}
    sector = 0
    for i, line in enumerate(lines):
        m = _SECTOR.match(line)
        if not m:
            continue
        floor, _, ceil, _ = m.groups()
        if floor not in (DOOR_FLOOR, EXIT_FLOOR):
            floor = rng.choice(FLATS)
            walls[sector] = rng.choice(WALLS)
        if int(ceil) == CEIL:
            ceil = height
        light = rng.randint(*LIGHT)
        lines[i] = (f'sector {{ texturefloor = "{floor}"; textureceiling = "{ceiling_flat}"; heightfloor = 0; '
                    f'heightceiling = {ceil}; lightlevel = {light}; }}')
        sector += 1
    for i, line in enumerate(lines):
        m = _SIDEDEF.match(line)
        if m and int(m[1]) in walls and m[2] not in (EXIT_WALL, JAMB):
            lines[i] = line.replace(f'texturemiddle = "{m[2]}"', f'texturemiddle = "{walls[int(m[1])]}"', 1)
    return "\n".join(lines)
