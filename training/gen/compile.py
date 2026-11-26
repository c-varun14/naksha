"""ASCII grid -> UDMF (namespace "zdoom") WAD, following the conventions of the organiser fixtures
(tests/fixtures/*.wad): 64-unit cells, one sector per connected floor region, door cells as
zero-height sectors with Door_Raise (12) / Door_LockedRaise (13) on both faces, the exit as a GATE3
cell entered across an Exit_Normal (243) line. Training-only tool: never imported by submission/.

Text row 0 is north: cell (r, c) spans x in [64c, 64c+64], y in [64(H-1-r), 64(H-r)].
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "env"))
from doom_env.wad import Lump, write_wad  # noqa: E402

CELL = 64
CEIL = 128

THINGS = {  # char -> (doomednum, default angle)
    "P": (1, 0),
    "r": (13, 180), "b": (5, 180), "y": (6, 180),  # red / blue / yellow keycards
    "z": (3004, 180), "s": (9, 180), "c": (65, 180), "i": (3001, 180), "d": (3002, 180),
    "h": (2011, 0), "m": (2012, 0), "a": (2018, 0), "k": (2007, 0), "l": (2008, 0),
}
LOCKS = {"R": (129, "DOORRED"), "B": (130, "DOORBLU"), "Y": (131, "DOORYEL")}
DOORS = {"D": (0, "BIGDOOR2"), **LOCKS}
WALL, EXIT = "#", "E"
# (floor, wall) per floor region, cycled in flood-fill order. The first two match the fixtures.
PALETTE = (
    ("FLOOR0_1", "STARTAN2"), ("FLAT14", "BROWN1"), ("FLOOR4_8", "STARTAN3"),
    ("FLAT5", "BROWN96"), ("FLOOR5_1", "STARTAN2"), ("FLAT1", "BROWN1"),
)
DOOR_FLOOR, EXIT_FLOOR, EXIT_WALL, JAMB = "FLAT5_4", "GATE3", "GRAY1", "DOORTRAK"
FLAGS = "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true;"


class GridError(ValueError):
    pass


def parse(grid: str) -> list[str]:
    rows = [r.rstrip("\n") for r in grid.strip("\n").split("\n")]
    width = max(len(r) for r in rows)
    rows = [r.ljust(width, WALL) for r in rows]
    known = set(THINGS) | set(DOORS) | {WALL, EXIT, "."}
    bad = {ch for r in rows for ch in r} - known
    if bad:
        raise GridError(f"unknown grid symbols: {sorted(bad)}")
    if sum(r.count("P") for r in rows) != 1:
        raise GridError("grid needs exactly one P")
    return rows


@dataclass
class _Sector:
    floor: str
    wall: str
    ceil: int


def _sectors(rows: list[str]):
    """Return (sector id per cell or -1 for wall, list of _Sector, kind per sector)."""
    h, w = len(rows), len(rows[0])
    sid = [[-1] * w for _ in range(h)]
    sectors: list[_Sector] = []
    kinds: list[str] = []  # "floor" | door char | "exit"
    regions = 0
    for r in range(h):
        for c in range(w):
            ch = rows[r][c]
            if ch == WALL or sid[r][c] != -1:
                continue
            if ch in DOORS:
                sid[r][c] = len(sectors)
                sectors.append(_Sector(DOOR_FLOOR, JAMB, 0))
                kinds.append(ch)
                continue
            if ch == EXIT:
                sid[r][c] = len(sectors)
                sectors.append(_Sector(EXIT_FLOOR, EXIT_WALL, CEIL))
                kinds.append("exit")
                continue
            floor, wall = PALETTE[regions % len(PALETTE)]
            regions += 1
            s = len(sectors)
            sectors.append(_Sector(floor, wall, CEIL))
            kinds.append("floor")
            stack = [(r, c)]
            sid[r][c] = s
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and sid[ny][nx] == -1:
                        nch = rows[ny][nx]
                        if nch != WALL and nch not in DOORS and nch != EXIT:
                            sid[ny][nx] = s
                            stack.append((ny, nx))
    return sid, sectors, kinds


def compile_grid(grid: str, *, player_angle: int | None = None) -> str:
    """Return the TEXTMAP text for an ASCII grid."""
    rows = parse(grid)
    h, w = len(rows), len(rows[0])
    sid, sectors, kinds = _sectors(rows)

    def corners(r, c):
        x0, x1 = c * CELL, (c + 1) * CELL
        y1 = (h - r) * CELL
        y0 = y1 - CELL
        return x0, x1, y0, y1

    def cw_edge(r, c, side):
        """Edge of cell (r, c) on `side`, oriented so the cell is on the right (front)."""
        x0, x1, y0, y1 = corners(r, c)
        return {"N": ((x0, y1), (x1, y1)), "E": ((x1, y1), (x1, y0)),
                "S": ((x1, y0), (x0, y0)), "W": ((x0, y0), (x0, y1))}[side]

    raw = []  # (p1, p2, front_sid, back_sid or None, kind)
    seen_pairs = set()
    for r in range(h):
        for c in range(w):
            a = sid[r][c]
            if a < 0:
                continue
            for side, (dr, dc) in (("N", (-1, 0)), ("E", (0, 1)), ("S", (1, 0)), ("W", (0, -1))):
                nr, nc = r + dr, c + dc
                b = sid[nr][nc] if 0 <= nr < h and 0 <= nc < w else -1
                if b == a:
                    continue
                if b < 0:
                    raw.append((*cw_edge(r, c, side), a, None, "wall"))
                    continue
                key = (min(a, b), max(a, b), (min(r, nr), min(c, nc), side in "NS"))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                ka, kb = kinds[a], kinds[b]
                # Front faces away from the door / exit cell, into the room (as in the fixtures).
                if ka in DOORS and kb in DOORS:
                    raise GridError(f"adjacent door cells at {(r, c)}")
                if (ka in DOORS and kb == "exit") or (kb in DOORS and ka == "exit"):
                    raise GridError(f"door next to exit at {(r, c)}")
                if ka in DOORS or ka == "exit":
                    # (r, c) is the door/exit: emit from the neighbour's side so front = neighbour.
                    opp = {"N": "S", "S": "N", "E": "W", "W": "E"}[side]
                    p1, p2 = cw_edge(nr, nc, opp)
                    raw.append((p1, p2, b, a, ka if ka in DOORS else "exit"))
                elif kb in DOORS or kb == "exit":
                    p1, p2 = cw_edge(r, c, side)
                    raw.append((p1, p2, a, b, kb if kb in DOORS else "exit"))
                else:
                    p1, p2 = cw_edge(r, c, side)
                    raw.append((p1, p2, a, b, "pass"))

    # Merge collinear wall segments of the same sector, only through vertices no other line uses.
    use = {}
    for p1, p2, *_ in raw:
        use[p1] = use.get(p1, 0) + 1
        use[p2] = use.get(p2, 0) + 1
    walls = [e for e in raw if e[4] == "wall"]
    other = [e for e in raw if e[4] != "wall"]
    starts = {(e[0], e[2], _dir(e)): e for e in walls}
    ends = {(e[1], e[2], _dir(e)): e for e in walls}
    merged = []
    for e in walls:
        k = (e[0], e[2], _dir(e))
        if use[e[0]] == 2 and k in ends:
            continue  # not a chain start: it continues the edge ending at its first vertex
        cur = e
        while use[cur[1]] == 2 and (cur[1], cur[2], _dir(cur)) in starts:
            cur = starts[(cur[1], cur[2], _dir(cur))]
        merged.append((e[0], cur[1], e[2], None, "wall"))
    edges = merged + other

    verts: dict[tuple, int] = {}

    def vid(p):
        if p not in verts:
            verts[p] = len(verts)
        return verts[p]

    sidedefs, linedefs = [], []
    for p1, p2, front, back, kind in edges:
        v1, v2 = vid(p1), vid(p2)
        fs = sectors[front]
        if back is None:
            tex = fs.wall
            sidedefs.append(f'sidedef {{ sector = {front}; texturetop = "-"; texturemiddle = "{tex}"; texturebottom = "-"; }}')
            linedefs.append(f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {len(sidedefs) - 1}; blocking = true; dontpegbottom = true; }}")
            continue
        if kind in DOORS:
            lock, dtex = DOORS[kind]
            sidedefs.append(f'sidedef {{ sector = {front}; texturetop = "{dtex}"; texturemiddle = "-"; texturebottom = "-"; }}')
            sidedefs.append(f'sidedef {{ sector = {back}; texturetop = "-"; texturemiddle = "-"; texturebottom = "-"; }}')
            if lock:
                special = f"special = 13; arg0 = 0; arg1 = 16; arg2 = 0; arg3 = {lock}; playeruse = true; monsteruse = false; repeatspecial = true;"
            else:
                special = "special = 12; arg0 = 0; arg1 = 16; arg2 = 150; playeruse = true; monsteruse = true; repeatspecial = true;"
        else:
            sidedefs.append(f'sidedef {{ sector = {front}; texturetop = "-"; texturemiddle = "-"; texturebottom = "-"; }}')
            sidedefs.append(f'sidedef {{ sector = {back}; texturetop = "-"; texturemiddle = "-"; texturebottom = "-"; }}')
            special = "special = 243; playercross = true;" if kind == "exit" else ""
        linedefs.append(f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {len(sidedefs) - 2}; sideback = {len(sidedefs) - 1}; twosided = true; {special} }}".replace("  ", " "))

    things = []
    for r in range(h):
        for c in range(w):
            ch = rows[r][c]
            if ch in THINGS:
                num, ang = THINGS[ch]
                if ch == "P" and player_angle is not None:
                    ang = int(player_angle)
                x = c * CELL + CELL / 2
                y = (h - 1 - r) * CELL + CELL / 2
                things.append(f"thing {{ x = {x:.1f}; y = {y:.1f}; angle = {ang}; type = {num}; {FLAGS} }}")

    out = ['namespace = "zdoom";']
    out += [f"vertex {{ x = {float(x):.1f}; y = {float(y):.1f}; }}" for (x, y), _ in sorted(verts.items(), key=lambda kv: kv[1])]
    out += [f'sector {{ texturefloor = "{s.floor}"; textureceiling = "CEIL1_1"; heightfloor = 0; heightceiling = {s.ceil}; lightlevel = 192; }}' for s in sectors]
    out += sidedefs + linedefs + things
    return "\n".join(out) + "\n"


def _dir(e):
    (x1, y1), (x2, y2) = e[0], e[1]
    return ((x2 > x1) - (x2 < x1), (y2 > y1) - (y2 < y1))


def write_map(grid: str, path: str | Path, *, player_angle: int | None = None, map_name: str = "MAP01") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = compile_grid(grid, player_angle=player_angle)
    write_wad(path, [Lump(map_name, b""), Lump("TEXTMAP", text.encode("ascii")), Lump("ENDMAP", b"")])
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compile an ASCII grid file to a UDMF WAD")
    ap.add_argument("grid_file")
    ap.add_argument("out_wad")
    ap.add_argument("--angle", type=int, default=None)
    a = ap.parse_args()
    print(write_map(Path(a.grid_file).read_text(), a.out_wad, player_angle=a.angle))
