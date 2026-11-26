"""Render the game view and audience map using RGB arrays."""

import math
import struct
import zlib

import numpy as np

from doom_env.spaces import NUM_RAYS, RAY_RANGE
from training.gen.compile import CELL

HEIGHT = 360
GAME_WIDTH = 480
MAP_WIDTH = 360
BACKGROUND = (19, 24, 31)
FLOOR = (78, 87, 98)
KEY_COLORS = {"r": (255, 90, 93), "b": (88, 172, 255), "y": (255, 211, 84)}
CELL_COLORS = {
    "#": (32, 38, 47), "D": (160, 106, 64), "E": (50, 176, 106),
    "R": (175, 52, 59), "B": (48, 105, 179), "Y": (177, 138, 31),
}
RAY_ANGLES = np.arange(NUM_RAYS) * (math.tau / NUM_RAYS)


def encode_png(rgb: np.ndarray) -> bytes:
    height, width, channels = rgb.shape
    if channels != 3 or rgb.dtype != np.uint8 or not height or not width:
        raise ValueError("expected a nonempty uint8 RGB image")

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    scanlines = b"".join(b"\0" + row.tobytes() for row in rgb)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(scanlines, 1)) + chunk(b"IEND", b""))


def blank_frame(*, has_memory: bool = False) -> np.ndarray:
    frame = np.empty((HEIGHT, GAME_WIDTH + MAP_WIDTH * (1 + has_memory), 3), dtype=np.uint8)
    frame[:] = BACKGROUND
    return frame


class AudienceMap:
    def __init__(self, rows: list[str]):
        self.rows = rows
        self.height, self.width = len(rows), len(rows[0])
        self.cell_px = (MAP_WIDTH - 24) // max(self.height, self.width)
        self.left = (MAP_WIDTH - self.width * self.cell_px) // 2
        self.top = (HEIGHT - self.height * self.cell_px) // 2
        colors = np.array([[CELL_COLORS.get(ch, FLOOR) for ch in row] for row in rows])
        lit = np.minimum(colors + 42, 255)
        self.colors = colors.astype(np.uint8)
        self.lit_colors = lit.astype(np.uint8)
        self.seen = np.zeros((self.height, self.width), dtype=bool)
        self.trail = np.zeros((HEIGHT, MAP_WIDTH), dtype=bool)
        self.position = None
        self.heading = 0.0
        self.keys = {(r, c): ch for r, row in enumerate(rows)
                     for c, ch in enumerate(row) if ch in KEY_COLORS}

    def _mark_rays(self, x, y, heading, distances):
        # Work in grid coordinates. The midpoint between successive grid-line
        # crossings identifies every cell the ray traverses, including short crossings.
        lengths = np.maximum(0, distances.astype(float) * RAY_RANGE - 0.001) / CELL
        dxs = np.cos(heading + RAY_ANGLES) * lengths
        dys = -np.sin(heading + RAY_ANGLES) * lengths
        for dx, dy in zip(dxs, dys):
            cuts = [0.0, 1.0]
            for start, delta in ((x, dx), (y, dy)):
                lo, hi = sorted((start, start + delta))
                for edge in range(math.floor(lo) + 1, math.ceil(hi)):
                    cuts.append((edge - start) / delta)
            cuts.sort()
            for a, b in zip(cuts, cuts[1:]):
                if a == b:
                    continue
                t = (a + b) / 2
                c, r = math.floor(x + t * dx), math.floor(y + t * dy)
                if 0 <= r < self.height and 0 <= c < self.width and self.rows[r][c] != "#":
                    self.seen[r, c] = True

    def _pixel(self, x, y):
        return self.left + x * self.cell_px, self.top + y * self.cell_px

    def observe(self, pose, rays=None):
        x, y = pose["x"] / CELL, self.height - pose["y"] / CELL
        self.heading = math.radians(pose["angle"])
        if rays is not None:
            self._mark_rays(x, y, self.heading, rays["distance"])
        position = self._pixel(x, y)
        if self.position is not None:
            px, py = self.position
            nx, ny = position
            count = math.ceil(max(abs(nx - px), abs(ny - py))) + 1
            xs = np.rint(np.linspace(px, nx, count)).astype(int)
            ys = np.rint(np.linspace(py, ny, count)).astype(int)
            inside = (0 <= xs) & (xs < MAP_WIDTH) & (0 <= ys) & (ys < HEIGHT)
            self.trail[ys[inside], xs[inside]] = True
        self.position = position

    def pick_up(self, symbol, pose):
        cells = [cell for cell, key in self.keys.items() if key == symbol]
        if cells:
            x, y = pose["x"] / CELL, self.height - pose["y"] / CELL
            cell = min(cells, key=lambda rc: (x - rc[1] - .5)**2 + (y - rc[0] - .5)**2)
            del self.keys[cell]

    def draw(self, panel):
        colors = np.where(self.seen[:, :, None], self.lit_colors, self.colors)
        tiles = colors.repeat(self.cell_px, axis=0).repeat(self.cell_px, axis=1)
        if self.cell_px >= 6:
            tiles[::self.cell_px] = BACKGROUND
            tiles[:, ::self.cell_px] = BACKGROUND
        height, width = tiles.shape[:2]
        panel[self.top:self.top + height, self.left:self.left + width] = tiles
        panel[self.trail] = (88, 206, 217)

        for (r, c), key in self.keys.items():
            x, y = self._pixel(c + .5, r + .5)
            radius = max(1, min(8, self.cell_px // 4))
            ys, xs = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            x, y = int(x), int(y)
            panel[y-radius:y+radius+1, x-radius:x+radius+1][xs**2 + ys**2 <= radius**2] = KEY_COLORS[key]
        if self.position is not None:
            self._draw_player(panel)

    def _draw_player(self, panel):
        x, y = self.position
        size = max(4, min(13, self.cell_px * .42))
        left, right = max(0, int(x - size)), min(MAP_WIDTH, math.ceil(x + size + 1))
        top, bottom = max(0, int(y - size)), min(HEIGHT, math.ceil(y + size + 1))
        if left >= right or top >= bottom:
            return
        ys, xs = np.ogrid[top:bottom, left:right]
        c, s = math.cos(self.heading), math.sin(self.heading)
        forward = (xs - x) * c - (ys - y) * s
        sideways = (xs - x) * s + (ys - y) * c
        triangle = (forward >= -size * .62) & (np.abs(sideways) <= (size - forward) * .42)
        panel[top:bottom, left:right][triangle] = (239, 255, 255)


def composite(rgb: np.ndarray, audience: AudienceMap, memory_rgb: np.ndarray | None = None) -> np.ndarray:
    frame = blank_frame(has_memory=memory_rgb is not None)
    frame[:, :GAME_WIDTH] = rgb.repeat(3, axis=0).repeat(3, axis=1)
    audience.draw(frame[:, GAME_WIDTH:GAME_WIDTH + MAP_WIDTH])
    if memory_rgb is not None:
        height, width = memory_rgb.shape[:2]
        scale = min(HEIGHT / height, MAP_WIDTH / width)
        h, w = max(1, round(height * scale)), max(1, round(width * scale))
        rows, cols = np.arange(h) * height // h, np.arange(w) * width // w
        top = (HEIGHT - h) // 2
        left = GAME_WIDTH + MAP_WIDTH + (MAP_WIDTH - w) // 2
        frame[top:top + h, left:left + w] = memory_rgb[rows[:, None], cols]
    return frame
