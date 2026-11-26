"""Render the game view and audience map using RGB arrays."""

import math
import struct
import zlib
from functools import lru_cache
from textwrap import wrap

import numpy as np

from demo.font import draw_text, text_width
from doom_env.spaces import NUM_RAYS, RAY_RANGE
from training.gen.compile import CELL

HEIGHT = 900
GAME_WIDTH = 1200
PANEL_WIDTH = 400
MAP_WIDTH, MAP_HEIGHT = 360, 344
BACKGROUND = (14, 20, 28)
MAP_BACKGROUND = (9, 14, 20)
FLOOR = (52, 66, 80)
EXPLORED = (108, 130, 147)
TEXT = (240, 247, 249)
MUTED = (167, 186, 201)
ACCENT = (88, 226, 233)
GREEN = (115, 231, 167)
TRACK = (39, 54, 68)
KEY_COLORS = {"r": (255, 90, 93), "b": (88, 172, 255), "y": (255, 211, 84)}
CELL_COLORS = {
    "#": (23, 33, 43), "D": (156, 101, 61), "E": (46, 173, 105),
    "R": (194, 63, 70), "B": (60, 124, 201), "Y": (196, 154, 44),
}
INTENTION_LABELS = {
    "explore": "Explore", "get_key": "Get key", "go_door": "Go to door",
    "go_exit": "Go to exit", "fight": "Fight",
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


def blank_frame() -> np.ndarray:
    frame = np.empty((HEIGHT, GAME_WIDTH + PANEL_WIDTH, 3), dtype=np.uint8)
    frame[:] = BACKGROUND
    frame[:, :GAME_WIDTH] = MAP_BACKGROUND
    return frame


class AudienceMap:
    def __init__(self, rows: list[str]):
        self.rows = rows
        self.height, self.width = len(rows), len(rows[0])
        self.cell_px = min(MAP_WIDTH / self.width, MAP_HEIGHT / self.height)
        width, height = round(self.width * self.cell_px), round(self.height * self.cell_px)
        self.left = (MAP_WIDTH - width) // 2
        self.top = (MAP_HEIGHT - height) // 2
        self.columns = ((np.arange(width) + .5) * self.width / width).astype(int)
        self.tile_rows = ((np.arange(height) + .5) * self.height / height).astype(int)
        self.colors = np.array([[CELL_COLORS.get(ch, FLOOR) for ch in row] for row in rows], dtype=np.uint8)
        self.lit_colors = np.where((self.colors == FLOOR).all(axis=2, keepdims=True),
                                   np.array(EXPLORED, dtype=np.uint8), self.colors)
        self.seen = np.zeros((self.height, self.width), dtype=bool)
        self.trail = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=bool)
        self.position = None
        self.heading = 0.0
        self.keys = {(r, c): ch for r, row in enumerate(rows)
                     for c, ch in enumerate(row) if ch in KEY_COLORS}
        self.labels = {(r, c): "EXIT" if ch == "E" else ch for r, row in enumerate(rows)
                       for c, ch in enumerate(row) if ch in "RBYE"}

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
            for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                xx, yy = xs + dx, ys + dy
                inside = (0 <= xx) & (xx < MAP_WIDTH) & (0 <= yy) & (yy < MAP_HEIGHT)
                self.trail[yy[inside], xx[inside]] = True
        self.position = position

    def pick_up(self, symbol, pose):
        cells = [cell for cell, key in self.keys.items() if key == symbol]
        if cells:
            x, y = pose["x"] / CELL, self.height - pose["y"] / CELL
            cell = min(cells, key=lambda rc: (x - rc[1] - .5)**2 + (y - rc[0] - .5)**2)
            del self.keys[cell]

    def draw(self, panel):
        colors = np.where(self.seen[:, :, None], self.lit_colors, self.colors)
        tiles = colors[self.tile_rows[:, None], self.columns]
        if self.cell_px >= 6:
            tiles[np.r_[True, np.diff(self.tile_rows) != 0]] = BACKGROUND
            tiles[:, np.r_[True, np.diff(self.columns) != 0]] = BACKGROUND
        height, width = tiles.shape[:2]
        panel[self.top:self.top + height, self.left:self.left + width] = tiles
        panel[self.trail] = ACCENT

        for (r, c), label in self.labels.items():
            x, y = map(int, self._pixel(c + .5, r + .5))
            scale = max(1, min(3, int(self.cell_px - 2) // max(7, text_width(label, 1))))
            _center_text(panel, x, y, label, MAP_BACKGROUND, scale)
        for (r, c), key in self.keys.items():
            x, y = map(int, self._pixel(c + .5, r + .5))
            _key_icon(panel, x, y, key, max(1, min(2, int(self.cell_px) // 16)))
        if self.position is not None:
            self._draw_player(panel)

    def _draw_player(self, panel):
        x, y = self.position
        size = max(10, min(20, self.cell_px * .58))
        left, right = max(0, int(x - size)), min(MAP_WIDTH, math.ceil(x + size + 1))
        top, bottom = max(0, int(y - size)), min(MAP_HEIGHT, math.ceil(y + size + 1))
        if left >= right or top >= bottom:
            return
        ys, xs = np.ogrid[top:bottom, left:right]
        c, s = math.cos(self.heading), math.sin(self.heading)
        forward = (xs - x) * c - (ys - y) * s
        sideways = (xs - x) * s + (ys - y) * c
        half_width = (size - forward) * .46
        # A notched arrow with a dark outline stays legible over the trail.
        tail = -size * .65 + np.maximum(0, size * .38 - np.abs(sideways))
        arrow = (forward >= tail) & (np.abs(sideways) <= half_width)
        region = panel[top:bottom, left:right]
        region[arrow] = MAP_BACKGROUND
        region[arrow & (forward >= tail + 2) & (np.abs(sideways) <= half_width - 1.5)] = TEXT


@lru_cache(maxsize=4)
def _bilinear_axis(source, target):
    pos = np.clip((np.arange(target, dtype=np.float32) + .5) * source / target - .5, 0, source - 1)
    lower = pos.astype(int)
    return lower, np.minimum(lower + 1, source - 1), (pos - lower).astype(np.float32)


def _draw_game(frame, rgb):
    source_height, source_width = rgb.shape[:2]
    scale = min(GAME_WIDTH / source_width, HEIGHT / source_height)
    width, height = max(1, round(source_width * scale)), max(1, round(source_height * scale))
    left, top = (GAME_WIDTH - width) // 2, (HEIGHT - height) // 2
    x0, x1, wx = _bilinear_axis(source_width, width)
    y0, y1, wy = _bilinear_axis(source_height, height)
    source = rgb.astype(np.float32)
    wx = wx[None, :, None]
    wy = wy[:, None, None]
    horizontal = source[:, x0] * (1 - wx) + source[:, x1] * wx
    frame[top:top+height, left:left+width] = np.rint(horizontal[y0] * (1 - wy) + horizontal[y1] * wy)


def _center_text(panel, x, y, text, color, scale):
    draw_text(panel, x - text_width(text, scale) // 2, y - 7 * scale // 2, text, color, scale)


def _key_icon(panel, x, y, key, scale):
    color = KEY_COLORS[key]
    xs = slice(max(0, x - 5 * scale), x + 4 * scale)
    panel[max(0, y - 5 * scale):y + 5 * scale, xs] = color
    panel[max(0, y - scale):y + scale, x + 4 * scale:x + 9 * scale] = color
    panel[y:y + 3 * scale, x + 7 * scale:x + 9 * scale] = color
    _center_text(panel, x, y, key, MAP_BACKGROUND, scale)


def _draw_status(panel, state):
    panel[455:457, 20:380] = TRACK
    draw_text(panel, 20, 475, "What it's thinking", TEXT)
    intentions = state.get("intention", {})
    values = [intentions.get(name, 0) for name in INTENTION_LABELS]
    top = int(np.argmax(values)) if any(values) else -1
    for i, (label, probability) in enumerate(zip(INTENTION_LABELS.values(), values)):
        y = 513 + 35 * i
        color = ACCENT if i == top else MUTED
        if i == top:
            panel[y-5:y+26, 12:388] = (24, 49, 58)
        draw_text(panel, 20, y, label, color)
        panel[y+5:y+16, 208:309] = TRACK
        panel[y+5:y+16, 208:208+round(101 * probability)] = color
        percentage = f"{probability:.0%}"
        draw_text(panel, 380 - text_width(percentage, 3), y, percentage, color)

    panel[690:692, 20:380] = TRACK
    draw_text(panel, 20, 711, "Keys", TEXT)
    keys = list(dict.fromkeys(key[0] for key in state.get("keys", [])))
    if keys:
        for i, key in enumerate(keys):
            x = 116 + 44 * i
            panel[706:736, x:x+30] = KEY_COLORS[key]
            _center_text(panel, x + 15, 721, key, MAP_BACKGROUND, 3)
    else:
        draw_text(panel, 116, 711, "None", MUTED)
    draw_text(panel, 20, 754, "Health", TEXT)
    health = state.get("health") or 0
    panel[760:770, 142:309] = TRACK
    panel[760:770, 142:142+round(167 * min(health, 100) / 100)] = GREEN if health > 30 else KEY_COLORS["r"]
    label = str(health)
    draw_text(panel, 380 - text_width(label, 3), 754, label, TEXT)

    if event := state.get("latest_event"):
        color = GREEN if state.get("outcome") == "exit" else ACCENT
        panel[802:880, 20:380] = (21, 40, 48)
        panel[802:880, 20:24] = color
        scale = 4 if text_width(event, 4) <= 330 else 3
        lines = wrap(event, width=(330 + scale) // (6 * scale))
        line_height = 7 * scale + 6
        y = 841 - (len(lines) * line_height - 6) // 2
        for row, line in enumerate(lines):
            draw_text(panel, 36, y + row * line_height, line, color, scale)


def composite(rgb: np.ndarray, audience: AudienceMap, state: dict | None = None) -> np.ndarray:
    frame = blank_frame()
    _draw_game(frame, rgb)
    panel = frame[:, GAME_WIDTH:]
    panel[:, :2] = TRACK
    draw_text(panel, 20, 18, "LEVEL MAP", TEXT, scale=4)
    draw_text(panel, 20, 54, "FOR YOU - NAKSHA CAN'T SEE IT", MUTED if "MUTED" in globals() else TEXT, scale=2)
    map_panel = panel[78:78+MAP_HEIGHT, 20:380]
    map_panel[:] = MAP_BACKGROUND
    audience.draw(map_panel)
    _draw_status(panel, state or {})
    return frame
