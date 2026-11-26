"""Episode-local spatial memory and its policy and odometry networks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .actions import NUM_ACTIONS
from .model import HIDDEN, EnsembleNet, NakshaNet, _VERSIONS

MEMORY_VERSION = "grid-v1"
CROP_SHAPE = (16, 24, 24)
CELL_SIZE = 32
GRID_SIZE = 192
POOL = 4
_ODOMETRY_SCALE = (32.0, 32.0, 14.0)

_angles = np.arange(64) * (2 * np.pi / 64)
_RAY_COS = np.round(np.cos(_angles), 15)
_RAY_SIN = np.round(np.sin(_angles), 15)
_RAY_SAMPLES = np.arange(0, 1024, 16, dtype=np.float64)
_CROP_ROW, _CROP_COL = np.indices(CROP_SHAPE[1:])
_CROP_FWD = (12 - _CROP_ROW).ravel()
_CROP_RIGHT = (_CROP_COL - 12).ravel()
_RAY_CHANNEL = np.array([0, 1, 2, 3, 5, 4])
_ENTITY_CHANNEL = np.array([-1, 5, 5, 5, -1, 4, 4, 4, 4, 7, -1, 4])
_COLORS = np.array([
    [73, 76, 81], [219, 221, 224], [177, 124, 73], [242, 144, 55],
    [82, 176, 119], [230, 79, 77], [78, 112, 155], [249, 218, 91],
], dtype=np.uint8)


class OdometryNet(nn.Module):
    """Estimate motion in the previous observation's forward/left frame."""

    def __init__(self) -> None:
        super().__init__()
        self.rays = nn.Sequential(
            nn.Conv1d(14, 24, 5, stride=2, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(24, 32, 5, stride=2, padding=2, padding_mode="circular"), nn.ReLU(),
            nn.Conv1d(32, 32, 3, stride=2, padding=1, padding_mode="circular"), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 8, 96), nn.ReLU(),
        )
        self.motion = nn.Sequential(
            nn.Linear(96 + 3 + NUM_ACTIONS, 96), nn.ReLU(),
            nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 3),
        )
        self.register_buffer("scale", torch.tensor(_ODOMETRY_SCALE), persistent=False)

    def forward(self, prev_rays: Tensor, rays: Tensor, velocities: Tensor, prev_action: Tensor) -> Tensor:
        encoded = self.rays(torch.cat((prev_rays, rays), dim=-1).transpose(1, 2))
        return self.motion(torch.cat((encoded, velocities, prev_action), dim=-1)) * self.scale


class GridMemory:
    """Integrate ego-frame motion and maintain binary occupancy and semantic channels."""

    def __init__(self) -> None:
        self.grid = np.zeros((8, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        coarse_size = GRID_SIZE // POOL
        self._coarse = np.zeros((8, coarse_size, coarse_size), dtype=np.uint8)
        self.reset()

    def reset(self) -> None:
        self.grid.fill(0)
        self._coarse.fill(0)
        self._x = self._y = self._heading = 0.0
        self._cos, self._sin = 1.0, 0.0

    def pose(self) -> tuple[float, float, float]:
        return self._x, self._y, self._heading

    def _write(self, channel: int | np.ndarray, dx: np.ndarray, dy: np.ndarray) -> None:
        col = np.floor((self._x + dx) / CELL_SIZE).astype(np.intp) + GRID_SIZE // 2
        row = np.floor((self._y + dy) / CELL_SIZE).astype(np.intp) + GRID_SIZE // 2
        inside = (row >= 0) & (row < GRID_SIZE) & (col >= 0) & (col < GRID_SIZE)
        if isinstance(channel, np.ndarray):
            channel = channel[inside]
        row, col = row[inside], col[inside]
        self.grid[channel, row, col] = 1
        # A binary write also sets the max of its containing coarse cell.
        self._coarse[channel, row // POOL, col // POOL] = 1

    def step(
        self, rays_dist: np.ndarray, rays_type: np.ndarray, ents_feat: np.ndarray,
        ents_type: np.ndarray, ents_mask: np.ndarray, delta: tuple[float, float, float] | np.ndarray,
    ) -> None:
        fwd, left, turn = map(float, delta)
        self._x += self._cos * fwd - self._sin * left
        self._y += self._sin * fwd + self._cos * left
        self._heading = (self._heading + turn) % 360.0
        heading = np.deg2rad(self._heading)
        self._cos = round(float(np.cos(heading)), 15)
        self._sin = round(float(np.sin(heading)), 15)

        self.grid[5].fill(0)
        self._coarse[5].fill(0)
        dx = self._cos * _RAY_COS - self._sin * _RAY_SIN
        dy = self._sin * _RAY_COS + self._cos * _RAY_SIN
        distance = np.clip(rays_dist, 0, 1) * 1024
        ray, sample = np.nonzero(_RAY_SAMPLES < distance[:, None])
        ranges = _RAY_SAMPLES[sample]
        self._write(0, dx[ray] * ranges, dy[ray] * ranges)
        hits = (distance < 1024) & (rays_type > 0)
        self._write(_RAY_CHANNEL[rays_type[hits]], dx[hits] * distance[hits], dy[hits] * distance[hits])

        channels = _ENTITY_CHANNEL[ents_type]
        visible = ents_mask.astype(bool) & (channels >= 0)
        dist, sin, cos = ents_feat[visible, :3].T
        forward, left = 1024 * dist * cos, 1024 * dist * sin
        self._write(channels[visible], self._cos * forward - self._sin * left,
                    self._sin * forward + self._cos * left)
        self._write(6, np.zeros(1), np.zeros(1))

    def crops(self) -> np.ndarray:
        dx = self._cos * _CROP_FWD + self._sin * _CROP_RIGHT
        dy = self._sin * _CROP_FWD - self._cos * _CROP_RIGHT
        crops = np.zeros((16, 24 * 24), dtype=np.uint8)
        for offset, grid, cell in ((0, self.grid, CELL_SIZE), (8, self._coarse, CELL_SIZE * POOL)):
            size = grid.shape[1]
            col = np.floor(dx + self._x / cell).astype(np.intp) + size // 2
            row = np.floor(dy + self._y / cell).astype(np.intp) + size // 2
            inside = (row >= 0) & (row < size) & (col >= 0) & (col < size)
            crops[offset:offset + 8, inside] = grid[:, row[inside], col[inside]]
        return crops.reshape(CROP_SHAPE)

    def render_rgb(self, scale: int = 2) -> np.ndarray:
        """Render a north-up view, trimmed to the explored cells and agent."""
        if scale < 1:
            raise ValueError("scale must be positive")
        rgb = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
        for channel in (0, 6, 1, 2, 3, 4, 5, 7):
            rgb[self.grid[channel].astype(bool)] = _COLORS[channel]
        x, y = np.floor(np.array((self._x, self._y)) / CELL_SIZE).astype(int) + GRID_SIZE // 2
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            tip_x = x + round(self._cos)
            tip_y = y + round(self._sin)
            if 0 <= tip_x < GRID_SIZE and 0 <= tip_y < GRID_SIZE:
                rgb[tip_y, tip_x] = (116, 210, 251)
            rgb[y, x] = (43, 169, 241)
        rows, cols = np.nonzero(rgb.any(axis=2))
        if rows.size:
            rgb = rgb[max(0, rows.min() - 1):rows.max() + 2, max(0, cols.min() - 1):cols.max() + 2]
        else:
            rgb = rgb[GRID_SIZE // 2:GRID_SIZE // 2 + 1, GRID_SIZE // 2:GRID_SIZE // 2 + 1]
        return rgb[::-1].repeat(scale, axis=0).repeat(scale, axis=1)


class SensorMemory:
    """Track an episode's map using the same encoded inputs as the policy."""

    def __init__(self) -> None:
        self.grid = GridMemory()
        self.prev_rays: np.ndarray | None = None

    def reset(self) -> None:
        self.grid.reset()
        self.prev_rays = None

    @torch.inference_mode()
    def step(self, encoded: dict[str, np.ndarray], odometry: OdometryNet) -> np.ndarray:
        rays = encoded["rays"]
        ents = encoded["ents"]
        delta = (0.0, 0.0, 0.0)
        if self.prev_rays is not None:
            delta = odometry(
                torch.from_numpy(self.prev_rays)[None], torch.from_numpy(rays)[None],
                torch.from_numpy(encoded["vitals"][14:17])[None],
                torch.from_numpy(encoded["prev_action"])[None],
            )[0].numpy()
        self.grid.step(rays[:, 0], rays[:, 1:].argmax(axis=-1), ents[:, :5],
                       ents[:, 5:].argmax(axis=-1), encoded["ent_mask"], delta)
        self.prev_rays = rays
        return self.grid.crops()


class NakshaMemNet(NakshaNet):
    def __init__(self) -> None:
        super().__init__()
        self.memory_encoder = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 3 * 3, 128), nn.ReLU(),
        )
        self.core[0] = nn.Linear(self.core[0].in_features + 128, HIDDEN)
        self.odometry = OdometryNet()

    def forward(self, inputs: dict[str, Tensor], state: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        time, batch = inputs["vitals"].shape[:2]
        rays = self.rays(inputs["rays"].flatten(0, 1).transpose(1, 2))
        ents = self.ents(inputs["ents"].flatten(0, 1))
        ents = torch.cat((ents, self.empty_entity.expand(time * batch, -1, -1)), dim=1)
        mask = inputs["ent_mask"].flatten(0, 1).bool()
        mask = torch.cat((mask, torch.ones_like(mask[:, :1])), dim=1)
        scores = (ents @ self.entity_query) * (64 ** -0.5)
        weights = scores.masked_fill(~mask, -torch.inf).softmax(dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), ents).squeeze(1)

        vitals = self.vitals(inputs["vitals"].flatten(0, 1))
        prev_action = self.prev_action(inputs["prev_action"].flatten(0, 1))
        pixels = self.pixels(inputs["pixels"].flatten(0, 1).float() / 255)
        memory = self.memory_encoder(inputs["memory"].flatten(0, 1).float())
        z = self.core(torch.cat((rays, pooled, vitals, prev_action, pixels, memory), dim=-1))
        z, _ = self.memory(z.reshape(time, batch, HIDDEN), None)
        z, state = self.gru(z, state)
        return {
            "logits": self.logits(z),
            "value": self.value(z).squeeze(-1),
            "intention": self.intention(z),
            "exit_vis": self.exit_vis(z).squeeze(-1),
            "exit_dir": self.exit_dir(z).tanh(),
        }, state


def load_any(
    path: str | Path, device: str | torch.device = "cpu",
) -> tuple[NakshaNet, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    meta = checkpoint.get("meta", {})
    for name, expected in _VERSIONS.items():
        if meta.get(name) != expected:
            raise ValueError(f"{name} mismatch: expected {expected}, got {meta.get(name)!r}")
    version = meta.get("memory")
    if meta.get("ensemble"):
        model = EnsembleNet(int(meta["ensemble"]))
    elif version is None:
        model = NakshaNet()
    elif version == MEMORY_VERSION:
        model = NakshaMemNet()
    else:
        raise ValueError(f"unsupported memory version {version!r}")
    model.to(device).load_state_dict(checkpoint["state_dict"])
    return model, meta


def pack_crops(crops: np.ndarray) -> np.ndarray:
    return np.packbits(crops, axis=-1)


def unpack_crops(bits: np.ndarray) -> np.ndarray:
    return np.unpackbits(bits, axis=-1, count=CROP_SHAPE[-1])
