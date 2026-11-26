"""Fit observation-only odometry from pose-labelled episodes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import smooth_l1_loss

import training.paths  # noqa: F401
from naksha.actions import NUM_ACTIONS, table
from naksha.memory import OdometryNet

ODOMETRY_PATH = Path("runs/memory/odometry.pt")
WINDOW = 200
_RAY_TYPES = np.eye(6, dtype=np.float32)
_ACTIONS = np.eye(NUM_ACTIONS, dtype=np.float32)
_TURNS = 14.06 * (table()[:, 4] - table()[:, 5])


def episode_paths(dirs: list[str | Path]) -> list[Path]:
    paths = []
    for directory in dirs:
        episodes = sorted(Path(directory).absolute().glob("*.npz"))
        if not episodes:
            raise ValueError(f"no episodes in {directory}")
        paths.extend(episodes)
    return list(dict.fromkeys(paths))


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + 180) % 360 - 180


def motion_targets(pose: np.ndarray) -> np.ndarray:
    dx, dy, turn = np.diff(pose, axis=0).T
    heading = np.deg2rad(pose[:-1, 2])
    cos, sin = np.cos(heading), np.sin(heading)
    return np.stack((cos * dx + sin * dy, -sin * dx + cos * dy, _wrap(turn)), axis=1)


class _Episodes:
    """Store half-precision ranges and type IDs, decoding one minibatch at a time."""

    def __init__(self, paths: list[Path]) -> None:
        distances, types, velocities, actions, targets = [], [], [], [], []
        self.poses: list[np.ndarray] = []
        self.slices: list[slice] = []
        offset = 0
        for path in paths:
            with np.load(path, allow_pickle=False) as episode:
                pose = episode["pose"]
                if len(pose) < 2:
                    continue
                rays = episode["rays"]
                distances.append(rays[:, :, 0].astype(np.float16))
                types.append(rays[:, :, 1:].argmax(axis=-1).astype(np.uint8))
                velocities.append(episode["vitals"][:, 14:17].copy())
                actions.append(episode["prev_action"].argmax(axis=-1).astype(np.uint8))
                targets.append(motion_targets(pose))
            self.poses.append(pose)
            self.slices.append(slice(offset + 1, offset + len(pose)))
            offset += len(pose)
        if not self.slices:
            raise ValueError("odometry requires episodes with at least two steps")
        self.distance = np.concatenate(distances)
        self.types = np.concatenate(types)
        self.velocity = np.concatenate(velocities)
        self.action = np.concatenate(actions)
        self.targets = np.empty((offset, 3), np.float32)
        for segment, target in zip(self.slices, targets):
            self.targets[segment] = target

    def batch(self, indices: np.ndarray) -> tuple[torch.Tensor, ...]:
        def rays(rows: np.ndarray) -> np.ndarray:
            return np.concatenate((self.distance[rows, :, None].astype(np.float32),
                                   _RAY_TYPES[self.types[rows]]), axis=-1)

        return tuple(torch.from_numpy(value) for value in (
            rays(indices - 1), rays(indices), self.velocity[indices], _ACTIONS[self.action[indices]],
        ))


def _window_drift(motion: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Integrate once, then align the predicted heading and position at each window start.
    head = np.r_[0.0, np.cumsum(motion[:, 2], dtype=np.float64)]
    predicted_pos = np.r_[0j, np.cumsum(
        (motion[:, 0] + 1j * motion[:, 1]) * np.exp(1j * np.deg2rad(head[:-1])),
    )]
    predicted_delta = predicted_pos[WINDOW:] - predicted_pos[:-WINDOW]
    aligned = predicted_delta * np.exp(1j * np.deg2rad(pose[:-WINDOW, 2] - head[:-WINDOW]))
    actual_pos = pose[:, 0] + 1j * pose[:, 1]
    actual_delta = actual_pos[WINDOW:] - actual_pos[:-WINDOW]
    actual_turn = pose[WINDOW:, 2] - pose[:-WINDOW, 2]
    heading_error = _wrap(head[WINDOW:] - head[:-WINDOW] - actual_turn)
    return np.abs(aligned - actual_delta), np.abs(heading_error)


def _evaluate(
    net: OdometryNet, data: _Episodes, held_out: np.ndarray, batch_size: int,
) -> dict:
    totals = np.zeros((2, 3))
    drift = np.zeros((2, 2))
    steps = windows = 0
    net.eval()
    with torch.inference_mode():
        for index in held_out:
            segment = data.slices[index]
            indices = np.arange(segment.start, segment.stop)
            output = []
            for start in range(0, len(indices), batch_size):
                output.append(net(*data.batch(indices[start:start + batch_size])).numpy())
            learned = np.concatenate(output)
            naive = np.column_stack((4.4 * 20 * data.velocity[indices, :2],
                                     _TURNS[data.action[indices]]))
            pose = data.poses[index]
            for i, prediction in enumerate((learned, naive)):
                error = prediction - data.targets[indices]
                error[:, 2] = _wrap(error[:, 2])
                totals[i] += np.abs(error).sum(axis=0)
                if len(indices) >= WINDOW:
                    position, heading = _window_drift(prediction, pose)
                    drift[i] += position.sum(), heading.sum()
            steps += len(indices)
            windows += max(0, len(indices) - WINDOW + 1)
    return {
        "held_out_episodes": len(held_out), "held_out_steps": steps, "windows_200": windows,
        **{name: {
            "mae_fwd": float(totals[i, 0] / steps), "mae_left": float(totals[i, 1] / steps),
            "mae_theta_deg": float(totals[i, 2] / steps),
            "drift_position": float(drift[i, 0] / windows) if windows else None,
            "drift_heading_deg": float(drift[i, 1] / windows) if windows else None,
        } for i, name in enumerate(("learned", "naive"))},
    }


def train(
    dirs: list[str | Path], minutes: float, *, out: Path = ODOMETRY_PATH,
    max_updates: int | None = None, batch_size: int = 256,
) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = _Episodes(episode_paths(dirs))
    count = len(data.slices)
    if count < 2:
        raise ValueError("odometry requires at least two eligible episodes for the held-out split")
    order = rng.permutation(count)
    val_size = min(count - 1, max(1, round(count * 0.1)))
    train_indices = np.concatenate([
        np.arange(data.slices[i].start, data.slices[i].stop) for i in order[val_size:]
    ])
    net = OdometryNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    updates = 0
    deadline = time.monotonic() + minutes * 60
    print(f"Training on {len(train_indices):,} transitions from {count - val_size:,} episodes", flush=True)
    while time.monotonic() < deadline and (max_updates is None or updates < max_updates):
        rng.shuffle(train_indices)
        for start in range(0, len(train_indices), batch_size):
            indices = train_indices[start:start + batch_size]
            target = torch.from_numpy(data.targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss = smooth_l1_loss(net(*data.batch(indices)) / net.scale, target / net.scale)
            loss.backward()
            optimizer.step()
            updates += 1
            if updates % 1000 == 0:
                print(f"Update {updates:,}, training loss {float(loss.detach()):.5f}", flush=True)
            if time.monotonic() >= deadline or (max_updates is not None and updates >= max_updates):
                break
    metrics = {"updates": updates, **_evaluate(net, data, order[:val_size], batch_size)}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "metrics": metrics}, out)
    return metrics


def load_odometry(path: str | Path) -> OdometryNet:
    net = OdometryNet()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
    return net.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--minutes", type=float, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.minutes) or args.minutes <= 0:
        parser.error("minutes must be finite and positive")
    print(json.dumps(train(args.data, args.minutes), indent=2))


if __name__ == "__main__":
    main()
