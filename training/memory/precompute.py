"""Build packed spatial-memory crops from predicted episode trajectories."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from training.memory.odometry_train import ODOMETRY_PATH, episode_paths, load_odometry
from naksha.memory import CROP_SHAPE, GridMemory, OdometryNet, pack_crops

PACKED_SHAPE = (*CROP_SHAPE[:-1], (CROP_SHAPE[-1] + 7) // 8)
_odometry: OdometryNet


def sidecar_path(path: Path) -> Path:
    return path.parent.with_name(path.parent.name + "_mem") / f"{path.stem}.mem.npz"


def _init_worker(odometry: Path) -> None:
    global _odometry
    torch.set_num_threads(1)
    _odometry = load_odometry(odometry)


@torch.inference_mode()
def episode_crops(episode: dict[str, np.ndarray], odometry: OdometryNet) -> np.ndarray:
    rays = episode["rays"]
    vitals = episode["vitals"]
    previous = episode["prev_action"]
    deltas = np.zeros((len(rays), 3), dtype=np.float32)
    for start in range(1, len(rays), 512):
        stop = min(start + 512, len(rays))
        deltas[start:stop] = odometry(
            torch.from_numpy(rays[start - 1:stop - 1]), torch.from_numpy(rays[start:stop]),
            torch.from_numpy(vitals[start:stop, 14:17]), torch.from_numpy(previous[start:stop]),
        ).numpy()

    ents = episode["ents"]
    ray_types = rays[:, :, 1:].argmax(axis=-1)
    ent_types = ents[:, :, 5:].argmax(axis=-1)
    memory = GridMemory()
    packed = np.empty((len(rays), *PACKED_SHAPE), dtype=np.uint8)
    for i, delta in enumerate(deltas):
        memory.step(rays[i, :, 0], ray_types[i], ents[i, :, :5], ent_types[i],
                    episode["ent_mask"][i], delta)
        packed[i] = pack_crops(memory.crops())
    return packed


def _precompute(path: Path) -> int:
    with np.load(path, allow_pickle=False) as episode:
        observations = {key: episode[key] for key in ("rays", "ents", "ent_mask", "vitals", "prev_action")}
    packed = episode_crops(observations, _odometry)
    out = sidecar_path(path)
    with NamedTemporaryFile(dir=out.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        try:
            np.savez_compressed(file, memory_bits=packed)
            file.flush()
            temporary.replace(out)
        finally:
            temporary.unlink(missing_ok=True)
    return out.stat().st_size


def precompute(dirs: list[str | Path], odometry: Path, procs: int) -> int:
    paths = episode_paths(dirs)
    for path in {sidecar_path(path).parent for path in paths}:
        path.mkdir(parents=True, exist_ok=True)
    if procs == 1:
        _init_worker(odometry)
        return _report(map(_precompute, paths), len(paths))
    with ProcessPoolExecutor(max_workers=procs, initializer=_init_worker, initargs=(odometry,)) as pool:
        return _report(pool.map(_precompute, paths), len(paths))


def _report(sizes: Iterable[int], count: int) -> int:
    total = 0
    for i, size in enumerate(sizes, 1):
        total += size
        if i % 100 == 0 and i < count:
            print(f"Processed {i:,}/{count:,} episodes", flush=True)
    print(f"Wrote {count:,} episodes, {total:,} bytes ({total / 2**20:.2f} MiB)", flush=True)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--odometry", type=Path, default=ODOMETRY_PATH)
    parser.add_argument("--procs", type=int, default=6)
    args = parser.parse_args()
    if args.procs < 1:
        parser.error("procs must be positive")
    precompute(args.data, args.odometry, args.procs)


if __name__ == "__main__":
    main()
