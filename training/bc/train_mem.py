"""Train the recurrent policy by behavior cloning with precomputed spatial memory."""

from __future__ import annotations

import argparse
import csv
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from training.bc.train import BURN_IN, _git_sha, _masked_mean
from training.data.shards import _FIELDS
from training.memory.odometry_train import ODOMETRY_PATH, episode_paths, load_odometry
from training.memory.precompute import PACKED_SHAPE, sidecar_path
from naksha import features, model
from naksha.memory import MEMORY_VERSION, NakshaMemNet, load_any, unpack_crops

_TARGETS = ("label", "intention", "exit_vis", "exit_dir")
_FIELDS_USED = {key: _FIELDS[key] for key in (*features.KEYS, *_TARGETS)}


@lru_cache(maxsize=24)
def _read_episode(path: Path, bits_path: Path, length: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        episode = {key: archive[key] for key in _FIELDS_USED}
    for key, (shape, dtype) in _FIELDS_USED.items():
        if episode[key].shape != (length, *shape) or episode[key].dtype != dtype:
            raise ValueError(f"{path}: invalid shape or dtype for {key}")
    with np.load(bits_path, allow_pickle=False) as archive:
        bits = archive["memory_bits"]
    if bits.shape != (length, *PACKED_SHAPE) or bits.dtype != np.uint8:
        raise ValueError(f"{bits_path}: expected uint8 memory_bits of shape {(length, *PACKED_SHAPE)}")
    return episode, bits


class MemoryDataset:
    """Keep a bounded cache of shards and unpack only the sampled sequences."""

    def __init__(self, dirs: list[str | Path]) -> None:
        self.episodes: list[tuple[Path, Path, int]] = []
        for path in episode_paths(dirs):
            with np.load(path, allow_pickle=False) as archive:
                length = len(archive["action"])
            if length <= BURN_IN:
                continue
            bits_path = sidecar_path(path)
            if not bits_path.is_file():
                raise FileNotFoundError(f"missing memory crops for {path}: {bits_path}")
            self.episodes.append((path, bits_path, length))

    def split(self) -> tuple[np.ndarray, np.ndarray]:
        count = len(self.episodes)
        if count < 2:
            raise ValueError(f"training requires two episodes with at least {BURN_IN + 1} steps each")
        order = np.random.default_rng(0).permutation(count)
        val_size = min(count - 1, max(1, round(count * 0.1)))
        return order[val_size:], order[:val_size]

    def sample(
        self, indices: np.ndarray, batch: int, length: int, rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        result = {key: np.zeros((length, batch, *shape), dtype)
                  for key, (shape, dtype) in _FIELDS_USED.items()}
        result["valid"] = np.zeros((length, batch), dtype=bool)
        packed = np.zeros((length, batch, *PACKED_SHAPE), dtype=np.uint8)
        for column, index in enumerate(rng.choice(indices, size=batch)):
            entry = self.episodes[index]
            episode_length = entry[2]
            start = int(rng.integers(episode_length - BURN_IN))
            size = min(episode_length - start, length)
            segment = slice(start, start + size)
            episode, bits = _read_episode(*entry)
            for key, array in episode.items():
                result[key][:size, column] = array[segment]
            result["valid"][:size, column] = True
            packed[:size, column] = bits[segment]
        result["memory"] = unpack_crops(packed)
        return result


def _batch_loss(
    net: NakshaMemNet, chunk: dict[str, np.ndarray], device: torch.device,
) -> tuple[Tensor, Tensor]:
    inputs = {key: torch.from_numpy(chunk[key]).to(device) for key in (*features.KEYS, "memory")}
    state = net.initial_state(chunk["valid"].shape[1], device)
    with torch.no_grad():
        _, state = net({key: data[:BURN_IN] for key, data in inputs.items()}, state)
    out, _ = net({key: data[BURN_IN:] for key, data in inputs.items()}, state)

    targets = {key: torch.from_numpy(chunk[key][BURN_IN:]).to(device)
               for key in ("valid", *_TARGETS)}
    valid = targets["valid"]
    label = targets["label"].long()
    labelled = valid & (label >= 0)
    action_ce = F.cross_entropy(out["logits"].flatten(0, 1), label.masked_fill(~labelled, 0).flatten(),
                                reduction="none").reshape_as(label)
    intention = targets["intention"].long().masked_fill(~labelled, 0)
    intention_ce = F.cross_entropy(out["intention"].flatten(0, 1), intention.flatten(),
                                   reduction="none").reshape_as(label)
    visible = targets["exit_vis"]
    exit_bce = F.binary_cross_entropy_with_logits(out["exit_vis"], visible.float(), reduction="none")
    visible_mask = valid & (visible == 1)
    direction = targets["exit_dir"].float().masked_fill(~visible_mask.unsqueeze(-1), 0)
    exit_mse = (out["exit_dir"] - direction).square().mean(dim=-1)
    loss = (_masked_mean(action_ce, labelled) + 0.2 * _masked_mean(intention_ce, labelled)
            + 0.1 * _masked_mean(exit_bce, valid) + 0.1 * _masked_mean(exit_mse, visible_mask))
    accuracy = _masked_mean(out["logits"].argmax(dim=-1) == label, labelled)
    return loss, accuracy


def _build_model(
    init: Path | None, odometry: Path, device: torch.device,
) -> NakshaMemNet:
    net = NakshaMemNet()
    if init:
        initial, _ = load_any(init)
        if isinstance(initial, NakshaMemNet):
            net = initial
        else:
            weights = initial.state_dict()
            core_weight = weights.pop("core.0.weight")
            net.load_state_dict(weights, strict=False)
            with torch.no_grad():
                net.core[0].weight[:, :core_weight.shape[1]].copy_(core_weight)
    net.odometry.load_state_dict(load_odometry(odometry).state_dict())
    net.odometry.requires_grad_(False)
    return net.to(device)


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dataset = MemoryDataset(args.data)
    train_indices, val_indices = dataset.split()
    rng = np.random.default_rng()
    chunk_len = args.seq + BURN_IN
    val_chunk = dataset.sample(val_indices, args.batch, chunk_len, np.random.default_rng(1))
    net = _build_model(args.init, args.odometry, device)
    optimizer = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=1e-4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path("runs") / args.out.stem / "log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best = {"data": args.data, "git_sha": _git_sha(), "val_loss": float("inf"),
            "memory": MEMORY_VERSION}

    def validate(update: int) -> tuple[float, float]:
        net.eval()
        with torch.inference_mode():
            loss, acc = (float(value) for value in _batch_loss(net, val_chunk, device))
        if loss < best["val_loss"]:
            best.update(updates=update, val_loss=loss, val_acc=acc)
            model.save_checkpoint(args.out, net, best)
        net.train()
        return loss, acc

    updates = 0
    train_metrics: tuple[float | str, float | str] = ("", "")
    with log_path.open("w", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(("update", "loss", "train_acc", "val_loss", "val_acc"))
        deadline = time.monotonic() + args.minutes * 60
        while time.monotonic() < deadline:
            chunk = dataset.sample(train_indices, args.batch, chunk_len, rng)
            optimizer.zero_grad(set_to_none=True)
            loss, acc = _batch_loss(net, chunk, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            updates += 1
            if updates % 200 == 0:
                train_metrics = (float(loss.detach()), float(acc))
                val_metrics = validate(updates) if updates % 1_000 == 0 else ("", "")
                writer.writerow((updates, *train_metrics, *val_metrics))
                log.flush()
        if updates == 0 or updates % 1_000:
            val_metrics = validate(updates)
            if updates % 200:
                train_metrics = (float(loss.detach()), float(acc))
            writer.writerow((updates, *train_metrics, *val_metrics))
    return {**best, "total_updates": updates, "out": str(args.out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--odometry", type=Path, default=ODOMETRY_PATH)
    args = parser.parse_args()
    if not np.isfinite(args.minutes) or args.minutes < 0 or min(args.batch, args.seq) < 1:
        parser.error("minutes must be finite and nonnegative, and batch and seq must be positive")
    print(json.dumps(train(args)))


if __name__ == "__main__":
    main()
