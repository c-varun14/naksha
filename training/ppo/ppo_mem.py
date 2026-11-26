"""Fine-tune a grid-v1 Naksha checkpoint with recurrent PPO."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections.abc import Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical

from training.ppo import ppo
from naksha.memory import CROP_SHAPE, MEMORY_VERSION, NakshaMemNet, SensorMemory, load_any, pack_crops

PACKED_SHAPE = (*CROP_SHAPE[:-1], (CROP_SHAPE[-1] + 7) // 8)


class Collector(ppo.Collector):
    def __init__(self, args: argparse.Namespace, worker: int, episode_counter) -> None:
        super().__init__(args, worker, episode_counter)
        self.model = NakshaMemNet().eval().requires_grad_(False)
        self.memory = SensorMemory()
        self.bootstrap_crops: np.ndarray | None = None

    def _start_episode(self) -> None:
        super()._start_episode()
        self.memory.reset()
        self.bootstrap_crops = None

    def _forward(self, features: dict[str, np.ndarray]) -> tuple[dict[str, Tensor], Tensor]:
        crops = self.bootstrap_crops
        self.bootstrap_crops = None
        if crops is None:
            crops = self.memory.step(features, self.model.odometry)
        if self.position < len(self.packed):
            self.packed[self.position] = pack_crops(crops)
            self.position += 1
        else:
            # The bootstrap observation is also the first one in the next
            # rollout. Reuse its map crop, but evaluate with the new weights.
            self.bootstrap_crops = crops
        inputs = {key: torch.from_numpy(value)[None, None] for key, value in features.items()}
        inputs["memory"] = torch.from_numpy(crops)[None, None]
        return self.model(inputs, self.hidden)

    def collect(self) -> ppo.Rollout:
        self.position = 0
        self.packed = np.empty((self.args.steps_per_worker, *PACKED_SHAPE), dtype=np.uint8)
        rollout = super().collect()
        rollout.features["memory_bits"] = self.packed
        return rollout


def _worker(pipe: Connection, args: argparse.Namespace, index: int, episode_counter, scratch: Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(args.seed_offset + index * 100_000 + episode_counter.value)
    cwd = scratch / str(os.getpid())
    cwd.mkdir(exist_ok=True)
    os.chdir(cwd)
    collector = Collector(args, index, episode_counter)
    try:
        while (weights := pipe.recv()) is not None:
            collector.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            pipe.send(collector.collect())
    except (EOFError, BrokenPipeError):
        pass
    finally:
        collector.close()
        pipe.close()


class Workers(ppo.Workers):
    def _start(self, index: int) -> ppo.Worker:
        parent, child = self.ctx.Pipe()
        process = self.ctx.Process(
            target=_worker, args=(child, self.args, index, self.counters[index], self.scratch),
            daemon=True,
        )
        try:
            process.start()
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()
        worker = ppo.Worker(process, parent)
        self.running[index] = worker
        return worker


class PackedPolicy(torch.nn.Module):
    """Decode only the replay segment or minibatch entering a forward pass."""

    def __init__(self, model: NakshaMemNet, device: torch.device) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("bit_order", torch.arange(7, -1, -1, dtype=torch.uint8, device=device))

    def initial_state(self, batch: int, device: torch.device) -> Tensor:
        return self.model.initial_state(batch, device)

    def forward(self, inputs: dict[str, Tensor], hidden: Tensor) -> tuple[dict[str, Tensor], Tensor]:
        inputs = inputs.copy()
        bits = inputs.pop("memory_bits")
        inputs["memory"] = ((bits.unsqueeze(-1) >> self.bit_order) & 1).flatten(-2)[..., :CROP_SHAPE[-1]]
        return self.model(inputs, hidden)


def _update(
    model: PackedPolicy, reference: PackedPolicy, optimizer: torch.optim.Optimizer,
    rollouts: dict[int, ppo.Rollout], reference_hidden: dict[int, Tensor], args: argparse.Namespace,
) -> dict[str, float]:
    device = model.bit_order.device
    sequences = []
    features = {key: [] for key in (*ppo.KEYS, "memory_bits")}
    old_actions, old_log_probs, old_values, ref_log_probs = [], [], [], []
    advantages, targets = [], []
    offset = 0
    for worker, rollout in sorted(rollouts.items()):
        inputs = {key: torch.from_numpy(value).to(device) for key, value in rollout.features.items()}
        ref, reference_hidden[worker] = ppo._reference_log_probs(
            reference, inputs, rollout, reference_hidden.get(worker),
        )
        ref_log_probs.append(ref)
        for key in features:
            features[key].append(inputs[key])
        old_actions.append(rollout.actions)
        old_log_probs.append(rollout.log_probs)
        old_values.append(rollout.values[:-1])
        ends = [start for start, _ in rollout.chunks[1:]] + [len(rollout.actions)]
        for (start, hidden), end in zip(rollout.chunks, ends, strict=True):
            advantage, target = ppo.gae(
                rollout.rewards[start:end], rollout.values[start:end + 1], rollout.dones[start:end],
                args.gamma, args.lam,
            )
            sequences.append((offset + start, end - start, hidden))
            advantages.append(advantage)
            targets.append(target)
        offset += len(rollout.actions)

    inputs = {key: torch.cat(parts) for key, parts in features.items()}
    actions, log_probs, values, advantage, target = (
        torch.from_numpy(np.concatenate(parts)).to(device)
        for parts in (old_actions, old_log_probs, old_values, advantages, targets)
    )
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    ref_log_prob = torch.cat(ref_log_probs)
    starts, lengths, states = zip(*sequences, strict=True)
    starts = torch.tensor(starts, device=device)
    lengths = torch.tensor(lengths, device=device)
    states = torch.from_numpy(np.concatenate(states, axis=1)).to(device)
    times = torch.arange(max(length for _, length, _ in sequences), device=device)[:, None]

    # Accumulate metrics by valid transition count, including short terminal chunks.
    totals = torch.zeros(3, device=device)
    for _ in range(args.epochs):
        order = torch.randperm(len(sequences), device=device)
        for batch in order.split(args.minibatch_seqs):
            length = lengths[batch]
            valid = times < length
            indices = starts[batch] + torch.minimum(times, length - 1)
            out, _ = model({key: value[indices] for key, value in inputs.items()}, states[:, batch])
            indices = indices[valid]
            dist = Categorical(logits=out["logits"][valid])
            ratio = (dist.log_prob(actions[indices]) - log_probs[indices]).exp()
            unclipped = ratio * advantage[indices]
            clipped = ratio.clamp(1 - args.clip, 1 + args.clip) * advantage[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()

            prediction = out["value"][valid]
            value_clipped = values[indices] + (prediction - values[indices]).clamp(-args.clip, args.clip)
            value_loss = torch.maximum(
                (prediction - target[indices]).square(), (value_clipped - target[indices]).square(),
            ).mean()
            entropy = dist.entropy().mean()
            ref = ref_log_prob[indices]
            kl = (ref.exp() * (ref - dist.logits)).sum(dim=-1).mean()
            loss = policy_loss + 0.5 * value_loss - args.ent_coef * entropy + args.kl_coef * kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            totals += torch.stack((kl.detach(), entropy.detach(), value_loss.detach())) * len(indices)
    return dict(zip(("kl", "entropy", "value_loss"), (totals / (offset * args.epochs)).tolist(), strict=True))


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    net, initial_meta = load_any(args.init)
    if not isinstance(net, NakshaMemNet):
        raise ValueError(f"--init must be a {MEMORY_VERSION} checkpoint")
    reference, _ = load_any(args.init, device)
    net.to(device)
    net.odometry.eval().requires_grad_(False)
    policy = PackedPolicy(net, device)
    reference = PackedPolicy(reference.eval().requires_grad_(False), device)
    optimizer = torch.optim.Adam((p for p in policy.parameters() if p.requires_grad), lr=args.lr)
    reference_hidden: dict[int, Tensor] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path("runs/ppo") / args.out.stem / "log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"init": str(args.init), "iterations": 0, "steps": 0, "memory": MEMORY_VERSION}
    if "temperature" in initial_meta:
        meta["temperature"] = initial_meta["temperature"]
    deadline = perf_counter() + 60 * args.minutes
    with TemporaryDirectory(prefix="naksha-ppo-mem-") as scratch, log_path.open("w", newline="", encoding="utf-8") as log:
        weights = Path(scratch) / "policy.pt"
        workers = Workers(args, Path(scratch))
        writer = csv.DictWriter(log, fieldnames=ppo.METRICS)
        writer.writeheader()
        try:
            while True:
                started = perf_counter()
                torch.save({key: value.detach().cpu() for key, value in net.state_dict().items()}, weights)
                rollouts = workers.collect(weights)
                if rollouts:
                    losses = _update(policy, reference, optimizer, rollouts, reference_hidden, args)
                    steps = sum(len(rollout.actions) for rollout in rollouts.values())
                    meta["steps"] += steps
                    meta["iterations"] += 1
                    writer.writerow({
                        "iteration": meta["iterations"],
                        "steps": meta["steps"],
                        **ppo.episode_metrics(rollouts),
                        "kills": sum(rollout.kills for rollout in rollouts.values()),
                        "novelty_cells": sum(rollout.novelty_cells for rollout in rollouts.values()),
                        **losses,
                        "fps": steps / (perf_counter() - started),
                    })
                    log.flush()
                    if meta["iterations"] % 10 == 0:
                        ppo.save_checkpoint(args.out, net, meta)
                if perf_counter() >= deadline:
                    break
        finally:
            workers.close()
    ppo.save_checkpoint(args.out, net, meta)
    return {**meta, "out": str(args.out), "log": str(log_path), "worker_restarts": workers.restarts}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True, type=Path, help=f"initial {MEMORY_VERSION} checkpoint")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--minutes", required=True, type=float, help="wall-clock budget (0 runs one iteration)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps-per-worker", type=int, default=256)
    parser.add_argument("--families", nargs="+", choices=ppo.FAMILIES, default=ppo.FAMILIES)
    parser.add_argument("--sizes", nargs="+", choices=ppo.SIZES, default=ppo.SIZES)
    parser.add_argument("--seed-offset", type=int, default=300_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--clip", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--minibatch-seqs", type=int, default=32)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if min(args.workers, args.steps_per_worker, args.epochs, args.minibatch_seqs, args.seq) < 1:
        parser.error("workers, steps-per-worker, epochs, minibatch-seqs, and seq must be positive")
    if any(not math.isfinite(value) or value < 0 for value in (args.minutes, args.kl_coef, args.ent_coef)):
        parser.error("minutes, kl-coef, and ent-coef must be finite and nonnegative")
    if not (math.isfinite(args.lr) and args.lr > 0 and 0 < args.clip < 1):
        parser.error("lr must be finite and positive and clip must be between 0 and 1")
    if not (0 <= args.gamma <= 1 and 0 <= args.lam <= 1 and args.seed_offset >= 0):
        parser.error("gamma and lam must be in [0, 1] and seed-offset must be nonnegative")
    if args.init.resolve() == args.out.resolve():
        parser.error("init and out must be different paths")
    if torch.device(args.device).type == "cpu":
        torch.set_num_threads(1)
    try:
        summary = train(args)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
