"""Fine-tune a Naksha imitation checkpoint with recurrent PPO."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import os
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical

import training.paths  # noqa: F401
from doom_env import make_env
from naksha.actions import to_buttons
from naksha.features import KEYS, SHAPES, encode
from naksha.model import NakshaNet, load_checkpoint, save_checkpoint
from training.gen.compile import CELL, write_map
from training.gen.families import generate

FAMILIES = ("arena", "rooms", "locks", "gauntlet")
SIZES = ("s", "m", "l")
LOG = logging.getLogger(__name__)
METRICS = ("iteration", "steps", "return_mean", "exit_rate", "kills", "novelty_cells",
           "kl", "entropy", "value_loss", "fps")


def gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return advantages and value targets. The last value bootstraps the chunk."""
    advantages = np.empty(len(rewards), dtype=np.float32)
    carry = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        discount = gamma * (not dones[t])
        delta = rewards[t] + discount * values[t + 1] - values[t]
        carry = delta + discount * lam * carry
        advantages[t] = carry
    return advantages, advantages + values[:-1]


def _cell(state: dict) -> tuple[int, int]:
    return int(state["x"] // CELL), int(state["y"] // CELL)


class EpisodeReward:
    def __init__(self, state: dict, door_locks: dict[int, int]) -> None:
        self.previous = state
        self.door_locks = door_locks
        self.opened = {sector for sector in door_locks if state["sector_ceiling"][sector] > 8}
        self.cells = {_cell(state)}
        self.total = 0.0

    def step(self, state: dict) -> tuple[float, int, int]:
        previous = self.previous
        reward = -0.0005
        if state["exit_reached"] and not previous["exit_reached"]:
            ticks = state["episode_ticks"]
            reward += 1.0 + 0.5 * max(0, ticks - state["tic"]) / ticks
        if state["dead"] and not previous["dead"]:
            reward -= 1.0
        reward += 0.5 * np.count_nonzero(np.logical_and(state["keys"], np.logical_not(previous["keys"])))
        for sector, lock in self.door_locks.items():
            if sector not in self.opened and previous["sector_ceiling"][sector] <= 8 < state["sector_ceiling"][sector]:
                self.opened.add(sector)
                reward += 0.5 if lock > 0 else 0.2
        cell = _cell(state)
        novel = int(cell not in self.cells)
        self.cells.add(cell)
        kills = max(0, state["kills"] - previous["kills"])
        reward += 0.02 * novel + 0.1 * kills - 0.003 * max(0, previous["health"] - state["health"])
        reward = float(np.clip(reward, -2.0, 2.0))
        self.total += reward
        self.previous = state
        return reward, kills, novel


@dataclass
class Rollout:
    features: dict[str, np.ndarray]
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    starts_episode: bool
    chunks: list[tuple[int, np.ndarray]]
    episodes: list[tuple[float, bool]]
    kills: int
    novelty_cells: int


class Collector:
    def __init__(self, args: argparse.Namespace, worker: int, episode_counter) -> None:
        self.args = args
        self.worker = worker
        self.episode_counter = episode_counter
        self.model = NakshaNet().eval()
        self.env = None

    def _start_episode(self) -> None:
        k = self.episode_counter.value
        self.episode_counter.value += 1
        args = self.args
        seed = args.seed_offset + self.worker * 100_000 + k
        # Cycle through every family at each size before advancing to the next size.
        family = args.families[(self.worker + k) % len(args.families)]
        size = args.sizes[(self.worker + k // len(args.families)) % len(args.sizes)]
        spec = generate(family, seed, size)
        wad = write_map(spec["grid"], "map.wad", player_angle=spec["player_angle"])
        self.env = make_env(wad, "MAP01", episode_ticks=spec["episode_ticks"], render_buffers=True)
        self.obs, _ = self.env.reset(seed=seed)
        self.reward = EpisodeReward(self.env.privileged_state(), self.env.mapinfo.door_sectors)
        self.hidden = self.model.initial_state(1)
        self.prev_action = 0

    def _forward(self, features: dict[str, np.ndarray]) -> tuple[dict[str, Tensor], Tensor]:
        inputs = {key: torch.from_numpy(features[key])[None, None] for key in KEYS}
        return self.model(inputs, self.hidden)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    @torch.inference_mode()
    def collect(self) -> Rollout:
        n = self.args.steps_per_worker
        rollout = Rollout(
            {key: np.empty((n, *shape), dtype) for key, (shape, dtype) in SHAPES.items()},
            np.empty(n, np.int64), np.empty(n, np.float32), np.empty(n + 1, np.float32),
            np.empty(n, np.float32), np.empty(n, np.bool_), self.env is None, [], [], 0, 0,
        )
        for t in range(n):
            if self.env is None:
                self._start_episode()
            if t == 0 or t - rollout.chunks[-1][0] == self.args.seq or rollout.dones[t - 1]:
                rollout.chunks.append((t, self.hidden.numpy().copy()))
            features = encode(self.obs, self.prev_action)
            for key in KEYS:
                rollout.features[key][t] = features[key]
            out, self.hidden = self._forward(features)
            dist = Categorical(logits=out["logits"][0, 0])
            action = dist.sample()
            self.prev_action = action.item()
            rollout.actions[t] = self.prev_action
            rollout.log_probs[t] = dist.log_prob(action).item()
            rollout.values[t] = out["value"].item()
            self.obs, _, terminated, truncated, _ = self.env.step(to_buttons(self.prev_action))
            state = self.env.privileged_state()
            reward, kills, novel = self.reward.step(state)
            rollout.rewards[t] = reward
            rollout.kills += kills
            rollout.novelty_cells += novel
            rollout.dones[t] = terminated or truncated
            if rollout.dones[t]:
                rollout.episodes.append((self.reward.total, state["exit_reached"]))
                self.close()
        rollout.values[n] = 0.0
        if self.env is not None:
            out, _ = self._forward(encode(self.obs, self.prev_action))
            rollout.values[n] = out["value"].item()
        return rollout


def _worker(
    pipe: Connection, args: argparse.Namespace, worker: int, episode_counter, scratch: Path,
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(args.seed_offset + worker * 100_000 + episode_counter.value)
    cwd = scratch / str(os.getpid())
    cwd.mkdir(exist_ok=True)
    os.chdir(cwd)
    collector = Collector(args, worker, episode_counter)
    try:
        while (weights := pipe.recv()) is not None:
            collector.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            pipe.send(collector.collect())
    except (EOFError, BrokenPipeError):
        pass
    finally:
        collector.close()
        pipe.close()


@dataclass
class Worker:
    process: BaseProcess
    pipe: Connection

    def close(self) -> None:
        self.pipe.close()
        self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        self.process.close()


class Workers:
    def __init__(self, args: argparse.Namespace, scratch: Path) -> None:
        self.args = args
        self.scratch = scratch
        self.ctx = mp.get_context("spawn")
        # Each counter has one writer. Replacements start after the old writer exits.
        self.counters = [self.ctx.Value("q", 0, lock=False) for _ in range(args.workers)]
        self.running: dict[int, Worker] = {}
        self.restarts = 0

    def _start(self, index: int) -> Worker:
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
        worker = Worker(process, parent)
        self.running[index] = worker
        return worker

    def collect(self, weights: Path) -> dict[int, Rollout]:
        results: dict[int, Rollout] = {}
        retries: set[int] = set()
        pending = {
            (self.running.get(index) or self._start(index)).pipe: index
            for index in range(self.args.workers)
        }
        for pipe in pending:
            try:
                pipe.send(weights)
            except OSError:
                # The read end will be ready with EOF when the worker exits.
                pass
        while pending:
            for pipe in wait(pending):
                index = pending.pop(pipe)
                try:
                    results[index] = pipe.recv()
                except (EOFError, OSError) as error:
                    self.running.pop(index).close()
                    if index in retries:
                        LOG.warning("PPO worker %d failed again (%r); will restart next iteration", index, error)
                        continue
                    retries.add(index)
                    self.restarts += 1
                    LOG.warning("Restarting PPO worker %d after rollout failure (%r)", index, error)
                    replacement = self._start(index)
                    try:
                        replacement.pipe.send(weights)
                    except OSError:
                        pass
                    pending[replacement.pipe] = index
        return results

    def close(self) -> None:
        for worker in self.running.values():
            try:
                worker.pipe.send(None)
            except OSError:
                pass
        for worker in self.running.values():
            worker.close()


@torch.no_grad()
def _reference_log_probs(
    reference: NakshaNet, features: dict[str, Tensor], rollout: Rollout, hidden: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Replay each episode segment once with the frozen policy's own recurrent state."""
    device = features["vitals"].device
    if hidden is None or rollout.starts_episode:
        hidden = reference.initial_state(1, device)
    ends = (np.flatnonzero(rollout.dones) + 1).tolist()
    if not ends or ends[-1] != len(rollout.dones):
        ends.append(len(rollout.dones))
    start = 0
    log_probs = []
    for end in ends:
        out, hidden = reference({key: value[start:end, None] for key, value in features.items()}, hidden)
        log_probs.append(out["logits"][:, 0].log_softmax(dim=-1))
        if rollout.dones[end - 1]:
            hidden = reference.initial_state(1, device)
        start = end
    return torch.cat(log_probs), hidden


def _update(
    model: NakshaNet, reference: NakshaNet, optimizer: torch.optim.Optimizer,
    rollouts: dict[int, Rollout], reference_hidden: dict[int, Tensor], args: argparse.Namespace,
) -> dict[str, float]:
    device = next(model.parameters()).device
    sequences = []
    features = {key: [] for key in KEYS}
    old_actions, old_log_probs, old_values, ref_log_probs = [], [], [], []
    advantages, targets = [], []
    offset = 0
    for worker, rollout in sorted(rollouts.items()):
        inputs = {key: torch.from_numpy(value).to(device) for key, value in rollout.features.items()}
        ref, reference_hidden[worker] = _reference_log_probs(
            reference, inputs, rollout, reference_hidden.get(worker),
        )
        ref_log_probs.append(ref)
        for key in KEYS:
            features[key].append(inputs[key])
        old_actions.append(rollout.actions)
        old_log_probs.append(rollout.log_probs)
        old_values.append(rollout.values[:-1])
        ends = [start for start, _ in rollout.chunks[1:]] + [len(rollout.actions)]
        for (start, hidden), end in zip(rollout.chunks, ends, strict=True):
            advantage, target = gae(
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
    times = torch.arange(max(end for _, end, _ in sequences), device=device)[:, None]

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
    model, _ = load_checkpoint(args.init, args.device)
    reference, _ = load_checkpoint(args.init, args.device)
    reference.eval().requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    reference_hidden: dict[int, Tensor] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path("runs/ppo") / args.out.stem / "log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"init": str(args.init), "iterations": 0, "steps": 0}
    deadline = perf_counter() + 60 * args.minutes
    with TemporaryDirectory(prefix="naksha-ppo-") as scratch, log_path.open("w", newline="", encoding="utf-8") as log:
        weights = Path(scratch) / "policy.pt"
        workers = Workers(args, Path(scratch))
        writer = csv.DictWriter(log, fieldnames=METRICS)
        writer.writeheader()
        try:
            # Complete one rollout/update with a zero-minute budget for smoke runs.
            while True:
                started = perf_counter()
                torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, weights)
                rollouts = workers.collect(weights)
                if rollouts:
                    losses = _update(model, reference, optimizer, rollouts, reference_hidden, args)
                    episodes = [episode for rollout in rollouts.values() for episode in rollout.episodes]
                    steps = sum(len(rollout.actions) for rollout in rollouts.values())
                    meta["steps"] += steps
                    meta["iterations"] += 1
                    writer.writerow({
                        "iteration": meta["iterations"],
                        "steps": meta["steps"],
                        "return_mean": sum(reward for reward, _ in episodes) / len(episodes) if episodes else None,
                        "exit_rate": sum(exited for _, exited in episodes) / len(episodes) if episodes else None,
                        "kills": sum(rollout.kills for rollout in rollouts.values()),
                        "novelty_cells": sum(rollout.novelty_cells for rollout in rollouts.values()),
                        **losses,
                        "fps": steps / (perf_counter() - started),
                    })
                    log.flush()
                    if meta["iterations"] % 10 == 0:
                        save_checkpoint(args.out, model, meta)
                if perf_counter() >= deadline:
                    break
        finally:
            workers.close()
    save_checkpoint(args.out, model, meta)
    return {**meta, "out": str(args.out), "log": str(log_path), "worker_restarts": workers.restarts}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--minutes", required=True, type=float, help="wall-clock budget (0 runs one iteration)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps-per-worker", type=int, default=256)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=FAMILIES)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=SIZES)
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
    print(json.dumps(train(args)))


if __name__ == "__main__":
    main()
