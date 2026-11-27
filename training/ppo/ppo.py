"""Fine-tune a Naksha imitation checkpoint with recurrent PPO."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import os
import random
from collections import deque
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
from training.eval.stock import STOCK_MAPS, build_specs
from training.gen.compile import CELL, write_map
from training.gen import families
from training.gen.families import generate

FAMILIES = ("arena", "rooms", "locks", "gauntlet")
SIZES = ("s", "m", "l")
LOG = logging.getLogger(__name__)
METRICS = ("iteration", "steps", "return_mean", "exit_rate", "kills", "novelty_cells",
           "kl", "entropy", "value_loss", "fps", "stock_exit_rate", "generated_exit_rate")


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


def transform_grid(grid: str, turns: int, mirror: bool) -> str:
    """Rotate an ASCII grid by `turns` quarter turns clockwise, then mirror it left-right if asked. Doors,
    keys and walls keep their neighbours, so a valid map stays valid."""
    rows = grid.strip("\n").split("\n")
    width = max(len(r) for r in rows)
    rows = [r.ljust(width, "#") for r in rows]
    for _ in range(turns % 4):
        rows = ["".join(col) for col in zip(*rows[::-1])]
    if mirror:
        rows = [r[::-1] for r in rows]
    return "\n".join(rows)


def _cell(state: dict) -> tuple[int, int]:
    return int(state["x"] // CELL), int(state["y"] // CELL)


class ScoreReward:
    """Reward = the per-step change in the map score, as the judge suite scores it, divided by 45 (the exit's
    points), so every trade-off (fight or go, health or kills, explore or exit, key order) is priced the way
    it is scored and the value function learns when to protect health. Per map: exit 45 plus efficiency
    5 x health/100 + 5 x time left; progression 20 x (keys + doors opened) / (keys + doors); exploration
    budget x 128-unit areas visited / reachable areas (budget 15, +20 with no keys or doors, +10 with no
    monsters); combat 10 x kills / monsters; damage -min(5, 0.02 x damage taken); early death -10 x time
    left. Generated maps only (the grid gives the totals). Optional stall penalty as in EpisodeReward."""

    MONSTERS, KEY_INDEX, BLOCK, SCALE = "zscid", {"b": 0, "y": 1, "r": 2}, 128, 45.0

    def __init__(self, state: dict, door_locks: dict[int, int], grid: str, *, stall_penalty: float = 0.0) -> None:
        rows = [r for r in grid.strip("\n").split("\n")]
        tiles = "".join(rows)
        self.keys = [self.KEY_INDEX[c] for c in self.KEY_INDEX if c in tiles]
        self.doors = list(door_locks)
        self.monsters = sum(tiles.count(c) for c in self.MONSTERS)
        self.areas = {(c * CELL // self.BLOCK, (len(rows) - 1 - r) * CELL // self.BLOCK)
                      for r, row in enumerate(rows) for c, ch in enumerate(row) if ch != "#"}
        items = len(self.keys) + len(self.doors)
        self.budget = 15 + 20 * (items == 0) + 10 * (self.monsters == 0)
        self.visited: set[tuple[int, int]] = set()
        self.opened: set[int] = set()
        self.stall_penalty = stall_penalty
        self.stale = 0
        self.cells = {_cell(state)}
        self.previous = state
        self.phi = self._potential(state)
        self.total = 0.0

    def _potential(self, state: dict) -> float:
        block = (math.floor(state["x"] / self.BLOCK), math.floor(state["y"] / self.BLOCK))
        if block in self.areas:
            self.visited.add(block)
        if state.get("sector_ceiling") is not None and state.get("sector_floor") is not None:
            for sector in self.doors:
                if state["sector_ceiling"][sector] > state["sector_floor"][sector] + 56:
                    self.opened.add(sector)
        items = len(self.keys) + len(self.doors)
        achieved = sum(state["keys"][i] > 0 for i in self.keys) + len(self.opened)
        points = 20 * achieved / items if items else 0.0
        points += self.budget * len(self.visited) / max(1, len(self.areas))
        points += 10 * state["kills"] / self.monsters if self.monsters else 0.0
        points -= min(5.0, 0.02 * float(state.get("damage_taken", 0.0)))
        return points / self.SCALE

    def step(self, state: dict) -> tuple[float, int, int]:
        previous = self.previous
        phi = self._potential(state)
        reward = phi - self.phi
        self.phi = phi
        ticks = state["episode_ticks"]
        time_left = max(0, ticks - state["tic"]) / ticks
        if state["exit_reached"] and not previous["exit_reached"]:
            reward += (45 + 5 * min(max(0.0, float(state["health"])), 100) / 100 + 5 * time_left) / self.SCALE
        if state["dead"] and not previous["dead"]:
            reward -= 10 * time_left / self.SCALE
        cell = _cell(state)
        novel = int(cell not in self.cells)
        self.cells.add(cell)
        self.stale = 0 if novel else self.stale + 1
        if self.stall_penalty and self.stale >= EpisodeReward.STALL_STEPS:
            reward -= self.stall_penalty
        kills = max(0, state["kills"] - previous["kills"])
        reward = float(np.clip(reward, -2.0, 2.0))
        self.total += reward
        self.previous = state
        return reward, kills, novel


class EpisodeReward:
    """Shaped PPO reward. `survival` adds the official score's survival terms (health bonus at the exit, a
    larger penalty for dying early, double damage cost, healing while hurt); `stall_penalty` is charged per
    step once STALL_STEPS steps pass without a new cell."""

    STALL_STEPS = 150

    def __init__(self, state: dict, door_locks: dict[int, int], *, stock: bool = False,
                 survival: bool = False, stall_penalty: float = 0.0) -> None:
        self.survival = survival
        self.stall_penalty = stall_penalty
        self.stale = 0
        self.previous = state
        self.door_locks = door_locks
        self.door_thresholds = {
            sector: state["sector_ceiling"][sector] + 32 if stock else 8
            for sector in door_locks
        }
        self.opened = {
            sector for sector, threshold in self.door_thresholds.items()
            if state["sector_ceiling"][sector] > threshold
        }
        self.cells = {_cell(state)}
        self.total = 0.0

    def step(self, state: dict) -> tuple[float, int, int]:
        previous = self.previous
        reward = -0.0005
        if state["exit_reached"] and not previous["exit_reached"]:
            ticks = state["episode_ticks"]
            reward += 1.0 + 0.5 * max(0, ticks - state["tic"]) / ticks
            if self.survival:
                reward += 0.5 * max(0.0, float(state["health"])) / 100.0
        if state["dead"] and not previous["dead"]:
            reward -= 1.0
            if self.survival:
                ticks = state["episode_ticks"]
                reward -= max(0, ticks - state["tic"]) / ticks
        reward += 0.5 * np.count_nonzero(np.logical_and(state["keys"], np.logical_not(previous["keys"])))
        for sector, lock in self.door_locks.items():
            threshold = self.door_thresholds[sector]
            if (sector not in self.opened
                    and previous["sector_ceiling"][sector] <= threshold < state["sector_ceiling"][sector]):
                self.opened.add(sector)
                reward += 0.5 if lock > 0 else 0.2
        cell = _cell(state)
        novel = int(cell not in self.cells)
        self.cells.add(cell)
        kills = max(0, state["kills"] - previous["kills"])
        lost = max(0, previous["health"] - state["health"])
        reward += 0.02 * novel + 0.1 * kills - (0.006 if self.survival else 0.003) * lost
        if self.survival and previous["health"] < 100:
            reward += 0.004 * min(max(0, state["health"] - previous["health"]), 100 - previous["health"])
        self.stale = 0 if novel else self.stale + 1
        if self.stall_penalty and self.stale >= self.STALL_STEPS:
            reward -= self.stall_penalty
        reward = float(np.clip(reward, -2.0, 2.0))
        self.total += reward
        self.previous = state
        return reward, kills, novel


class ProgressShaping:
    """Training-only reward for walking progress toward the next objective, added to another reward.

    The objective is the exit when it can be reached with the keys held, otherwise the nearest key not yet
    held; distance is the shortest 4-connected walk over the map grid from the true pose (locked doors pass
    only with their key, other doors always). Each cell closer earns `scale` and each cell farther costs the
    same, so any round trip nets exactly zero. Picking up a key re-targets without a jump: the locked door
    it opens joins the route, so the agent is paid to walk back to it."""

    LOCKS = {"B": 0, "Y": 1, "R": 2}
    KEYS = {"b": 0, "y": 1, "r": 2}
    MAX_STEP = 2  # cells; a bigger jump (re-entering the grid) resets the baseline instead of paying

    def __init__(self, base, state: dict, grid: str, scale: float) -> None:
        rows = grid.strip("\n").split("\n")
        self.free = {(c, len(rows) - 1 - r): ch for r, row in enumerate(rows) for c, ch in enumerate(row) if ch != "#"}
        self.base, self.scale, self.progress = base, scale, 0.0
        self.fields: dict[tuple[bool, ...], tuple[dict, dict]] = {}
        self.held = self._held(state)
        self.distance = self._distance(state)

    @staticmethod
    def _held(state: dict) -> tuple[bool, ...]:
        return tuple(bool(k > 0) for k in np.asarray(state["keys"])[:3])

    def _bfs(self, sources: list[tuple[int, int]], held: tuple[bool, ...]) -> dict[tuple[int, int], int]:
        def passable(cell: tuple[int, int]) -> bool:
            ch = self.free.get(cell)
            return ch is not None and (ch not in self.LOCKS or held[self.LOCKS[ch]])

        dist = {cell: 0 for cell in sources if passable(cell)}
        queue = deque(dist)
        while queue:
            x, y = queue.popleft()
            for cell in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if cell not in dist and passable(cell):
                    dist[cell] = dist[(x, y)] + 1
                    queue.append(cell)
        return dist

    def _distance(self, state: dict) -> int | None:
        if self.held not in self.fields:
            exits = [cell for cell, ch in self.free.items() if ch == "E"]
            keys = [cell for cell, ch in self.free.items() if ch in self.KEYS and not self.held[self.KEYS[ch]]]
            self.fields[self.held] = (self._bfs(exits, self.held), self._bfs(keys, self.held))
        exit_field, key_field = self.fields[self.held]
        cell = _cell(state)
        return exit_field[cell] if cell in exit_field else key_field.get(cell)

    def step(self, state: dict) -> tuple[float, int, int]:
        reward, kills, novel = self.base.step(state)
        held = self._held(state)
        if held != self.held:
            self.held = held
            self.distance = self._distance(state)
            return reward, kills, novel
        distance = self._distance(state)
        if distance is not None and self.distance is not None and abs(self.distance - distance) <= self.MAX_STEP:
            delta = self.scale * (self.distance - distance)
            reward += delta
            self.progress += delta
        if distance is not None:
            self.distance = distance
        return reward, kills, novel

    @property
    def total(self) -> float:
        return self.base.total + self.progress


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
    episodes: list[tuple[float, bool, bool]]
    kills: int
    novelty_cells: int


def episode_metrics(rollouts: dict[int, Rollout]) -> dict[str, float | None]:
    episodes = [episode for rollout in rollouts.values() for episode in rollout.episodes]
    stock_exits = [exited for _, exited, stock in episodes if stock]
    generated_exits = [exited for _, exited, stock in episodes if not stock]
    return {
        "return_mean": sum(reward for reward, _, _ in episodes) / len(episodes) if episodes else None,
        "exit_rate": sum(stock_exits + generated_exits) / len(episodes) if episodes else None,
        "stock_exit_rate": sum(stock_exits) / len(stock_exits) if stock_exits else None,
        "generated_exit_rate": sum(generated_exits) / len(generated_exits) if generated_exits else None,
    }


class Collector:
    def __init__(self, args: argparse.Namespace, worker: int, episode_counter) -> None:
        self.args = args
        self.worker = worker
        self.episode_counter = episode_counter
        self.model = NakshaNet().eval()
        self.env = None
        self.stock_specs = build_specs(getattr(args, "stock_maps", ()))
        extra = getattr(args, "extra_suite", None)
        self.extra_specs = json.loads(Path(extra).read_text()) if extra else []

    def _start_episode(self) -> None:
        k = self.episode_counter.value
        self.episode_counter.value += 1
        args = self.args
        seed = args.seed_offset + self.worker * 100_000 + k
        rng = random.Random(seed) if self.stock_specs else None
        self.stock = rng is not None and rng.random() < args.stock_frac
        if self.stock:
            spec = rng.choice(self.stock_specs)
            wad = spec["wad"]
        else:
            family = args.families[(self.worker + k) % len(args.families)]
            size = args.sizes[(self.worker + k // len(args.families)) % len(args.sizes)]
            sides = getattr(args, "maze_sides", None)
            if sides:
                families.MAZE_SIDES = dict(zip(SIZES, sides))
            spec = generate(family, seed, size)
            pick = random.Random(seed + 104_729)
            if self.extra_specs and pick.random() < getattr(args, "extra_frac", 0.0):
                base = pick.choice(self.extra_specs)
                spec = {"grid": transform_grid(base["grid"], pick.randrange(4), pick.random() < 0.5),
                        "player_angle": pick.choice((0, 90, 180, 270)), "episode_ticks": base["episode_ticks"]}
            restyle = random.Random(seed + 7_919).random() < getattr(args, "look_frac", 0.0)
            wad = write_map(spec["grid"], "map.wad", player_angle=spec["player_angle"],
                            look_seed=seed if restyle else None)
        self.env = make_env(wad, spec.get("map", "MAP01"), episode_ticks=spec["episode_ticks"], render_buffers=True)
        self.obs, _ = self.env.reset(seed=seed)
        if getattr(args, "score_reward", False) and not self.stock:
            self.reward = ScoreReward(self.env.privileged_state(), self.env.mapinfo.door_sectors, spec["grid"],
                                      stall_penalty=getattr(args, "stall_penalty", 0.0))
        else:
            self.reward = EpisodeReward(self.env.privileged_state(), self.env.mapinfo.door_sectors, stock=self.stock,
                                        survival=getattr(args, "survival", False),
                                        stall_penalty=getattr(args, "stall_penalty", 0.0))
        progress = getattr(args, "progress_reward", 0.0)
        if progress and not self.stock:
            self.reward = ProgressShaping(self.reward, self.env.privileged_state(), spec["grid"], progress)
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
                rollout.episodes.append((self.reward.total, state["exit_reached"], self.stock))
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
    survival, stall = getattr(args, "survival", False), getattr(args, "stall_penalty", 0.0)
    score = getattr(args, "score_reward", False)
    progress = getattr(args, "progress_reward", 0.0)
    if survival or stall or score or progress:
        meta["reward"] = {"survival": survival, "stall_penalty": stall, "score_reward": score,
                          "progress_reward": progress}
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
                    steps = sum(len(rollout.actions) for rollout in rollouts.values())
                    meta["steps"] += steps
                    meta["iterations"] += 1
                    writer.writerow({
                        "iteration": meta["iterations"],
                        "steps": meta["steps"],
                        **episode_metrics(rollouts),
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
    parser.add_argument("--families", nargs="+", choices=(*FAMILIES, "maze", "irregular", "comb", "locksfirst", "ambush"), default=FAMILIES)
    parser.add_argument("--stock-maps", nargs="+", type=str.upper, choices=STOCK_MAPS, default=())
    parser.add_argument("--stock-frac", type=float, default=0.0,
                        help="probability of sampling a stock map at each episode reset")
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=SIZES)
    parser.add_argument("--maze-sides", nargs=3, type=int, metavar=("S", "M", "L"),
                        help="maze lattice side per size (default 15 25 35)")
    parser.add_argument("--look-frac", type=float, default=0.0,
                        help="probability that a generated map gets a random appearance (textures, light, ceiling)")
    parser.add_argument("--survival", action="store_true",
                        help="survival reward terms: health bonus at the exit, early-death penalty, 2x damage cost, healing")
    parser.add_argument("--extra-suite", type=Path,
                        help="JSON list of map specs (grid, episode_ticks) to train on, randomly rotated and mirrored")
    parser.add_argument("--extra-frac", type=float, default=0.0,
                        help="probability that a generated-map episode uses a map from --extra-suite instead")
    parser.add_argument("--score-reward", action="store_true",
                        help="on generated maps, reward = per-step change in the judge-style map score / 45")
    parser.add_argument("--stall-penalty", type=float, default=0.0,
                        help="per-step penalty once 150 steps pass without a new cell")
    parser.add_argument("--progress-reward", type=float, default=0.0,
                        help="on generated maps, reward per grid cell of walking progress toward the next objective")
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
    if not 0 <= args.stock_frac <= 1:
        parser.error("stock-frac must be in [0, 1]")
    if not 0 <= args.look_frac <= 1:
        parser.error("look-frac must be in [0, 1]")
    if not 0 <= args.extra_frac <= 1 or (args.extra_frac and not args.extra_suite):
        parser.error("extra-frac must be in [0, 1] and needs --extra-suite")
    if args.extra_suite:
        args.extra_suite = args.extra_suite.resolve()  # workers run in a scratch directory
    if args.score_reward and args.survival:
        parser.error("--score-reward and --survival are alternatives")
    if not (math.isfinite(args.progress_reward) and args.progress_reward >= 0):
        parser.error("progress-reward must be finite and nonnegative")
    if not (math.isfinite(args.stall_penalty) and args.stall_penalty >= 0):
        parser.error("stall-penalty must be finite and nonnegative")
    if args.stock_frac and not args.stock_maps:
        parser.error("stock-maps is required when stock-frac is positive")
    if args.init.resolve() == args.out.resolve():
        parser.error("init and out must be different paths")
    if torch.device(args.device).type == "cpu":
        torch.set_num_threads(1)
    print(json.dumps(train(args)))


if __name__ == "__main__":
    main()
