"""Run seeded evaluation episodes and summarize results by map family."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

import training.paths  # noqa: F401
from doom_env import make_env
from naksha import actions
from training.gen.compile import CELL, write_map

Policy = Callable[[dict, dict], np.ndarray]
Episode = tuple[dict, list[float]]


def make_policy(kind: str, **kw) -> Policy:
    if kind == "random":
        rng = np.random.default_rng(kw.get("seed"))
        return lambda obs, priv: actions.to_buttons(rng.integers(actions.NUM_ACTIONS))
    if kind == "agent":
        from submission.agent import DoomAgent

        env = kw["env"]
        agent = DoomAgent(env.observation_space, env.action_space, "cpu", weights=kw.get("weights"))
        agent.reset()
        return lambda obs, priv: agent.act(obs)
    if kind == "teacher":
        from training.teacher.teacher import Teacher

        teacher = Teacher(kw["spec"])
        teacher.reset(kw["env"])
        return lambda obs, priv: actions.to_buttons(teacher.act(obs, priv)[0])
    raise ValueError(f"unknown policy: {kind}")


def run_episode(
    spec: dict, wad_path: str | Path, policy_kind: str, seed: int, weights: str | Path | None = None,
) -> dict:
    return _run_episode(spec, wad_path, policy_kind, seed, weights)[0]


def _run_episode(
    spec: dict, wad_path: str | Path, policy_kind: str, seed: int, weights: str | Path | None,
) -> Episode:
    cells: set[tuple[int, int]] = set()
    act_times: list[float] = []
    env = make_env(wad_path, spec.get("map", "MAP01"), episode_ticks=spec["episode_ticks"], render_buffers=True)
    # Door sectors come from the map's own linedef specials, so this also works for maps without a grid.
    door_lock = dict(env.mapinfo.door_sectors)
    door_ids = sorted(door_lock)
    locked = np.array([door_lock[i] > 0 for i in door_ids], dtype=bool)
    opened = np.zeros(len(door_ids), dtype=bool)
    try:
        obs, _ = env.reset(seed=seed)
        policy = make_policy(policy_kind, env=env, spec=spec, seed=seed, weights=weights)
        terminated = truncated = False
        while True:
            priv = env.privileged_state()
            cells.add((int(priv["x"] // CELL), int(priv["y"] // CELL)))
            opened |= priv["sector_ceiling"][door_ids] > 8
            if terminated or truncated:
                break
            start = perf_counter()
            buttons = policy(obs, priv)
            act_times.append((perf_counter() - start) * 1_000)
            obs, _, terminated, truncated, _ = env.step(buttons)
        p50, p99 = np.percentile(act_times, [50, 99])
        return {
            "id": spec.get("id"), "family": spec["family"], "seed": seed,
            "exit": priv["exit_reached"], "dead": priv["dead"], "tics": len(act_times) * 4,
            "keys": int(np.count_nonzero(obs["vitals"][8:14])),
            "doors_opened": int(opened.sum()), "locked_opened": int((opened & locked).sum()),
            "cells_visited": len(cells), "kills": priv["kills"], "damage_taken": priv["damage_taken"],
            "act_ms_p50": float(p50), "act_ms_p99": float(p99),
        }, act_times
    finally:
        env.close()


def evaluate(
    specs: list[dict], policy_kind: str, seeds: Iterable[int], procs: int,
    weights: str | Path | None = None, *, suite: str = "default",
) -> list[dict]:
    return [row for row, _ in _evaluate(specs, policy_kind, seeds, procs, weights, suite)]


def _evaluate(
    specs: list[dict], policy_kind: str, seeds: Iterable[int], procs: int,
    weights: str | Path | None, suite: str,
) -> list[Episode]:
    if procs < 1:
        raise ValueError("procs must be positive")
    if policy_kind == "agent" and weights is None:
        weights = os.environ.get("NAKSHA_WEIGHTS")
    weights = Path(weights).resolve() if weights else None
    seeds = tuple(seeds)
    tasks = []
    for spec in specs:
        if policy_kind == "teacher" and not spec.get("grid"):
            continue  # the reference policy needs the map grid
        wad = spec.get("wad")
        if not wad:
            path = Path("data/maps") / suite / f"{spec['id']}.wad"
            wad = write_map(spec["grid"], path, player_angle=spec.get("player_angle"))
        wad = Path(wad).resolve()
        tasks.extend((spec, wad, policy_kind, seed, weights) for seed in seeds)
    if not tasks:
        return []
    with TemporaryDirectory(prefix="naksha-eval-") as scratch:
        with mp.get_context("spawn").Pool(min(procs, len(tasks)), _prepare_worker, (scratch,)) as pool:
            return pool.starmap(_run_episode, tasks, chunksize=1)


def _prepare_worker(scratch: str) -> None:
    # ViZDoom creates runtime files relative to the process's working directory.
    cwd = Path(scratch) / str(os.getpid())
    cwd.mkdir()
    os.chdir(cwd)


def _print_table(episodes: list[Episode]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    timings: dict[str, list[float]] = defaultdict(list)
    for row, act_times in episodes:
        family = row["family"]
        groups[family].append(row)
        timings[family].extend(act_times)
    print("| Family | n | Exit % | Mean keys | Mean locked opened | Mean doors opened | "
          "Mean cells visited | Mean kills | Death % | Act ms p99 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for family, group in sorted(groups.items()):
        fields = ("exit", "keys", "locked_opened", "doors_opened", "cells_visited", "kills", "dead")
        means = [np.mean([row[field] for row in group]) for field in fields]
        means[0] *= 100
        means[-1] *= 100
        p99 = np.percentile(timings[family], 99)
        metrics = " | ".join(f"{value:.2f}" for value in (*means, p99))
        print(f"| {family} | {len(group)} | {metrics} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--policy", choices=("random", "agent", "teacher"), required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--procs", type=int, default=11)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if min(args.seeds, args.procs) < 1:
        parser.error("seeds and procs must be positive")
    specs = json.loads(args.suite.read_text(encoding="utf-8"))
    episodes = _evaluate(specs, args.policy, range(args.seeds), args.procs, args.weights, args.suite.stem)
    rows = [row for row, _ in episodes]
    path = Path("runs/eval") / f"{args.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    _print_table(episodes)


if __name__ == "__main__":
    main()
