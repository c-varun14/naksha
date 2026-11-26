"""Collect, train, and select DAgger checkpoints on the development suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


SCORE_WEIGHTS = {
    "exit": 3.0,
    "keys": 1.0,
    "locked_opened": 1.0,
    "doors_opened": 0.3,
    "cells_visited": 0.02,
    "kills": 0.2,
    "dead": -1.0,
}


def score_rows(rows: Sequence[Mapping[str, float]]) -> float:
    if not rows:
        raise ValueError("cannot score an empty evaluation")
    return math.fsum(
        math.fsum(weight * row.get(metric, 0) for metric, weight in SCORE_WEIGHTS.items())
        for row in rows
    ) / len(rows)


def _run_module(module: str, args: list[str], log_path: Path) -> None:
    command = [sys.executable, "-m", module, *args]
    print(f"Running {module} (log: {log_path})", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def _collected_episodes(log_path: Path) -> int:
    last_line = ""
    with log_path.open(encoding="utf-8") as log:
        for line in log:
            if line.strip():
                last_line = line
    summary = json.loads(last_line)
    return sum(family["n"] for family in summary["by_family"].values())


def _save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _select_data_dirs(
    data_dirs: Sequence[str], max_steps: int, step_counts: dict[Path, int],
) -> list[str]:
    selected = []
    remaining = max_steps
    for directory in data_dirs:
        path = Path(directory)
        if path not in step_counts:
            episodes = list(path.glob("*.npz"))
            if not episodes:
                raise ValueError(f"no episodes in {directory}")
            step_counts[path] = sum(
                json.loads(episode.with_suffix(".json").read_text(encoding="utf-8"))["steps"]
                for episode in episodes
            )
        count = step_counts[path]
        if count > remaining:
            break
        selected.append(directory)
        remaining -= count
    return selected


def _evaluate(checkpoint: str, name: str, args: argparse.Namespace, log_path: Path) -> float:
    _run_module("training.eval.run", [
        "--suite", "data/suites/dev.json", "--policy", "agent", "--weights", checkpoint,
        "--seeds", str(args.dev_seeds), "--procs", str(args.procs), "--name", name,
    ], log_path)
    result = Path("runs/eval") / f"{name}.json"
    return score_rows(json.loads(result.read_text(encoding="utf-8")))


def _loop(args: argparse.Namespace) -> dict:
    args.work.mkdir(parents=True, exist_ok=True)
    state_path = args.work / "state.json"
    resuming = state_path.exists()
    if resuming:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        saved = state["config"]
        if args.dev_seeds is not None and args.dev_seeds != saved["dev_seeds"]:
            raise ValueError(f"dev-seeds must match the evaluations saved in {state_path}")
        args.dev_seeds = saved["dev_seeds"]
        args.init = saved["init"]
        if args.data_dirs is None:
            args.data_dirs = saved["data_dirs"]
        if args.seed_offset is None:
            args.seed_offset = saved["seed_offset"]
    else:
        args.dev_seeds = 2 if args.dev_seeds is None else args.dev_seeds
        args.data_dirs = args.data_dirs or []
        args.seed_offset = 200000 if args.seed_offset is None else args.seed_offset
        state = {"round": 0, "best_checkpoint": args.init, "scores": {}, "history": []}

    config = {
        "init": args.init,
        "episodes": args.episodes,
        "betas": args.betas,
        "train_minutes": args.train_minutes,
        "data_dirs": args.data_dirs,
        "seed_offset": args.seed_offset,
        "dev_seeds": args.dev_seeds,
        "teacher": args.teacher,
        "mirror": args.mirror,
        "max_steps": args.max_steps,
    }
    state["config"] = config
    if not resuming:
        _save_state(state_path, state)

    history = state["history"]
    scores = state["scores"]
    step_counts: dict[Path, int] = {}
    work_id = hashlib.sha256(str(args.work.resolve()).encode()).hexdigest()[:16]
    for r in range(state["round"], args.rounds):
        round_dir = args.work / f"round{r}"
        round_dir.mkdir(exist_ok=True)
        if len(history) == r:
            previous_stop = max((item["seed_stop"] for item in history), default=args.seed_offset)
            start = max(args.seed_offset, previous_stop)
            history.append({
                "round": r, "beta": args.betas[min(r, len(args.betas) - 1)],
                "seed_start": start, "seed_stop": start + args.episodes,
                "data_dir": f"data/dagger_r{r}", "checkpoint": str(round_dir / "ckpt.pt"),
                "incumbent": state["best_checkpoint"], "teacher": args.teacher, "trained": False,
            })
            _save_state(state_path, state)

        entry = history[r]
        checkpoint = entry["checkpoint"]
        incumbent = entry["incumbent"]
        if "episodes" not in entry:
            start, stop = entry["seed_start"], entry["seed_stop"]
            collect_log = round_dir / "collect.log"
            _run_module("training.data.collect", [
                "--out", entry["data_dir"], "--seeds", str(start), str(stop), "--procs", str(args.procs),
                "--mode", "dagger", "--beta", str(entry["beta"]), "--weights", incumbent, "--round", str(r),
                "--teacher", entry["teacher"],
            ], collect_log)
            collected = _collected_episodes(collect_log)
            if collected <= 0:
                raise ValueError(f"round {r} collected no episodes; see {collect_log}")
            if collected != stop - start:
                print(f"Round {r} collected {collected} of {stop - start} requested episodes", flush=True)
            entry["episodes"] = collected
            _save_state(state_path, state)

        if not entry["trained"]:
            data_dirs = _select_data_dirs(
                [*(item["data_dir"] for item in reversed(history)), *args.data_dirs],
                args.max_steps, step_counts,
            )
            if not data_dirs:
                raise ValueError(f"{entry['data_dir']} exceeds the training budget of {args.max_steps} steps")
            _run_module("training.bc.train", [
                "--init", incumbent, "--data", *data_dirs, "--out", checkpoint,
                "--minutes", str(args.train_minutes), "--mirror", str(args.mirror),
            ], round_dir / "train.log")
            entry["trained"] = True
            entry["mirror"] = args.mirror
            _save_state(state_path, state)

        for role, weights in (("candidate", checkpoint), ("incumbent", incumbent)):
            if weights not in scores:
                name = f"dagger-{work_id}-r{r}-{role}"
                scores[weights] = _evaluate(weights, name, args, round_dir / f"eval_{role}.log")
                _save_state(state_path, state)

        candidate_score, incumbent_score = scores[checkpoint], scores[incumbent]
        if candidate_score > incumbent_score:
            state["best_checkpoint"] = checkpoint
        best = state["best_checkpoint"]
        entry.update(candidate_score=candidate_score, incumbent_score=incumbent_score, best_checkpoint=best)
        state["round"] = r + 1
        _save_state(state_path, state)
        print(f"Round {r} | episodes {entry['episodes']} | beta {entry['beta']:g} | candidate {candidate_score:.4f}"
              f" | incumbent {incumbent_score:.4f} | best {best}", flush=True)

    best = state["best_checkpoint"]
    return {"best_checkpoint": best, "score": scores[best]}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", help="initial checkpoint for a new run")
    parser.add_argument("--rounds", type=int, default=3, help="total number of rounds")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.5, 0.25, 0.0])
    parser.add_argument("--train-minutes", type=float, default=15.0)
    parser.add_argument("--data-dirs", nargs="+")
    parser.add_argument("--work", type=Path, default=Path("runs/dagger"))
    parser.add_argument("--procs", type=int, default=10)
    parser.add_argument("--seed-offset", type=int)
    parser.add_argument("--dev-seeds", type=int)
    parser.add_argument("--teacher", choices=["v1", "v2"], default="v1")
    parser.add_argument("--mirror", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1_100_000, help="training data step budget")
    args = parser.parse_args(argv)
    if args.init is None and not (args.work / "state.json").is_file():
        parser.error("--init is required for a new run")
    if min(args.rounds, args.episodes, args.procs, args.max_steps) < 1 or (
        args.dev_seeds is not None and args.dev_seeds < 1
    ):
        parser.error("rounds, episodes, procs, dev-seeds, and max-steps must be positive")
    if any(not 0 <= beta <= 1 for beta in args.betas):
        parser.error("betas must be between 0 and 1")
    if not 0 <= args.mirror <= 1:
        parser.error("mirror must be between 0 and 1")
    if not math.isfinite(args.train_minutes) or args.train_minutes < 0:
        parser.error("train-minutes must be finite and nonnegative")
    if args.seed_offset is not None and args.seed_offset < 0:
        parser.error("seed-offset must be nonnegative")
    print(json.dumps(_loop(args)))


if __name__ == "__main__":
    main()
