import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.dagger import loop  # noqa: E402


def _write_steps(directory, *counts):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i, count in enumerate(counts):
        (directory / f"{i}.npz").touch()
        (directory / f"{i}.json").write_text(json.dumps({"steps": count}))


def test_score_rows():
    rows = [
        {"exit": True, "keys": 2, "locked_opened": 1, "doors_opened": 3,
         "cells_visited": 20, "kills": 4, "dead": False},
        {"exit": False, "keys": 0, "locked_opened": 0, "doors_opened": 1,
         "cells_visited": 5, "kills": 0, "dead": True},
    ]
    assert loop.score_rows(rows) == pytest.approx((8.1 - 0.6) / 2)


def test_step_budget_selects_prefix_and_caches_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    newest, older, oldest, base = "newest", "older", "oldest", "base"
    _write_steps(newest, 30, 50)
    _write_steps(older, 20)
    _write_steps(oldest, 40)
    _write_steps(base, 5)
    dirs = [newest, older, oldest, base]
    cache = {}
    assert loop._select_data_dirs(dirs, 105, cache) == [newest, older]
    assert cache == {Path(newest): 80, Path(older): 20, Path(oldest): 40}

    read_text = Path.read_text

    def read_base(path, *args, **kwargs):
        assert path.parent == Path(base)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_base)
    assert loop._select_data_dirs(dirs, 79, cache) == []
    assert loop._select_data_dirs(dirs, 100, cache) == [newest, older]
    assert loop._select_data_dirs(dirs, 145, cache) == dirs


def test_resume_rounds(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    state_path = Path("runs/dagger/state.json")
    initial = "runs/ckpt/bc0.pt"
    checkpoints = [f"runs/dagger/round{r}/ckpt.pt" for r in range(3)]
    evaluations = {initial: False, checkpoints[0]: True, checkpoints[1]: False, checkpoints[2]: True}
    options = ["--init", initial, "--data-dirs", "data/bc0", "--episodes", "4", "--betas", "0.5", "0.25"]
    _write_steps("data/bc0", 100)
    calls = []
    fail_next = {("training.bc.train", initial), ("training.eval.run", initial)}

    def run(command, *, stdout, stderr, check):
        assert command[:2] == [sys.executable, "-m"]
        assert stderr == subprocess.STDOUT
        assert check
        module, *args = command[2:]
        policy_flag = "--init" if module == "training.bc.train" else "--weights"
        weights = args[args.index(policy_flag) + 1]
        calls.append((module, weights, args))
        if (module, weights) in fail_next:
            fail_next.remove((module, weights))
            raise subprocess.CalledProcessError(1, command)
        if module == "training.data.collect":
            r = int(args[args.index("--round") + 1])
            assert Path(stdout.name) == state_path.parent / f"round{r}/collect.log"
            assert args[args.index("--mode") + 1] == "dagger"
            _write_steps(args[args.index("--out") + 1], 60, 70)
            summary = {"by_family": {"arena": {"n": 2}, "locks": {"n": 1 if r == 0 else 2}}}
            stdout.write("progress\n" + json.dumps(summary) + "\n")
        elif module == "training.bc.train":
            out = Path(args[args.index("--out") + 1])
            assert Path(stdout.name) == out.with_name("train.log")
            out.touch()
        elif module == "training.eval.run":
            assert args[:4] == ["--suite", "data/suites/dev.json", "--policy", "agent"]
            assert args[args.index("--seeds") + 1] == "2"
            name = args[args.index("--name") + 1]
            result = Path("runs/eval") / f"{name}.json"
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(json.dumps([{"exit": evaluations[weights]}]), encoding="utf-8")
        else:
            pytest.fail(f"unexpected module {module}")

    monkeypatch.setattr(loop.subprocess, "run", run)

    with pytest.raises(subprocess.CalledProcessError):
        loop.main(options)
    state = json.loads(state_path.read_text())
    assert state["round"] == 0
    assert state["history"][0]["episodes"] == 3
    assert not state["history"][0]["trained"]

    with pytest.raises(subprocess.CalledProcessError):
        loop.main(options)
    state = json.loads(state_path.read_text())
    assert state["round"] == 0
    assert state["history"][0]["trained"]
    assert state["scores"] == {checkpoints[0]: 3.0}

    loop.main(options)
    output = capsys.readouterr().out.splitlines()
    assert "Round 0 collected 3 of 4 requested episodes" in output
    assert len([line for line in output if line.startswith(tuple(f"Round {r} |" for r in range(3)))]) == 3
    assert json.loads(output[-1]) == {"best_checkpoint": checkpoints[0], "score": 3.0}

    collects = [(weights, args) for module, weights, args in calls if module == "training.data.collect"]
    assert [weights for weights, _ in collects] == [initial, checkpoints[0], checkpoints[0]]
    for r, (_, args) in enumerate(collects):
        seed_index = args.index("--seeds")
        assert args[seed_index + 1:seed_index + 3] == [str(200000 + 4 * r), str(200004 + 4 * r)]
        assert args[args.index("--out") + 1] == f"data/dagger_r{r}"
        assert args[args.index("--beta") + 1] == ["0.5", "0.25", "0.25"][r]

    trains = [(weights, args) for module, weights, args in calls if module == "training.bc.train"]
    assert [weights for weights, _ in trains] == [initial, initial, checkpoints[0], checkpoints[0]]
    for r, (_, args) in enumerate(trains[1:]):
        assert args[args.index("--data") + 1:args.index("--out")] == [
            *(f"data/dagger_r{i}" for i in range(r, -1, -1)), "data/bc0",
        ]

    evaluated = [weights for module, weights, _ in calls if module == "training.eval.run"]
    assert evaluated == [checkpoints[0], initial, initial, *checkpoints[1:]]
    state = json.loads(state_path.read_text())
    assert state["round"] == 3
    assert [entry["best_checkpoint"] for entry in state["history"]] == [checkpoints[0]] * 3

    call_count = len(calls)
    loop.main(options)
    assert len(calls) == call_count
    assert json.loads(capsys.readouterr().out) == {"best_checkpoint": checkpoints[0], "score": 3.0}


def test_continues_legacy_run_with_new_round_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    work = Path("existing")
    work.mkdir()
    best = "existing/round2/ckpt.pt"
    state_path = work / "state.json"
    history = []
    for r in range(3):
        data_dir = f"data/dagger_r{r}"
        _write_steps(data_dir, 100)
        history.append({
            "round": r, "seed_start": 500 + 600 * r, "seed_stop": 1100 + 600 * r,
            "data_dir": data_dir, "trained": True,
        })
    _write_steps("data/bc0", 100)
    state_path.write_text(json.dumps({
        "round": 3, "best_checkpoint": best, "scores": {best: 2.0}, "history": history,
        "config": {"init": "initial.pt", "data_dirs": ["data/bc0"], "dev_seeds": 2,
                   "seed_offset": 500, "episodes": 600, "betas": [0.5, 0.25, 0.0],
                   "train_minutes": 15.0},
    }))
    collects, trains = [], []

    def run_module(module, args, log_path):
        if module == "training.data.collect":
            collects.append(args)
            _write_steps(args[args.index("--out") + 1], 100)
            log_path.write_text(json.dumps({"by_family": {"arena": {"n": 700}}}))
        else:
            assert module == "training.bc.train"
            trains.append(args)

    monkeypatch.setattr(loop, "_run_module", run_module)
    monkeypatch.setattr(loop, "_evaluate", lambda *args: 1.0)
    loop.main(["--work", str(work), "--rounds", "6", "--episodes", "700",
               "--teacher", "v2", "--betas", "0.0", "--mirror", "0.5", "--max-steps", "250"])

    assert len(collects) == len(trains) == 3
    for r, (collect, train) in enumerate(zip(collects, trains), start=3):
        start = 2300 + (r - 3) * 700
        seeds = collect.index("--seeds")
        assert collect[seeds + 1:seeds + 3] == [str(start), str(start + 700)]
        assert collect[collect.index("--round") + 1] == str(r)
        assert collect[collect.index("--teacher") + 1] == "v2"
        assert collect[collect.index("--beta") + 1] == "0.0"
        assert collect[collect.index("--weights") + 1] == best
        assert train[train.index("--init") + 1] == best
        assert train[train.index("--mirror") + 1] == "0.5"
        assert train[train.index("--data") + 1:train.index("--out")] == [
            f"data/dagger_r{r}", f"data/dagger_r{r - 1}",
        ]
    state = json.loads(state_path.read_text())
    assert state["round"] == 6
    assert state["history"][:3] == history
