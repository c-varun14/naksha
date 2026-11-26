import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.dagger import loop  # noqa: E402


def test_score_rows():
    rows = [
        {"exit": True, "keys": 2, "locked_opened": 1, "doors_opened": 3,
         "cells_visited": 20, "kills": 4, "dead": False},
        {"exit": False, "keys": 0, "locked_opened": 0, "doors_opened": 1,
         "cells_visited": 5, "kills": 0, "dead": True},
    ]
    assert loop.score_rows(rows) == pytest.approx((8.1 - 0.6) / 2)


def test_resume_rounds(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    state_path = Path("runs/dagger/state.json")
    initial = "runs/ckpt/bc0.pt"
    checkpoints = [f"runs/dagger/round{r}/ckpt.pt" for r in range(3)]
    evaluations = {initial: False, checkpoints[0]: True, checkpoints[1]: False, checkpoints[2]: True}
    options = ["--init", initial, "--data-dirs", "data/bc0", "--episodes", "4", "--betas", "0.5", "0.25"]
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
            "data/bc0", *(f"data/dagger_r{i}" for i in range(r + 1)),
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
