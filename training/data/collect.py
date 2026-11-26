"""Collect labelled episodes on generated maps.

mode teacher: the reference policy drives and labels every step (behaviour cloning data).
mode dagger:  the student drives (with probability 1-beta per step), the reference policy labels
              every step, so the student learns to recover from its own mistakes.

uv run --no-sync python -m training.data.collect --out data/bc0 --seeds 0 400 --procs 11
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import numpy as np

from training import paths  # noqa: F401

ROOT = paths.ROOT
_student = {}


def _student_model(weights: str):
    if weights not in _student:
        import torch
        from naksha.model import load_checkpoint
        torch.set_num_threads(1)
        model, _ = load_checkpoint(weights, "cpu")
        model.eval()
        _student[weights] = model
    return _student[weights]


def collect_one(task: dict) -> dict:
    from doom_env import make_env
    from naksha import features as F
    from naksha.actions import to_buttons
    from training.data.shards import write_episode
    from training.gen.compile import write_map
    from training.gen.families import generate
    from training.teacher.teacher import Teacher
    from training.teacher.teacher2 import Teacher2

    fam, size, seed = task["family"], task["size"], task["seed"]
    teacher_version = task.get("teacher", "v1")
    spec = generate(fam, seed, size)
    wad = write_map(spec["grid"], ROOT / f"data/maps/train/{fam}-{size}-{seed}.wad", player_angle=spec["player_angle"])
    env = make_env(str(wad), "MAP01", episode_ticks=spec["episode_ticks"], render_buffers=True)
    rng = np.random.default_rng(seed + 7919 * task.get("round", 0))
    obs, _ = env.reset(seed=int(seed))
    teacher = (Teacher2 if teacher_version == "v2" else Teacher)(spec, seed)
    teacher.reset(env)

    student, state = None, None
    if task["mode"] == "dagger":
        import torch
        student = _student_model(task["weights"])
        state = student.initial_state(1, "cpu")

    rec = {k: [] for k in (*F.KEYS, "action", "label", "intention", "exit_vis", "exit_dir", "pose", "reward", "done")}
    prev = 0
    total_r = 0.0
    while True:
        priv = env.privileged_state()
        last_use = teacher.last_use
        label, intent, info = teacher.act(obs, priv)
        feats = F.encode(obs, prev)
        executed = label
        if student is not None:
            with torch.inference_mode():
                inp = {k: torch.from_numpy(np.asarray(v)[None, None]) for k, v in feats.items()}
                out, state = student(inp, state)
                s_act = int(out["logits"][0, 0].argmax())
            if rng.random() >= task["beta"]:
                executed = s_act
            teacher.executed(executed, label, last_use)
        obs, r, term, trunc, info_env = env.step(to_buttons(executed))
        total_r += r
        for k in F.KEYS:
            rec[k].append(feats[k])
        rec["action"].append(executed)
        rec["label"].append(label)
        rec["intention"].append(intent)
        rec["exit_vis"].append(info["exit_vis"])
        rec["exit_dir"].append(info["exit_dir"])
        rec["pose"].append((priv["x"], priv["y"], priv["angle"]))
        rec["reward"].append(r)
        rec["done"].append(term or trunc)
        prev = executed
        if term or trunc:
            break
    end = env.privileged_state()
    env.close()
    steps = {k: np.stack(rec[k]) for k in F.KEYS}
    steps["action"] = np.asarray(rec["action"], np.int16)
    steps["label"] = np.asarray(rec["label"], np.int16)
    steps["intention"] = np.asarray(rec["intention"], np.int8)
    steps["exit_vis"] = np.asarray(rec["exit_vis"], np.int8)
    steps["exit_dir"] = np.asarray(rec["exit_dir"], np.float16)
    steps["pose"] = np.asarray(rec["pose"], np.float32)
    steps["reward"] = np.asarray(rec["reward"], np.float32)
    steps["done"] = np.asarray(rec["done"], bool)
    meta = {"map": f"{fam}-{size}-{seed}", "family": fam, "size": size, "seed": seed, "policy": task["mode"],
            "beta": task.get("beta", 1.0), "exit": bool(info_env["exit_reached"]), "dead": end["dead"],
            "keys": int(sum(end["keys"])), "locks": len(spec["locks"]), "kills": end["kills"], "steps": len(steps["label"]),
            "teacher": teacher_version, "FEATURES_VERSION": F.FEATURES_VERSION}
    name = f"{fam}-{size}-{seed}-{task['mode']}{task.get('round', 0)}"
    write_episode(task["out"], name, steps, meta)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", nargs="+", default=["arena", "rooms", "locks", "gauntlet"])
    ap.add_argument("--weights_family", nargs="+", type=float, default=None, help="relative share per family")
    ap.add_argument("--sizes", nargs="+", default=["s", "m", "l"])
    ap.add_argument("--seeds", nargs=2, type=int, default=[0, 100])
    ap.add_argument("--procs", type=int, default=11)
    ap.add_argument("--mode", choices=["teacher", "dagger"], default="teacher")
    ap.add_argument("--teacher", choices=["v1", "v2"], default="v1")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--round", type=int, default=0)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tasks = []
    for seed in range(*a.seeds):
        fam = a.families[seed % len(a.families)]
        size = a.sizes[(seed // len(a.families)) % len(a.sizes)]
        tasks.append({"family": fam, "size": size, "seed": seed, "mode": a.mode, "beta": a.beta,
                      "weights": a.weights, "out": a.out, "round": a.round, "teacher": a.teacher})
    t0 = time.time()
    stats, steps = Counter(), 0
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=50) as pool:
        for i, meta in enumerate(pool.imap_unordered(collect_one, tasks), 1):
            steps += meta["steps"]
            stats[(meta["family"], "n")] += 1
            stats[(meta["family"], "exit")] += meta["exit"]
            if i % 50 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"{i}/{len(tasks)} episodes, {steps} steps, {steps / el:.0f} steps/s", flush=True)
    summary = {f: {"n": stats[(f, "n")], "exit_rate": round(stats[(f, "exit")] / max(1, stats[(f, "n")]), 3)} for f in a.families}
    print(json.dumps({"out": a.out, "steps": steps, "minutes": round((time.time() - t0) / 60, 1), "by_family": summary}))


if __name__ == "__main__":
    main()
