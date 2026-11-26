"""Measure the reference policy on generated maps: exit rate, keys, deaths per family.

uv run --no-sync python -m training.teacher.check --families locks rooms --n 20 --procs 11
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

from training import paths  # noqa: F401

ROOT = paths.ROOT


def run_one(args):
    family, size, seed, record, teacher = args
    from doom_env import make_env
    from naksha.actions import to_buttons
    from training.gen.compile import write_map
    from training.gen.families import generate

    if teacher == "v2":
        from training.teacher.teacher2 import Teacher2 as Teacher
    else:
        from training.teacher.teacher import Teacher

    spec = generate(family, seed, size)
    wad = write_map(spec["grid"], ROOT / f"data/maps/check/{family}-{size}-{seed}.wad", player_angle=spec["player_angle"])
    env = make_env(str(wad), "MAP01", episode_ticks=spec["episode_ticks"], render_buffers=False)
    obs, _ = env.reset(seed=seed)
    t = Teacher(spec, seed)
    t.reset(env)
    intents = defaultdict(int)
    steps = 0
    while True:
        priv = env.privileged_state()
        a, it, _ = t.act(obs, priv)
        intents[it] += 1
        obs, _, term, trunc, info = env.step(to_buttons(a))
        steps += 1
        if term or trunc:
            break
    priv = env.privileged_state()
    env.close()
    return {"family": family, "size": size, "seed": seed, "exit": bool(info["exit_reached"]), "dead": priv["dead"],
            "keys": int(sum(priv["keys"])), "locks": len(spec["locks"]), "steps": steps, "kills": priv["kills"],
            "intents": dict(intents), "grid": spec["grid"] if record else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["arena", "rooms", "locks", "gauntlet"])
    ap.add_argument("--sizes", nargs="+", default=["m"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--procs", type=int, default=11)
    ap.add_argument("--teacher", choices=("v1", "v2"), default="v1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tasks = [(f, s, a.start + i, True, a.teacher) for f in a.families for s in a.sizes for i in range(a.n)]
    with mp.get_context("spawn").Pool(a.procs) as pool:
        rows = pool.map(run_one, tasks)
    agg = defaultdict(lambda: defaultdict(float))
    for r in rows:
        k = f"{r['family']}-{r['size']}"
        agg[k]["n"] += 1
        agg[k]["exit"] += r["exit"]
        agg[k]["dead"] += r["dead"]
        agg[k]["keys"] += r["keys"]
        agg[k]["locks"] += r["locks"]
        agg[k]["steps"] += r["steps"]
    print("| family | n | exit % | dead % | keys/locks | mean steps |")
    print("|---|---|---|---|---|---|")
    for k, v in agg.items():
        n = v["n"]
        print(f"| {k} | {int(n)} | {100 * v['exit'] / n:.0f} | {100 * v['dead'] / n:.0f} | {v['keys'] / n:.2f}/{v['locks'] / n:.2f} | {v['steps'] / n:.0f} |")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
