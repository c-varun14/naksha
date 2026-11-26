"""Evaluate agent and random policies on the bundled Freedoom 2 stock maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from training.eval.run import evaluate

from doom_env.wrapper import IWAD


STOCK_MAPS = tuple(f"MAP{i:02d}" for i in range(1, 33))
HELD_OUT_MAPS = STOCK_MAPS[:5]


def build_specs(map_names=HELD_OUT_MAPS) -> list[dict]:
    """Use the bundled IWAD as both the base game and the scenario WAD."""
    return [
        {
            "id": f"freedoom-{map_name}",
            "family": "freedoom",
            "wad": IWAD,
            "map": map_name,
            "episode_ticks": 4200,
            "grid": None,
        }
        for map_name in map_names
    ]


def _table(specs: list[dict], rows: list[dict]) -> str:
    lines = [
        "| Map | n | Exits | Mean cells visited | Mean kills | Deaths | Mean doors opened |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    groups = [(spec["map"], [row for row in rows if row["id"] == spec["id"]]) for spec in specs]
    for label, group in [*groups, ("All maps", rows)]:
        exits = sum(row["exit"] for row in group)
        deaths = sum(row["dead"] for row in group)
        cells, kills, doors = (mean(row[field] for row in group)
                               for field in ("cells_visited", "kills", "doors_opened"))
        lines.append(
            f"| {label} | {len(group)} | {exits} | {cells:.2f} | {kills:.2f} | {deaths} | {doors:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", nargs="+", choices=("agent", "random"), default=("agent", "random"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--procs", type=int, choices=range(1, 4), default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/dl2"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be positive")

    specs = build_specs()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = [
        "# Freedoom 2 stock maps",
        "",
        f"MAP01–MAP05, episode seeds 0–{args.seeds - 1}, and a 4,200-tic limit.",
        "The official wrapper loads the bundled Freedoom 2 IWAD as its scenario WAD.",
        "Cells are 64×64 map-unit position bins.",
        "The shared evaluator counts door sectors whose absolute ceiling exceeds 8 at any observation, "
        "so this metric can include doors already open at spawn or doors on elevated floors.",
    ]
    for policy in dict.fromkeys(args.policy):
        rows = evaluate(specs, policy, range(args.seeds), args.procs, args.weights)
        path = args.out_dir / f"stock-{policy}.json"
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        report.extend(("", f"## {policy.title()}", "", _table(specs, rows)))
    rendered = "\n".join(report) + "\n"
    (args.out_dir / "stock.md").write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
