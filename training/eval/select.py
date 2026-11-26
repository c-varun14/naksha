"""Select a checkpoint using the development suite and held-out Freedoom maps."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from training.dagger.loop import score_rows
from training.eval.run import evaluate
from training.eval.stock import HELD_OUT_MAPS, STOCK_MAPS, build_specs

DEV_TOLERANCE = 0.15


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", nargs="+", required=True, type=Path)
    parser.add_argument("--dev", type=Path, default=Path("data/suites/dev.json"))
    parser.add_argument("--stock", nargs="+", type=str.upper, choices=STOCK_MAPS, default=HELD_OUT_MAPS)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--procs", type=int, choices=range(1, 4), default=3)
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    for path in (args.dev, *args.candidates):
        if not path.is_file():
            parser.error(f"file not found: {path}")
    dev_specs = json.loads(args.dev.read_text(encoding="utf-8"))
    if not dev_specs:
        parser.error("--dev must contain at least one map")

    stock_specs = build_specs(args.stock)
    seeds = range(args.seeds)
    results = []
    for checkpoint in args.candidates:
        dev = evaluate(dev_specs, "agent", seeds, args.procs, checkpoint, suite=args.dev.stem)
        stock = evaluate(stock_specs, "agent", seeds, args.procs, checkpoint)
        cells = mean(row["cells_visited"] for row in stock)
        exits = sum(row["exit"] for row in stock)
        deaths = sum(row["dead"] for row in stock)
        results.append({
            "checkpoint": str(checkpoint), "dev_score": score_rows(dev),
            "stock_cells": cells, "stock_exits": exits,
            "stock_kills": sum(row["kills"] for row in stock), "stock_deaths": deaths,
            "stock_score": cells + 20 * exits - 5 * deaths,
            "dev_rows": dev, "stock_rows": stock,
        })

    best_dev = max(row["dev_score"] for row in results)
    for row in results:
        row["eligible"] = row["dev_score"] >= best_dev - DEV_TOLERANCE
    best = max((row for row in results if row["eligible"]),
               key=lambda row: (row["stock_score"], row["dev_score"]))
    print("| Checkpoint | Dev score | Stock exits | Stock mean cells visited | Stock kills | Stock deaths | Stock score | Eligible |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in results:
        print(f"| {row['checkpoint']} | {row['dev_score']:.4f} | {row['stock_exits']} | "
              f"{row['stock_cells']:.2f} | {row['stock_kills']} | {row['stock_deaths']} | "
              f"{row['stock_score']:.2f} | {'yes' if row['eligible'] else 'no'} |")
    print(f"Best checkpoint: {best['checkpoint']}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path("runs/eval") / f"select-{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "best": best["checkpoint"], "best_dev_score": best_dev, "dev_tolerance": DEV_TOLERANCE,
        "dev": str(args.dev), "stock_maps": list(args.stock), "seeds": list(seeds), "candidates": results,
    }
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
