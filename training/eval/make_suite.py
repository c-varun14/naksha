"""Generate a JSON suite of procedural Doom maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.gen.families import generate


def _seed_range(value: str) -> range:
    try:
        first, last = map(int, value.split("-"))
        if first <= last:
            return range(first, last + 1)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected an ascending inclusive seed range A-B")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    families = ("arena", "rooms", "locks", "gauntlet")
    sizes = ("s", "m", "l")
    parser.add_argument("--families", nargs="+", choices=families, default=families)
    parser.add_argument("--sizes", nargs="+", choices=sizes, default=sizes)
    parser.add_argument("--seeds", type=_seed_range, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    specs = [
        {**generate(family, seed, size), "id": f"{family}-{size}-{seed}"}
        for family in args.families for size in args.sizes for seed in args.seeds
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(specs, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
