"""Generate a JSON suite of procedural Doom maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training.gen.families import generate

FAMILIES = ("arena", "rooms", "locks", "gauntlet")
SIZES = ("s", "m", "l")


def _spec(family: str, seed: int, size: str) -> dict:
    return {**generate(family, seed, size), "id": f"{family}-{size}-{seed}"}


def _preset(name: str) -> list[dict]:
    base, count = {"holdout": (100000, 5), "dev": (50000, 3)}[name]
    specs = [
        _spec(family, base + 1000 * i + 100 * j + k, size)
        for i, family in enumerate(FAMILIES)
        for j, size in enumerate(SIZES)
        for k in range(count)
    ]
    if name == "holdout":
        specs.append({
            "id": "organiser-two_rooms", "family": "organiser", "seed": 0, "size": "s",
            "wad": "maps/two_rooms.wad", "episode_ticks": 2100, "grid": None, "locks": [],
        })
    return specs


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
    parser.add_argument("--families", nargs="+", choices=FAMILIES)
    parser.add_argument("--sizes", nargs="+", choices=SIZES)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seeds", type=_seed_range)
    source.add_argument("--preset", choices=("holdout", "dev"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", metavar="SHA", help="fail if the written SHA-256 differs")
    args = parser.parse_args()
    if args.preset:
        if args.families or args.sizes:
            parser.error("--preset fixes the families and sizes")
        content = json.dumps(_preset(args.preset), indent=1)
    else:
        specs = [
            _spec(family, seed, size)
            for family in (args.families or FAMILIES)
            for size in (args.sizes or SIZES)
            for seed in args.seeds
        ]
        content = json.dumps(specs, indent=2) + "\n"

    encoded = content.encode("utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(encoded)
    sha = hashlib.sha256(encoded).hexdigest()
    print(sha)
    if args.check is not None and sha != args.check.lower():
        parser.exit(1, f"SHA-256 mismatch: expected {args.check}, got {sha}\n")


if __name__ == "__main__":
    main()
