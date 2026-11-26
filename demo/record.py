"""Record a headless Naksha episode as numbered PNG frames and JSON telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from vizdoom import ViZDoomErrorException

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.server import (  # noqa: E402
    DEFAULT_EPISODE_TICKS, FRAME_SKIP, load_maps, play_episode, validate_grid, weights_path, write_map,
)


def record(spec: dict, out: Path, weights: Path):
    """Record the live demo layout, including the sensor map for memory checkpoints."""
    grid = spec["grid"]
    rows = validate_grid(grid)
    states = []

    def publish(state, frame):
        (out / f"frame-{state['step']:06d}.png").write_bytes(frame)
        states.append(state)

    with TemporaryDirectory(prefix="naksha-record-") as temp:
        wad = write_map(grid, Path(temp) / "map.wad", player_angle=spec.get("player_angle"))
        out.mkdir(parents=True, exist_ok=True)
        events = play_episode(wad, rows, spec.get("episode_ticks", DEFAULT_EPISODE_TICKS),
                              weights, publish, seed=spec.get("seed"))
    data = {"id": spec.get("id"), "grid": grid, "outcome": states[-1]["outcome"],
            "states": states, "events": events}
    (out / "episode.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([
            ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-framerate", f"35/{FRAME_SKIP}",
            "-start_number", "0", "-i", str(out / "frame-%06d.png"), "-frames:v", str(len(states)),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out / "episode.mp4"),
        ], check=True)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--grid", type=Path, metavar="FILE")
    source.add_argument("--spec-id", metavar="ID")
    parser.add_argument("--maps", type=Path, metavar="FILE")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument("--weights", type=Path, metavar="PATH")
    args = parser.parse_args()
    try:
        if args.grid is not None:
            spec = {"grid": args.grid.read_text(encoding="utf-8")}
        else:
            if args.maps is None:
                parser.error("--spec-id requires --maps FILE")
            spec = next((spec for spec in load_maps(args.maps) if spec["id"] == args.spec_id), None)
            if spec is None:
                parser.error(f"map ID {args.spec_id!r} was not found in {args.maps}")
        data = record(spec, args.out, weights_path(args.weights))
    except (OSError, ValueError, RuntimeError, ViZDoomErrorException, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(f"Recorded {len(data['states'])} frames in {args.out} ({data['outcome']}).")


if __name__ == "__main__":
    main()
