"""Record a headless Naksha episode as PNG frames, JSON telemetry, and MP4."""

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
    DEFAULT_EPISODE_TICKS, load_maps, play_episode, prepare_map, validate_memory, validate_ticks,
    weights_path,
)


def record(spec: dict, out: Path, weights: Path):
    """Record the same composite served by the live demo."""
    ticks = validate_ticks(spec.get("episode_ticks", DEFAULT_EPISODE_TICKS))
    memory = validate_memory(spec.get("memory", "on"))
    states = []

    def publish(state, frame):
        (out / f"frame-{state['step']:06d}.png").write_bytes(frame)
        states.append(state)

    with TemporaryDirectory(prefix="naksha-record-") as temp:
        wad = Path(temp) / "map.wad"
        rows, map_name = prepare_map(spec, wad)
        out.mkdir(parents=True, exist_ok=True)
        events = play_episode(wad, rows, ticks, weights, publish, map_name=map_name,
                              seed=spec.get("seed"), memory=memory)
    source = {"wad": str(spec["wad"]), "map": map_name} if rows is None else {"grid": spec["grid"]}
    data = {"id": spec.get("id"), **source, "memory": memory, "outcome": states[-1]["outcome"],
            "spectator_desync_step": states[-1]["spectator_desync_step"],
            "states": states, "events": events}
    (out / "episode.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([
            ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-threads", "1", "-framerate", "12",
            "-start_number", "0", "-i", str(out / "frame-%06d.png"), "-frames:v", str(len(states)),
            "-c:v", "libx264", "-threads", "1", "-filter_threads", "1", "-pix_fmt", "yuv420p",
            str(out / "episode.mp4"),
        ], check=True)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--grid", type=Path, metavar="FILE")
    source.add_argument("--wad", type=Path, metavar="FILE")
    source.add_argument("--spec-id", metavar="ID")
    parser.add_argument("--map", default="MAP01", metavar="NAME", help="map name for --wad")
    parser.add_argument("--maps", type=Path, metavar="FILE")
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    parser.add_argument("--weights", type=Path, metavar="PATH")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int, help="episode duration in game tics")
    parser.add_argument("--memory", choices=("on", "off"), default="on")
    args = parser.parse_args()
    try:
        if args.grid is not None:
            spec = {"id": args.grid.stem, "grid": args.grid.read_text(encoding="utf-8")}
        elif args.wad is not None:
            spec = {"wad": args.wad, "map": args.map}
        else:
            if args.maps is None:
                parser.error("--spec-id requires --maps FILE")
            spec = next((spec for spec in load_maps(args.maps) if spec["id"] == args.spec_id), None)
            if spec is None:
                parser.error(f"map ID {args.spec_id!r} was not found in {args.maps}")
        if args.seed is not None:
            spec = {**spec, "seed": args.seed}
        if args.ticks is not None:
            spec = {**spec, "episode_ticks": args.ticks}
        spec = {**spec, "memory": args.memory}
        data = record(spec, args.out, weights_path(args.weights))
    except (OSError, ValueError, RuntimeError, ViZDoomErrorException, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(f"Recorded {len(data['states'])} frames in {args.out} ({data['outcome']}).")


if __name__ == "__main__":
    main()
