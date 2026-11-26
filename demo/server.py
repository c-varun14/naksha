"""Run the local Naksha demo with the standard-library HTTP server."""

from __future__ import annotations

import argparse
import json
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread, TIMEOUT_MAX
from time import monotonic
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))

import training.paths  # noqa: E402, F401
from demo.render import AudienceMap, blank_frame, composite, encode_png  # noqa: E402
from doom_env import make_env  # noqa: E402
from doom_env.spaces import FRAME_SKIP, VITALS_FIELDS  # noqa: E402
from doom_env.wrapper import DEFAULT_EPISODE_TICKS  # noqa: E402
from naksha.actions import NAMES  # noqa: E402
from naksha.model import INTENTIONS  # noqa: E402
from submission.agent import DoomAgent  # noqa: E402
from training.gen.compile import GridError, parse, write_map  # noqa: E402
from training.gen.families import solvable  # noqa: E402

INDEX = ROOT / "demo/index.html"
BUILT_INS = (
    {"id": "Simple door", "family": "rooms",
     "grid": "###########\n#....#....#\n#P...D..E.#\n#....#....#\n###########"},
    {"id": "Red key dead end", "family": "locks",
     "grid": "#############\n#P....R...E.#\n###.###.....#\n###.#########\n###r#########\n#############"},
    {"id": "Two zombiemen", "family": "arena",
     "grid": "###########\n#.....z...#\n#P......E.#\n#.....z...#\n###########"},
)
KEYS = tuple(field.removeprefix("key_").replace("_", " ") for field in VITALS_FIELDS[8:14])
MAX_GRID_SIDE = 128
MAX_REQUEST_BYTES = 64 * 1024


def load_maps(path: Path | None) -> list[dict]:
    if path is None:
        return []
    maps = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(maps, list):  # maps without an ASCII grid (e.g. stock WADs) cannot be shown here
        maps = [spec for spec in maps if isinstance(spec, dict) and isinstance(spec.get("grid"), str)]
    if not isinstance(maps, list) or any(
        not isinstance(spec, dict)
        or any(not isinstance(spec.get(field), str) for field in ("id", "family", "grid"))
        for spec in maps
    ):
        raise ValueError("maps file must contain a list of specs with id, family, and grid")
    return maps


def validate_grid(grid: str) -> list[str]:
    if not isinstance(grid, str):
        raise GridError("grid must be a string")
    lines = grid.strip("\n").split("\n")
    if len(lines) > MAX_GRID_SIDE or max(map(len, lines)) > MAX_GRID_SIDE:
        raise GridError(f"grid must fit within {MAX_GRID_SIDE} × {MAX_GRID_SIDE} cells")
    rows = parse(grid)
    if not solvable(rows):
        raise GridError("the exit E must be reachable from P with the available keys")
    return rows


def weights_path(weights: Path | None) -> Path:
    path = Path(weights or os.environ.get("NAKSHA_WEIGHTS") or ROOT / "submission/weights/naksha.pt")
    if not path.is_file():
        raise FileNotFoundError(f"Weights not found at {path}. Pass --weights with a checkpoint.")
    return path.resolve()


def initial_state(outcome=None) -> dict:
    return {
        "step": 0, "tic": 0, "health": None, "keys": [], "kills": 0,
        "intention": dict.fromkeys(INTENTIONS, 0.0), "exit_vis": 0.0,
        "macro": None, "outcome": outcome, "events": [],
    }


def play_episode(wad, rows, ticks, weights, publish, *, stop=None, speed=None, seed=None):
    """Play from official observations and publish rendered frames with audience telemetry."""
    stop = stop if stop is not None else Event()
    os.environ.setdefault("ALSOFT_DRIVERS", "null")
    env = make_env(wad, "MAP01", episode_ticks=ticks, render_buffers=True)
    try:
        agent = DoomAgent(env.observation_space, env.action_space, "cpu", weights=weights)
        agent.reset()
        obs, _ = env.reset(seed=seed)
        audience = AudienceMap(rows)
        pose = env.privileged_state()
        audience.observe(pose, obs["rays"])
        door_sectors = pose["door_lock"] >= 0
        open_doors = pose["sector_ceiling"][door_sectors] > pose["sector_floor"][door_sectors]
        held = obs["vitals"][8:14] > 0
        events = []
        state = initial_state("running")
        state.update(tic=pose["tic"], health=int(pose["health"]), kills=pose["kills"])
        if not stop.is_set():
            publish(state, encode_png(composite(obs["rgb"], audience)))
        step = 0
        start = monotonic()
        while not stop.is_set():
            step += 1
            if speed is not None:
                deadline = start + step * FRAME_SKIP / (35 * speed)
                remaining = deadline - monotonic()
                while remaining > 0 and not stop.wait(min(remaining, TIMEOUT_MAX)):
                    remaining = deadline - monotonic()
                if stop.is_set():
                    break
            action = agent.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            pose = env.privileged_state()
            finished = terminated or truncated
            # The wrapper repeats the previous observation on the terminal step.
            audience.observe(pose, None if finished else obs["rays"])

            new_held = obs["vitals"][8:14] > 0
            for i, key in enumerate(KEYS):
                if new_held[i] and not held[i]:
                    color = key.split()[0]
                    audience.pick_up(color[0], pose)
                    events.append(f"picked {color} key at step {step}")
            held = new_held

            doors = pose["sector_ceiling"][door_sectors] > pose["sector_floor"][door_sectors]
            events.extend(f"opened door at step {step}" for _ in range(int((doors & ~open_doors).sum())))
            open_doors = doors

            outcome = "running"
            if pose["exit_reached"]:
                outcome = "exit"
                events.append(f"exit reached at step {step}")
            elif pose["dead"]:
                outcome = "dead"
                events.append(f"died at step {step}")
            elif truncated:
                outcome = "timeout"
                events.append(f"time ran out at step {step}")

            state = {
                "step": step, "tic": pose["tic"], "health": max(0, int(pose["health"])),
                "keys": [key for key, has_key in zip(KEYS, held) if has_key], "kills": pose["kills"],
                "intention": dict(zip(INTENTIONS, map(float, agent.last_out["intention"]))),
                "exit_vis": agent.last_out["exit_vis"],
                "macro": NAMES[int(agent.last_out["logits"].argmax())],
                "outcome": outcome, "events": events[-12:],
            }
            publish(state, encode_png(composite(obs["rgb"], audience)))
            if finished:
                break
        return events
    finally:
        env.close()


class DemoServer(ThreadingHTTPServer):
    def __init__(self, address=("127.0.0.1", 8765), *, weights=None, maps=None):
        self.weights = weights
        self.maps = [*BUILT_INS, *load_maps(maps)]
        self.snapshot_lock = Lock()
        self.run_lock = Lock()
        self.state = initial_state()
        self.frame = encode_png(blank_frame())
        self.run_id = 0
        self.worker = None
        self.stop = Event()
        super().__init__(address, DemoHandler)

    def snapshot(self):
        with self.snapshot_lock:
            return self.state, self.frame

    def start_run(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        grid = payload.get("grid")
        rows = validate_grid(grid)
        spec = next((spec for spec in self.maps if spec["grid"] == grid), {})
        angle = payload.get("angle", spec.get("player_angle"))
        ticks = payload.get("ticks", spec.get("episode_ticks", DEFAULT_EPISODE_TICKS))
        speed = payload.get("speed", 1.0)
        if angle is not None and type(angle) is not int:
            raise ValueError("angle must be a whole number in degrees")
        if type(ticks) is not int or ticks < 1 or ticks > 2**31 - 1:
            raise ValueError("ticks must be a positive 32-bit whole number")
        if type(speed) not in (float, int) or not 0 < speed <= 4 or not math.isfinite(speed):
            raise ValueError("speed must be greater than zero and no more than 4×")

        temp = TemporaryDirectory(prefix="naksha-demo-")
        try:
            wad = write_map(grid, Path(temp.name) / "map.wad", player_angle=None if angle is None else angle % 360)
            weights = weights_path(self.weights)
            with self.run_lock:
                self._stop_episode()
                self.stop = Event()
                self.run_id += 1
                run_id = self.run_id
                with self.snapshot_lock:
                    self.state = {**initial_state("running"), "run_id": run_id}
                    self.frame = encode_png(blank_frame())
                self.worker = Thread(
                    target=self._episode, args=(temp, wad, rows, ticks, weights, speed, spec.get("seed"), run_id),
                    daemon=True,
                )
                self.worker.start()
                return run_id
        except BaseException:
            temp.cleanup()
            raise

    def _episode(self, temp, wad, rows, ticks, weights, speed, seed, run_id):
        def publish(state, frame):
            with self.snapshot_lock:
                self.state = {**state, "run_id": run_id}
                self.frame = frame

        with temp:
            try:
                play_episode(wad, rows, ticks, weights, publish, stop=self.stop, speed=speed, seed=seed)
            except Exception as exc:
                with self.snapshot_lock:
                    self.state = {**self.state, "outcome": None, "error": f"Episode failed: {exc}"}

    def _stop_episode(self):
        self.stop.set()
        if self.worker is not None:
            self.worker.join()
            self.worker = None

    def server_close(self):
        super().server_close()
        with self.run_lock:
            self._stop_episode()


class DemoHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Frame polling would otherwise dominate the local terminal.
        pass

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, payload):
        self._send(status, "application/json", json.dumps(payload).encode())

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", INDEX.read_bytes())
        elif path == "/maps":
            self._json(200, [{field: spec[field] for field in ("id", "family", "grid")}
                             for spec in self.server.maps])
        elif path == "/state":
            self._json(200, self.server.snapshot()[0])
        elif path == "/frame.png":
            self._send(200, "image/png", self.server.snapshot()[1])
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        if urlsplit(self.path).path != "/run":
            self._json(404, {"error": "Not found"})
            return
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("send the run as application/json")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_REQUEST_BYTES:
                raise ValueError("invalid run request size")
            run_id = self.server.start_run(json.loads(self.rfile.read(size)))
        except (ValueError, RecursionError) as exc:
            self._json(400, {"error": str(exc)})
        except OSError as exc:
            self._json(500, {"error": str(exc)})
        else:
            self._json(202, {"run_id": run_id})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--maps", type=Path, help="optional JSON list of example map specs")
    args = parser.parse_args()
    try:
        server = DemoServer(("127.0.0.1", args.port), weights=weights_path(args.weights), maps=args.maps)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    with server:
        print(f"Naksha demo listening on http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
