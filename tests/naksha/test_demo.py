import json
from pathlib import Path
import struct
import sys
from threading import Thread
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zlib

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo.hd_env import Lockstep, open_games  # noqa: E402
from demo.server import BUILT_INS, DemoServer, write_map  # noqa: E402
from demo.render import ACCENT, AudienceMap, composite, encode_png  # noqa: E402
from doom_env.spaces import BUTTONS, FRAME_SKIP  # noqa: E402
from naksha.memory import MEMORY_VERSION, NakshaMemNet  # noqa: E402
from naksha.model import INTENTIONS, NakshaNet, save_checkpoint  # noqa: E402


def test_png_round_trip():
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    png = encode_png(rgb)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    chunks = {}
    while pos < len(png):
        size, kind = struct.unpack_from(">I4s", png, pos)
        data = png[pos + 8:pos + 8 + size]
        assert struct.unpack_from(">I", png, pos + 8 + size)[0] == zlib.crc32(kind + data)
        chunks[kind] = data
        pos += size + 12
    assert pos == len(png)
    assert list(chunks) == [b"IHDR", b"IDAT", b"IEND"]
    assert struct.unpack(">IIBBBBB", chunks[b"IHDR"]) == (4, 3, 8, 2, 0, 0, 0)
    raw = zlib.decompress(chunks[b"IDAT"])
    assert len(raw) == 3 * (1 + 4 * 3)
    pixels = np.frombuffer(raw, np.uint8).reshape(3, 13)
    assert not pixels[:, 0].any()
    np.testing.assert_array_equal(pixels[:, 1:].reshape(rgb.shape), rgb)


def test_composite_shape_and_scaling():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    rgb[0, 1] = (21, 54, 89)
    audience = AudienceMap(BUILT_INS[0]["grid"].splitlines())
    image = composite(rgb, audience)
    assert image.shape == (900, 1600, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image[0, 0], (0, 0, 0))
    np.testing.assert_array_equal(image[0, 7], (10, 27, 44))
    assert len(np.unique(image[:, 1200:].reshape(-1, 3), axis=0)) > 3


def test_showcase_door_spectator_lockstep(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALSOFT_DRIVERS", "null")
    grid = (Path(__file__).resolve().parents[2] / "showcase/01-door.txt").read_text()
    wad = write_map(grid, tmp_path / "door.wad")
    audience = AudienceMap(grid.splitlines())
    with open_games(wad, episode_ticks=63 * FRAME_SKIP) as (official, spectator):
        games = Lockstep(official, spectator)
        obs = games.reset(seed=0)
        initial_x = games.pose["x"]
        action = np.zeros(official.action_space.shape, dtype=np.int8)
        for step in range(60):
            action[BUTTONS.index("MOVE_FORWARD")] = step < 10
            obs, terminated, truncated, spectator_finished = games.step(action)
            assert not (terminated or truncated or spectator_finished)
            pose = spectator.privileged_state()
            assert abs(games.pose["x"] - pose["x"]) <= 0.01
            assert abs(games.pose["y"] - pose["y"]) <= 0.01
            assert games.desync_step is None
            audience.observe(games.pose, obs["rays"])
        assert games.pose["x"] > initial_x
        assert obs["rgb"].shape == (120, 160, 3)
        assert obs["depth"].shape == (120, 160)
        rgb = games.frame(obs)
        assert rgb.shape == (480, 640, 3)
        assert composite(rgb, audience).shape == (900, 1600, 3)

        spectator_pose = spectator.privileged_state
        with monkeypatch.context() as patch:
            patch.setattr(spectator, "privileged_state", lambda: {**spectator_pose(), "x": -1})
            obs, *_ = games.step(action)
            assert games.desync_step == 61
            assert games.frame(obs) is obs["rgb"]

        obs, *_ = games.step(action)
        assert games.desync_step == 61
        assert games.frame(obs) is obs["rgb"]


@pytest.fixture
def serve_demo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with DemoServer(("127.0.0.1", 0)) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield server, f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


def post_run(url, **payload):
    request = Request(f"{url}/run", json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return urlopen(request, timeout=5)


def test_run_requires_player_start(serve_demo):
    _, url = serve_demo
    with pytest.raises(HTTPError) as exc:
        post_run(url, grid="#####\n#..E#\n#####")
    with exc.value as response:
        assert response.code == 400
        assert "exactly one P" in json.load(response)["error"]


@pytest.mark.parametrize("has_memory", [False, True], ids=["feed-forward", "memory"])
def test_episode_updates_state(serve_demo, tmp_path, has_memory):
    server, url = serve_demo
    server.weights = tmp_path / "random.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        net = NakshaMemNet() if has_memory else NakshaNet()
        save_checkpoint(server.weights, net, {"memory": MEMORY_VERSION} if has_memory else {})
    with urlopen(f"{url}/maps", timeout=5) as response:
        assert json.load(response) == list(BUILT_INS)
    with post_run(url, grid=BUILT_INS[0]["grid"], ticks=30 * FRAME_SKIP, speed=4) as response:
        assert response.status == 202
        run_id = json.load(response)["run_id"]

    deadline = monotonic() + 10
    while monotonic() < deadline:
        with urlopen(f"{url}/state", timeout=5) as response:
            state = json.load(response)
        assert "error" not in state, state
        if state["outcome"] != "running":
            break
        sleep(0.02)

    assert state["run_id"] == run_id
    assert state["has_memory"] is has_memory
    assert state["step"] == 30
    assert state["outcome"] == "timeout"
    assert set(state["intention"]) == set(INTENTIONS)
    assert sum(state["intention"].values()) == pytest.approx(1)
    assert 0 <= state["exit_vis"] <= 1
    assert state["macro"]
    assert state["events"]
    assert state["spectator_desync_step"] is None
    with urlopen(f"{url}/frame.png", timeout=5) as response:
        assert response.headers.get_content_type() == "image/png"
        png = response.read()
    width, height = struct.unpack_from(">II", png, 16)
    assert (width, height) == (1600, 900)
    size, kind = struct.unpack_from(">I4s", png, 33)
    assert kind == b"IDAT"
    pixels = np.frombuffer(zlib.decompress(png[41:41 + size]), np.uint8)
    panel = pixels.reshape(height, 1 + width * 3)[:, 1 + 1200 * 3:].reshape(-1, 3)
    assert (panel == ACCENT).all(axis=1).any()
