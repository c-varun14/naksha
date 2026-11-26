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

from demo.agent import DoomAgent, MemoryOffAgent  # noqa: E402
from demo.hd_env import Lockstep, open_games  # noqa: E402
from demo.server import BUILT_INS, DemoServer, write_map  # noqa: E402
from demo.render import (  # noqa: E402
    ACCENT, CELL_COLORS, MEMORY_OFF_RED, TRACK, WALL, AudienceMap, SectorMap, composite, encode_png,
)
from demo.wad_source import WAD_SHORTCUTS  # noqa: E402
from doom_env.spaces import BUTTONS, FRAME_SKIP  # noqa: E402
from doom_env.wrapper import IWAD  # noqa: E402
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


def test_memory_off_resets_recurrent_state_before_every_act(monkeypatch):
    class Model:
        def initial_state(self, batch_size):
            assert batch_size == 1
            return [0]

    def act(self, obs):
        self.state[0] += obs
        self.last_out = {"state": self.state[0]}
        return self.state[0]

    monkeypatch.setattr(DoomAgent, "act", act)
    agent = MemoryOffAgent.__new__(MemoryOffAgent)
    agent.model = Model()
    agent.state = [99]
    assert agent.act(2) == 2
    assert agent.act(3) == 3
    assert agent.last_out == {"state": 3}


def test_freedoom_spectator_lockstep(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALSOFT_DRIVERS", "null")
    with open_games(IWAD, "MAP01", episode_ticks=60 * FRAME_SKIP) as (official, spectator):
        games = Lockstep(official, spectator)
        obs = games.reset(seed=0)
        audience = SectorMap(official.sector_polygons(), official.mapinfo.door_sectors)
        audience.observe(games.pose)
        initial_position = audience.position
        action = np.zeros(official.action_space.shape, dtype=np.int8)
        action[BUTTONS.index("MOVE_FORWARD")] = 1
        for _ in range(40):
            obs, terminated, truncated, spectator_finished = games.step(action)
            assert not (terminated or truncated or spectator_finished)
            audience.observe(games.pose)
        assert games.desync_step is None
        assert audience.position != initial_position
        assert audience.trail.any()
        rgb = games.frame(obs)
        assert rgb.shape == (480, 640, 3)
        image = composite(rgb, audience)
        assert image.shape == (900, 1600, 3)
        panel = image[:, 1200:]
        for color in (WALL, CELL_COLORS["D"], ACCENT):
            assert (panel == color).all(axis=-1).any()


def test_sector_map_draws_shared_locked_door_in_red():
    boundary = np.array([[0, 0, 0, 1]])
    sectors = [(0, boundary), (1, boundary)]
    audience = SectorMap(sectors, {0: 1})
    assert (audience.level == CELL_COLORS["R"]).all(axis=-1).any()
    assert not (audience.level == WALL).all(axis=-1).any()


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


@pytest.mark.parametrize(("has_memory", "memory", "use_wad"), [
    (False, None, False), (True, "off", False), (True, "on", True),
], ids=["default", "memory-off", "wad"])
def test_episode_updates_state(serve_demo, tmp_path, has_memory, memory, use_wad):
    server, url = serve_demo
    server.weights = tmp_path / "random.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        net = NakshaMemNet() if has_memory else NakshaNet()
        save_checkpoint(server.weights, net, {"memory": MEMORY_VERSION} if has_memory else {})
    with urlopen(f"{url}/maps", timeout=5) as response:
        assert json.load(response) == list(BUILT_INS)
    with urlopen(f"{url}/wad-shortcuts", timeout=5) as response:
        assert json.load(response) == WAD_SHORTCUTS
    source = {"wad": IWAD, "map": "MAP02"} if use_wad else {"grid": BUILT_INS[0]["grid"]}
    if memory is not None:
        source["memory"] = memory
    with post_run(url, **source, ticks=30 * FRAME_SKIP, speed=4) as response:
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
    assert state["memory"] == (memory or "on")
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
    np.testing.assert_array_equal(panel[0], MEMORY_OFF_RED if memory == "off" else TRACK)


@pytest.mark.parametrize(("payload", "message"), [
    ({"grid": BUILT_INS[0]["grid"], "memory": "invalid"}, "memory"),
    ({"grid": BUILT_INS[0]["grid"], "wad": IWAD}, "either grid or wad"),
    ({"wad": "relative.wad"}, "absolute"),
    ({"wad": IWAD, "map": "MISSING"}, "does not contain map MISSING"),
])
def test_run_rejects_invalid_map_options(serve_demo, payload, message):
    _, url = serve_demo
    with pytest.raises(HTTPError) as exc:
        post_run(url, **payload)
    with exc.value as response:
        assert response.code == 400
        assert message in json.load(response)["error"]


def test_run_rejects_unreadable_or_invalid_wad(serve_demo, tmp_path):
    _, url = serve_demo
    invalid = tmp_path / "invalid.wad"
    invalid.write_bytes(b"not a WAD")
    directory = tmp_path / "directory.wad"
    directory.mkdir()
    for path, message in [
        (tmp_path / "file.txt", "must end in .wad"),
        (tmp_path / "missing.wad", "Cannot read WAD file"),
        (directory, "expected a regular file"),
        (invalid, "not a WAD file"),
    ]:
        with pytest.raises(HTTPError) as exc:
            post_run(url, wad=str(path))
        with exc.value as response:
            assert response.code == 400
            assert message in json.load(response)["error"]
