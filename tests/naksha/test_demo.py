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

from demo.server import BUILT_INS, DemoServer  # noqa: E402
from demo.render import AudienceMap, composite, encode_png  # noqa: E402
from doom_env.spaces import FRAME_SKIP  # noqa: E402
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
    assert image.shape == (360, 840, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image[:3, 3:6], np.broadcast_to(rgb[0, 1], (3, 3, 3)))
    assert len(np.unique(image[:, 480:].reshape(-1, 3), axis=0)) > 3


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


def test_episode_updates_state(serve_demo, tmp_path):
    server, url = serve_demo
    server.weights = tmp_path / "random.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        save_checkpoint(server.weights, NakshaNet(), {})
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
    assert state["step"] == 30
    assert state["outcome"] == "timeout"
    assert set(state["intention"]) == set(INTENTIONS)
    assert sum(state["intention"].values()) == pytest.approx(1)
    assert 0 <= state["exit_vis"] <= 1
    assert state["macro"]
    assert state["events"]
    with urlopen(f"{url}/frame.png", timeout=5) as response:
        assert response.headers.get_content_type() == "image/png"
        assert struct.unpack_from(">II", response.read(), 16) == (840, 360)
