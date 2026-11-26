import json
from pathlib import Path
import sys
from threading import Thread
from urllib.request import urlopen

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from demo.server import BUILT_INS, DemoServer  # noqa: E402
from training.gen.compile import parse, write_map  # noqa: E402
from training.gen.families import solvable  # noqa: E402


SHOWCASE = ROOT / "showcase"
GRIDS = sorted(SHOWCASE.glob("*.txt"))


def test_showcase_has_grids():
    assert GRIDS


@pytest.mark.parametrize("path", GRIDS, ids=lambda path: path.stem)
def test_showcase_grid_compiles(path, tmp_path):
    grid = path.read_text(encoding="ascii")
    rows = parse(grid)
    assert sum(row.count("P") for row in rows) == 1
    assert any("E" in row for row in rows)
    assert solvable(rows)
    assert write_map(grid, tmp_path / f"{path.stem}.wad").stat().st_size > 0


def test_showcase_is_listed_before_other_maps(tmp_path):
    extra = {"id": "Extra", "family": "rooms", "grid": BUILT_INS[0]["grid"]}
    maps = tmp_path / "maps.json"
    maps.write_text(json.dumps([extra]), encoding="utf-8")
    with DemoServer(("127.0.0.1", 0), maps=maps, showcase=SHOWCASE) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/maps", timeout=5) as response:
                listed = json.load(response)
        finally:
            server.shutdown()
            thread.join()
    expected = [
        {"id": path.stem, "family": "showcase", "grid": path.read_text(encoding="ascii").strip("\n")}
        for path in GRIDS
    ]
    assert listed == [*expected, *BUILT_INS, extra]
