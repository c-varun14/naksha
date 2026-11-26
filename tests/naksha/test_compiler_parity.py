"""Run the organiser's own wrapper tests (tests/test_env.py) on maps built by OUR compiler.

If our grid -> UDMF compiler matches the organiser conventions, every one of their assertions
(ray distances, door / locked-door hit types, entity order, key pickup + locked door + exit) holds.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import test_env as organiser  # noqa: E402

from training.gen.compile import write_map  # noqa: E402

CASES = [
    (name, fn)
    for name, fn in inspect.getmembers(organiser, inspect.isfunction)
    if name.startswith("test_") and "tmp_path" in inspect.signature(fn).parameters
]


@pytest.mark.parametrize("name,fn", CASES, ids=[c[0] for c in CASES])
def test_organiser_case_on_our_compiler(name, fn, tmp_path, monkeypatch):
    counter = {"n": 0}

    def our_fixture_wad(grid: str) -> str:
        counter["n"] += 1
        return str(write_map(grid, tmp_path / f"ours_{counter['n']}.wad"))

    monkeypatch.setattr(organiser, "fixture_wad", our_fixture_wad)
    fn(tmp_path)
    assert counter["n"] >= 1, "case did not build a map"
