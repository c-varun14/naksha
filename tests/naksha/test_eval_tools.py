import hashlib
import json
import sys
from pathlib import Path

import pytest

from training.eval import make_suite, report, stock

HOLDOUT_SHA = "346227e923dfe48a88a4cc559f34d8a34a178b76154f1423f370ed3f6a7900c9"


def test_holdout_preset_matches_reference_hash(tmp_path, monkeypatch, capsys):
    path = tmp_path / "holdout.json"
    monkeypatch.setattr(sys, "argv", [
        "make_suite", "--preset", "holdout", "--check", HOLDOUT_SHA, "--out", str(path),
    ])
    make_suite.main()

    assert hashlib.sha256(path.read_bytes()).hexdigest() == HOLDOUT_SHA
    assert capsys.readouterr().out.strip() == HOLDOUT_SHA


def test_report_wilson_interval_from_eval_rows():
    rows = [
        {
            "family": "arena", "exit": i < 3, "dead": i >= 9, "keys": 0,
            "locked_opened": 0, "doors_opened": 0, "cells_visited": 5, "kills": 1,
            "act_ms_p99": (i + 1) / 2,
        }
        for i in range(10)
    ]
    result = report.build_report({"agent": rows}, ["arena"])
    stats = result["families"]["arena"]["agent"]

    assert stats["n"] == 10
    assert stats["exit_pct"] == 30
    assert stats["exit_ci_95"] == pytest.approx([10.8, 60.3], abs=0.1)
    assert stats["death_pct"] == 10
    assert stats["act_ms_p99"] == 5
    assert result["score_rows"]["agent"] == pytest.approx(1.1)


def test_stock_specs_use_freedoom_stock_maps():
    from doom_env.wad import list_maps, read_wad

    specs = stock.build_specs()
    maps = [f"MAP{i:02d}" for i in range(1, 6)]
    assert [spec["map"] for spec in specs] == maps
    assert json.loads(json.dumps(specs)) == specs
    wad = Path(specs[0]["wad"])
    assert wad.name == "freedoom2.wad"
    assert set(maps).issubset(list_maps(read_wad(wad)))
    for spec, map_name in zip(specs, maps):
        assert spec == {
            "id": f"freedoom-{map_name}", "family": "freedoom", "wad": str(wad),
            "map": map_name, "episode_ticks": 4200, "grid": None,
        }
