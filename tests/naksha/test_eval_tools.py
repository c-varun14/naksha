import hashlib
import json
import sys
from pathlib import Path

import pytest

from training.eval import make_suite, report, select, stock

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


@pytest.mark.parametrize("second_dev,second_cells,expected", [
    (42, 80, "first.pt"),  # Falls outside the dev margin despite a higher stock score.
    (43, 30, "second.pt"),  # Higher stock score within the dev margin.
    (43, 21, "first.pt"),  # Stock scores tie, so dev breaks the tie.
])
def test_select_checkpoints(tmp_path, monkeypatch, capsys, second_dev, second_cells, expected):
    monkeypatch.chdir(tmp_path)
    paths = [Path("first.pt"), Path("second.pt")]
    for path in paths:
        path.touch()
    dev = Path("tiny-dev.json")
    dev.write_text(json.dumps([{"id": "arena"}]), encoding="utf-8")
    calls = []

    def evaluate(specs, policy, seeds, procs, weights, *, suite="default"):
        calls.append((weights, [spec["id"] for spec in specs], suite))
        assert (policy, list(seeds), procs) == ("agent", [0, 1], 1)
        if suite == dev.stem:
            return [{"cells_visited": 50 if weights == paths[0] else second_dev}] * len(seeds)
        if weights == paths[0]:
            return [
                {"cells_visited": 5, "exit": True, "dead": False, "kills": 2},
                {"cells_visited": 7, "exit": False, "dead": True, "kills": 1},
            ]
        return [{"cells_visited": second_cells, "exit": False, "dead": False, "kills": 0}] * len(seeds)

    monkeypatch.setattr(select, "evaluate", evaluate)
    select.main([
        "--candidates", *map(str, paths), "--dev", str(dev), "--stock", "MAP01",
        "--seeds", "2", "--procs", "1",
    ])
    assert calls == [(path, ids, suite) for path in paths
                     for ids, suite in [(["arena"], dev.stem), (["freedoom-MAP01"], "default")]]
    report_path, = Path("runs/eval").glob("select-*.json")
    result = json.loads(report_path.read_text(encoding="utf-8"))
    assert result["best"] == expected
    first, second = result["candidates"]
    assert [first[field] for field in ("stock_exits", "stock_cells", "stock_kills", "stock_deaths")] == [1, 6, 3, 1]
    assert first["stock_score"] == 21
    assert second["eligible"] == (second_dev >= 43)
    output = capsys.readouterr().out
    assert "Stock mean cells visited" in output
    assert f"Best checkpoint: {expected}" in output
