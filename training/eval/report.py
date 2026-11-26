"""Report family metrics and DAgger scores from saved evaluation rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean

from training.dagger.loop import score_rows

MEAN_FIELDS = {
    "keys": "Mean keys",
    "locked_opened": "Mean locked opened",
    "doors_opened": "Mean doors opened",
    "cells_visited": "Mean cells visited",
    "kills": "Mean kills",
}


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    """Return the bounds of a two-sided Wilson 95% interval as proportions."""
    if n < 1 or not 0 <= successes <= n:
        raise ValueError("expected 0 <= successes <= n and n > 0")
    z = 1.959963984540054
    rate = successes / n
    denominator = 1 + z**2 / n
    center = (rate + z**2 / (2 * n)) / denominator
    margin = z * sqrt((rate * (1 - rate) + z**2 / (4 * n)) / n) / denominator
    return (0.0 if successes == 0 else center - margin,
            1.0 if successes == n else center + margin)


def summarize_rows(rows: list[dict]) -> dict:
    n = len(rows)
    exits = sum(row["exit"] for row in rows)
    low, high = wilson_interval(exits, n)
    return {
        "n": n,
        "exit_pct": 100 * exits / n,
        "exit_ci_95": [100 * low, 100 * high],
        **{f"mean_{field}": mean(row[field] for row in rows) for field in MEAN_FIELDS},
        "death_pct": 100 * mean(row["dead"] for row in rows),
        "act_ms_p99": max(row["act_ms_p99"] for row in rows),
    }


def build_report(runs: dict[str, list[dict]], families: list[str]) -> dict:
    groups = defaultdict(lambda: defaultdict(list))
    for name, rows in runs.items():
        for row in rows:
            groups[row["family"]][name].append(row)
    order = dict.fromkeys([*families, *sorted(groups)])
    return {
        "score_rows": {name: score_rows(rows) for name, rows in runs.items()},
        "families": {
            family: {name: summarize_rows(rows) for name, rows in groups[family].items()}
            for family in order if family in groups
        },
    }


def _escape(cell: str) -> str:
    return cell.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict) -> str:
    lines = ["## DAgger scores", "", "| Run | score_rows |", "| --- | ---: |"]
    for name, score in report["score_rows"].items():
        lines.append(f"| {_escape(name)} | {score:.4f} |")

    headers = ["Run", "n", "Exit % [Wilson 95% CI]", *MEAN_FIELDS.values(), "Death %", "Act ms p99 (max)"]
    header = "| " + " | ".join(headers) + " |"
    rule = "| --- |" + " ---: |" * (len(headers) - 1)
    for family, runs in report["families"].items():
        lines.extend(("", f"## {_escape(family)}", "", header, rule))
        for name, row in runs.items():
            low, high = row["exit_ci_95"]
            metrics = [row[f"mean_{field}"] for field in MEAN_FIELDS]
            metrics.extend((row["death_pct"], row["act_ms_p99"]))
            values = " | ".join(f"{value:.2f}" for value in metrics)
            lines.append(
                f"| {_escape(name)} | {row['n']} | {row['exit_pct']:.2f} [{low:.2f}, {high:.2f}] | {values} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", metavar="NAME=PATH", required=True)
    parser.add_argument("--families", nargs="+", default=[], help="list these families first, in order")
    parser.add_argument("--format", choices=("md", "json"), default="md")
    args = parser.parse_args()

    runs = {}
    for value in args.runs:
        name, separator, path = value.partition("=")
        if not (name and separator and path) or name in runs:
            parser.error(f"--runs requires unique NAME=PATH entries (got {value!r})")
        try:
            rows = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parser.error(f"{path}: {exc}")
        if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
            parser.error(f"{path} must contain a non-empty list of episode rows")
        runs[name] = rows
    try:
        report = build_report(runs, args.families)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"invalid evaluation rows: {exc}")
    print(json.dumps(report, indent=2) if args.format == "json" else render_markdown(report))


if __name__ == "__main__":
    main()
