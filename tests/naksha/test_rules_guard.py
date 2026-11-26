from pathlib import Path


def test_submission_rules_guard() -> None:
    forbidden = ("atan2", "privileged_state", "heapq", "dijkstra", "a_star", "astar", "bfs(",
                 "import training", "from training")
    submission = Path(__file__).resolve().parents[2] / "submission"
    violations = []
    for path in sorted(submission.rglob("*.py")):
        source = path.read_text(encoding="utf-8").lower()
        violations.extend((path.relative_to(submission), token) for token in forbidden if token in source)
    assert not violations, violations
