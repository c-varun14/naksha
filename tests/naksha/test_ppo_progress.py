import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submission"))

import training.paths  # noqa: E402, F401
from training.gen.compile import CELL  # noqa: E402
from training.ppo.ppo import ProgressShaping  # noqa: E402

# Row 1 of this grid is y = 1 in cell units (grid rows count down from the top).
GRID = "##########\n#r..P.R.E#\n##########"


class Flat:
    total = 0.0

    def step(self, state):
        return 0.0, 0, 0


def at(col, keys=(0, 0, 0)):
    return {"x": (col + 0.5) * CELL, "y": 1.5 * CELL, "keys": np.array(keys)}


def walk(shaping, cols, keys=(0, 0, 0)):
    return sum(shaping.step(at(c, keys))[0] for c in cols)


def test_without_the_key_it_is_paid_to_walk_to_the_key_not_the_locked_door():
    shaping = ProgressShaping(Flat(), at(4), GRID, 0.01)
    assert walk(shaping, [3, 2, 1]) == 0.03
    shaping = ProgressShaping(Flat(), at(4), GRID, 0.01)
    assert walk(shaping, [5]) == -0.01


def test_after_the_key_it_is_paid_to_walk_back_through_the_lock_to_the_exit():
    shaping = ProgressShaping(Flat(), at(4), GRID, 0.01)
    walk(shaping, [3, 2, 1])
    assert shaping.step(at(1, (0, 0, 1)))[0] == 0.0  # picking up the key re-targets without a jump
    back = walk(shaping, [2, 3, 4, 5, 6, 7, 8], (0, 0, 1))
    assert np.isclose(back, 0.07)
    assert np.isclose(shaping.total, 0.10)


def test_pacing_earns_nothing():
    shaping = ProgressShaping(Flat(), at(4), GRID, 0.01)
    assert np.isclose(walk(shaping, [3, 4, 3, 4, 3, 4]), 0.0)


def test_off_grid_steps_are_skipped_and_the_base_reward_passes_through():
    class Base(Flat):
        def step(self, state):
            return 0.5, 1, 1

    shaping = ProgressShaping(Base(), at(4), GRID, 0.01)
    off = {"x": 4.5 * CELL, "y": 0.5 * CELL, "keys": np.zeros(3)}  # a wall cell
    assert shaping.step(off) == (0.5, 1, 1)
    assert np.isclose(shaping.step(at(3))[0], 0.51)


def test_a_round_trip_nets_zero_even_with_a_long_retreat_and_a_jump_resets_the_baseline():
    grid = "############\n#P........E#\n############"
    shaping = ProgressShaping(Flat(), at(5), grid, 0.01)
    assert np.isclose(walk(shaping, [4, 3, 2, 3, 4, 5]), 0.0)
    assert shaping.step(at(9))[0] == 0.0  # a 4-cell jump pays nothing
    assert np.isclose(shaping.step(at(10))[0], 0.01)


from training.ppo.ppo import FrontierShaping  # noqa: E402

DEAD_END = "###########\n#E.....P..#\n###########"


def test_frontier_dead_end_excursion_costs_coming_back_pays_and_a_new_sighting_retargets():
    shaping = FrontierShaping(Flat(), at(7), DEAD_END, 0.01)
    assert shaping.distance == 3  # cols 5-9 seen from col 7; the nearest unseen cell is col 4
    assert np.isclose(shaping.step(at(8))[0], -0.01)  # deeper into the explored dead end
    assert np.isclose(shaping.step(at(9))[0], -0.01)
    assert np.isclose(walk(shaping, [8, 7]), 0.02)  # walking back out pays it back
    assert shaping.step(at(6))[0] == 0.0  # col 4 comes into view: re-target, no payment
    assert shaping.distance == 3


def test_frontier_circling_explored_cells_nets_zero():
    shaping = FrontierShaping(Flat(), at(7), DEAD_END, 0.01)
    assert np.isclose(walk(shaping, [8, 9, 8, 7, 8, 9, 8, 7]), 0.0)


def test_frontier_switches_to_the_exit_once_seen_so_returning_to_it_pays():
    grid = "##############\n#P.E.........#\n##############"  # the exit is within sight of the start
    shaping = FrontierShaping(Flat(), at(1), grid, 0.01)
    assert shaping.mode == "exit" and shaping.distance == 2
    away = walk(shaping, [2, 3, 4, 5, 6])  # walking past the exit costs, even while new cells come into view
    assert np.isclose(away, -0.01 * 3 + 0.01 * 2)  # cols 2, 3 approach (+2); cols 4-6 retreat (-3)
    back = walk(shaping, [5, 4, 3])
    assert np.isclose(back, 0.03)


def test_frontier_ignores_an_exit_it_cannot_reach_yet():
    grid = "##########\n#P.R.E...#\n##########"  # the exit is behind a red lock the agent has no key for
    shaping = FrontierShaping(Flat(), at(1), grid, 0.01)
    assert shaping.mode == "frontier"


def test_frontier_targets_only_exits_it_has_seen():
    grid = "###############\n#E.P..........E#\n###############"  # one exit in sight, one far away
    shaping = FrontierShaping(Flat(), at(3), grid, 0.01)
    assert shaping.mode == "exit" and shaping.distance == 2
    assert np.isclose(shaping.step(at(4))[0], -0.01)  # walking toward the unseen exit is not paid
