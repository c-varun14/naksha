"""Macro action table: the agent picks one macro per step; it maps to the 9 official buttons.

Frozen interface (see reports/INTERFACES.md). Bump ACTIONS_VERSION on any change; weights record it.
Button order follows doom_env.spaces.BUTTONS:
  0 MOVE_FORWARD, 1 MOVE_BACKWARD, 2 MOVE_LEFT, 3 MOVE_RIGHT, 4 TURN_LEFT, 5 TURN_RIGHT,
  6 ATTACK, 7 USE, 8 SELECT_NEXT_WEAPON (never pressed).
"""

from __future__ import annotations

import numpy as np

ACTIONS_VERSION = 1

MACROS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("noop", ()),
    ("forward", (0,)),
    ("back", (1,)),
    ("strafe_left", (2,)),
    ("strafe_right", (3,)),
    ("turn_left", (4,)),
    ("turn_right", (5,)),
    ("forward_turn_left", (0, 4)),
    ("forward_turn_right", (0, 5)),
    ("attack", (6,)),
    ("attack_turn_left", (6, 4)),
    ("attack_turn_right", (6, 5)),
    ("forward_attack", (0, 6)),
    ("use", (7,)),
)
NUM_ACTIONS = len(MACROS)
NAMES = tuple(name for name, _ in MACROS)
INDEX = {name: i for i, name in enumerate(NAMES)}

_TABLE = np.zeros((NUM_ACTIONS, 9), np.int8)
for _i, (_, _buttons) in enumerate(MACROS):
    _TABLE[_i, list(_buttons)] = 1
_TABLE.setflags(write=False)


def to_buttons(macro: int) -> np.ndarray:
    """Official MultiBinary(9) action for a macro id (a copy, safe to hand to the env)."""
    return _TABLE[int(macro)].copy()


def table() -> np.ndarray:
    """Read-only (NUM_ACTIONS, 9) table, for vectorised use."""
    return _TABLE
