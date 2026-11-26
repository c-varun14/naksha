"""Map compiler symbols beyond the organiser test layouts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.gen.compile import compile_grid  # noqa: E402


def test_barrel_symbol_compiles_to_an_explosive_barrel():
    text = compile_grid("#######\n#P.x.E#\n#######")
    assert "type = 2035;" in text
