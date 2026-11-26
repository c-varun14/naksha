"""Import helper: the inference package lives in submission/naksha so the agent is self-contained."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "submission", ROOT / "env", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
