"""Make both the package imports and legacy test imports available to pytest."""

import sys
from pathlib import Path

project = Path(__file__).resolve().parents[1]
for path in (project, project / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
