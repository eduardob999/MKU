"""Pytest bootstrap.

Insert ``src/`` ahead of everything on ``sys.path`` so ``import ivette`` resolves
to the package under ``src/ivette`` and not the root-level ``ivette.py`` launcher.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))
