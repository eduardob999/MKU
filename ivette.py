#!/usr/bin/env python3
"""
Ivette launcher.

Usage:

    python ivette.py

or after installation:

    python -m ivette
"""

import sys
from pathlib import Path


def main():
    """
    Add src/ to Python path and launch Ivette.
    """

    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"

    sys.path.insert(
        0,
        str(src_path)
    )

    from ivette.__main__ import main as ivette_main

    ivette_main()


if __name__ == "__main__":
    main()