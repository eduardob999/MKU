#!/usr/bin/env python3
"""
Ivette launcher.

Usage:

    python ivette.py                  # launch the interactive app
    python ivette.py --show-defaults  # print every default parameter and exit

or after installation:

    python -m ivette
"""

import sys
from pathlib import Path


def main():
    """
    Add src/ to Python path and launch Ivette (or print defaults and exit).
    """

    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"

    sys.path.insert(
        0,
        str(src_path)
    )

    if "--show-defaults" in sys.argv[1:]:
        from ivette.core.params import format_defaults
        print(format_defaults())
        return

    from ivette.__main__ import main as ivette_main

    ivette_main()


if __name__ == "__main__":
    main()