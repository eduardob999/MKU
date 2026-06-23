"""Shared CLI context and header rendering.

The single source of truth for the interactive session state — previously
duplicated between ``__main__`` and the orphaned ``cli/header.py``.
"""


class IvetteContext:
    """Mutable state shown in the header of every menu screen."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.mode = "Structure Sets"
        self.active_set = None
        self.active_compound_set = None
        self.active_run = None
        self.info = {}


context = IvetteContext()


def render_header():
    print()
    print("=" * 60)
    print("IVETTE")
    print(f"Mode: {context.mode}")
    if context.active_set:
        print(f"Set: {context.active_set}")
    if context.active_compound_set:
        print(f"Compounds: {context.active_compound_set}")
    if context.active_run:
        print(f"Run: {context.active_run}")
    for key, value in context.info.items():
        print(f"{key}: {value}")
    print("=" * 60)
    print()
