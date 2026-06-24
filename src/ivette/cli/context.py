"""Shared CLI context and header rendering.

The single source of truth for the interactive session state. Header rendering
is delegated to :mod:`ivette.cli.ui` so all styling lives in one place.
"""

from ivette.cli import ui


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


def render_header(clear=True):
    """Render the context header panel (optionally clearing the screen first)."""
    if clear:
        ui.clear()
    ui.header(context)
