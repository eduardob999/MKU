"""Rich + questionary UI toolkit for the Ivette CLI.

A single place for look-and-feel: a themed Rich console, an app header panel,
arrow-key menus (with a numbered fallback when there is no real terminal),
styled prompts, tables, panels, spinners and progress bars. Menus should talk
to the user only through this module so the styling stays consistent.

The whole session runs inside :func:`fullscreen` — the terminal's alternate
screen buffer — so the UI stays fixed in place (like ``vim``/``htop``) and the
original terminal contents are restored on exit.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

import questionary
from questionary import Separator
from questionary import Style as QStyle
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

ACCENT = "#5fd7ff"

THEME = Theme({
    "accent": f"bold {ACCENT}",
    "heading": "bold white",
    "success": "bold green",
    "warn": "bold yellow",
    "error": "bold red",
    "muted": "grey62",
})

console = Console(theme=THEME)

# questionary widgets styled to match the Rich accent.
_QSTYLE = QStyle([
    ("qmark", f"fg:{ACCENT} bold"),
    ("question", "bold"),
    ("pointer", f"fg:{ACCENT} bold"),
    ("highlighted", f"fg:{ACCENT} bold"),
    ("selected", f"fg:{ACCENT}"),
    ("answer", f"fg:{ACCENT} bold"),
    ("instruction", "fg:#808080 italic"),
    ("separator", "fg:#d7af5f bold"),   # section dividers — amber, distinct from cyan selection
])

# Sentinel returned by select() when the user cancels (Esc / Ctrl-C).
CANCEL = object()

# questionary can map at most this many list items to keyboard shortcuts
# (digits 1-9,0 then a-z). Past this we silently drop number shortcuts and the
# user navigates with the arrow keys.
_MAX_SHORTCUTS = 36

# Colour used for section headers in the non-interactive fallback list.
_SECTION_COLOR = "#d7af5f"


class _Section:
    """A non-selectable header/divider for use inside :func:`select` choices."""
    __slots__ = ("title",)

    def __init__(self, title):
        self.title = title


def section(title):
    """A visible section divider to interleave with ``(label, value)`` choices.

    Drop one into the list passed to :func:`select` to group the options
    underneath it, e.g. ``[ui.section("Gaussian"), ("Run opt", "opt"), …]``.
    """
    return _Section(title)


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


# ---------------------------------------------------------------------------
# Screen / header
# ---------------------------------------------------------------------------

@contextmanager
def fullscreen():
    """Run the app inside the terminal's alternate screen buffer.

    Keeps the interface fixed in place and restores the user's terminal (and
    cursor) on exit. A no-op when stdout isn't a real terminal.
    """
    use_alt = _interactive()
    if use_alt:
        console.set_alt_screen(True)
    try:
        yield
    finally:
        if use_alt:
            console.set_alt_screen(False)
        console.show_cursor(True)


def clear():
    console.clear()


def banner():
    """The application splash shown on the top-level screen."""
    title = Text()
    title.append("◆ ", style="accent")
    title.append("IVETTE", style="bold white")
    body = Text("chemoinformatics workbench", style="muted")
    console.print(
        Panel(body, title=title, border_style="accent",
              box=box.DOUBLE, padding=(1, 4), expand=False)
    )


def header(context):
    """Render the context header (mode + breadcrumbs + info) as a panel."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="muted", justify="right", no_wrap=True)
    grid.add_column(style="white")
    grid.add_row("Mode", f"[accent]{context.mode}[/accent]")
    if context.active_set:
        grid.add_row("Set", str(context.active_set))
    if context.active_compound_set:
        grid.add_row("Compounds", str(context.active_compound_set))
    if context.active_run:
        grid.add_row("Run", str(context.active_run))
    for key, value in context.info.items():
        grid.add_row(str(key), str(value))

    title = Text()
    title.append("◆ ", style="accent")
    title.append("IVETTE", style="bold white")
    console.print(
        Panel(grid, title=title, title_align="left",
              border_style="accent", box=box.ROUNDED, padding=(1, 2))
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print(*args, **kwargs):  # noqa: A001 - intentional console.print shim
    console.print(*args, **kwargs)


def rule(title=""):
    console.rule(f"[accent]{title}[/accent]" if title else "", style="accent")


def success(message):
    console.print(f"[success]✔[/success] {message}")


def warn(message):
    console.print(f"[warn]![/warn] {message}")


def error(message):
    console.print(f"[error]✘[/error] {message}")


def info(message):
    console.print(f"[accent]ℹ[/accent] {message}")


def note(message):
    console.print(f"[muted]{message}[/muted]")


def panel(renderable, *, title=None, border_style="accent", subtitle=None):
    console.print(
        Panel(renderable, title=title, subtitle=subtitle,
              border_style=border_style, box=box.ROUNDED, padding=(1, 2))
    )


def table(columns, rows, *, title=None, caption=None):
    """Build and print a Rich table.

    ``columns`` is a list of names or (name, style) tuples; ``rows`` is a list
    of row sequences (cells are coerced to str).
    """
    t = Table(title=title, caption=caption, box=box.SIMPLE_HEAVY,
              header_style="accent", title_style="heading",
              expand=False, pad_edge=False)
    for col in columns:
        if isinstance(col, tuple):
            name, style = col
            t.add_column(name, style=style, overflow="fold")
        else:
            t.add_column(col, overflow="fold")
    for row in rows:
        t.add_row(*[str(c) for c in row])
    console.print(t)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def select(message, choices, *, default=None):
    """Arrow- and number-key menu.

    ``choices`` is a list of ``(label, value)`` pairs, optionally interleaved
    with :func:`section` dividers to group the options into visible sections.
    The user can move the highlight with ↑/↓ **or** by pressing the number/letter
    shown in front of each option, then Enter to confirm.

    Returns the selected value, or :data:`CANCEL` if the user backs out. Falls
    back to a numbered prompt when stdin/stdout isn't a TTY.
    """
    items = list(choices)
    # Real, selectable options (section dividers excluded), in display order.
    pairs = [it for it in items if not isinstance(it, _Section)]

    if _interactive():
        qchoices, idx = [], 0
        for it in items:
            if isinstance(it, _Section):
                line = f"── {it.title} ──" if it.title else "─" * 12
                qchoices.append(Separator(line))
            else:
                qchoices.append(questionary.Choice(title=it[0], value=idx))
                idx += 1
        answer = questionary.select(
            message,
            choices=qchoices,
            default=default,
            style=_QSTYLE,
            qmark="❯",
            pointer="▶",
            instruction="(↑/↓ or number, then Enter)",
            # Number/letter shortcuts when the list is short enough to map them;
            # j/k vim-nav off so it never clashes with auto-assigned letter keys.
            use_shortcuts=len(pairs) <= _MAX_SHORTCUTS,
            use_arrow_keys=True,
            use_jk_keys=False,
            use_indicator=False,
        ).ask()
        if answer is None:
            return CANCEL
        return pairs[answer][1]

    # Non-interactive fallback: numbered list with section headers.
    console.print(f"[accent]?[/accent] [bold]{message}[/bold]")
    n = 0
    for it in items:
        if isinstance(it, _Section):
            line = f"── {it.title} ──" if it.title else "─" * 12
            console.print(f"   [{_SECTION_COLOR}]{line}[/{_SECTION_COLOR}]")
        else:
            n += 1
            console.print(f"   [accent]{n:>2}[/accent]  {it[0]}")
    raw = input("Select a number: ").strip()
    try:
        idx = int(raw) - 1
    except ValueError:
        return CANCEL
    if 0 <= idx < len(pairs):
        return pairs[idx][1]
    return CANCEL


def _is_number(text, cast):
    try:
        cast(text)
        return True
    except ValueError:
        return False


def ask_text(message, default=None):
    if _interactive():
        ans = questionary.text(
            message, default="" if default is None else str(default),
            style=_QSTYLE, qmark="❯",
        ).ask()
        if ans is None:
            return default
        ans = ans.strip()
        return ans if ans else (default if default is not None else "")
    raw = input(f"  {message}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    return raw if raw else (default if default is not None else "")


def _ask_number(message, default, cast):
    if _interactive():
        ans = questionary.text(
            message,
            default=str(default),
            validate=lambda t: t.strip() == "" or _is_number(t, cast)
            or f"Enter a valid {cast.__name__}",
            style=_QSTYLE, qmark="❯",
        ).ask()
        if ans is None or ans.strip() == "":
            return default
        return cast(ans.strip())
    while True:
        raw = input(f"  {message} [{default}]: ").strip()
        if raw == "":
            return default
        if _is_number(raw, cast):
            return cast(raw)
        console.print(f"[warn]  ! expected {cast.__name__}[/warn]")


def ask_int(message, default):
    return _ask_number(message, default, int)


def ask_float(message, default):
    return _ask_number(message, default, float)


def confirm(message, default=True):
    if _interactive():
        ans = questionary.confirm(message, default=default, style=_QSTYLE, qmark="❯").ask()
        return default if ans is None else ans
    hint = "Y/n" if default else "y/N"
    raw = input(f"  {message} [{hint}]: ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes")


def pause(message="Press Enter to continue"):
    if _interactive():
        questionary.press_any_key_to_continue(f"{message} …", style=_QSTYLE).ask()
    else:
        input(f"{message}... ")


# ---------------------------------------------------------------------------
# Progress / status
# ---------------------------------------------------------------------------

def status(message):
    """Spinner context manager for an indeterminate wait."""
    return console.status(f"[accent]{message}[/accent]", spinner="dots")


def progress():
    """A configured :class:`rich.progress.Progress` for batch loops."""
    return Progress(
        SpinnerColumn(style="accent"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40, complete_style="accent", finished_style="success"),
        MofNCompleteColumn(),
        TextColumn("[muted]elapsed[/muted]"),
        TimeElapsedColumn(),
        TextColumn("[muted]eta[/muted]"),
        TimeRemainingColumn(),
        console=console,
    )
