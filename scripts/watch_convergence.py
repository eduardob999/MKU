#!/usr/bin/env python3
"""
watch_convergence.py
Watch a Gaussian 16 optimisation log in real time.
The screen redraws in place (like htop). Each convergence criterion gets
its own 2D bar chart (log y-axis, dashed threshold line, step number x-axis).
Bars turn yellow then green as they approach / cross the threshold.

Usage:
    python watch_convergence.py 10701.log
    python watch_convergence.py 10701.log --interval 3
    python watch_convergence.py 10701.log --once        # finished log, no loop
    python watch_convergence.py 10701.log --height 10   # taller plots
    python watch_convergence.py 10701.log --no-color
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── ANSI ──────────────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"

HOME         = "\033[H"    # cursor to top-left
CLEAR        = "\033[2J\033[H"
ERASE_DOWN   = "\033[J"    # erase from cursor to end of screen
HIDE_CURSOR  = "\033[?25l"
SHOW_CURSOR  = "\033[?25h"

_use_color = True

def c(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if _use_color else text


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Criterion:
    name:      str
    value:     float
    threshold: float
    converged: bool

    @property
    def ratio(self) -> float:
        return self.value / self.threshold if self.threshold else float("inf")


@dataclass
class OptStep:
    step:     int
    energy:   Optional[float]
    delta_e:  Optional[float]
    criteria: list[Criterion]

    @property
    def all_converged(self) -> bool:
        return all(cr.converged for cr in self.criteria)

    @property
    def n_converged(self) -> int:
        return sum(cr.converged for cr in self.criteria)


# ── Parser ────────────────────────────────────────────────────────────────────

_SCF_RE  = re.compile(r"SCF Done:\s+E\(\S+\)\s+=\s+([-\d.]+)")
_CONV_RE = re.compile(
    r"^\s*(Maximum Force|RMS\s+Force|Maximum Displacement|RMS\s+Displacement)"
    r"\s+([\d.]+)\s+([\d.]+)\s+(YES|NO)",
    re.MULTILINE,
)

_SHORT_NAME = {
    "Maximum Force":        "Max Force",
    "RMS Force":            "RMS Force",
    "Maximum Displacement": "Max Disp ",
    "RMS Displacement":     "RMS Disp ",
}

_CRIT_COLOR = [BLUE, CYAN, YELLOW, MAGENTA]


def parse_steps(text: str) -> list[OptStep]:
    energies = [float(m.group(1)) for m in _SCF_RE.finditer(text)]
    steps: list[OptStep] = []
    buf:   list[Criterion] = []
    step_no = 0

    for m in _CONV_RE.finditer(text):
        name      = re.sub(r"\s+", " ", m.group(1).strip())
        val       = float(m.group(2))
        thr       = float(m.group(3))
        converged = m.group(4) == "YES"
        buf.append(Criterion(name=name, value=val, threshold=thr, converged=converged))

        if len(buf) == 4:
            energy = energies[step_no] if step_no < len(energies) else None
            prev_e = energies[step_no - 1] if step_no > 0 else None
            delta  = (energy - prev_e) if energy is not None and prev_e is not None else None
            step_no += 1
            steps.append(OptStep(step=step_no, energy=energy, delta_e=delta, criteria=buf[:]))
            buf.clear()

    return steps


# ── 2D bar chart ──────────────────────────────────────────────────────────────

YLABEL_W = 9    # fixed width for y-axis label column


def bar_plot_2d(
    values:    list[float],
    threshold: float,
    width:     int,
    height:    int,
    color:     str,
) -> list[str]:
    """
    Return a list of terminal lines forming a 2D bar chart.

    - Log y-axis.  Threshold shown as a red dashed line spanning full width.
    - Bars coloured with *color* normally, YELLOW within 1.5× threshold,
      GREEN once below threshold.
    - Left-aligned: step 1 is the leftmost bar.  Future steps are blank space.
    - Each line is  YLABEL_W + 2 + width  visible characters wide.
    """
    valid = [v for v in values if v and v > 0]
    if not valid:
        return [" " * (YLABEL_W + 2 + width)]

    max_v   = max(valid) * 1.15
    min_v   = min(min(valid) * 0.5, threshold * 0.3)
    log_max = math.log10(max_v)
    log_min = math.log10(min_v)
    log_rng = max(log_max - log_min, 1e-9)
    log_thr = math.log10(threshold)

    def to_row(v: float) -> int:   # 0 = bottom, height = top
        lv = math.log10(max(v, min_v * 0.99))
        return max(0, min(height, round((lv - log_min) / log_rng * height)))

    thr_row = to_row(threshold)

    # sample values to *width* columns, left-aligned (right-pad with None)
    n = len(values)
    if n >= width:
        sampled: list[Optional[float]] = [
            values[round(i * (n - 1) / (width - 1))] if width > 1 else values[-1]
            for i in range(width)
        ]
    else:
        sampled = list(values) + [None] * (width - n)   # type: ignore[list-item]

    FULL = "█"
    DASH = "╌"

    # grid[row][col] = (char, color_str);  row 0 = bottom baseline
    grid: list[list[tuple[str, str]]] = [
        [(" ", "") for _ in range(width)] for _ in range(height + 1)
    ]

    for ci, v in enumerate(sampled):
        if v is None:
            continue
        bar_h = to_row(v)
        ratio = v / threshold
        if ratio <= 1.0:
            bar_col = GREEN
        elif ratio <= 1.5:
            bar_col = YELLOW
        else:
            bar_col = color
        for row in range(1, bar_h + 1):
            grid[row][ci] = (FULL, bar_col)

    # draw threshold line (dashed where bar doesn't reach, solid colour where it does)
    for ci in range(width):
        ch, bc = grid[thr_row][ci]
        if ch == FULL:
            grid[thr_row][ci] = (FULL, bc)   # keep bar colour, threshold implied
        else:
            grid[thr_row][ci] = (DASH, RED)

    lines: list[str] = []

    for row in range(height, -1, -1):   # top → bottom
        # y-axis label
        if row == height:
            yl = f"{max_v:.1e}"
            yl_str = c(DIM, f"{yl:>{YLABEL_W}}")
        elif row == thr_row:
            yl = f"{threshold:.1e}"
            yl_str = c(RED, f"{yl:>{YLABEL_W}}")
        elif row == 1:
            yl = f"{min_v:.1e}"
            yl_str = c(DIM, f"{yl:>{YLABEL_W}}")
        else:
            yl_str = " " * YLABEL_W

        # bar row
        bar_str = ""
        for ci in range(width):
            ch, bc = grid[row][ci]
            if row == 0:
                bar_str += c(DIM, "─")
            elif bc:
                bar_str += c(bc, ch)
            else:
                bar_str += ch

        lines.append(f"{yl_str} │{bar_str}")

    # x-axis step labels
    pad    = " " * (YLABEL_W + 2)
    s_end  = str(n)
    xaxis  = f"{pad}{c(DIM, '1')}{c(DIM, s_end.rjust(width - 1))}"
    lines.append(xaxis)

    return lines


# ── Renderer ──────────────────────────────────────────────────────────────────

def render(
    steps:    list[OptStep],
    log_path: str,
    interval: float,
    now:      str,
    finished: bool,
    error:    bool,
    height:   int,
) -> str:
    tw = shutil.get_terminal_size((100, 40)).columns
    # bar width = terminal width minus y-axis label + border
    bar_w = max(20, tw - YLABEL_W - 4)

    out: list[str] = [""]

    # ── Status bar ────────────────────────────────────────────────────────────
    if finished:
        status = c(GREEN + BOLD, "✓ Normal termination")
    elif error:
        status = c(RED + BOLD,   "✗ Error termination")
    else:
        status = c(CYAN,         "● Running")

    n_steps = len(steps)
    n_conv  = steps[-1].n_converged if steps else 0
    info    = (
        f"  {status}   "
        f"{c(BOLD, Path(log_path).name)}   "
        f"step {n_steps}   "
        f"converged {n_conv}/4   "
        f"polled {now}"
    )
    if interval > 0:
        info += c(DIM, f"  (every {interval:.0f}s — Ctrl-C to stop)")
    out.append(info)

    # ── Energy line ───────────────────────────────────────────────────────────
    if steps:
        last = steps[-1]
        if last.energy is not None:
            e_str = f"  E = {last.energy:.8f} Ha"
            if last.delta_e is not None:
                sign  = "+" if last.delta_e >= 0 else ""
                dcol  = RED if last.delta_e > 0 else GREEN
                e_str += c(dcol, f"   ΔE {sign}{last.delta_e:.3e} Ha")
            out.append(e_str)

    out.append("")

    # ── One plot per criterion ────────────────────────────────────────────────
    for ci, col in enumerate(_CRIT_COLOR):
        if not steps:
            break
        cr   = steps[-1].criteria[ci]
        name = _SHORT_NAME.get(cr.name, cr.name)

        # header line
        pct     = cr.ratio * 100
        pct_col = GREEN if cr.converged else (YELLOW if pct <= 150 else RED)
        yn      = c(GREEN + BOLD, "YES ✓") if cr.converged else c(RED, " NO")
        header  = (
            f"  {c(col + BOLD, name)}"
            f"   value {cr.value:.4e}"
            f"   threshold {cr.threshold:.4e}"
            f"   {c(pct_col, f'{pct:.1f}% of thr')}"
            f"   {yn}"
        )
        out.append(header)

        history = [s.criteria[ci].value for s in steps]
        for line in bar_plot_2d(history, cr.threshold, bar_w, height, col):
            out.append(line)

        out.append("")

    return "\n".join(out)


# ── Watch loop ────────────────────────────────────────────────────────────────

def watch(log_path: str, interval: float, height: int) -> None:
    path = Path(log_path)
    text = ""

    if _use_color:
        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.flush()

    def _restore(_sig=None, _frame=None):
        if _use_color:
            sys.stdout.write(SHOW_CURSOR + "\n")
            sys.stdout.flush()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _restore)
    signal.signal(signal.SIGTERM, _restore)

    first = True

    while True:
        finished = error = False

        if path.exists():
            new_text = path.read_text(errors="replace")
            if new_text != text:
                text = new_text
            finished = "Normal termination" in text
            error    = "Error termination"  in text or "Segmentation fault" in text

        steps  = parse_steps(text)
        now    = time.strftime("%H:%M:%S")
        screen = render(steps, str(path), interval, now, finished, error, height)

        sys.stdout.write(CLEAR if first else HOME)
        first = False
        sys.stdout.write(screen)
        sys.stdout.write(ERASE_DOWN)
        sys.stdout.flush()

        if finished or error:
            if _use_color:
                sys.stdout.write(SHOW_CURSOR + "\n")
            break

        time.sleep(interval)


def load_once(log_path: str, height: int) -> None:
    text     = Path(log_path).read_text(errors="replace")
    steps    = parse_steps(text)
    finished = "Normal termination" in text
    error    = "Error termination"  in text
    if not steps:
        print("No convergence data found.")
        return
    print(render(steps, log_path, 0, time.strftime("%H:%M:%S"), finished, error, height))


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Watch Gaussian 16 convergence — 2D bar charts per criterion"
    )
    parser.add_argument("log",           help="Path to Gaussian .log file")
    parser.add_argument("--interval",    type=float, default=5.0,
                        help="Poll interval in seconds (default: 5)")
    parser.add_argument("--height",      type=int,   default=7,
                        help="Plot height in terminal rows (default: 7)")
    parser.add_argument("--once",        action="store_true",
                        help="Print report for a completed log and exit")
    parser.add_argument("--no-color",    action="store_true",
                        help="Disable ANSI colour and cursor control")
    args = parser.parse_args(argv)

    global _use_color
    if args.no_color or not sys.stdout.isatty():
        _use_color = False

    if args.once:
        load_once(args.log, args.height)
    else:
        watch(args.log, args.interval, args.height)

    return 0


if __name__ == "__main__":
    sys.exit(main())
