#!/usr/bin/env python3
"""
watch_convergence_matplotlib.py

Live Matplotlib monitor for Gaussian 16 optimization convergence.

Usage:
    python watch_convergence_matplotlib.py calc.log
    python watch_convergence_matplotlib.py calc.log --interval 5
    python watch_convergence_matplotlib.py calc.log --once
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

# Force a GUI backend instead of Agg
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


@dataclass
class Criterion:
    name: str
    value: float
    threshold: float
    converged: bool

    @property
    def ratio(self):
        return self.value / self.threshold


@dataclass
class OptStep:
    step: int
    energy: Optional[float]
    criteria: list[Criterion]

    @property
    def n_converged(self):
        return sum(x.converged for x in self.criteria)


_SCF_RE = re.compile(r"SCF Done:\s+E\(\S+\)\s+=\s+([-\d.]+)")
_CONV_RE = re.compile(
    r"^\s*(Maximum Force|RMS\s+Force|Maximum Displacement|RMS\s+Displacement)"
    r"\s+([\d.]+)\s+([\d.]+)\s+(YES|NO)",
    re.MULTILINE,
)

NAMES = [
    "Maximum Force",
    "RMS Force",
    "Maximum Displacement",
    "RMS Displacement",
]


def parse_steps(text):
    energies = [float(x.group(1)) for x in _SCF_RE.finditer(text)]

    steps = []
    buf = []

    for m in _CONV_RE.finditer(text):
        buf.append(
            Criterion(
                m.group(1),
                float(m.group(2)),
                float(m.group(3)),
                m.group(4) == "YES",
            )
        )

        if len(buf) == 4:
            i = len(steps)
            steps.append(
                OptStep(
                    i + 1,
                    energies[i] if i < len(energies) else None,
                    buf[:],
                )
            )
            buf.clear()

    return steps


class GaussianViewer:
    def __init__(self, logfile, interval):
        self.path = Path(logfile)
        self.interval = interval
        self.last_text = ""

        self.fig, self.axs = plt.subplots(
            2, 2, figsize=(12, 8)
        )
        self.axs = self.axs.flatten()

        self.fig.canvas.manager.set_window_title(
            f"Gaussian convergence - {self.path.name}"
        )

        self.animation = FuncAnimation(
            self.fig,
            self.update,
            interval=interval * 1000,
            cache_frame_data=False,
        )

    def update(self, _):
        if not self.path.exists():
            return

        text = self.path.read_text(errors="replace")

        if text == self.last_text:
            return

        self.last_text = text
        steps = parse_steps(text)

        for ax in self.axs:
            ax.clear()

        if not steps:
            self.fig.suptitle("Waiting for optimization data...")
            return

        for i, ax in enumerate(self.axs):
            vals = [s.criteria[i].value for s in steps]
            thr = steps[-1].criteria[i].threshold

            x = range(1, len(vals) + 1)

            colors = [
                "green" if v <= thr
                else "gold" if v <= 1.5 * thr
                else "steelblue"
                for v in vals
            ]

            ax.bar(x, vals, color=colors)
            ax.axhline(
                thr,
                linestyle="--",
                color="red",
                label="threshold",
            )

            ax.set_yscale("log")
            ax.set_title(
                steps[-1].criteria[i].name
            )
            ax.set_xlabel("Optimization step")
            ax.set_ylabel("Value")
            ax.legend()

        last = steps[-1]

        status = (
            "✓ Normal termination"
            if "Normal termination" in text
            else "✗ Error termination"
            if "Error termination" in text
            else "● Running"
        )

        energy = ""
        if last.energy is not None:
            energy = f" | E = {last.energy:.8f} Ha"

        self.fig.suptitle(
            f"{status} | step {last.step} | "
            f"converged {last.n_converged}/4"
            f"{energy}\n"
            f"Updated {time.strftime('%H:%M:%S')}"
        )

        self.fig.tight_layout()

    def show(self):
        # Keep a hard reference to the animation object.
        # Otherwise matplotlib garbage-collects it.
        self.animation._start()

        plt.show(block=True)


def once(logfile):
    text = Path(logfile).read_text(errors="replace")
    steps = parse_steps(text)

    if not steps:
        print("No convergence data found.")
        return

    for s in steps:
        print(
            s.step,
            [f"{c.value:.3e}" for c in s.criteria]
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument(
        "--interval",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--once",
        action="store_true",
    )

    args = parser.parse_args()

    if args.once:
        once(args.log)
    else:
        GaussianViewer(
            args.log,
            args.interval,
        ).show()


if __name__ == "__main__":
    main()
