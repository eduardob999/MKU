#!/usr/bin/env python3
"""
control_room.py

A non-interactive Matplotlib "control room" for the Ivette Gaussian pipeline.
One fixed window, several panels, refreshed live — focused on performance:

    ┌ CPU thread scaling ┬ Preopt: worth it? ┬ Batch progress ──────┐
    ├ Live opt convergence ┬ Frequency progress ┬ Thermochemistry ──┤
    └────────────────────────────────────────────────────────────────┘

Performance note: each panel fingerprints its own data files (path + mtime +
size) and is only re-parsed/redrawn when that data actually changes, so an idle
panel costs nothing per tick.

Data sources (all under --data-dir, default <repo>/data/sdfs):
    gaussian_benchmark.json              CPU scaling + preopt comparison
    **/gaussian/**/checkpoint.json       batch progress / throughput
    **/<cid>_opt.log                     optimization convergence
    **/<cid>_freq.log                    frequency progress + thermochemistry

Usage:
    python scripts/control_room.py                 # live, auto-refresh
    python scripts/control_room.py --interval 3
    python scripts/control_room.py --once          # render one PNG and exit
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

# Pick a backend before importing pyplot: Agg for one-shot PNG (headless safe),
# an interactive GUI backend otherwise.
_ONESHOT = "--once" in sys.argv or "--save" in sys.argv
if _ONESHOT:
    matplotlib.use("Agg")
else:
    for _backend in ("QtAgg", "TkAgg"):
        try:
            matplotlib.use(_backend)
            break
        except Exception:
            continue

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ──────────────────────────────────────────────────────────────────────────────
# Theme
# ──────────────────────────────────────────────────────────────────────────────

BG = "#0f1419"
PANEL = "#161b22"
FG = "#c9d1d9"
MUTED = "#8b949e"
ACCENT = "#5fd7ff"
GREEN = "#3fb950"
RED = "#f85149"
GOLD = "#d29922"
PURPLE = "#bc8cff"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": MUTED,
    "axes.titlecolor": FG,
    "text.color": FG,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": "#21262d",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
})


# ──────────────────────────────────────────────────────────────────────────────
# Parsers (self-contained — only matplotlib is required)
# ──────────────────────────────────────────────────────────────────────────────

CONV_NAMES = ["Maximum Force", "RMS Force", "Maximum Displacement", "RMS Displacement"]
CONV_COLORS = [ACCENT, GREEN, GOLD, PURPLE]

_SCF_RE = re.compile(r"SCF Done:\s+E\(\S+\)\s+=\s+([-\d.]+)")
_CONV_RE = re.compile(
    r"^\s*(Maximum Force|RMS\s+Force|Maximum Displacement|RMS\s+Displacement)"
    r"\s+([\d.]+)\s+([\d.]+)\s+(YES|NO)",
    re.MULTILINE,
)
_FREQ_RE = re.compile(r"Frequencies --\s+([-\d.\s]+)")
_NATOMS_RE = re.compile(r"NAtoms=\s+(\d+)")


@dataclass
class Criterion:
    name: str
    value: float
    threshold: float
    converged: bool


@dataclass
class OptStep:
    step: int
    energy: Optional[float]
    criteria: list[Criterion]

    @property
    def n_converged(self) -> int:
        return sum(c.converged for c in self.criteria)


def parse_opt_steps(text: str) -> list[OptStep]:
    energies = [float(m.group(1)) for m in _SCF_RE.finditer(text)]
    steps: list[OptStep] = []
    buf: list[Criterion] = []
    for m in _CONV_RE.finditer(text):
        buf.append(Criterion(m.group(1), float(m.group(2)), float(m.group(3)), m.group(4) == "YES"))
        if len(buf) == 4:
            i = len(steps)
            steps.append(OptStep(i + 1, energies[i] if i < len(energies) else None, buf[:]))
            buf.clear()
    return steps


def parse_frequencies(text: str) -> list[float]:
    freqs: list[float] = []
    for m in _FREQ_RE.finditer(text):
        for tok in m.group(1).split():
            try:
                freqs.append(float(tok))
            except ValueError:
                pass
    return freqs


def parse_natoms(text: str) -> Optional[int]:
    m = _NATOMS_RE.search(text)
    return int(m.group(1)) if m else None


def _grab(text: str, label: str) -> Optional[float]:
    m = re.search(re.escape(label) + r"\s*([-\d.]+)", text)
    return float(m.group(1)) if m else None


def parse_thermo(text: str) -> dict:
    out = {
        "temp": _grab(text, "Temperature"),
        "zpe": _grab(text, "Zero-point correction="),
        "te": _grab(text, "Thermal correction to Energy="),
        "th": _grab(text, "Thermal correction to Enthalpy="),
        "tg": _grab(text, "Thermal correction to Gibbs Free Energy="),
        "e_zpe": _grab(text, "Sum of electronic and zero-point Energies="),
        "h": _grab(text, "Sum of electronic and thermal Enthalpies="),
        "g": _grab(text, "Sum of electronic and thermal Free Energies="),
    }
    m = re.search(r"^\s*Total\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text, re.MULTILINE)
    out["entropy"] = float(m.group(3)) if m else None
    return out


def termination(text: str) -> str:
    if "Normal termination" in text:
        return "done"
    if "Error termination" in text:
        return "error"
    return "running"


# ──────────────────────────────────────────────────────────────────────────────
# Data discovery
# ──────────────────────────────────────────────────────────────────────────────

def fp(path: Optional[Path]):
    """Fingerprint a file by (path, mtime, size); None if absent."""
    if not path or not path.exists():
        return None
    st = path.stat()
    return (str(path), st.st_mtime, st.st_size)


def latest(root: Path, pattern: str) -> Optional[Path]:
    """Most-recently-modified file matching ``pattern`` under ``root``."""
    best, best_mtime = None, -1.0
    for p in root.rglob(pattern):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if mtime > best_mtime:
            best, best_mtime = p, mtime
    return best


def latest_finished_freq(root: Path) -> Optional[Path]:
    """Most-recent *_freq.log that has thermochemistry (Normal termination)."""
    candidates = sorted(root.rglob("*_freq.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates[:20]:
        try:
            tail = p.read_text(errors="replace")[-6000:]
        except OSError:
            continue
        if "Normal termination" in tail and "Free Energies" in tail:
            return p
    return candidates[0] if candidates else None


# ──────────────────────────────────────────────────────────────────────────────
# Panels
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Panel:
    ax: plt.Axes
    root: Path
    bench: Path
    _fp: object = None
    _drawn: bool = False

    def fingerprint(self):
        return None

    def render(self):
        ...

    def tick(self) -> bool:
        """Re-render only if the underlying data changed. Returns True if drawn."""
        cur = self.fingerprint()
        if self._drawn and cur == self._fp:
            return False
        self._fp = cur
        self.ax.clear()
        try:
            self.render()
        except Exception as exc:  # never let one panel kill the dashboard
            self._empty(f"error: {exc}")
        self._drawn = True
        return True

    # -- shared helpers ----------------------------------------------------
    def _empty(self, message: str):
        self.ax.set_axis_off()
        self.ax.text(0.5, 0.5, message, ha="center", va="center",
                     color=MUTED, style="italic", transform=self.ax.transAxes)

    def _title(self, text: str):
        self.ax.set_title(text, loc="left")


class CpuScalingPanel(Panel):
    def fingerprint(self):
        return fp(self.bench)

    def render(self):
        rows = _benchmark_rows(self.bench, "threads")
        if not rows:
            self._title("CPU thread scaling")
            return self._empty("no thread-scaling benchmark yet")
        rows = sorted(rows, key=lambda r: r["threads"])
        threads = [r["threads"] for r in rows]
        secs = [r["run_seconds"] for r in rows]

        ax = self.ax
        ax.plot(threads, secs, "-o", color=ACCENT, lw=2, label="wall time")
        best_i = min(range(len(secs)), key=lambda i: secs[i])
        ax.scatter([threads[best_i]], [secs[best_i]], s=120, facecolor="none",
                   edgecolor=GREEN, lw=2, zorder=5)
        ax.annotate(f"optimal\n{threads[best_i]} threads", (threads[best_i], secs[best_i]),
                    textcoords="offset points", xytext=(6, 10), color=GREEN, fontsize=8)
        ax.set_xlabel("threads (%nprocshared)")
        ax.set_ylabel("seconds", color=ACCENT)
        ax.grid(True, alpha=0.3)

        spd = ax.twinx()
        speedup = [secs[0] / s for s in secs]
        spd.plot(threads, speedup, "--", color=MUTED, lw=1.2, label="speedup")
        spd.plot(threads, [t / threads[0] for t in threads], ":", color="#444c56",
                 lw=1, label="ideal")
        spd.set_ylabel("speedup ×", color=MUTED)
        spd.tick_params(colors=MUTED)
        self._title("CPU thread scaling")


class PreoptPanel(Panel):
    def fingerprint(self):
        return fp(self.bench)

    def render(self):
        rows = _benchmark_rows(self.bench, "preopt")
        self._title("Preopt — is it worth it?")
        if not rows:
            return self._empty("no preopt benchmark yet")
        # newest entry's rows already; order none/pm7/6-31G as found
        labels = [r.get("preopt_mode", "?") for r in rows]
        pre = [r.get("preopt_seconds") or 0.0 for r in rows]
        dft = [r.get("dft_seconds") for r in rows]
        has_dft = any(d is not None for d in dft)
        x = range(len(rows))

        ax = self.ax
        ax.bar(x, pre, color=MUTED, label="preopt")
        if has_dft:
            dft0 = [d or 0.0 for d in dft]
            ax.bar(x, dft0, bottom=pre, color=ACCENT, label="DFT opt+freq")
            totals = [p + d for p, d in zip(pre, dft0)]
            best = min(range(len(totals)), key=lambda i: totals[i] if rows[i].get("success") else 1e18)
            for i, r in enumerate(rows):
                steps = r.get("opt_steps")
                tag = f"{totals[i]:.0f}s"
                if steps is not None:
                    tag += f"\n{steps} steps"
                ax.text(i, totals[i], tag, ha="center", va="bottom", fontsize=8,
                        color=GREEN if i == best else FG)
            ax.bar([best], [totals[best]], facecolor="none", edgecolor=GREEN, lw=2)
            ax.set_ylabel("total wall time (s)")
        else:
            ax.set_ylabel("preopt time (s)")
            ax.text(0.5, 0.92, "DFT-per-mode not measured yet — re-run benchmark",
                    transform=ax.transAxes, ha="center", color=GOLD, fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.2)


class BatchPanel(Panel):
    def _checkpoint(self) -> Optional[Path]:
        return latest(self.root, "checkpoint.json")

    def fingerprint(self):
        return fp(self._checkpoint())

    def render(self):
        self._title("Batch progress")
        ck = self._checkpoint()
        if not ck:
            return self._empty("no active batch")
        counts, total, throughput, eta = _batch_stats(ck)
        done, failed, running = counts["done"], counts["failed"], counts["running"]
        not_started = max(0, total - done - failed - running)

        ax = self.ax
        segs = [("done", done, GREEN), ("failed", failed, RED),
                ("running", running, GOLD), ("queued", not_started, "#30363d")]
        left = 0
        for name, val, color in segs:
            if val:
                ax.barh(0, val, left=left, color=color, label=f"{name} {val}")
                left += val
        ax.set_xlim(0, max(total, 1))
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(loc="lower center", ncol=4, fontsize=7, framealpha=0.2,
                  bbox_to_anchor=(0.5, -0.35))

        pct = 100 * done / total if total else 0
        sub = f"{done}/{total} done ({pct:.0f}%)"
        if throughput:
            sub += f"  ·  {throughput:.1f} mol/h"
        if eta:
            sub += f"  ·  ETA {_fmt_dur(eta)}"
        ax.text(0.5, 0.62, sub, transform=ax.transAxes, ha="center", color=FG, fontsize=9)
        ax.text(0.5, 1.02, ck.parent.relative_to(self.root).as_posix(),
                transform=ax.transAxes, ha="center", color=MUTED, fontsize=7)


class OptConvergencePanel(Panel):
    def _log(self) -> Optional[Path]:
        return latest(self.root, "*_opt.log") or latest(self.root, "*.log")

    def fingerprint(self):
        return fp(self._log())

    def render(self):
        log = self._log()
        if not log:
            self._title("Opt convergence")
            return self._empty("no optimization running")
        steps = parse_opt_steps(log.read_text(errors="replace"))
        cid = log.stem.replace("_opt", "")
        if not steps:
            self._title(f"Opt convergence · {cid}")
            return self._empty("waiting for first step…")

        ax = self.ax
        xs = range(1, len(steps) + 1)
        for i, (name, color) in enumerate(zip(CONV_NAMES, CONV_COLORS)):
            ratio = [s.criteria[i].value / s.criteria[i].threshold for s in steps]
            ax.plot(xs, ratio, "-o", ms=3, color=color, lw=1.4, label=name)
        ax.axhline(1.0, ls="--", color=RED, lw=1, label="threshold")
        ax.set_yscale("log")
        ax.set_xlabel("optimization step")
        ax.set_ylabel("value / threshold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.2)

        last = steps[-1]
        e = f"  E={last.energy:.6f}" if last.energy is not None else ""
        self._title(f"Opt · {cid} · step {last.step} · {last.n_converged}/4 converged{e}")


class FreqProgressPanel(Panel):
    def _log(self) -> Optional[Path]:
        return latest(self.root, "*_freq.log")

    def fingerprint(self):
        return fp(self._log())

    def render(self):
        self.ax.set_axis_off()
        log = self._log()
        if not log:
            self._title("Frequency progress")
            return self._empty("no frequency job")
        text = log.read_text(errors="replace")
        freqs = parse_frequencies(text)
        natoms = parse_natoms(text)
        total = (3 * natoms - 6) if natoms else None
        done = len(freqs)
        imaginary = sum(1 for f in freqs if f < 0)
        state = termination(text)
        cid = log.stem.replace("_freq", "")

        frac = (done / total) if total else (1.0 if state == "done" else 0.0)
        frac = min(max(frac, 0.0), 1.0)
        ax = self.ax
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        bar_color = RED if imaginary else (GREEN if state == "done" else ACCENT)
        ax.add_patch(plt.Rectangle((0.05, 0.45), 0.9, 0.16, color="#30363d"))
        ax.add_patch(plt.Rectangle((0.05, 0.45), 0.9 * frac, 0.16, color=bar_color))
        modes = f"{done}/{total}" if total else f"{done}"
        ax.text(0.5, 0.53, f"{frac*100:.0f}%", ha="center", va="center",
                color=BG if frac > 0.15 else FG, fontsize=11, fontweight="bold")
        ax.text(0.05, 0.72, f"modes computed: {modes}", color=FG, fontsize=9)
        status_txt = {"done": "✓ normal termination", "error": "✗ error termination",
                      "running": "● running"}[state]
        status_col = {"done": GREEN, "error": RED, "running": GOLD}[state]
        ax.text(0.05, 0.28, status_txt, color=status_col, fontsize=9)
        if imaginary:
            ax.text(0.05, 0.12, f"⚠ {imaginary} imaginary frequency(ies) — saddle point",
                    color=RED, fontsize=8)
        elif freqs:
            ax.text(0.05, 0.12, f"lowest mode {min(freqs):.1f} cm⁻¹ — true minimum",
                    color=MUTED, fontsize=8)
        self._title(f"Frequency · {cid}")


class ThermoPanel(Panel):
    def _log(self) -> Optional[Path]:
        return latest_finished_freq(self.root)

    def fingerprint(self):
        return fp(self._log())

    def render(self):
        self.ax.set_axis_off()
        self._title("Thermochemistry")
        log = self._log()
        if not log:
            return self._empty("no completed frequency job")
        t = parse_thermo(log.read_text(errors="replace"))
        cid = log.stem.replace("_freq", "")
        rows = [
            ("Temperature", t["temp"], "K"),
            ("Zero-point corr.", t["zpe"], "Ha"),
            ("Thermal→Energy", t["te"], "Ha"),
            ("Thermal→Enthalpy", t["th"], "Ha"),
            ("Thermal→Gibbs", t["tg"], "Ha"),
            ("E + ZPE", t["e_zpe"], "Ha"),
            ("Enthalpy H", t["h"], "Ha"),
            ("Gibbs G", t["g"], "Ha"),
            ("Entropy S", t["entropy"], "cal/mol·K"),
        ]
        y = 0.86
        self.ax.text(0.03, 0.97, cid, color=ACCENT, fontsize=9, fontweight="bold",
                     transform=self.ax.transAxes)
        for label, val, unit in rows:
            shown = f"{val:.6f}" if isinstance(val, float) and abs(val) < 1e4 else (
                f"{val:.3f}" if isinstance(val, float) else "—")
            self.ax.text(0.03, y, label, color=MUTED, fontsize=8, transform=self.ax.transAxes)
            self.ax.text(0.62, y, shown, color=FG, fontsize=8, transform=self.ax.transAxes,
                         family="monospace")
            self.ax.text(0.93, y, unit, color=MUTED, fontsize=7, transform=self.ax.transAxes,
                         ha="right")
            y -= 0.095


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark / batch helpers
# ──────────────────────────────────────────────────────────────────────────────

def _benchmark_rows(bench: Path, stage: str) -> list[dict]:
    """Rows from the newest benchmark entry whose key contains stage=<stage>."""
    if not bench.exists():
        return []
    try:
        data = json.loads(bench.read_text()).get("benchmarks", {})
    except (OSError, json.JSONDecodeError):
        return []
    entries = [(k, v) for k, v in data.items() if f"stage={stage}" in k]
    if not entries:
        return []
    entries.sort(key=lambda kv: kv[1].get("updated", ""), reverse=True)
    return entries[0][1].get("runs", [])


def _parse_iso(s: str) -> Optional[float]:
    try:
        return datetime.fromisoformat(s).timestamp()
    except (TypeError, ValueError):
        return None


def _batch_stats(ck: Path):
    """Return (Counter(done/failed/running), total, throughput_per_h, eta_seconds)."""
    try:
        data = json.loads(ck.read_text())
    except (OSError, json.JSONDecodeError):
        return Counter(), 0, None, None

    times: list[float] = []
    if isinstance(data.get("jobs"), dict):
        jobs = data["jobs"]
        created = _parse_iso(data.get("created", ""))
        if created:
            times.append(created)

        def classify(v):
            s = (v.get("status", "") if isinstance(v, dict) else "").lower()
            if s in {"completed", "done", "success", "finished", "ok"}:
                return "done"
            if s in {"failed", "error", "aborted"}:
                return "failed"
            return "running"
        for v in jobs.values():
            ts = _parse_iso(v.get("updated", "")) if isinstance(v, dict) else None
            if ts:
                times.append(ts)
    else:
        jobs = data

        def classify(v):
            return "done" if (isinstance(v, dict) and v.get("success")) else "failed"

    counts = Counter(classify(v) for v in jobs.values())
    # total molecules = SDFs in the set dir (runs/<sdf_id>), falling back to jobs seen
    sdf_dir = ck.parents[2] if len(ck.parents) >= 3 else ck.parent
    total = len(list(sdf_dir.glob("*.sdf"))) or len(jobs)

    throughput = eta = None
    done = counts["done"]
    if done >= 1 and len(times) >= 2:
        span = max(times) - min(times)
        if span > 0:
            rate = done / span               # molecules per second
            throughput = rate * 3600.0
            pending = max(0, total - done - counts["failed"])
            eta = pending / rate if rate else None
    return counts, total, throughput, eta


def _fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ──────────────────────────────────────────────────────────────────────────────
# Control room
# ──────────────────────────────────────────────────────────────────────────────

class ControlRoom:
    def __init__(self, data_root: Path, interval: float):
        self.root = data_root
        self.bench = data_root / "gaussian_benchmark.json"
        self.interval = interval

        self.fig = plt.figure(figsize=(16, 9), constrained_layout=True)
        self.fig.canvas.manager.set_window_title("Ivette — Control Room")
        gs = self.fig.add_gridspec(2, 3)
        axes = [self.fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

        self.panels = [
            CpuScalingPanel(axes[0], data_root, self.bench),
            PreoptPanel(axes[1], data_root, self.bench),
            BatchPanel(axes[2], data_root, self.bench),
            OptConvergencePanel(axes[3], data_root, self.bench),
            FreqProgressPanel(axes[4], data_root, self.bench),
            ThermoPanel(axes[5], data_root, self.bench),
        ]
        self._suptitle = self.fig.suptitle("", color=FG, fontsize=13, fontweight="bold")

    def update(self, _=None):
        # Only panels whose data changed are re-parsed/redrawn.
        for panel in self.panels:
            panel.tick()
        self._suptitle.set_text(
            f"◆ IVETTE CONTROL ROOM      {self.root}      "
            f"updated {time.strftime('%H:%M:%S')}"
        )

    def run_live(self):
        self.update()
        self._anim = FuncAnimation(
            self.fig, self.update, interval=self.interval * 1000,
            cache_frame_data=False,
        )
        plt.show(block=True)

    def run_once(self, out: Path):
        self.update()
        out.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out, dpi=120, facecolor=BG)
        print(f"Saved {out}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Ivette Gaussian control room")
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data" / "sdfs",
                        help="Directory holding benchmark JSON + run logs")
    parser.add_argument("--interval", type=float, default=4.0, help="Refresh seconds")
    parser.add_argument("--once", action="store_true", help="Render one PNG and exit")
    parser.add_argument("--save", type=Path, default=None,
                        help="PNG path for --once (default: <data-dir>/control_room.png)")
    args = parser.parse_args()

    room = ControlRoom(args.data_dir, args.interval)
    if args.once or args.save:
        out = args.save or (args.data_dir / "control_room.png")
        room.run_once(out)
    else:
        room.run_live()


if __name__ == "__main__":
    main()
