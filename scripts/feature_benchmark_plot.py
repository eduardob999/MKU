#!/usr/bin/env python3
"""
feature_benchmark_plot.py

Visualise a feature-selection benchmark (from
ivette.core.feature_benchmark.run_feature_selection_benchmark, saved as JSON).

Four views, color used as a real dimension throughout:
  1. CV R² by method  — scaffold (honest) bars with the random (optimistic)
     score overlaid as a marker, grouped by before/after DFT. The bar→marker
     gap is the leakage.
  2. Performance vs complexity — scatter of kept features vs scaffold R²,
     colour = method, marker = before/after DFT.
  3. Leakage map — random−scaffold gap per method/block, colour = gap size.
  4. Feature relevance heatmap — top features × method, colour = importance.

Usage:
  python scripts/feature_benchmark_plot.py result.json            # interactive
  python scripts/feature_benchmark_plot.py result.json --save out.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

_HEADLESS = "--save" in sys.argv
if _HEADLESS:
    matplotlib.use("Agg")
else:
    for _b in ("QtAgg", "TkAgg"):
        try:
            matplotlib.use(_b)
            break
        except Exception:
            continue

import matplotlib.pyplot as plt

BG, PANEL, FG, MUTED = "#0f1419", "#161b22", "#c9d1d9", "#8b949e"
ACCENT, GREEN, RED, GOLD, PURPLE = "#5fd7ff", "#3fb950", "#f85149", "#d29922", "#bc8cff"
METHOD_COLORS = {"none": MUTED, "variance": GOLD, "correlation": PURPLE,
                 "univariate": ACCENT, "model": GREEN}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": "#30363d",
    "axes.labelcolor": MUTED, "axes.titlecolor": FG, "text.color": FG,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#21262d",
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
})


def _methods_in(results):
    seen = []
    for r in results:
        if r["method"] not in seen:
            seen.append(r["method"])
    return seen


def _get(results, block, method, key):
    for r in results:
        if r["block"] == block and r["method"] == method:
            return r.get(key)
    return None


def panel_method_bars(ax, data):
    grouping = data.get("grouping", "scaffold")
    results = data["results"]
    methods = _methods_in(results)
    blocks = [b for b in ("without_dft", "with_dft") if any(r["block"] == b for r in results)]
    x = range(len(methods))
    width = 0.8 / max(len(blocks), 1)
    block_color = {"without_dft": "#3a4250", "with_dft": ACCENT}
    for bi, block in enumerate(blocks):
        offs = [i + bi * width - 0.4 + width / 2 for i in x]
        scaf = [(_get(results, block, m, "cv_r2_scaffold") or 0.0) for m in methods]
        rnd = [_get(results, block, m, "cv_r2_random") for m in methods]
        ax.bar(offs, scaf, width=width, color=block_color.get(block, MUTED),
               label=f"{block.replace('_', ' ')} ({grouping})", edgecolor="#0f1419")
        # random score overlaid as a hollow marker — the gap to the bar = leakage
        ax.scatter(offs, [v if v is not None else 0.0 for v in rnd], s=40,
                   facecolor="none", edgecolor=GOLD, lw=1.5, zorder=5,
                   label="random" if bi == 0 else None)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("CV R²")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=6, framealpha=0.2, loc="lower left")
    ax.set_title(f"CV R² by selection method  (bar = {grouping}, ○ = random)", loc="left")


def panel_perf_vs_complexity(ax, data):
    grouping = data.get("grouping", "scaffold")
    results = data["results"]
    markers = {"without_dft": "o", "with_dft": "^"}
    for r in results:
        y = r.get("cv_r2_scaffold")
        if y is None:
            continue
        ax.scatter(r["n_features"], y, s=120,
                   color=METHOD_COLORS.get(r["method"], FG),
                   marker=markers.get(r["block"], "o"),
                   edgecolor="#0f1419", lw=0.6, alpha=0.9)
        ax.annotate(r["method"], (r["n_features"], y), fontsize=6, color=MUTED,
                    textcoords="offset points", xytext=(4, 4))
    ax.set_xscale("symlog")
    ax.set_xlabel("kept features (symlog)")
    ax.set_ylabel(f"{grouping} CV R²")
    ax.grid(True, alpha=0.3)
    # legends: colour = method, marker = block
    from matplotlib.lines import Line2D
    mleg = [Line2D([], [], marker="o", ls="", color=c, label=m)
            for m, c in METHOD_COLORS.items() if any(r["method"] == m for r in results)]
    bleg = [Line2D([], [], marker=mk, ls="", color=FG, label=b.replace("_", " "))
            for b, mk in markers.items() if any(r["block"] == b for r in results)]
    ax.legend(handles=mleg + bleg, fontsize=6, framealpha=0.2, loc="lower right")
    ax.set_title("Performance vs complexity  (colour = method, shape = DFT)", loc="left")


def panel_leakage(ax, data):
    grouping = data.get("grouping", "scaffold")
    results = data["results"]
    methods = _methods_in(results)
    blocks = [b for b in ("without_dft", "with_dft") if any(r["block"] == b for r in results)]
    grid = []
    for block in blocks:
        row = []
        for m in methods:
            rnd, scaf = _get(results, block, m, "cv_r2_random"), _get(results, block, m, "cv_r2_scaffold")
            row.append((rnd - scaf) if (rnd is not None and scaf is not None) else float("nan"))
        grid.append(row)
    im = ax.imshow(grid, aspect="auto", cmap="inferno")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels([b.replace("_", " ") for b in blocks], fontsize=8)
    for i in range(len(blocks)):
        for j in range(len(methods)):
            v = grid[i][j]
            if v == v:  # not NaN
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.5 else FG, fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(f"random − {grouping}", color=MUTED)
    ax.set_title("Leakage map  (higher = more over-optimistic)", loc="left")


def panel_relevance(ax, data):
    rel = data.get("relevance", {})
    # prefer the with_dft block; keep methods that produced scores
    keys = [k for k in rel if k.startswith("with_dft:")] or list(rel)
    if not keys:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no feature-relevance scores", ha="center", va="center",
                color=MUTED, style="italic")
        ax.set_title("Feature relevance", loc="left")
        return
    # union of top features across the selected methods
    feats = {}
    for k in keys:
        for f, s in rel[k].items():
            feats[f] = max(feats.get(f, 0.0), s)
    top_feats = [f for f, _ in sorted(feats.items(), key=lambda kv: kv[1], reverse=True)[:15]]
    cols = keys
    grid = [[rel[k].get(f, 0.0) for k in cols] for f in top_feats]
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.split(":")[-1] for c in cols], rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels([f[:22] for f in top_feats], fontsize=6)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("relevance", color=MUTED)
    ax.set_title("Top feature relevance  (colour = importance)", loc="left")


def render(data, out=None):
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    grouping = data.get("grouping", "scaffold")
    title = (f"◆ Feature-selection benchmark — {data.get('target','?')}   "
             f"n={data.get('n_samples','?')}, {grouping} groups="
             f"{data.get('n_groups', data.get('n_scaffold_groups','?'))}")
    fig.suptitle(title, color=FG, fontsize=13, fontweight="bold")
    gs = fig.add_gridspec(2, 2)
    panel_method_bars(fig.add_subplot(gs[0, 0]), data)
    panel_perf_vs_complexity(fig.add_subplot(gs[0, 1]), data)
    panel_leakage(fig.add_subplot(gs[1, 0]), data)
    panel_relevance(fig.add_subplot(gs[1, 1]), data)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, facecolor=BG)
        print(f"Saved {out}")
    else:
        plt.show(block=True)


def main():
    p = argparse.ArgumentParser(description="Plot a feature-selection benchmark result")
    p.add_argument("result", help="Benchmark result JSON")
    p.add_argument("--save", default=None, help="PNG output path (headless)")
    args = p.parse_args()
    data = json.loads(Path(args.result).read_text())
    if "error" in data:
        sys.exit(f"Benchmark has no results: {data['error']}")
    render(data, args.save)


if __name__ == "__main__":
    main()
