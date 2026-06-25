"""Figure builders for the Ivette "Results & Reports" explorer.

Each ``fig_*`` returns ``(figure, keep)`` where ``keep`` holds references to any
interactive widgets (so they survive garbage collection). Figures share a dark
theme for a cohesive, appealing look and degrade gracefully when data is missing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

from ivette.util.paths import COMPOUND_DIR, GAUSSIAN_BENCHMARK_FILE, STRUCTURE_DIR
from ivette.util.storage import COMPOUNDS, MODELS, DATASETS, STRUCTURES
from ivette.util.text import slugify

# ── theme ─────────────────────────────────────────────────────────────────────
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

_SEQ = [ACCENT, GREEN, GOLD, PURPLE, RED, "#79c0ff"]


def _new_fig(title, figsize=(15, 9)):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    try:
        fig.canvas.manager.set_window_title(f"Ivette — {title}")
    except Exception:
        pass
    fig.suptitle(title, color=FG, fontsize=14, fontweight="bold")
    return fig


def _empty(ax, message):
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", style="italic",
            color=MUTED, transform=ax.transAxes)


def _grid(ax):
    ax.grid(True, alpha=0.3)


# ──────────────────────────────────────────────────────────────────────────────
# Structure libraries
# ──────────────────────────────────────────────────────────────────────────────

def fig_structure_library(structure_id):
    info = STRUCTURES.get(structure_id)
    df = pd.read_csv(STRUCTURE_DIR / info["file"])
    fig = _new_fig(f"Structure library · {info['name']} ({structure_id})", figsize=(13, 6))
    ax1, ax2 = fig.subplots(1, 2)

    if "RingSize" in df:
        counts = df["RingSize"].value_counts().sort_index()
        ax1.bar([str(i) for i in counts.index], counts.values, color=ACCENT)
    ax1.set_title(f"Ring size  ({len(df)} structures)")
    ax1.set_xlabel("ring size")
    ax1.set_ylabel("count")
    _grid(ax1)

    if "HeteroatomCount" in df:
        counts = df["HeteroatomCount"].value_counts().sort_index()
        ax2.bar([str(i) for i in counts.index], counts.values, color=PURPLE)
    ax2.set_title("Heteroatom count")
    ax2.set_xlabel("heteroatoms")
    ax2.set_ylabel("count")
    _grid(ax2)
    return fig, []


# ──────────────────────────────────────────────────────────────────────────────
# Compound libraries (PubChem physchem properties)
# ──────────────────────────────────────────────────────────────────────────────

def fig_compound_library(compound_id):
    info = COMPOUNDS.get(compound_id)
    df = pd.read_csv(COMPOUND_DIR / info["file"])
    fig = _new_fig(f"Compound library · {info['name']} ({compound_id}) — {len(df)} compounds")
    axes = fig.subplots(2, 3).flatten()

    hist_props = ["MolecularWeight", "XLogP", "TPSA", "Complexity", "RotatableBondCount"]
    for ax, prop in zip(axes, hist_props):
        if prop in df:
            vals = pd.to_numeric(df[prop], errors="coerce").dropna()
            ax.hist(vals, bins=24, color=ACCENT, edgecolor=BG)
            ax.set_title(prop)
            ax.axvline(vals.median(), color=GOLD, ls="--", lw=1,
                       label=f"median {vals.median():.1f}")
            ax.legend(fontsize=7, framealpha=0.2)
        else:
            _empty(ax, f"{prop} missing")
        _grid(ax)

    # chemical-space scatter: MW vs XLogP coloured by TPSA
    ax = axes[5]
    if {"MolecularWeight", "XLogP"} <= set(df.columns):
        mw = pd.to_numeric(df["MolecularWeight"], errors="coerce")
        logp = pd.to_numeric(df["XLogP"], errors="coerce")
        tpsa = pd.to_numeric(df.get("TPSA"), errors="coerce") if "TPSA" in df else None
        sc = ax.scatter(mw, logp, c=tpsa, cmap="viridis", s=18, alpha=0.8)
        ax.set_xlabel("Molecular weight")
        ax.set_ylabel("XLogP")
        ax.set_title("Chemical space")
        if tpsa is not None:
            cb = fig.colorbar(sc, ax=ax, fraction=0.046)
            cb.set_label("TPSA", color=MUTED)
            cb.ax.yaxis.set_tick_params(color=MUTED)
    else:
        _empty(ax, "MW/XLogP missing")
    _grid(ax)
    return fig, []


# ──────────────────────────────────────────────────────────────────────────────
# Property datasets
# ──────────────────────────────────────────────────────────────────────────────

def _read_cleaning_report(path):
    out = {}
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    for label in ("Original rows", "Rows retained", "Rows removed",
                  "Unique compounds", "Unique properties before cleaning",
                  "Unique properties after cleaning"):
        m = re.search(re.escape(label) + r":\s*(\d+)", text)
        if m:
            out[label] = int(m.group(1))
    return out


def fig_property_dataset(dataset_id):
    info = DATASETS.get(dataset_id)
    out_dir = Path(info["output_dir"])
    fig = _new_fig(f"Property dataset · {info['name']} ({dataset_id})")
    (ax_funnel, ax_cov), (ax_avail, ax_dist) = fig.subplots(2, 2)

    # 1) cleaning funnel
    rep = _read_cleaning_report(out_dir / "cleaning_report.csv")
    if rep:
        stages = ["Original rows", "Rows retained"]
        vals = [rep.get(s, 0) for s in stages]
        bars = ax_funnel.bar(["parsed", "cleaned"], vals, color=[MUTED, GREEN])
        for b, v in zip(bars, vals):
            ax_funnel.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom")
        kept = rep.get("Rows retained", 0)
        orig = rep.get("Original rows", 1) or 1
        ax_funnel.set_title(f"Cleaning funnel — kept {100*kept/orig:.0f}% "
                            f"({rep.get('Unique compounds','?')} compounds)")
    else:
        _empty(ax_funnel, "no cleaning report")
    _grid(ax_funnel)

    # 2) property coverage (compounds per property)
    summ = out_dir / "summary.csv"
    if summ.exists():
        s = pd.read_csv(summ)
        cov = (s.groupby("StandardPropertyName")["CID"].nunique()
               .sort_values(ascending=False).head(15).iloc[::-1])
        ax_cov.barh(cov.index, cov.values, color=ACCENT)
        ax_cov.set_title("Property coverage (top 15)")
        ax_cov.set_xlabel("# compounds")
    else:
        _empty(ax_cov, "no summary.csv")

    # 3) data availability from report.csv
    rep_csv = out_dir / "report.csv"
    if rep_csv.exists():
        r = pd.read_csv(rep_csv)
        total = len(r)
        nist = int(r["NIST_Found"].astype(str).str.lower().isin(["true", "1"]).sum()) \
            if "NIST_Found" in r else 0
        pubmed = int((pd.to_numeric(r.get("PubMed_Abstract_Match_Count"), errors="coerce")
                      .fillna(0) > 0).sum()) if "PubMed_Abstract_Match_Count" in r else 0
        ax_avail.bar(["compounds", "NIST hit", "PubMed hit"], [total, nist, pubmed],
                     color=[MUTED, ACCENT, PURPLE])
        ax_avail.set_title("Literature availability")
        ax_avail.set_ylabel("count")
    else:
        _empty(ax_avail, "no report.csv")
    _grid(ax_avail)

    # 4) numeric value spread for the most-covered property
    cleaned = out_dir / "cleaned.csv"
    if cleaned.exists():
        c = pd.read_csv(cleaned)
        if {"StandardPropertyName", "NumericValue"} <= set(c.columns):
            top = c["StandardPropertyName"].value_counts().head(6).index.tolist()
            pairs = []
            for p in top:
                vals = pd.to_numeric(c[c["StandardPropertyName"] == p]["NumericValue"],
                                     errors="coerce").dropna().values
                if len(vals):
                    pairs.append((p[:10], vals))
            if pairs:
                bp = ax_dist.boxplot([v for _, v in pairs], patch_artist=True)
                ax_dist.set_xticks(range(1, len(pairs) + 1))
                ax_dist.set_xticklabels([lbl for lbl, _ in pairs], rotation=30, fontsize=7)
                for patch in bp["boxes"]:
                    patch.set_facecolor(ACCENT)
                    patch.set_alpha(0.6)
                ax_dist.set_title("Value spread — top properties")
            else:
                _empty(ax_dist, "no numeric values")
        else:
            _empty(ax_dist, "cleaned.csv missing columns")
    else:
        _empty(ax_dist, "no cleaned.csv")
    _grid(ax_dist)
    return fig, []


# ──────────────────────────────────────────────────────────────────────────────
# Models (interactive: cycle the target shown in the importance panel)
# ──────────────────────────────────────────────────────────────────────────────

def fig_model(model_id):
    info = MODELS.get(model_id)
    out_dir = Path(info["output_dir"])
    report = out_dir / "model_report.csv"
    fig = _new_fig(f"Model · {info['name']} ({model_id})")
    (ax_r2, ax_n), (ax_fit, ax_imp) = fig.subplots(2, 2)

    if not report.exists():
        for ax in (ax_r2, ax_n, ax_fit, ax_imp):
            _empty(ax, "no model_report.csv")
        return fig, []

    df = pd.read_csv(report).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)

    # 1) CV R² per target (top 20)
    top = df.head(20).iloc[::-1]
    colors = [GREEN if v >= 0.5 else GOLD if v >= 0.2 else RED for v in top["cv_r2_mean"]]
    ax_r2.barh(range(len(top)), top["cv_r2_mean"],
               xerr=top.get("cv_r2_std"), color=colors, ecolor=MUTED, capsize=2)
    ax_r2.set_yticks(range(len(top)))
    ax_r2.set_yticklabels([t.replace("ChEMBL:", "")[:34] for t in top["target"]], fontsize=7)
    ax_r2.axvline(0, color=MUTED, lw=0.8)
    ax_r2.set_title(f"Cross-validated R²  (top {len(top)}/{len(df)} targets)")
    ax_r2.set_xlabel("CV R²")
    _grid(ax_r2)

    # 2) samples vs CV R²
    ax_n.scatter(df["n_samples"], df["cv_r2_mean"], c=df["cv_r2_mean"],
                 cmap="viridis", s=22, alpha=0.85)
    ax_n.set_xscale("log")
    ax_n.set_xlabel("training samples (log)")
    ax_n.set_ylabel("CV R²")
    ax_n.set_title("Does more data help?")
    _grid(ax_n)

    # 3) train vs CV R² (overfit diagnostic)
    if "train_r2" in df:
        ax_fit.scatter(df["train_r2"], df["cv_r2_mean"], color=ACCENT, s=22, alpha=0.85)
        lim = [min(df["cv_r2_mean"].min(), 0), 1.02]
        ax_fit.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="no overfit")
        ax_fit.set_xlim(0, 1.02)
        ax_fit.set_ylim(*lim)
        ax_fit.legend(fontsize=7, framealpha=0.2)
    ax_fit.set_xlabel("train R²")
    ax_fit.set_ylabel("CV R²")
    ax_fit.set_title("Generalization (train vs CV)")
    _grid(ax_fit)

    # 4) interactive feature importance per target
    targets = df["target"].tolist()
    state = {"i": 0}

    def draw_importance():
        ax_imp.clear()
        target = targets[state["i"]]
        path = out_dir / f"{slugify(target)}_importance.csv"
        if path.exists():
            d = pd.read_csv(path).head(15).iloc[::-1]
            ax_imp.barh(range(len(d)), d["importance"], color=PURPLE)
            ax_imp.set_yticks(range(len(d)))
            ax_imp.set_yticklabels(d["feature"], fontsize=7)
            ax_imp.set_xlabel("importance")
        else:
            _empty(ax_imp, "no importance file")
        cv = df.iloc[state["i"]]["cv_r2_mean"]
        ax_imp.set_title(f"Feature importance · {target.replace('ChEMBL:', '')[:30]}\n"
                         f"target {state['i']+1}/{len(targets)} · CV R²={cv:.3f}")
        _grid(ax_imp)
        fig.canvas.draw_idle()

    draw_importance()

    keep = []
    try:
        ax_prev = fig.add_axes([0.62, 0.005, 0.07, 0.03])
        ax_next = fig.add_axes([0.70, 0.005, 0.07, 0.03])
        b_prev = Button(ax_prev, "◀ prev", color="#21262d", hovercolor=ACCENT)
        b_next = Button(ax_next, "next ▶", color="#21262d", hovercolor=ACCENT)
        for b in (b_prev, b_next):
            b.label.set_color(FG)
            b.label.set_fontsize(8)

        def step(delta):
            state["i"] = (state["i"] + delta) % len(targets)
            draw_importance()

        b_prev.on_clicked(lambda _e: step(-1))
        b_next.on_clicked(lambda _e: step(1))
        keep = [b_prev, b_next]
    except Exception:
        pass
    return fig, keep


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian benchmarks
# ──────────────────────────────────────────────────────────────────────────────

def _benchmark_rows(stage):
    import json
    if not GAUSSIAN_BENCHMARK_FILE.exists():
        return []
    try:
        data = json.loads(GAUSSIAN_BENCHMARK_FILE.read_text()).get("benchmarks", {})
    except Exception:
        return []
    entries = sorted(((k, v) for k, v in data.items() if f"stage={stage}" in k),
                     key=lambda kv: kv[1].get("updated", ""), reverse=True)
    return entries[0][1].get("runs", []) if entries else []


def fig_benchmarks():
    fig = _new_fig("Gaussian benchmarks", figsize=(13, 6))
    ax_cpu, ax_pre = fig.subplots(1, 2)

    rows = sorted(_benchmark_rows("threads"), key=lambda r: r.get("threads", 0))
    if rows:
        threads = [r["threads"] for r in rows]
        secs = [r["run_seconds"] for r in rows]
        ax_cpu.plot(threads, secs, "-o", color=ACCENT, lw=2)
        bi = min(range(len(secs)), key=lambda i: secs[i])
        ax_cpu.scatter([threads[bi]], [secs[bi]], s=130, facecolor="none",
                       edgecolor=GREEN, lw=2)
        ax_cpu.annotate(f"optimal {threads[bi]}", (threads[bi], secs[bi]),
                        textcoords="offset points", xytext=(6, 8), color=GREEN)
        ax_cpu.set_xlabel("threads")
        ax_cpu.set_ylabel("seconds")
    else:
        _empty(ax_cpu, "no thread benchmark")
    ax_cpu.set_title("CPU thread scaling")
    _grid(ax_cpu)

    rows = _benchmark_rows("preopt")
    if rows:
        labels = [r.get("preopt_mode", "?") for r in rows]
        pre = [r.get("preopt_seconds") or 0.0 for r in rows]
        dft = [r.get("dft_seconds") for r in rows]
        x = range(len(rows))
        ax_pre.bar(x, pre, color=MUTED, label="preopt")
        if any(d is not None for d in dft):
            dft0 = [d or 0.0 for d in dft]
            ax_pre.bar(x, dft0, bottom=pre, color=ACCENT, label="DFT")
            totals = [p + d for p, d in zip(pre, dft0)]
            best = min(range(len(totals)), key=lambda i: totals[i])
            for i, r in enumerate(rows):
                steps = r.get("opt_steps")
                tag = f"{totals[i]:.0f}s" + (f"\n{steps} steps" if steps is not None else "")
                ax_pre.text(i, totals[i], tag, ha="center", va="bottom", fontsize=8,
                            color=GREEN if i == best else FG)
            ax_pre.set_ylabel("total wall time (s)")
        else:
            ax_pre.text(0.5, 0.9, "DFT-per-mode not measured yet", transform=ax_pre.transAxes,
                        ha="center", color=GOLD, fontsize=8)
            ax_pre.set_ylabel("preopt time (s)")
        ax_pre.set_xticks(list(x))
        ax_pre.set_xticklabels(labels, fontsize=8)
        ax_pre.legend(fontsize=7, framealpha=0.2)
    else:
        _empty(ax_pre, "no preopt benchmark")
    ax_pre.set_title("Preopt — is it worth it?")
    _grid(ax_pre)
    return fig, []


def fig_dft_descriptor_set(dft_id):
    from ivette.util.storage import load_dft_descriptor_set
    info, df = load_dft_descriptor_set(dft_id)
    fig = _new_fig(f"DFT descriptor set \u00b7 {info['name']} ({dft_id}) "
                   f"\u2014 {len(df)} compounds")
    axes = fig.subplots(2, 3).flatten()
    props = ["gibbs_G", "enthalpy_H", "entropy_S", "zpe_correction",
             "lowest_freq", "n_imaginary"]
    for ax, prop in zip(axes, props):
        if prop not in df:
            _empty(ax, f"{prop} missing")
        elif prop == "n_imaginary":
            vals = pd.to_numeric(df[prop], errors="coerce").fillna(0).astype(int)
            counts = vals.value_counts().sort_index()
            ax.bar([str(i) for i in counts.index], counts.values,
                   color=RED if (vals > 0).any() else GREEN)
            ax.set_xlabel("# imaginary modes")
            ax.set_title(prop)
        else:
            vals = pd.to_numeric(df[prop], errors="coerce").dropna()
            ax.hist(vals, bins=20, color=ACCENT, edgecolor=BG)
            if len(vals):
                ax.axvline(vals.median(), color=GOLD, ls="--", lw=1,
                           label=f"median {vals.median():.3g}")
                ax.legend(fontsize=7, framealpha=0.2)
            ax.set_title(prop)
        _grid(ax)
    return fig, []


BUILDERS = {
    "structure_library": fig_structure_library,
    "compound_library": fig_compound_library,
    "property_dataset": fig_property_dataset,
    "model": fig_model,
    "benchmarks": fig_benchmarks,
    "dft_descriptor_set": fig_dft_descriptor_set,
}
