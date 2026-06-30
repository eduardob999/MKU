"""Benchmark feature-selection methods for one target, before vs after DFT.

Runs each selection configuration (none / variance / correlation / univariate /
model) on the same data, scores it with dual cross-validation (random +
scaffold), and records the kept-feature count and per-feature relevance. The
structured result feeds the comparison plots (which method wins, how much DFT
helps, which features matter). UI-free.
"""

from __future__ import annotations

import pandas as pd

from ivette.core.train_xgboost_emfp import (
    build_model,
    classify_columns,
    generate_emfp_dataframe,
)
from ivette.core.modeling import (
    apply_transform,
    cluster_groups,
    decide_transform,
    evaluate_cv,
    scaffold_groups,
    select_features,
)
from ivette.core.params import FeatureSelectionParams, TrainingParams

ALL_METHODS = ["none", "variance", "correlation", "univariate", "model"]


def _method_configs(fsp: FeatureSelectionParams) -> dict:
    """One FeatureSelectionParams per benchmarked method (each isolates one knob)."""
    return {
        "none": FeatureSelectionParams(method="none", variance_threshold=0.0,
                                       correlation_threshold=1.0, k_best=0),
        "variance": FeatureSelectionParams(method="none",
                                           variance_threshold=fsp.variance_threshold or 1e-6,
                                           correlation_threshold=1.0, k_best=0),
        "correlation": FeatureSelectionParams(method="none", variance_threshold=0.0,
                                              correlation_threshold=fsp.correlation_threshold,
                                              k_best=0),
        "univariate": FeatureSelectionParams(method="univariate", score_func=fsp.score_func,
                                             k_best=fsp.k_best, variance_threshold=0.0,
                                             correlation_threshold=1.0),
        "model": FeatureSelectionParams(method="model", k_best=fsp.k_best,
                                        variance_threshold=0.0, correlation_threshold=1.0),
    }


def _build_matrices(source, target, dft_df, smiles_col, tp, grouping="cluster"):
    """Return (base_X, augmented_X|None, y, groups) or None if unusable."""
    df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    if target not in df.columns or smiles_col not in df.columns or "CID" not in df.columns:
        return None
    descriptor_features, _ = classify_columns(df, smiles_col)
    sub = df.dropna(subset=[target]).reset_index(drop=True)
    if len(sub) < tp.min_samples:
        return None

    transform = decide_transform(sub[target], min_samples=tp.min_samples,
                                 dynamic_range=tp.log_dynamic_range)
    y = apply_transform(sub[target], transform)
    fp = generate_emfp_dataframe(sub[smiles_col], radius=tp.radius, nbits=tp.nbits)
    base = pd.concat(
        [sub[descriptor_features].reset_index(drop=True), fp.reset_index(drop=True)], axis=1)
    groups = (cluster_groups(sub[smiles_col]) if grouping == "cluster"
              else scaffold_groups(sub[smiles_col]))

    augmented = None
    if dft_df is not None:
        dft_cols = [c for c in dft_df.columns if c != "CID"]
        d = dft_df.copy()
        d["CID"] = d["CID"].astype(str)
        joined = (sub[["CID"]].astype({"CID": str})
                  .merge(d, on="CID", how="left")[dft_cols].reset_index(drop=True))
        augmented = pd.concat([base.reset_index(drop=True), joined], axis=1)
    return base, augmented, y, groups


def run_feature_selection_benchmark(source, target, dft_df=None, *,
                                    smiles_col="SMILES", tp=None, fsp=None, methods=None,
                                    grouping="cluster"):
    """Benchmark selection methods × {without DFT, with DFT} for one target.

    Returns a dict with ``results`` (one row per block×method: dual-CV scores +
    feature count) and ``relevance`` (top features per block×method), or an
    ``error`` key when the target is unusable.
    """
    tp = tp or TrainingParams()
    fsp = fsp or FeatureSelectionParams()
    methods = methods or ALL_METHODS

    built = _build_matrices(source, target, dft_df, smiles_col, tp, grouping)
    if built is None:
        return {"error": "insufficient data or missing CID/SMILES/target columns",
                "target": target}
    base, augmented, y, groups = built
    configs = _method_configs(fsp)

    blocks = [("without_dft", base)]
    if augmented is not None:
        blocks.append(("with_dft", augmented))

    results, relevance = [], {}
    for block_name, X in blocks:
        for method in methods:
            cfg = configs[method]
            X_sel, kept, scores = select_features(X, y, cfg)
            cv = evaluate_cv(lambda: build_model(tp), X_sel, y, groups, strategy="both",
                             max_folds=tp.cv_max_folds, min_reliable_samples=tp.min_reliable_samples)
            results.append({
                "block": block_name,
                "method": method,
                "n_features": len(kept),
                "cv_r2_random": cv["cv_r2_random"],
                "cv_r2_scaffold": cv["cv_r2_scaffold"],
                "reliable": cv["reliable"],
            })
            if scores:
                top = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:20])
                relevance[f"{block_name}:{method}"] = {k: float(v) for k, v in top.items()}

    return {
        "target": target,
        "n_samples": int(len(y)),
        "grouping": grouping,
        "n_groups": int(len(set(groups))),
        "results": results,
        "relevance": relevance,
    }


def best_method(result, *, metric="cv_r2_scaffold", prefer_block="with_dft"):
    """Return the name of the best-scoring method in a benchmark ``result``.

    Judged by the honest (scaffold) score, preferring the with-DFT block when
    present. Returns ``None`` if there are no usable scores.
    """
    rows = result.get("results", [])
    blocks = {r["block"] for r in rows}
    block = prefer_block if prefer_block in blocks else (next(iter(blocks), None))
    scored = [(r["method"], r.get(metric)) for r in rows
              if r["block"] == block and r.get(metric) is not None]
    if not scored:
        return None
    return max(scored, key=lambda kv: kv[1])[0]


def best_config(result, fsp=None, **kw):
    """``FeatureSelectionParams`` configured for the benchmark's winning method.

    Keeps the k_best / thresholds from ``fsp`` (or defaults); only the ``method``
    (and matching filter knobs) are set from the benchmark winner.
    """
    fsp = fsp or FeatureSelectionParams()
    winner = best_method(result, **kw)
    if winner is None or winner == "none":
        return FeatureSelectionParams(method="none", variance_threshold=0.0,
                                      correlation_threshold=1.0, k_best=0)
    if winner == "variance":
        return FeatureSelectionParams(method="none",
                                      variance_threshold=fsp.variance_threshold or 1e-6,
                                      correlation_threshold=1.0, k_best=0)
    if winner == "correlation":
        return FeatureSelectionParams(method="none", variance_threshold=0.0,
                                      correlation_threshold=fsp.correlation_threshold, k_best=0)
    # univariate or model
    return FeatureSelectionParams(method=winner, score_func=fsp.score_func,
                                  k_best=fsp.k_best, variance_threshold=0.0,
                                  correlation_threshold=1.0)
