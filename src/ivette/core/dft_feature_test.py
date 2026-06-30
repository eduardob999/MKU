"""Ablation test: do DFT descriptors add predictive value for one target?

For a single (model, target), trains the SAME XGBoost model twice — once on the
existing features (physchem descriptors + eMFP fingerprints) and once with the
parsed DFT descriptors left-joined on CID — and compares cross-validated R². Also
reports how much the model leans on the DFT block via feature importance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ivette.core.train_xgboost_emfp import (
    MIN_SAMPLES,
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


def compare_target_with_dft(source, target, dft_df, *,
                            smiles_col="SMILES", radius=2, nbits=512, fsp=None,
                            grouping="cluster", cluster_cutoff=0.4,
                            cluster_fp_radius=2, cluster_fp_bits=1024, model_factory=None):
    """Return a dict comparing baseline vs DFT-augmented CV R² for ``target``.

    ``source`` is the model's training data (a DataFrame, or a CSV path) which
    must already carry CID, SMILES, the physchem descriptors and the target
    column; ``dft_df`` is the parsed DFT descriptor table (with a CID column).
    """
    df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    if target not in df.columns:
        return {"error": f"target '{target}' not in source dataset"}
    if smiles_col not in df.columns or "CID" not in df.columns:
        return {"error": "source dataset missing CID/SMILES columns"}

    descriptor_features, _ = classify_columns(df, smiles_col)
    sub = df.dropna(subset=[target]).reset_index(drop=True)
    n = len(sub)
    if n < MIN_SAMPLES:
        return {"error": f"only {n} samples (need >= {MIN_SAMPLES})", "n_samples": n}

    # Per-target transform (log only positive, wide-range targets) — never a
    # blind log10 that would destroy signed properties.
    transform = decide_transform(sub[target])
    y = apply_transform(sub[target], transform)

    # base features: physchem descriptors + eMFP fingerprints
    fp = generate_emfp_dataframe(sub[smiles_col], radius=radius, nbits=nbits)
    base = pd.concat(
        [sub[descriptor_features].reset_index(drop=True), fp.reset_index(drop=True)],
        axis=1,
    )

    # DFT block, left-joined on CID
    dft_cols = [c for c in dft_df.columns if c != "CID"]
    dft = dft_df.copy()
    dft["CID"] = dft["CID"].astype(str)
    joined = (sub[["CID"]].astype({"CID": str})
              .merge(dft, on="CID", how="left")[dft_cols].reset_index(drop=True))
    n_covered = int(joined.notna().any(axis=1).sum())
    augmented = pd.concat([base.reset_index(drop=True), joined], axis=1)

    # Optional feature selection (e.g. the benchmark's best method), applied to
    # each matrix so the comparison reflects the same selection the model uses.
    if fsp is not None:
        base = select_features(base, y, fsp)[0]
        augmented = select_features(augmented, y, fsp)[0]

    # Evaluate under random + the chosen grouping. cluster (default) = "predict
    # new analogs of this family"; scaffold = novel-chemotype stress test. The
    # random-vs-grouped gap is the leakage / over-fit meter.
    mf = model_factory or build_model
    groups = (cluster_groups(sub[smiles_col], cutoff=cluster_cutoff,
                             radius=cluster_fp_radius, fp_bits=cluster_fp_bits)
              if grouping == "cluster" else scaffold_groups(sub[smiles_col]))
    base = evaluate_cv(mf, base, y, groups, strategy="both")
    aug = evaluate_cv(mf, augmented, y, groups, strategy="both")

    def _delta(a, b):
        return None if (a is None or b is None) else (a - b)

    # importance share of the DFT block in the augmented model
    model = mf()
    model.fit(augmented, y)
    imp = pd.Series(model.feature_importances_, index=augmented.columns)
    dft_imp = imp[dft_cols].sort_values(ascending=False)

    return {
        "target": target,
        "n_samples": n,
        "n_dft_covered": n_covered,
        "transform": transform,
        "grouping": grouping,
        "cv_folds": base["cv_folds"],
        "n_groups": base["n_scaffold_groups"],
        "reliable": base["reliable"],
        "reliability_note": base["reliability_note"],
        # primary (grouped = honest within-family score when grouping=cluster)
        "baseline_cv_r2": base["cv_r2_mean"],
        "baseline_cv_std": base["cv_r2_std"] or 0.0,
        "augmented_cv_r2": aug["cv_r2_mean"],
        "augmented_cv_std": aug["cv_r2_std"] or 0.0,
        "delta_cv_r2": _delta(aug["cv_r2_mean"], base["cv_r2_mean"]),
        # random vs grouped, side by side
        "baseline_cv_r2_random": base["cv_r2_random"],
        "baseline_cv_r2_grouped": base["cv_r2_scaffold"],
        "augmented_cv_r2_random": aug["cv_r2_random"],
        "augmented_cv_r2_grouped": aug["cv_r2_scaffold"],
        "delta_cv_r2_random": _delta(aug["cv_r2_random"], base["cv_r2_random"]),
        "delta_cv_r2_grouped": _delta(aug["cv_r2_scaffold"], base["cv_r2_scaffold"]),
        "dft_total_importance": float(imp[dft_cols].sum()),
        "dft_importance": {k: float(v) for k, v in dft_imp.items()},
    }
