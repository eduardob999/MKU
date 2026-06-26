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
    cv_r2,
    decide_transform,
    scaffold_groups,
)


def compare_target_with_dft(source, target, dft_df, *,
                            smiles_col="SMILES", radius=2, nbits=512):
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

    # Same scaffold-grouped folds for both models so the comparison is fair and
    # not inflated by analog leakage.
    groups = scaffold_groups(sub[smiles_col])
    base_r2, base_std, folds, cv_method = cv_r2(build_model, base, y, groups)
    aug_r2, aug_std, _, _ = cv_r2(build_model, augmented, y, groups)

    # importance share of the DFT block in the augmented model
    model = build_model()
    model.fit(augmented, y)
    imp = pd.Series(model.feature_importances_, index=augmented.columns)
    dft_imp = imp[dft_cols].sort_values(ascending=False)

    return {
        "target": target,
        "n_samples": n,
        "n_dft_covered": n_covered,
        "transform": transform,
        "cv_method": cv_method,
        "cv_folds": folds,
        "baseline_cv_r2": base_r2,
        "baseline_cv_std": base_std,
        "augmented_cv_r2": aug_r2,
        "augmented_cv_std": aug_std,
        "delta_cv_r2": aug_r2 - base_r2,
        "dft_total_importance": float(imp[dft_cols].sum()),
        "dft_importance": {k: float(v) for k, v in dft_imp.items()},
    }
