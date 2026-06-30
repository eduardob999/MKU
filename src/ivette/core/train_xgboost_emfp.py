#!/usr/bin/env python3
"""
train_xgboost_emfp.py

Train one XGBoost model per target column using:

    - Molecular descriptors
    - eMFP (Morgan count fingerprints)

Example:

python train_xgboost_emfp.py \
    --input merged_dataset.csv \
    --output-dir models \
    --radius 3 \
    --nbits 2048
"""

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

from ivette.util.columns import is_target_column
from ivette.util.text import slugify
from ivette.core.modeling import (
    apply_transform,
    cluster_groups,
    cross_conformal,
    decide_transform,
    evaluate_cv,
    scaffold_groups,
    select_features,
    y_scramble,
)


# ============================================================
# Configuration
# ============================================================

# Columns that are identifiers, not features or targets
ID_COLUMNS = {
    "CID", "InChIKey", "PubChemName", "PubChem_URL", "SMILES"
}

MIN_TARGET_COVERAGE = 0.05
MIN_SAMPLES = 30


# ============================================================
# Column Classification
# ============================================================

def classify_columns(df: pd.DataFrame, smiles_col: str):
    """
    Walk the DataFrame columns in order.
    Returns (feature_cols, target_cols).

    Strategy:
      - Skip known ID columns and the SMILES column.
      - Once we first encounter a target-like column, everything
        from that point on is treated as a target (or skipped).
      - Everything before the first target column that is numeric
        and not an ID is a feature.
    """
    feature_cols = []
    target_cols  = []
    passed_boundary = False

    for col in df.columns:

        if col in ID_COLUMNS or col == smiles_col:
            continue

        if is_target_column(col):
            passed_boundary = True
            target_cols.append(col)
            continue

        # The bare "ChEMBL:" sentinel marks the boundary but
        # is itself neither a feature nor a target
        if col == "ChEMBL:":
            passed_boundary = True
            continue

        if passed_boundary:
            # Columns after the boundary that don't match
            # is_target_column are unexpected — skip them
            continue

        # Before the boundary: keep numeric columns as features
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols, target_cols


# ============================================================
# Fingerprints
# ============================================================

def generate_emfp_dataframe(
    smiles_series,
    radius=3,
    nbits=2048
):
    """
    Generate Morgan count fingerprints as a DataFrame.
    """

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=nbits
    )

    fingerprints = []

    total = len(smiles_series)

    for idx, smiles in enumerate(smiles_series):

        if idx % 1000 == 0:
            print(f"  fingerprints: {idx}/{total}")

        mol = Chem.MolFromSmiles(str(smiles))

        if mol is None:
            fingerprints.append(
                np.zeros(nbits, dtype=np.uint16)
            )
            continue

        fp = generator.GetCountFingerprintAsNumPy(mol)

        fingerprints.append(fp)

    return pd.DataFrame(
        fingerprints,
        columns=[
            f"eMFP_{i}"
            for i in range(nbits)
        ]
    )


# ============================================================
# Model
# ============================================================

def build_model(tp=None):
    """Build the XGBoost regressor from a :class:`TrainingParams` (or defaults)."""
    from ivette.core.params import TrainingParams

    tp = tp or TrainingParams()
    return XGBRegressor(
        n_estimators=tp.n_estimators,
        max_depth=tp.max_depth,
        learning_rate=tp.learning_rate,
        subsample=tp.subsample,
        colsample_bytree=tp.colsample_bytree,
        reg_alpha=tp.reg_alpha,
        reg_lambda=tp.reg_lambda,
        min_child_weight=tp.min_child_weight,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )


# ============================================================
# Training
# ============================================================

def train_target(
    df,
    target_col,
    feature_columns,
    smiles_col="SMILES",
    model_factory=None,
    tp=None,
    fsp=None,
):
    # ``tp`` (TrainingParams) drives hyperparameters + CV/transform thresholds.
    # ``fsp`` (FeatureSelectionParams), when given, prunes the feature matrix
    # per target before fitting. ``model_factory`` overrides estimator
    # construction when given (e.g. a small/fast model in tests).
    from ivette.core.params import TrainingParams

    tp = tp or TrainingParams()
    if model_factory is None:
        model_factory = lambda: build_model(tp)

    # Keep SMILES alongside the features/target so we can scaffold-group the CV;
    # it is never used as a model input.
    cols = list(dict.fromkeys(feature_columns + [target_col, smiles_col]))

    subset = df[cols].copy()

    subset = subset.dropna(
        subset=[target_col]
    )

    n = len(subset)

    if n < tp.min_samples:
        return None

    X = subset[feature_columns]

    y_raw = subset[target_col]

    # Per-target transform: log only positive, wide-dynamic-range targets;
    # signed properties (logP, enthalpies, energies) stay linear.
    transform = decide_transform(y_raw, min_samples=tp.min_samples,
                                 dynamic_range=tp.log_dynamic_range)
    y = apply_transform(y_raw, transform)

    # Grouping for leakage-safe CV: scaffold (default) or cluster (within-family
    # middle ground). 'cluster' still reports random alongside the grouped score.
    if tp.cv_strategy == "cluster":
        groups, cv_grouping, eval_strategy = cluster_groups(subset[smiles_col]), "cluster", "both"
    else:
        groups, cv_grouping, eval_strategy = scaffold_groups(subset[smiles_col]), "scaffold", tp.cv_strategy

    # Optional per-target feature selection (e.g. the benchmark's best method).
    selected_cols = list(feature_columns)
    if fsp is not None:
        X, selected_cols, _scores = select_features(X, y, fsp)

    model = model_factory()

    cv = evaluate_cv(model_factory, X, y, groups,
                     strategy=eval_strategy, max_folds=tp.cv_max_folds,
                     min_reliable_samples=tp.min_reliable_samples,
                     n_repeats=tp.cv_repeats)

    # Optional uncertainty + sanity check (opt-in to keep default training fast).
    conformal_halfwidth = conformal_level = None
    if tp.conformal:
        conformal_halfwidth, conformal_level = cross_conformal(
            model_factory, X, y, alpha=tp.conformal_alpha, max_folds=tp.cv_max_folds)
    scramble_r2 = None
    if tp.y_scramble_runs and tp.y_scramble_runs > 0:
        scramble_r2 = y_scramble(model_factory, X, y,
                                 n_repeats=tp.y_scramble_runs, max_folds=tp.cv_max_folds)

    model.fit(X, y)

    pred = model.predict(X)

    report = {
        "target": target_col,
        "n_samples": int(n),
        "transform": transform,
        "fs_method": (fsp.method if fsp is not None else "none"),
        "n_features": len(selected_cols),
        "cv_strategy": tp.cv_strategy,
        "cv_grouping": cv_grouping,
        "cv_repeats": tp.cv_repeats,
        "cv_method": cv["cv_method"],
        "cv_folds": int(cv["cv_folds"] or 0),
        "n_scaffold_groups": cv["n_scaffold_groups"],
        "conformal_halfwidth": conformal_halfwidth,
        "conformal_level": conformal_level,
        "y_scramble_r2": scramble_r2,
        # primary (honest scaffold score when available) for sorting/back-compat
        "cv_r2_mean": float(cv["cv_r2_mean"]) if cv["cv_r2_mean"] is not None else float("nan"),
        "cv_r2_std": float(cv["cv_r2_std"]) if cv["cv_r2_std"] is not None else 0.0,
        # both splits reported side by side
        "cv_r2_random": cv["cv_r2_random"],
        "cv_r2_scaffold": cv["cv_r2_scaffold"],
        "reliable": cv["reliable"],
        "reliability_note": cv["reliability_note"],
        "train_r2": float(
            r2_score(y, pred)
        ),
        "train_rmse": float(
            np.sqrt(
                mean_squared_error(y, pred)
            )
        ),
        "train_mae": float(
            mean_absolute_error(y, pred)
        ),
    }

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    return model, report, importance


# ============================================================
# Main
# ============================================================

def main(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        default="models"
    )

    parser.add_argument(
        "--radius",
        type=int,
        default=3
    )

    parser.add_argument(
        "--nbits",
        type=int,
        default=2048
    )

    parser.add_argument(
        "--smiles-col",
        default="SMILES"
    )

    parser.add_argument(
        "--params-json",
        default=None,
        help="JSON file of TrainingParams (radius, nbits, hyperparameters, …). "
             "Overrides --radius/--nbits when given.",
    )
    parser.add_argument(
        "--dft-csv",
        default=None,
        help="CSV of DFT/redox descriptors (CID + feature columns) to join as features.",
    )
    parser.add_argument(
        "--fs-params-json",
        default=None,
        help="JSON file of FeatureSelectionParams applied per target before fitting.",
    )

    args = parser.parse_args(argv)

    # Training configuration: a params JSON (from the UI) is the full source of
    # truth; otherwise fall back to --radius/--nbits with documented defaults.
    from ivette.core.params import TrainingParams, FeatureSelectionParams, from_dict
    if args.params_json:
        tp = from_dict(TrainingParams, json.load(open(args.params_json)))
    else:
        tp = TrainingParams(radius=args.radius, nbits=args.nbits)
    fsp = from_dict(FeatureSelectionParams, json.load(open(args.fs_params_json))) \
        if args.fs_params_json else None
    print(f"Training params: radius={tp.radius} nbits={tp.nbits} "
          f"n_estimators={tp.n_estimators} max_depth={tp.max_depth} "
          f"lr={tp.learning_rate} min_samples={tp.min_samples}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")

    df = pd.read_csv(args.input)

    if args.smiles_col not in df.columns:
        raise ValueError(
            f"SMILES column '{args.smiles_col}' not found."
        )

    # Join DFT/redox descriptors (added explicitly to the feature list below so
    # column position never matters for their classification).
    dft_feature_columns = []
    if args.dft_csv and Path(args.dft_csv).exists():
        dft_df = pd.read_csv(args.dft_csv)
        dft_df["CID"] = dft_df["CID"].astype(str)
        df["CID"] = df["CID"].astype(str)
        dft_feature_columns = [c for c in dft_df.columns if c != "CID"]
        df = df.merge(dft_df, on="CID", how="left")
        print(f"  DFT/redox   : {len(dft_feature_columns)} features joined")

    print("\nClassifying columns...")

    descriptor_features, targets = classify_columns(
        df, args.smiles_col
    )

    print(f"  Descriptors : {len(descriptor_features)}")
    print(f"  Targets     : {len(targets)}")

    if not targets:
        print("No usable targets found. Exiting.")
        return

    print("\nGenerating eMFP fingerprints...")

    fp_df = generate_emfp_dataframe(
        df[args.smiles_col],
        radius=tp.radius,
        nbits=tp.nbits
    )

    df = pd.concat(
        [
            df.reset_index(drop=True),
            fp_df.reset_index(drop=True)
        ],
        axis=1
    )

    fingerprint_features = [
        c for c in df.columns if c.startswith("eMFP_")
    ]

    available_features = descriptor_features + dft_feature_columns + fingerprint_features

    print(f"  Fingerprints: {len(fingerprint_features)}")
    print(f"  Total feats : {len(available_features)}")
    if fsp is not None:
        print(f"  Feature selection: {fsp.method} (k_best={fsp.k_best})")
    print(f"\nFound {len(targets)} usable targets")

    reports = []

    for target in targets:

        print(f"\nTraining: {target}")

        result = train_target(
            df,
            target,
            available_features,
            smiles_col=args.smiles_col,
            tp=tp,
            fsp=fsp,
        )

        if result is None:
            print(f"  skipped (<{MIN_SAMPLES} samples)")
            continue

        model, report, importance = result

        safe_name = slugify(target)

        model_file      = output_dir / f"{safe_name}.joblib"
        importance_file = output_dir / f"{safe_name}_importance.csv"

        joblib.dump(model, model_file)
        importance.to_csv(importance_file, index=False)

        reports.append(report)

        print(f"  samples={report['n_samples']}")
        print(f"  CV R²={report['cv_r2_mean']:.4f}")

    if reports:

        report_df = (
            pd.DataFrame(reports)
            .sort_values("cv_r2_mean", ascending=False)
        )

        report_df.to_csv(
            output_dir / "model_report.csv",
            index=False
        )

        with open(output_dir / "model_report.json", "w") as f:
            json.dump(reports, f, indent=2)

        print("\nTop models:\n")
        print(report_df[["target", "n_samples", "cv_r2_mean"]])

    else:
        print("\nNo models were trained.")


if __name__ == "__main__":
    main()