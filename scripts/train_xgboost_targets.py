#!/usr/bin/env python3
"""
train_xgboost_targets.py

Train one XGBoost model per target column.

Example:
    python train_xgboost_targets.py \
        --input merged_dataset.csv \
        --output-dir models
"""

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

FEATURE_COLUMNS = [
    "MW",
    "LogP",
    "Complexity",
    "Covalently-Bonded Unit Count",
    "Defined Atom Stereocenter Count",
    "Defined Bond Stereocenter Count",
    "ExactMass",
    "Formal Charge",
    "Heavy Atom Count",
    "Hydrogen Bond Acceptor Count",
    "Hydrogen Bond Donor Count",
    "Isotope Atom Count",
    "MonoMass",
    "Rotatable Bond Count",
    "Topological Polar Surface Area",
    "Undefined Atom Stereocenter Count",
    "Undefined Bond Stereocenter Count",
    "Solubility",
]

MIN_TARGET_COVERAGE = 0.05
MIN_SAMPLES = 30


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def is_target_column(col):
    return (
        "ChEMBL:" in col
        or "IC50" in col
        or "EC50" in col
        or "Ki" in col
        or "Kd" in col
        or "Potency" in col
    )


def select_targets(df):
    targets = []

    for col in df.columns:
        if not is_target_column(col):
            continue

        coverage = df[col].notna().mean()

        if coverage >= MIN_TARGET_COVERAGE:
            targets.append(col)

    return targets


def build_model():
    return XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_target(df, target_col, feature_columns):

    cols = feature_columns + [target_col]

    subset = df[cols].copy()
    subset = subset.dropna(subset=[target_col])

    n = len(subset)

    if n < MIN_SAMPLES:
        return None

    X = subset[feature_columns]

    # XGBoost handles NaNs
    y = subset[target_col]

    # log-transform activity values
    y = np.log10(y.clip(lower=1e-12))

    model = build_model()

    folds = min(5, max(3, n // 10))

    cv = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )

    r2_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="r2"
    )

    model.fit(X, y)

    pred = model.predict(X)

    report = {
        "target": target_col,
        "n_samples": int(n),
        "cv_r2_mean": float(np.mean(r2_scores)),
        "cv_r2_std": float(np.std(r2_scores)),
        "train_r2": float(r2_score(y, pred)),
        "train_rmse": float(
            np.sqrt(mean_squared_error(y, pred))
        ),
        "train_mae": float(
            mean_absolute_error(y, pred)
        ),
    }

    return model, report


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        default="models"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(args.input)

    available_features = [
        c for c in FEATURE_COLUMNS
        if c in df.columns
    ]

    missing = set(FEATURE_COLUMNS) - set(available_features)

    if missing:
        print(
            f"Warning: missing features: "
            f"{len(missing)}"
        )

    print(
        f"Using {len(available_features)} features"
    )

    targets = select_targets(df)

    print(
        f"Found {len(targets)} usable targets"
    )

    reports = []

    for target in targets:

        print(f"\nTraining: {target}")

        result = train_target(
            df,
            target,
            available_features
        )

        if result is None:
            print("  skipped (<30 samples)")
            continue

        model, report = result

        safe_name = (
            target.replace("/", "_")
            .replace(":", "_")
            .replace("[", "")
            .replace("]", "")
            .replace(" ", "_")
        )

        model_file = (
            output_dir /
            f"{safe_name}.joblib"
        )

        joblib.dump(model, model_file)

        reports.append(report)

        print(
            f"  samples={report['n_samples']}"
        )
        print(
            f"  CV R²={report['cv_r2_mean']:.3f}"
        )

    if reports:

        report_df = pd.DataFrame(reports)

        report_df.sort_values(
            "cv_r2_mean",
            ascending=False
        ).to_csv(
            output_dir / "model_report.csv",
            index=False
        )

        with open(
            output_dir / "model_report.json",
            "w"
        ) as f:
            json.dump(
                reports,
                f,
                indent=2
            )

        print("\nTop models:")
        print(
            report_df[
                [
                    "target",
                    "n_samples",
                    "cv_r2_mean"
                ]
            ].sort_values(
                "cv_r2_mean",
                ascending=False
            )
        )

    else:
        print("\nNo models were trained.")


if __name__ == "__main__":
    main()