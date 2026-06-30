#!/usr/bin/env python3
"""Three-step training pipeline: fetch InChIKeys/SMILES, merge, train XGBoost."""

import argparse
from pathlib import Path

import pandas as pd

from ivette.core.fetch_inchikeys import main as fetch_inchikeys_main
from ivette.core.train_xgboost_emfp import main as train_xgboost_main


def main(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Thermo ML-ready dataset CSV"
    )

    parser.add_argument(
        "--workdir",
        default="data"
    )

    parser.add_argument(
        "--models-dir",
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
        default=1024
    )

    parser.add_argument(
        "--params-json",
        default=None,
        help="JSON file of TrainingParams forwarded to the trainer."
    )

    parser.add_argument(
        "--dft-csv",
        default=None,
        help="CSV of DFT/redox descriptors forwarded to the trainer."
    )

    parser.add_argument(
        "--fs-params-json",
        default=None,
        help="JSON file of FeatureSelectionParams forwarded to the trainer."
    )

    args = parser.parse_args(argv)


    project_root = Path(
        __file__
    ).resolve().parents[2]


    workdir = (
        project_root /
        args.workdir
    )

    workdir.mkdir(
        parents=True,
        exist_ok=True
    )


    models_dir = (
        project_root /
        args.models_dir
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    structures_file = (
        workdir /
        "structures.csv"
    )

    merged_file = (
        workdir /
        "training_dataset.csv"
    )


    # ==================================================
    # STEP 1
    # Fetch structures
    # ==================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Fetching structures"
    )

    print(
        "=" * 70
    )


    fetch_inchikeys_main(
        [
            "--input",
            args.input,

            "--output",
            str(structures_file)
        ]
    )


    # ==================================================
    # STEP 2
    # Merge
    # ==================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Merging datasets"
    )

    print(
        "=" * 70
    )


    df_dataset = pd.read_csv(
        args.input,
        dtype={
            "CID": str
        }
    )


    df_structures = pd.read_csv(
        structures_file,
        dtype={
            "CID": str
        }
    )


    if "CID" not in df_dataset.columns:

        raise ValueError(
            "Input dataset does not contain CID column."
        )


    if "CID" not in df_structures.columns:

        raise ValueError(
            "Structures file does not contain CID column."
        )


    df_merged = df_dataset.merge(
        df_structures,
        on="CID",
        how="left"
    )


    missing_smiles = int(
        df_merged["SMILES"]
        .isna()
        .sum()
    )


    df_merged.to_csv(
        merged_file,
        index=False
    )


    print(
        f"Rows merged: {len(df_merged)}"
    )

    print(
        f"Missing SMILES: {missing_smiles}"
    )

    print(
        f"Saved:\n{merged_file}"
    )


    # ==================================================
    # STEP 3
    # Train
    # ==================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "Training models"
    )

    print(
        "=" * 70
    )


    train_argv = [
        "--input",
        str(merged_file),

        "--output-dir",
        str(models_dir),

        "--radius",
        str(args.radius),

        "--nbits",
        str(args.nbits),
    ]

    if args.params_json:
        train_argv += ["--params-json", args.params_json]
    if args.dft_csv:
        train_argv += ["--dft-csv", args.dft_csv]
    if args.fs_params_json:
        train_argv += ["--fs-params-json", args.fs_params_json]

    train_xgboost_main(train_argv)


    print(
        "\nPipeline completed successfully."
    )

if __name__ == "__main__":
    main()