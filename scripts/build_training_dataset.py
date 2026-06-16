#!/usr/bin/env python3

from pathlib import Path
import argparse
import subprocess
import sys

import pandas as pd


def run_step(command, name):

    print(f"\n{'=' * 70}")
    print(name)
    print(f"{'=' * 70}")

    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(
            f"{name} failed with code "
            f"{result.returncode}"
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Original dataset CSV"
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

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    fetch_script = (
        script_dir /
        "fetch_inchikeys.py"
    )

    train_script = (
        script_dir /
        "train_xgboost_emfp.py"
    )

    for script in [
        fetch_script,
        train_script
    ]:

        if not script.exists():

            raise FileNotFoundError(
                f"Required script not found:\n{script}"
            )

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

    run_step(
        [
            sys.executable,
            str(fetch_script),
            "--input",
            args.input,
            "--output",
            str(structures_file)
        ],
        "Fetching structures"
    )

    # ==================================================
    # STEP 2
    # Merge
    # ==================================================

    print(f"\n{'=' * 70}")
    print("Merging datasets")
    print(f"{'=' * 70}")

    df_dataset = pd.read_csv(
        args.input,
        dtype={"CID": str}
    )

    df_structures = pd.read_csv(
        structures_file,
        dtype={"CID": str}
    )

    if "CID" not in df_dataset.columns:
        raise ValueError(
            "Input dataset does not contain a CID column."
        )

    if "CID" not in df_structures.columns:
        raise ValueError(
            "Structures file does not contain a CID column."
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
        f"Saved merged dataset:\n"
        f"{merged_file}"
    )

    # ==================================================
    # STEP 3
    # Train
    # ==================================================

    run_step(
        [
            sys.executable,
            str(train_script),
            "--input",
            str(merged_file),
            "--output-dir",
            str(models_dir),
            "--radius",
            str(args.radius),
            "--nbits",
            str(args.nbits)
        ],
        "Training models"
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()