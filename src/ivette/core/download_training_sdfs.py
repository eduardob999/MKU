#!/usr/bin/env python3
"""
download_training_sdfs.py
Download SDF files from PubChem for compounds that would be
trained by train_xgboost_targets.py (i.e. appear in at least
one target column that meets MIN_TARGET_COVERAGE / MIN_SAMPLES).

Usage:
    python download_training_sdfs.py \
        --input merged_dataset.csv \
        --output-dir sdfs
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

# ── must match train_xgboost_targets.py ──────────────────────────
MIN_TARGET_COVERAGE = 0.01
MIN_SAMPLES = 1


def is_target_column(col: str) -> bool:
    return any(k in col for k in ("ChEMBL:", "IC50", "EC50", "Ki", "Kd", "Potency"))


def select_targets(df: pd.DataFrame) -> list[str]:
    return [
        col for col in df.columns
        if is_target_column(col) and df[col].notna().mean() >= MIN_TARGET_COVERAGE
    ]


def qualifying_cids(df: pd.DataFrame, targets: list[str]) -> set[str]:
    """Return CIDs that appear in ≥1 target with enough non-null rows."""
    cids: set[str] = set()
    for target in targets:
        subset = df[["CID", target]].dropna(subset=[target])
        if len(subset) >= MIN_SAMPLES:
            cids.update(subset["CID"].astype(str).str.strip())
    return cids


def download_sdf(cid: str, out_dir: Path, use_3d: bool = True) -> bool:
    """Download SDF for one CID from PubChem. Returns True on success."""
    dim = "3D" if use_3d else "2D"
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
        f"/{cid}/SDF?record_type={dim.lower()}"
    )
    path = out_dir / f"{cid}.sdf"
    if path.exists():
        return True  # already downloaded

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404 and use_3d:
            # Fall back to 2D if no 3D conformer available
            return download_sdf(cid, out_dir, use_3d=False)
        resp.raise_for_status()
        path.write_bytes(resp.content)
        return True
    except Exception as exc:
        print(f"  WARNING: CID {cid} failed — {exc}")
        return False


def main(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="merged_dataset.csv"
    )

    parser.add_argument(
        "--output-dir",
        default="sdfs",
        help="Directory for SDF files"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Seconds between requests"
    )

    parser.add_argument(
        "--2d-only",
        dest="two_d",
        action="store_true",
        help="Download 2D SDFs only"
    )


    args = parser.parse_args(
        argv
    )


    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        f"Loading {args.input} ..."
    )


    df = pd.read_csv(
        args.input,
        low_memory=False
    )


    targets = select_targets(
        df
    )


    print(
        f"Qualifying targets : {len(targets)}"
    )


    cids = qualifying_cids(
        df,
        targets
    )


    print(
        f"\nUnique CIDs to download: {len(cids)}"
    )


    ok = 0
    fail = 0
    skip = 0


    for i, cid in enumerate(
        sorted(
            cids,
            key=lambda x: int(x)
            if x.isdigit()
            else 0
        ),
        1
    ):

        sdf_path = (
            out_dir /
            f"{cid}.sdf"
        )


        if sdf_path.exists():

            skip += 1

            continue


        success = download_sdf(
            cid,
            out_dir,
            use_3d=not args.two_d
        )


        if success:

            ok += 1

        else:

            fail += 1


        if i % 50 == 0:

            print(
                f"[{i}/{len(cids)}] "
                f"ok={ok} fail={fail} skip={skip}"
            )


        time.sleep(
            args.delay
        )


    print()

    print(
        f"Done. "
        f"Downloaded={ok} "
        f"Failed={fail} "
        f"Already present={skip}"
    )

    print(
        f"SDF files written to: {out_dir}"
    )


if __name__ == "__main__":
    main()