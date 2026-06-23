#!/usr/bin/env python3
"""Download SDF files from PubChem for compounds that qualify for training
(appear in at least one target column meeting MIN_TARGET_COVERAGE / MIN_SAMPLES).
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from ivette.util import http
from ivette.util.columns import select_targets

# ── must match the training thresholds ───────────────────────────
MIN_TARGET_COVERAGE = 0.01
MIN_SAMPLES = 1


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
    dim = "3d" if use_3d else "2d"
    url = f"{http.PUBCHEM_PUG}/compound/cid/{cid}/SDF?record_type={dim}"
    path = out_dir / f"{cid}.sdf"
    if path.exists():
        return True  # already downloaded

    try:
        resp = http.get(url)
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
    parser.add_argument("--input", required=True, help="merged_dataset.csv")
    parser.add_argument("--output-dir", default="sdfs", help="Directory for SDF files")
    parser.add_argument("--delay", type=float, default=0.25, help="Seconds between requests")
    parser.add_argument("--2d-only", dest="two_d", action="store_true",
                        help="Download 2D SDFs only")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input, low_memory=False)

    targets = select_targets(df, MIN_TARGET_COVERAGE)
    print(f"Qualifying targets : {len(targets)}")

    cids = qualifying_cids(df, targets)
    print(f"\nUnique CIDs to download: {len(cids)}")

    ok = fail = skip = 0
    ordered = sorted(cids, key=lambda x: int(x) if x.isdigit() else 0)
    for i, cid in enumerate(ordered, 1):
        if (out_dir / f"{cid}.sdf").exists():
            skip += 1
            continue
        if download_sdf(cid, out_dir, use_3d=not args.two_d):
            ok += 1
        else:
            fail += 1
        if i % 50 == 0:
            print(f"[{i}/{len(cids)}] ok={ok} fail={fail} skip={skip}")
        time.sleep(args.delay)

    print()
    print(f"Done. Downloaded={ok} Failed={fail} Already present={skip}")
    print(f"SDF files written to: {out_dir}")


if __name__ == "__main__":
    main()