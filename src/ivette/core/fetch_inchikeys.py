#!/usr/bin/env python3
"""Fetch InChIKeys (and SMILES) from PubChem for a list of CIDs."""

import argparse
import csv
import sys
import time

from ivette.util import http
from ivette.util.text import chunked

PROPERTIES = "InChIKey,IsomericSMILES,CanonicalSMILES".split(",")


def fetch_inchikeys(cids: list[str], max_retries: int = 5, debug: bool = False) -> dict[str, dict]:
    if not cids:
        return {}
    props = http.pubchem_fetch_properties(cids, PROPERTIES, max_retries=max_retries)
    if debug and props:
        print(f"  DEBUG first prop keys: {list(props[0].keys())}", file=sys.stderr)
    return {
        str(p["CID"]): {
            "InChIKey": p.get("InChIKey", ""),
            "SMILES": (
                p.get("IsomericSMILES")
                or p.get("CanonicalSMILES")
                or p.get("SMILES")
                or ""
            ),
        }
        for p in props
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Fetch InChIKeys from PubChem for CIDs in a CSV."
    )
    ap.add_argument("--input",     required=True,            help="Input CSV containing a CID column")
    ap.add_argument("--output",    default="cid_inchikey.csv", help="Output CSV (default: cid_inchikey.csv)")
    ap.add_argument("--cid-col",   default="CID",            help="Name of the CID column (default: CID)")
    ap.add_argument("--batch-size", type=int, default=200,   help="CIDs per request (default: 200)")
    ap.add_argument("--sleep",      type=float, default=0.5, help="Seconds between requests (default: 0.5)")
    ap.add_argument("--debug", action="store_true", help="Print raw API response keys for the first batch")
    args = ap.parse_args(argv)

    with open(args.input, newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or args.cid_col not in reader.fieldnames:
            raise SystemExit(
                f"Error: column '{args.cid_col}' not found in '{args.input}'.\n"
                f"Available columns: {sorted(reader.fieldnames or [])}"
            )
        cids = [row[args.cid_col].strip() for row in reader
                if row.get(args.cid_col, "").strip()]

    cids = list(dict.fromkeys(cids))  # deduplicate, preserve order
    print(f"Found {len(cids)} unique CIDs in '{args.input}'")

    results = {}
    total_batches = (len(cids) + args.batch_size - 1) // args.batch_size

    for i, batch in enumerate(chunked(cids, args.batch_size), start=1):
        print(f"  Fetching batch {i}/{total_batches} ({len(batch)} CIDs)...")
        try:
            batch_results = fetch_inchikeys(batch, max_retries=5, debug=args.debug and i == 1)
            if args.debug and i == 1 and batch_results:
                sample = next(iter(batch_results.values()))
                print(f"  Sample result: {sample}")
            results.update(batch_results)
        except Exception as e:
            print(f"  Batch {i} failed entirely: {e}", file=sys.stderr)
        time.sleep(args.sleep)

    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["CID", "InChIKey", "SMILES"])
        writer.writeheader()
        for cid in cids:
            entry = results.get(cid, {"InChIKey": "", "SMILES": ""})
            writer.writerow({"CID": cid, "InChIKey": entry["InChIKey"], "SMILES": entry["SMILES"]})

    matched = sum(1 for v in results.values() if v.get("InChIKey"))
    print(f"\nWrote {len(cids)} rows to '{args.output}' ({matched} InChIKeys fetched)")
    missing = len(cids) - matched
    if missing:
        print(f"Warning: {missing} CIDs returned no InChIKey", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
