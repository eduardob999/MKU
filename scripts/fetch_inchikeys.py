#!/usr/bin/env python3
"""
fetch_inchikeys.py

Fetch InChIKeys from PubChem for a list of CIDs.

Usage:
  python fetch_inchikeys.py --input wide.csv --output cid_inchikey.csv
"""

import argparse
import csv
import sys
import time

import requests

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def fetch_inchikeys(cids: list[str], max_retries: int = 5, debug: bool = False) -> dict[str, dict]:
    if not cids:
        return {}
    cid_str = ",".join(cids)
    url = f"{PUBCHEM_BASE}/compound/cid/{cid_str}/property/InChIKey,IsomericSMILES,CanonicalSMILES/JSON"

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            raw = resp.json()
            if debug and raw.get("PropertyTable", {}).get("Properties"):
                print(f"  DEBUG first prop keys: {list(raw['PropertyTable']['Properties'][0].keys())}",
                      file=sys.stderr)
            props = raw.get("PropertyTable", {}).get("Properties", [])
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
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            wait = min(4 * (attempt + 1), 30)
            print(f"  Connection error (attempt {attempt + 1}/{max_retries}), "
                  f"retrying in {wait}s: {e}", file=sys.stderr)
            time.sleep(wait)
        except requests.HTTPError as e:
            if resp.status_code == 400 and len(cids) > 1:
                mid = len(cids) // 2
                left  = fetch_inchikeys(cids[:mid], max_retries)
                right = fetch_inchikeys(cids[mid:], max_retries)
                return {**left, **right}
            elif resp.status_code in (429, 500, 502, 503, 504):
                wait = min(4 * (attempt + 1), 30)
                print(f"  HTTP {resp.status_code} (attempt {attempt + 1}/{max_retries}), "
                      f"retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
            else:
                raise

    raise requests.exceptions.ConnectionError(
        f"Failed to fetch properties after {max_retries} attempts for {len(cids)} CIDs"
    )


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
