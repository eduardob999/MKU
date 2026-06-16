#!/usr/bin/env python3
"""CLI to fetch pharmacology data (PubChem BioAssay + ChEMBL) for compounds listed in a CSV.

Input CSV must have columns `CID` and/or `InChIKey`.
Outputs `pharma_parsed.csv` by default with one activity row per line.
"""

import argparse
import csv
import sys
import time
from typing import List

from pharma_fetchers import fetch_pubchem_bioassays, fetch_chembl_activities_by_inchikey


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?", default="benzene.csv", help="Input CSV with CID/InChIKey")
    p.add_argument("--output", default="pharma_parsed.csv", help="Output CSV for pharmacology rows")
    p.add_argument("--max", type=int, default=200, help="Max compounds to process")
    p.add_argument("--chembl-target-cache", default="scripts/chembl_target_name_cache.json",
                   help="JSON cache file for ChEMBL target names")
    return p.parse_args()


def read_input(path: str):
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


def main():
    args = parse_args()
    rows = read_input(args.input)
    out_rows = []
    count = 0
    for r in rows:
        if args.max and count >= args.max:
            break
        cid = r.get('CID') or r.get('cid') or ''
        ik = r.get('InChIKey') or r.get('inchikey') or ''
        print(f'Fetching pharmacology for CID={cid} InChIKey={ik}')
        if cid:
            try:
                pc = fetch_pubchem_bioassays(cid)
                for a in pc:
                    a['CID'] = cid
                    a['InChIKey'] = ik
                    out_rows.append(a)
            except Exception as exc:
                print(f'PubChem fetch failed for CID {cid}: {exc}', file=sys.stderr)
        if ik:
            try:
                ch = fetch_chembl_activities_by_inchikey(ik, cache_path=args.chembl_target_cache)
                for a in ch:
                    a['CID'] = cid
                    a['InChIKey'] = ik
                    out_rows.append(a)
            except Exception as exc:
                print(f'ChEMBL fetch failed for InChIKey {ik}: {exc}', file=sys.stderr)
        count += 1
        time.sleep(0.2)

    # write CSV
    if out_rows:
        keys = sorted({k for d in out_rows for k in d.keys()})
        with open(args.output, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for row in out_rows:
                writer.writerow(row)
        print(f'Wrote pharmacology rows to {args.output}')
    else:
        print('No pharmacology rows retrieved')


if __name__ == '__main__':
    main()
