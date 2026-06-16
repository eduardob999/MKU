#!/usr/bin/env python3
"""
compute_fingerprints.py

Compute eMFP (count-based Morgan) fingerprints for compounds in a CSV file
containing a SMILES column.

Outputs a CSV with one count column per feature, keyed by CID, InChIKey, and SMILES.

Usage:
  python compute_fingerprints.py --input cid_inchikey.csv --output fingerprints.csv
  python compute_fingerprints.py --input cid_inchikey.csv --radius 3 --nbits 2048
"""

import argparse
import csv
import sys

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


# Generator cache — created once per (radius, nbits) combination
_gen_cache: dict[tuple, object] = {}


def get_generator(radius: int, nbits: int):
    key = (radius, nbits)
    if key not in _gen_cache:
        _gen_cache[key] = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=nbits
        )
    return _gen_cache[key]


def compute_emfp(smiles: str, radius: int, nbits: int) -> list[int] | None:
    """
    True eMFP: sum atom environment counts per radius level into a fixed-size vector.
    nbits here is the actual output dimensionality (e.g. 32, 64, 128).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = get_generator(radius, nbits)
    # GetCountFingerprint returns sparse counts hashed into nbits buckets
    # For true eMFP keep nbits small: 32-512
    fp = gen.GetCountFingerprint(mol)
    dense = [0] * nbits
    for idx, count in fp.GetNonzeroElements().items():
        dense[idx % nbits] += count  # fold into small vector
    return dense


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compute eMFP (count-based Morgan) fingerprints from a SMILES-containing CSV."
    )
    ap.add_argument("--input",      required=True,               help="Input CSV (must contain a SMILES column)")
    ap.add_argument("--output",     default="fingerprints.csv",  help="Output CSV (default: fingerprints.csv)")
    ap.add_argument("--smiles-col", default="SMILES",            help="Name of the SMILES column (default: SMILES)")
    ap.add_argument("--id-col",     default="CID",               help="Name of the ID column (default: CID)")
    ap.add_argument("--radius",     type=int, default=3,         help="Morgan radius (default: 3 = eMFP6)")
    ap.add_argument("--nbits",      type=int, default=2048,      help="Fingerprint feature count (default: 2048)")
    ap.add_argument("--keep-cols",  nargs="+", default=["InChIKey"],
                    help="Extra columns from input to carry through (default: InChIKey)")
    args = ap.parse_args(argv)

    with open(args.input, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise SystemExit(f"Error: '{args.input}' appears to be empty.")

        available = set(reader.fieldnames)
        if args.smiles_col not in available:
            raise SystemExit(
                f"Error: SMILES column '{args.smiles_col}' not found.\n"
                f"Available columns: {sorted(available)}"
            )

        carry = [c for c in args.keep_cols if c in available]
        missing_carry = set(args.keep_cols) - available
        if missing_carry:
            print(f"Warning: requested --keep-cols not found and will be skipped: {missing_carry}",
                  file=sys.stderr)

        rows = list(reader)

    feature_cols = [f"eMFP_{i}" for i in range(args.nbits)]
    id_cols = ([args.id_col] if args.id_col in available else []) + [args.smiles_col] + carry
    fieldnames = id_cols + feature_cols

    skipped = 0
    out_rows = []

    for row in rows:
        smiles = row.get(args.smiles_col, "").strip()
        if not smiles:
            skipped += 1
            continue

        counts = compute_emfp(smiles, radius=args.radius, nbits=args.nbits)
        if counts is None:
            print(f"Warning: could not parse SMILES '{smiles}' — skipping.", file=sys.stderr)
            skipped += 1
            continue

        out_row = {c: row.get(c, "") for c in id_cols}
        for i, v in enumerate(counts):
            out_row[f"eMFP_{i}"] = v
        out_rows.append(out_row)

    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} fingerprints to '{args.output}'")
    print(f"Radius: {args.radius} (eMFP{args.radius * 2}), features: {args.nbits}")
    if skipped:
        print(f"Skipped: {skipped} rows (invalid or missing SMILES)")


if __name__ == "__main__":
    raise SystemExit(main())
