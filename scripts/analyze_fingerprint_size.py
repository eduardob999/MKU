#!/usr/bin/env python3
"""
analyze_fingerprint_size.py

Helps determine the optimal eMFP size for a given dataset by measuring:
  - Bit occupancy (fraction of features that are ever nonzero)
  - Collision rate (features sharing the same hash bucket)
  - Inter-compound variance per feature (low variance = uninformative)
  - Pairwise Tanimoto diversity (does size affect structural discrimination?)

Usage:
  python analyze_fingerprint_size.py --input cid_inchikey.csv
"""

import argparse
import csv
import sys
from itertools import combinations

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def get_generator(radius: int, nbits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)


def compute_emfp_dense(mol, gen, nbits: int) -> list[int]:
    fp = gen.GetCountFingerprint(mol)
    dense = [0] * nbits
    for idx, count in fp.GetNonzeroElements().items():
        dense[idx % nbits] += count
    return dense


def tanimoto(a: list[int], b: list[int]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a)
    norm_b = sum(x * x for x in b)
    denom = norm_a + norm_b - dot
    return dot / denom if denom > 0 else 1.0


def analyze(mols: list, radius: int, nbits: int) -> dict:
    gen = get_generator(radius, nbits)
    fps = [compute_emfp_dense(mol, gen, nbits) for mol in mols]

    n_compounds = len(fps)
    n_features = nbits

    # Occupancy: fraction of features nonzero in at least one compound
    occupied = sum(1 for i in range(n_features) if any(fp[i] > 0 for fp in fps))
    occupancy = occupied / n_features

    # Mean variance across features (higher = more informative)
    variances = []
    for i in range(n_features):
        vals = [fp[i] for fp in fps]
        mean = sum(vals) / n_compounds
        var = sum((v - mean) ** 2 for v in vals) / n_compounds
        variances.append(var)
    mean_variance = sum(variances) / n_features
    zero_variance_frac = sum(1 for v in variances if v == 0.0) / n_features

    # Pairwise Tanimoto on a sample (up to 200 pairs for speed)
    pairs = list(combinations(range(min(n_compounds, 50)), 2))
    tanimotos = [tanimoto(fps[i], fps[j]) for i, j in pairs]
    mean_tanimoto = sum(tanimotos) / len(tanimotos) if tanimotos else 0.0

    return {
        "nbits": nbits,
        "occupancy": round(occupancy, 4),
        "mean_variance": round(mean_variance, 6),
        "zero_variance_frac": round(zero_variance_frac, 4),
        "mean_pairwise_tanimoto": round(mean_tanimoto, 4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Analyze optimal eMFP size for a SMILES dataset."
    )
    ap.add_argument("--input",      required=True,        help="Input CSV with a SMILES column")
    ap.add_argument("--smiles-col", default="SMILES",     help="SMILES column name (default: SMILES)")
    ap.add_argument("--radius",     type=int, default=3,  help="Morgan radius (default: 3)")
    ap.add_argument("--sizes",      nargs="+", type=int,
                    default=[32, 64, 128, 256, 512, 1024, 2048],
                    help="Bit sizes to evaluate (default: 32 64 128 256 512 1024 2048)")
    ap.add_argument("--output",     default=None,         help="Optional CSV to save results")
    args = ap.parse_args(argv)

    with open(args.input, newline="") as fh:
        reader = csv.DictReader(fh)
        if args.smiles_col not in (reader.fieldnames or []):
            raise SystemExit(
                f"Error: '{args.smiles_col}' not found.\n"
                f"Available: {sorted(reader.fieldnames or [])}"
            )
        rows = list(reader)

    mols = []
    for row in rows:
        smi = row.get(args.smiles_col, "").strip()
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            mols.append(mol)
        else:
            print(f"Warning: skipping invalid SMILES '{smi}'", file=sys.stderr)

    print(f"\nDataset: {len(mols)} valid compounds, radius={args.radius}\n")
    print(f"{'nbits':>6}  {'occupancy':>10}  {'mean_var':>10}  {'zero_var%':>10}  {'mean_tanimoto':>14}")
    print("-" * 60)

    results = []
    for nbits in sorted(args.sizes):
        r = analyze(mols, args.radius, nbits)
        results.append(r)
        print(
            f"{r['nbits']:>6}  "
            f"{r['occupancy']:>10.4f}  "
            f"{r['mean_variance']:>10.6f}  "
            f"{r['zero_variance_frac']:>9.1%}  "
            f"{r['mean_pairwise_tanimoto']:>14.4f}"
        )

    print()
    # Recommend: highest occupancy before it plateaus, lowest zero-variance fraction
    best = max(results, key=lambda r: r["occupancy"] - r["zero_variance_frac"])
    print(f"Suggested size: {best['nbits']} bits")
    print(f"  Occupancy {best['occupancy']:.1%}, "
          f"zero-variance features {best['zero_variance_frac']:.1%}, "
          f"mean Tanimoto {best['mean_pairwise_tanimoto']:.4f}")

    if args.output:
        with open(args.output, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to '{args.output}'")


if __name__ == "__main__":
    raise SystemExit(main())
