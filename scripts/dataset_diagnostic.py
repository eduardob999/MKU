#!/usr/bin/env python3
"""
dataset_diagnostic.py

Analyzes a wide-format CSV with chemical features and pharmacological targets.
Produces a sparsity report, target coverage summary, and feature completeness
to guide preprocessing and modeling decisions.

Usage:
  python dataset_diagnostic.py --input thermo_wide_values_only.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Default report sink — data/exports/ keeps the repo root clean.
_DEFAULT_DIAGNOSTIC = Path(__file__).resolve().parents[1] / "data" / "exports" / "diagnostic.txt"


# ── Column classification ────────────────────────────────────────────────────

IDENTITY_COLS = {"CID", "InChIKey", "PubChemName", "PubChem_URL"}

THERMO_COLS = {
    "MW", "LogP", "Tm", "Tb", "Hvap", "IE", "EA", "CCS",
    "ExactMass", "MonoMass", "Complexity", "Heavy Atom Count",
    "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount",
    "TPSA", "Rotatable Bond Count", "Formal Charge", "Isotope Atom Count",
    "Defined Atom Stereocenter Count", "Undefined Atom Stereocenter Count",
    "Covalently-Bonded Unit Count", "Hydrogen Bond Acceptor Count",
    "Hydrogen Bond Donor Count",
}

ANTIMICROBIAL_KEYWORDS = {
    "Staphylococcus", "Escherichia", "Enterococcus", "Proteus",
    "Providencia", "Pseudomonas", "Klebsiella", "Salmonella",
    "Bacillus", "MIC", "MBC", "TIME",
}

ANTIPARASITIC_KEYWORDS = {
    "Leishmania", "Trypanosoma", "Plasmodium", "Giardia",
    "Trichomonas","reductase", "oxidoreductase",
}

CYTOTOXICITY_KEYWORDS = {
    "HepG2", "Vero", "THP-1", "CC50", "GI50", "CD50",
    "MCF7", "HeLa", "A549", "HCT",
}


def classify_column(col: str) -> str:
    if col in IDENTITY_COLS:
        return "identity"
    if col in THERMO_COLS or not col.startswith("ChEMBL:"):
        return "feature"
    if any(k in col for k in ANTIMICROBIAL_KEYWORDS):
        return "target_antimicrobial"
    if any(k in col for k in ANTIPARASITIC_KEYWORDS):
        return "target_antiparasitic"
    if any(k in col for k in CYTOTOXICITY_KEYWORDS):
        return "target_cytotoxicity"
    return "target_other"


def parse_float(val: str):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Diagnose dataset sparsity and target coverage for ML preparation."
    )
    ap.add_argument("--input",   required=True,              help="Wide-format CSV")
    ap.add_argument("--output",  default=str(_DEFAULT_DIAGNOSTIC),   help="Report output path (default: data/exports/diagnostic.txt)")
    ap.add_argument("--target-min-coverage", type=float, default=0.05,
                    help="Min fraction of compounds with a value to keep a target (default: 0.05)")
    ap.add_argument("--feature-min-coverage", type=float, default=0.10,
                    help="Min fraction of compounds with a value to keep a feature (default: 0.10)")
    args = ap.parse_args(argv)

    with open(args.input, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    n = len(rows)
    if n == 0:
        raise SystemExit("Error: no data rows found.")

    print(f"Compounds: {n}")
    print(f"Columns:   {len(fieldnames)}")

    # Classify columns
    classified = defaultdict(list)
    for col in fieldnames:
        classified[classify_column(col)].append(col)

    # Coverage per column
    coverage = {}
    for col in fieldnames:
        filled = sum(1 for r in rows if parse_float(r.get(col)) is not None)
        coverage[col] = filled / n

    lines = []
    lines.append(f"Dataset Diagnostic Report")
    lines.append(f"{'='*60}")
    lines.append(f"Compounds : {n}")
    lines.append(f"Columns   : {len(fieldnames)}")
    lines.append("")

    # Summary per category
    for category in ["feature", "target_antimicrobial", "target_antiparasitic",
                     "target_cytotoxicity", "target_other"]:
        cols = classified[category]
        if not cols:
            continue
        coverages = [coverage[c] for c in cols]
        above_threshold = (
            args.target_min_coverage if "target" in category
            else args.feature_min_coverage
        )
        usable = [c for c in cols if coverage[c] >= above_threshold]

        lines.append(f"── {category.upper()} ({len(cols)} columns) ──")
        lines.append(f"   Usable (>= {above_threshold:.0%} coverage): {len(usable)}")
        lines.append(f"   Coverage range: "
                     f"{min(coverages):.1%} – {max(coverages):.1%}  "
                     f"median {sorted(coverages)[len(coverages)//2]:.1%}")
        lines.append("")

        # Show usable columns sorted by coverage descending
        for col in sorted(usable, key=lambda c: -coverage[c]):
            lines.append(f"   {coverage[col]:>6.1%}  {col}")
        lines.append("")

    # Feature completeness per compound
    feat_cols = classified["feature"]
    feat_completeness = []
    for r in rows:
        filled = sum(1 for c in feat_cols if parse_float(r.get(c)) is not None)
        feat_completeness.append(filled / len(feat_cols) if feat_cols else 0)
    mean_feat = sum(feat_completeness) / n
    compounds_sparse = sum(1 for v in feat_completeness if v < 0.3)

    lines.append(f"── FEATURE COMPLETENESS PER COMPOUND ──")
    lines.append(f"   Mean feature coverage per compound : {mean_feat:.1%}")
    lines.append(f"   Compounds with < 30% features filled: {compounds_sparse} ({compounds_sparse/n:.1%})")
    lines.append("")

    # Multi-target coverage: how many compounds have >= k targets
    target_cols = (
        classified["target_antimicrobial"] +
        classified["target_antiparasitic"] +
        classified["target_cytotoxicity"] +
        classified["target_other"]
    )
    target_counts = []
    for r in rows:
        count = sum(1 for c in target_cols if parse_float(r.get(c)) is not None)
        target_counts.append(count)

    lines.append(f"── TARGET COVERAGE PER COMPOUND ──")
    for threshold in [1, 2, 5, 10, 20]:
        n_above = sum(1 for c in target_counts if c >= threshold)
        lines.append(f"   >= {threshold:>2} target values: {n_above:>4} compounds ({n_above/n:.1%})")
    lines.append("")

    # Recommended targets (antimicrobial, above min coverage)
    recommended = [c for c in classified["target_antimicrobial"]
                   if coverage[c] >= args.target_min_coverage]
    lines.append(f"── RECOMMENDED ANTIMICROBIAL TARGETS ──")
    lines.append(f"   (>= {args.target_min_coverage:.0%} compound coverage)")
    if recommended:
        for col in sorted(recommended, key=lambda c: -coverage[c]):
            n_filled = int(coverage[col] * n)
            lines.append(f"   {coverage[col]:>6.1%}  ({n_filled:>4} compounds)  {col}")
    else:
        lines.append("   None meet the coverage threshold — consider lowering --target-min-coverage")
    lines.append("")

    # Next steps
    lines.append(f"── SUGGESTED NEXT STEPS ──")
    lines.append(f"   1. Use recommended targets above as y; train one model per target")
    lines.append(f"   2. Impute or drop features with < {args.feature_min_coverage:.0%} coverage")
    lines.append(f"   3. Join eMFP fingerprints on CID before modeling")
    lines.append(f"   4. Log-transform MIC/IC50/EC50 targets (typically log-normal distributed)")
    lines.append(f"   5. Stratified train/test split if any target has < 50 compounds")

    report = "\n".join(lines)
    print("\n" + report)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\nReport saved to '{args.output}'")


if __name__ == "__main__":
    raise SystemExit(main())
