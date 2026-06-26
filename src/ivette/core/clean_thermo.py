#!/usr/bin/env python3
"""Clean long-format thermo data for ML-ready output.

This script ingests a parsed CSV from PubChem/NIST scraping and produces:
- thermo_cleaned.csv
- thermo_summary.csv
- thermo_ml.csv
- rare_properties.csv
- cleaning_report.txt

The workflow is intentionally vectorized for scale and easy extension.
"""

import argparse
import re
from typing import Dict, List

import numpy as np
import pandas as pd

from ivette.util.text import extract_numeric
from ivette.util.paths import export_path

PARSER_ARTIFACT_NAMES = [
    "Quantity",
    "Value",
    "Units",
    "Phase change data",
    "Reaction data",
    "Gas phase ion energetics data",
    "Condensed phase thermochemistry data",
]

NON_QUANTITATIVE_CATEGORIES = [
    "Physical Description",
    "Color/Form",
    "Odor",
    "Taste",
    "Drugs",
    "Cosmetics",
    "Food Additives",
    "Fragrances",
    "Pesticides",
    "Dietary Ingredients",
    "Food Contact Substances",
    "Flavoring Agents",
    "Polymers",
    "Lipids",
    "Other Products",
    "Chemical Classes",
    "SpringerMaterials Properties",
]

GARBAGE_PATTERNS = [
    r"For more",
    r"please visit",
    r"DOI:",
    r"\{\'ExternalTableName\'",
]

PROPERTY_NAME_MAPPING: Dict[str, str] = {
    "molecular weight": "MW",
    "exact mass": "ExactMass",
    "monoisotopic mass": "MonoMass",
    "xlogp3": "LogP",
    "logp": "LogP",
    "boiling point": "Tb",
    "melting point": "Tm",
    "t boil": "Tb",
    "t fus": "Tm",
    "dissociation constants": "pKa",
    "ionization potential": "IE",
    "ie (ev)": "IE",
    "ea (ev)": "EA",
    "ae (ev)": "EA",
    "heat of vaporization": "Hvap",
    "Δ vap h° (kj/mol)": "Hvap",
    "Δ vap h (kj/mol)": "Hvap",
    "Δ fus h (kj/mol)": "Hfus",
    "Δ f h° gas (kj/mol)": "Hf_gas",
    "Δ f h° liquid (kj/mol)": "Hf_liq",
    "Δ f h° solid (kj/mol)": "Hf_sol",
    "proton affinity (kj/mol)": "PA",
    "gas basicity (kj/mol)": "GB",
    "collision cross section": "CCS",
}

UNIT_MAPPING: Dict[str, str] = {
    "°c": "C",
    "c": "C",
    "k": "K",
    "kj/mol": "kJ/mol",
    "j/mol*k": "J/mol/K",
    "j/mol k": "J/mol/K",
    "j/mol/k": "J/mol/K",
    "j/mol": "J/mol",
    "ev": "eV",
    "da": "Da",
    "g/mol": "g/mol",
    "a2": "A2",
    "Å²": "A2",
    "ang^2": "A2",
    "angstrom^2": "A2",
    "angstroms^2": "A2",
    "cm^3/mol": "cm^3/mol",
    "m^3/mol": "m^3/mol",
}

ML_PROPERTY_ORDER = [
    "MW",
    "LogP",
    "Tm",
    "Tb",
    "Hvap",
    "Hfus",
    "Hf_gas",
    "Hf_liq",
    "Hf_sol",
    "PA",
    "GB",
    "IE",
    "EA",
    "CCS",
]

SOURCE_PRIORITY = {
    "NIST": 1,
    "PubChem": 2,
}

# Canonical unit per standardized property. Values are converted into this unit
# BEFORE aggregation so that, e.g., a boiling point reported in °C (PubChem) and
# in K (NIST) are combined correctly instead of being medianed across units.
# Properties absent here are unitless (LogP, pKa) or unmanaged — left untouched.
CANONICAL_UNIT: Dict[str, str] = {
    "Tm": "K", "Tb": "K",
    "Hvap": "kJ/mol", "Hfus": "kJ/mol",
    "Hf_gas": "kJ/mol", "Hf_liq": "kJ/mol", "Hf_sol": "kJ/mol",
    "PA": "kJ/mol", "GB": "kJ/mol",
    "IE": "eV", "EA": "eV",
    "MW": "g/mol", "ExactMass": "g/mol", "MonoMass": "g/mol",
    "CCS": "A2",
}

# Conversions keyed by (CleanUnit, canonical_unit). Identity pairs included so a
# value already in the canonical unit passes through. A (from, to) pair that is
# absent means the units are incompatible → the value is dropped (can't trust it).
_UNIT_CONVERSIONS = {
    ("C", "K"): lambda v: v + 273.15,
    ("K", "K"): lambda v: v,
    ("J/mol", "kJ/mol"): lambda v: v / 1000.0,
    ("kJ/mol", "kJ/mol"): lambda v: v,
    ("Da", "g/mol"): lambda v: v,
    ("g/mol", "g/mol"): lambda v: v,
    ("eV", "eV"): lambda v: v,
    ("A2", "A2"): lambda v: v,
}

# Physical plausibility bounds (in canonical units). Values outside [lo, hi] are
# dropped as misparses. Generous on purpose — only impossible values are removed.
PHYSICAL_BOUNDS: Dict[str, tuple] = {
    "MW": (1.0, 2000.0), "ExactMass": (1.0, 2000.0), "MonoMass": (1.0, 2000.0),
    "Tm": (1.0, 1500.0), "Tb": (1.0, 1500.0),
    "LogP": (-10.0, 15.0),
    "Hvap": (0.0, 500.0), "Hfus": (0.0, 300.0),
    "Hf_gas": (-2000.0, 2000.0), "Hf_liq": (-2000.0, 2000.0), "Hf_sol": (-2000.0, 2000.0),
    "PA": (0.0, 2000.0), "GB": (0.0, 2000.0),
    "IE": (1.0, 30.0), "EA": (-10.0, 10.0),
    "CCS": (10.0, 1000.0),
    "pKa": (-10.0, 50.0),
}


def to_canonical_value(prop, value, clean_unit):
    """Convert a measurement into its property's canonical unit.

    Returns the converted value, or ``np.nan`` when the value is missing or its
    unit is incompatible with the canonical unit. Unitless/unmanaged properties
    and rows with no unit are returned unchanged (a missing unit is assumed to
    already be canonical — true for e.g. PubChem molecular weight).
    """
    canon = CANONICAL_UNIT.get(prop)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if not canon:
        return value
    if not isinstance(clean_unit, str) or not clean_unit.strip():
        return value
    fn = _UNIT_CONVERSIONS.get((clean_unit.strip(), canon))
    return fn(value) if fn is not None else np.nan


def add_canonical_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ``NumericValue`` with values converted to each property's canonical
    unit, and record that unit in ``CanonicalUnit``."""
    df = df.copy()
    converted = [
        to_canonical_value(p, v, u)
        for p, v, u in zip(df["StandardPropertyName"], df["NumericValue"], df["CleanUnit"])
    ]
    df["NumericValue"] = converted
    df["CanonicalUnit"] = df["StandardPropertyName"].map(CANONICAL_UNIT)
    return df


def apply_physical_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose canonical value falls outside its property's plausible range."""
    keep = pd.Series(True, index=df.index)
    vals = df["NumericValue"]
    for prop, (lo, hi) in PHYSICAL_BOUNDS.items():
        is_prop = df["StandardPropertyName"] == prop
        out_of_range = is_prop & vals.notna() & ((vals < lo) | (vals > hi))
        keep &= ~out_of_range
    return df[keep].copy()


def split_by_coverage(df: pd.DataFrame, min_fraction: float):
    """Partition property names into (frequent, sparse) by unique-compound coverage."""
    total = df["CID"].nunique() or 1
    cov = (df.dropna(subset=["StandardPropertyName"])
             .groupby("StandardPropertyName")["CID"].nunique() / float(total))
    frequent = set(cov[cov >= min_fraction].index)
    sparse = set(cov[cov < min_fraction].index)
    return frequent, sparse


URL_REGEX = re.compile(r"^\s*(https?://|www\.)", flags=re.IGNORECASE)
DB_REF_REGEX = re.compile(r"^\s*[A-Za-z]{1,10}:[^\s]+\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean thermo property data for ML and summary outputs.")
    parser.add_argument("input", nargs="?", default=export_path("thermo_parsed.csv"), help="Input long-format CSV file")
    parser.add_argument("--output", default=export_path("thermo_cleaned.csv"), help="Cleaned long-format output")
    parser.add_argument("--summary-output", default=export_path("thermo_summary.csv"), help="Compound/property summary statistics")
    parser.add_argument("--ml-output", default=export_path("thermo_ml.csv"), help="Machine learning wide-format dataset")
    parser.add_argument("--rare-output", default=export_path("rare_properties.csv"), help="Rare property frequency export")
    parser.add_argument("--sparse-output", default=export_path("thermo_ml_sparse.csv"),
                        help="Wide table of the near-empty (rare) property columns, kept out of training but preserved")
    parser.add_argument("--report-output", default=export_path("cleaning_report.txt"), help="Cleaning report text file")
    parser.add_argument("--min-coverage", type=float, default=0.005,
                        help="Properties present in fewer than this fraction of compounds are moved to --sparse-output")
    return parser.parse_args()


def load_data(csv_path: str) -> pd.DataFrame:
    dtype = {"CID": str, "InChIKey": str, "PubChemName": str, "PubChem_URL": str,
             "Source": str, "Section": str, "Subsection": str, "PropertyName": str,
             "PropertyValue": str, "PropertyUnit": str, "Reference": str,
             "Method": str, "Comment": str, "Condition": str,
             "ReactionEquation": str, "SourceURL": str}
    df = pd.read_csv(csv_path, dtype=dtype, keep_default_na=False, na_values=["", "NA", "nan"])
    df = df.replace({"": np.nan})
    return df


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().replace({"": np.nan})


def remove_parser_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["PropertyName"].astype(str).str.strip().isin(PARSER_ARTIFACT_NAMES)
    mask |= df["PropertyName"].astype(str).str.strip().eq(df["PropertyValue"].astype(str).str.strip())

    non_empty = df[["PropertyName", "PropertyValue", "Reference", "Method", "Comment", "Condition", "ReactionEquation", "SourceURL"]].notna().sum(axis=1)
    mask |= non_empty <= 1

    return df[~mask].copy()


def remove_non_quantitative_categories(df: pd.DataFrame) -> pd.DataFrame:
    name = df["PropertyName"].astype(str).str.strip()
    return df[~name.isin(NON_QUANTITATIVE_CATEGORIES)].copy()


def remove_garbage_values(df: pd.DataFrame) -> pd.DataFrame:
    value = df["PropertyValue"].astype(str)
    mask = pd.Series(False, index=df.index)
    for pattern in GARBAGE_PATTERNS:
        mask |= value.str.contains(pattern, case=False, na=False)
    mask |= value.str.match(URL_REGEX)
    mask |= value.str.match(DB_REF_REGEX)
    return df[~mask].copy()


def standardize_property_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return np.nan
    normalized = name.strip().lower()
    return PROPERTY_NAME_MAPPING.get(normalized, name.strip())


def clean_unit_text(unit: str) -> str:
    if not isinstance(unit, str) or not unit.strip():
        return np.nan
    text = unit.strip()
    text = text.replace("°", "°").replace("Å", "Å").replace("Å", "Å")
    text = text.replace("*", "/").replace(" ", "")
    lower = text.lower()
    if lower in UNIT_MAPPING:
        return UNIT_MAPPING[lower]
    if lower.endswith("/mol*k"):
        return "J/mol/K"
    if lower.endswith("mol/k"):
        return "J/mol/K"
    if lower.endswith("kj/mol"):
        return "kJ/mol"
    if lower.endswith("/mol"):
        return lower
    normalized = lower.replace("degc", "c").replace("degc", "c").replace("degreec", "c")
    return UNIT_MAPPING.get(normalized, unit.strip())


def extract_unit_from_value(value: str) -> str:
    if not isinstance(value, str):
        return np.nan
    candidates = ["°C", "K", "kJ/mol", "J/mol*K", "J/mol/K", "eV", "Da", "g/mol", "Å²", "A2", "cm^3/mol"]
    for candidate in candidates:
        if candidate in value:
            return clean_unit_text(candidate)
    suffix_match = re.search(r"(?:°C|K|kJ/mol|J/mol\*K|J/mol/K|eV|Da|g/mol|Å²|A2)\b", value)
    if suffix_match:
        return clean_unit_text(suffix_match.group(0))
    return np.nan


def standardize_units(df: pd.DataFrame) -> pd.DataFrame:
    unit_series = df["PropertyUnit"].astype(str).replace({"nan": ""})
    clean_units = unit_series.map(clean_unit_text)

    # Ensure the Series uses object dtype so np.nan assignments are valid.
    # pd.StringDtype rejects float NaN — convert to plain object to avoid that.
    clean_units = clean_units.astype(object)

    missing = clean_units.isna()
    if missing.any():
        inferred = df.loc[missing, "PropertyValue"].astype(str).map(extract_unit_from_value)
        # extract_unit_from_value returns np.nan for no-match; object dtype accepts it fine.
        clean_units.loc[missing] = inferred.values

    df["CleanUnit"] = clean_units.replace({"": np.nan})
    return df


def apply_standardization(df: pd.DataFrame) -> pd.DataFrame:
    df["PropertyName"] = normalize_text(df["PropertyName"])
    df["PropertyValue"] = normalize_text(df["PropertyValue"])
    df["Reference"] = normalize_text(df["Reference"])
    df["Method"] = normalize_text(df["Method"])
    df["Comment"] = normalize_text(df["Comment"])
    df["Condition"] = normalize_text(df["Condition"])
    df["ReactionEquation"] = normalize_text(df["ReactionEquation"])
    df["SourceURL"] = normalize_text(df["SourceURL"])
    df["StandardPropertyName"] = df["PropertyName"].map(standardize_property_name)
    # extract_numeric returns None for non-numeric; astype gives a float NaN column.
    df["NumericValue"] = df["PropertyValue"].map(extract_numeric).astype("float64")
    df = standardize_units(df)
    return df


def deduplicate_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse same compound/property measurements into a single row using the median value.

    When multiple numeric values exist for the same CID and standardized property name,
    this function selects a representative row and replaces its NumericValue with the group median.
    """
    df = df.copy()
    df["SourcePriority"] = df["Source"].map(SOURCE_PRIORITY).fillna(999).astype(int)
    df["HasReference"] = df["Reference"].notna().astype(int)

    median_values = df.groupby(["CID", "StandardPropertyName"], dropna=False)["NumericValue"].median()
    df = df.sort_values(
        by=["CID", "StandardPropertyName", "HasReference", "SourcePriority"],
        ascending=[True, True, False, True],
    )
    representative = df.drop_duplicates(subset=["CID", "StandardPropertyName"], keep="first")
    median_map = median_values.to_dict()
    representative["NumericValue"] = representative.set_index(["CID", "StandardPropertyName"]).index.map(median_map)
    representative = representative.drop(columns=["SourcePriority", "HasReference"])
    return representative


def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for each compound/property pair using median as the central tendency."""
    grouped = df.groupby(["CID", "StandardPropertyName"], dropna=False)["NumericValue"]
    summary = grouped.agg(
        count="count",
        median="median",
        mean="mean",
        std="std",
        min="min",
        max="max",
        iqr=lambda x: x.quantile(0.75) - x.quantile(0.25),
    ).reset_index()
    return summary


def generate_ml_dataset(df: pd.DataFrame, include_props=None) -> pd.DataFrame:
    """Generate ML-ready wide-format dataset with columns grouped as Median/Count/IQR per property.

    ``include_props`` restricts which standardized properties become columns
    (used to split the frequent/sparse tables); ``None`` keeps all.
    """
    if include_props is not None:
        df = df[df["StandardPropertyName"].isin(include_props)]
    if df.empty:
        return pd.DataFrame(columns=["CID"])
    grouped = df.groupby(["CID", "StandardPropertyName"], dropna=False)["NumericValue"]
    summary = grouped.agg(
        median="median",
        count="count",
        iqr=lambda x: x.quantile(0.75) - x.quantile(0.25),
    ).reset_index()

    pivot_med = summary.pivot(index="CID", columns="StandardPropertyName", values="median")
    pivot_count = summary.pivot(index="CID", columns="StandardPropertyName", values="count")
    pivot_iqr = summary.pivot(index="CID", columns="StandardPropertyName", values="iqr")

    pivot_med.columns = [f"{col}_Median" for col in pivot_med.columns]
    pivot_count.columns = [f"{col}_Count" for col in pivot_count.columns]
    pivot_iqr.columns = [f"{col}_IQR" for col in pivot_iqr.columns]

    # Build interleaved column order: Median/Count/IQR per property
    all_props = summary["StandardPropertyName"].dropna().unique()
    ordered_columns = ["CID"]
    for prop in ML_PROPERTY_ORDER:
        if prop in all_props:
            for suffix in ("_Median", "_Count", "_IQR"):
                col = f"{prop}{suffix}"
                if col in pivot_med.columns or col in pivot_count.columns or col in pivot_iqr.columns:
                    ordered_columns.append(col)

    result = pivot_med.join(pivot_count, how="outer").join(pivot_iqr, how="outer").reset_index()

    # Append any properties not in ML_PROPERTY_ORDER, also interleaved
    known = set(ML_PROPERTY_ORDER)
    extra_props = [p for p in all_props if p not in known]
    for prop in extra_props:
        for suffix in ("_Median", "_Count", "_IQR"):
            col = f"{prop}{suffix}"
            if col in result.columns:
                ordered_columns.append(col)

    final_cols = [c for c in ordered_columns if c in result.columns]
    return result[final_cols]


def export_rare_properties(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    compound_counts = df.dropna(subset=["StandardPropertyName"]).groupby("StandardPropertyName")["CID"].nunique()
    total_compounds = df["CID"].nunique()
    rare = compound_counts.reset_index(name="CompoundCount")
    rare["FractionOfCompounds"] = rare["CompoundCount"] / float(total_compounds)
    rare = rare.sort_values(by="FractionOfCompounds")
    rare_rows = rare[rare["FractionOfCompounds"] < 0.005].copy()
    rare_rows.to_csv(output_path, index=False)
    return rare_rows


def build_report(original_count: int, cleaned_df: pd.DataFrame, raw_property_count: int, rare_rows: pd.DataFrame) -> List[str]:
    unique_properties_before = raw_property_count
    unique_properties_after = cleaned_df["StandardPropertyName"].nunique(dropna=True)
    top_50 = cleaned_df["StandardPropertyName"].value_counts().head(50)

    report = [
        f"Original rows: {original_count}",
        f"Rows retained: {len(cleaned_df)}",
        f"Rows removed: {original_count - len(cleaned_df)}",
        f"Unique compounds: {cleaned_df['CID'].nunique()}",
        f"Unique properties before cleaning: {unique_properties_before}",
        f"Unique properties after cleaning: {unique_properties_after}",
        "Top 50 remaining properties:",
    ]
    report += [f"  {prop}: {count}" for prop, count in top_50.items()]
    report.append("")
    report.append(f"Rare properties (<0.5% compounds): {len(rare_rows)}")
    if not rare_rows.empty:
        report.append("Sample rare properties:")
        for _, row in rare_rows.head(20).iterrows():
            report.append(f"  {row['StandardPropertyName']}: {row['CompoundCount']} compounds ({row['FractionOfCompounds']:.4f})")
    return report


def main() -> None:
    args = parse_args()
    raw_df = load_data(args.input)
    original_count = len(raw_df)
    raw_property_count = raw_df["PropertyName"].nunique(dropna=True)

    df = remove_parser_artifacts(raw_df)
    df = remove_non_quantitative_categories(df)
    df = remove_garbage_values(df)
    df = apply_standardization(df)
    df = df[df["StandardPropertyName"].notna()].copy()
    # Convert every value to its property's canonical unit BEFORE aggregation,
    # then drop rows that couldn't be converted or are physically impossible.
    df = add_canonical_values(df)
    df = df[df["NumericValue"].notna()].copy()
    df = apply_physical_bounds(df)
    df = deduplicate_measurements(df)

    df.to_csv(args.output, index=False)

    summary_df = generate_summary(df)
    summary_df.to_csv(args.summary_output, index=False)

    # Split property columns: frequent ones feed training; near-empty ones are
    # preserved in a separate wide table instead of being deleted or left to
    # pollute the feature matrix.
    frequent_props, sparse_props = split_by_coverage(df, args.min_coverage)
    generate_ml_dataset(df, include_props=frequent_props).to_csv(args.ml_output, index=False)
    generate_ml_dataset(df, include_props=sparse_props).to_csv(args.sparse_output, index=False)

    rare_df = export_rare_properties(df, args.rare_output)
    report_lines = build_report(original_count, df, raw_property_count, rare_df)
    report_lines.append("")
    report_lines.append(f"Frequent properties (≥{args.min_coverage:.1%} compounds): {len(frequent_props)} → {args.ml_output}")
    report_lines.append(f"Sparse properties (<{args.min_coverage:.1%} compounds): {len(sparse_props)} → {args.sparse_output}")
    with open(args.report_output, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines))

    print("Cleaning complete.")
    print("\n".join(report_lines))
    print(f"Cleaned data saved to: {args.output}")
    print(f"Summary statistics saved to: {args.summary_output}")
    print(f"ML dataset saved to: {args.ml_output}")
    print(f"Rare property export saved to: {args.rare_output}")
    print(f"Cleaning report saved to: {args.report_output}")


if __name__ == "__main__":
    main()
