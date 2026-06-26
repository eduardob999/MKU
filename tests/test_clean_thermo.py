"""Tests for clean_thermo data-quality fixes: unit conversion, physical bounds,
and the frequent/sparse coverage split."""

import numpy as np
import pandas as pd
import pytest

from ivette.core import clean_thermo as C


# ── Unit conversion ──────────────────────────────────────────────────────────

def test_to_canonical_celsius_to_kelvin():
    assert C.to_canonical_value("Tb", 100.0, "C") == pytest.approx(373.15)
    assert C.to_canonical_value("Tm", 0.0, "C") == pytest.approx(273.15)
    assert C.to_canonical_value("Tb", 373.15, "K") == pytest.approx(373.15)


def test_to_canonical_energy_and_mass():
    assert C.to_canonical_value("Hvap", 50000.0, "J/mol") == pytest.approx(50.0)
    assert C.to_canonical_value("Hvap", 50.0, "kJ/mol") == pytest.approx(50.0)
    assert C.to_canonical_value("MW", 180.0, "Da") == pytest.approx(180.0)


def test_to_canonical_unitless_and_missing_unit_passthrough():
    assert C.to_canonical_value("LogP", 2.5, None) == pytest.approx(2.5)      # unitless
    assert C.to_canonical_value("MW", 180.0, None) == pytest.approx(180.0)    # missing unit → assume canonical


def test_to_canonical_incompatible_unit_is_dropped():
    # A temperature carrying an energy unit can't be trusted → NaN (dropped later).
    assert np.isnan(C.to_canonical_value("Tb", 50.0, "kJ/mol"))


# ── The headline bug: mixed-unit values no longer averaged across units ──────

def test_mixed_unit_boiling_point_is_not_corrupted():
    # CID 1 has Tb = 100 °C (PubChem) and 373 K (NIST). Old code: median(100,373)=236.5.
    df = pd.DataFrame({
        "CID": ["1", "1"],
        "StandardPropertyName": ["Tb", "Tb"],
        "NumericValue": [100.0, 373.0],
        "CleanUnit": ["C", "K"],
        "Source": ["PubChem", "NIST"],
        "Reference": ["a", "b"],
    })
    df = C.add_canonical_values(df)
    df = df[df["NumericValue"].notna()]
    df = C.apply_physical_bounds(df)
    df = C.deduplicate_measurements(df)
    ml = C.generate_ml_dataset(df)
    tb = float(ml["Tb_Median"].iloc[0])
    assert tb == pytest.approx(373.07, abs=0.5)     # both ~373 K
    assert tb > 350                                 # definitely not the bogus 236.5


# ── Physical bounds ──────────────────────────────────────────────────────────

def test_apply_physical_bounds_drops_impossible_values():
    df = pd.DataFrame({
        "CID": ["1", "2", "3"],
        "StandardPropertyName": ["MW", "MW", "Tb"],
        "NumericValue": [180.0, 5000.0, 5.0e9],      # ok, too heavy, absurd temperature
    })
    kept = C.apply_physical_bounds(df)
    assert set(kept["CID"]) == {"1"}


def test_apply_physical_bounds_leaves_unbounded_properties():
    df = pd.DataFrame({"CID": ["1"], "StandardPropertyName": ["SomeNovelProp"],
                       "NumericValue": [1e6]})
    assert len(C.apply_physical_bounds(df)) == 1     # no bound defined → kept


# ── Coverage split ───────────────────────────────────────────────────────────

def test_split_by_coverage():
    df = pd.DataFrame({
        "CID": ["1", "2", "3", "1"],
        "StandardPropertyName": ["MW", "MW", "MW", "RareProp"],
        "NumericValue": [1.0, 2.0, 3.0, 9.0],
    })
    frequent, sparse = C.split_by_coverage(df, min_fraction=0.5)
    assert frequent == {"MW"}        # 3/3 compounds
    assert sparse == {"RareProp"}    # 1/3 compounds


def test_frequent_and_sparse_tables_are_disjoint():
    df = pd.DataFrame({
        "CID": ["1", "2", "3", "1"],
        "StandardPropertyName": ["MW", "MW", "MW", "RareProp"],
        "NumericValue": [100.0, 110.0, 120.0, 9.0],
    })
    freq, sparse = C.split_by_coverage(df, 0.5)
    main = C.generate_ml_dataset(df, include_props=freq)
    sp = C.generate_ml_dataset(df, include_props=sparse)
    assert any(c.startswith("MW") for c in main.columns)
    assert not any(c.startswith("RareProp") for c in main.columns)   # rare kept OUT of training
    assert any(c.startswith("RareProp") for c in sp.columns)         # but preserved here
