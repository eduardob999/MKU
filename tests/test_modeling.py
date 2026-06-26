"""Tests for the shared modeling primitives (transforms + leakage-safe CV).

These pin down the two correctness fixes:
  * targets are transformed per-column from their values (signed properties are
    never log-transformed), and
  * cross-validation groups by Bemis-Murcko scaffold so analogs don't leak.
"""

import numpy as np
import pandas as pd

from ivette.core.modeling import (
    apply_transform,
    cv_r2,
    decide_transform,
    inverse_transform,
    scaffold_groups,
)


# ── Target transform ────────────────────────────────────────────────────────

def test_signed_target_is_not_logged():
    # logP/enthalpy-like: spans negative→positive. The old code did
    # log10(clip(>=1e-12)), collapsing every negative to a constant -12.
    rng = np.random.default_rng(0)
    y = pd.Series(rng.uniform(-3.0, 6.0, size=200))
    assert decide_transform(y) == "identity"
    out = apply_transform(y, "identity")
    assert (out < 0).any()                       # negatives preserved
    assert np.allclose(out, y.to_numpy())


def test_positive_wide_range_target_is_logged():
    # activity/concentration-like: strictly positive across many decades.
    rng = np.random.default_rng(1)
    y = pd.Series(10.0 ** rng.uniform(-3.0, 4.0, size=200))
    assert decide_transform(y) == "log10"
    assert np.allclose(apply_transform(y, "log10"), np.log10(y.to_numpy()))


def test_positive_narrow_range_target_stays_linear():
    # MW-like: positive but only ~50..800 (≈16×, well under the log threshold).
    rng = np.random.default_rng(2)
    y = pd.Series(rng.uniform(50.0, 800.0, size=200))
    assert decide_transform(y) == "identity"


def test_small_sample_is_identity():
    # Too few values to judge distribution → leave it linear.
    assert decide_transform(pd.Series([1.0, 1000.0, 1e6])) == "identity"


def test_transform_roundtrip():
    y = np.array([1.0, 10.0, 100.0])
    assert np.allclose(inverse_transform(apply_transform(y, "log10"), "log10"), y)
    assert np.allclose(inverse_transform(apply_transform(y, "identity"), "identity"), y)


# ── Scaffold grouping ───────────────────────────────────────────────────────

def test_scaffold_groups_share_label_for_same_ring_system():
    # toluene + ethylbenzene share the benzene Murcko scaffold; pyridine differs.
    groups = scaffold_groups(["Cc1ccccc1", "CCc1ccccc1", "c1ccncc1"])
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]


def test_scaffold_groups_unique_for_invalid_and_acyclic():
    # unparseable + acyclic molecules each get their own label (no mega-group).
    groups = scaffold_groups(["not_a_smiles", "CCO", "also bad"])
    assert len(set(groups)) == 3


# ── Cross-validation ────────────────────────────────────────────────────────

def test_cv_r2_uses_scaffold_grouping_when_possible():
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(120, 3)))
    y = (X[0] * 2.0 + rng.normal(scale=0.1, size=120)).to_numpy()
    groups = [f"scaf_{i % 8}" for i in range(120)]
    mean, std, folds, method = cv_r2(lambda: LinearRegression(), X, y, groups)
    assert method == "scaffold"
    assert 2 <= folds <= 5
    assert mean > 0.8


def test_cv_r2_falls_back_to_random_with_single_group():
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(60, 2)))
    y = (X[0] + rng.normal(scale=0.1, size=60)).to_numpy()
    groups = ["only_one"] * 60
    _, _, _, method = cv_r2(lambda: LinearRegression(), X, y, groups)
    assert method == "random"
