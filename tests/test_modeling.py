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
    evaluate_cv,
    inverse_transform,
    scaffold_groups,
    select_features,
)
from sklearn.linear_model import LinearRegression
from ivette.core.params import FeatureSelectionParams


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


# ── Dual-CV evaluation (random vs scaffold + reliability) ────────────────────

def test_evaluate_cv_reports_both_and_flags_small_n():
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(120, 3)))
    y = (X[0] * 2 + rng.normal(scale=0.1, size=120)).to_numpy()
    groups = [f"s{i % 10}" for i in range(120)]
    r = evaluate_cv(lambda: LinearRegression(), X, y, groups,
                    strategy="both", min_reliable_samples=50)
    assert r["cv_r2_random"] is not None and r["cv_r2_scaffold"] is not None
    assert r["cv_method"] == "scaffold"        # honest score is the primary
    assert r["reliable"] is True               # n=120, 10 scaffolds
    # a tiny subset is flagged unreliable
    r2 = evaluate_cv(lambda: LinearRegression(), X.iloc[:30], y[:30], groups[:30],
                     strategy="both", min_reliable_samples=50)
    assert r2["reliable"] is False


def test_evaluate_cv_random_only():
    rng = np.random.default_rng(6)
    X = pd.DataFrame(rng.normal(size=(80, 2)))
    y = (X[0] + rng.normal(scale=0.1, size=80)).to_numpy()
    groups = [f"s{i % 8}" for i in range(80)]
    r = evaluate_cv(lambda: LinearRegression(), X, y, groups, strategy="random")
    assert r["cv_r2_random"] is not None and r["cv_r2_scaffold"] is None
    assert r["cv_method"] == "random"


def test_evaluate_cv_repeated_averages_both_splits():
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(120, 3)))
    y = (X[0] * 2 + rng.normal(scale=0.1, size=120)).to_numpy()
    groups = [f"s{i % 10}" for i in range(120)]
    r = evaluate_cv(lambda: LinearRegression(), X, y, groups, strategy="both", n_repeats=3)
    assert r["cv_r2_scaffold"] is not None and r["cv_r2_random"] is not None


def test_evaluate_cv_scaffold_falls_back_to_random_with_one_group():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(60, 2)))
    y = (X[0] + rng.normal(scale=0.1, size=60)).to_numpy()
    groups = ["only"] * 60
    r = evaluate_cv(lambda: LinearRegression(), X, y, groups, strategy="scaffold")
    assert r["cv_r2_scaffold"] is None
    assert r["cv_r2_random"] is not None        # fallback when grouping impossible
    assert r["cv_method"] == "random"


# ── Within-family tools: cluster CV, applicability domain, conformal, scramble ─

def test_cluster_groups_groups_similar_and_separates_different():
    from ivette.core.modeling import cluster_groups
    smis = ["O=[N+]([O-])c1ccccc1", "O=[N+]([O-])c1ccccc1C",
            "O=[N+]([O-])c1ccc(C)cc1", "CCCCCCCCO"]
    g = cluster_groups(smis, cutoff=0.5)
    assert len(set(g[:3])) < 3          # the close nitroaromatics share a cluster
    assert g[3] not in g[:3]            # the aliphatic stands apart


def test_applicability_domain_flags_outsiders():
    from ivette.core.modeling import fit_applicability_domain, in_domain
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.normal(0, 1, size=(80, 4)))
    ad = fit_applicability_domain(X_train, k=5, percentile=95)
    near = pd.DataFrame(rng.normal(0, 1, size=(5, 4)))
    far = pd.DataFrame(rng.normal(20, 1, size=(5, 4)))
    assert in_domain(ad, near).mean() >= 0.6
    assert in_domain(ad, far).mean() == 0.0


def test_cross_conformal_returns_positive_halfwidth():
    from ivette.core.modeling import cross_conformal
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(100, 3)))
    y = (X[0] + rng.normal(scale=0.5, size=100)).to_numpy()
    hw, level = cross_conformal(lambda: LinearRegression(), X, y, alpha=0.2)
    assert hw > 0 and abs(level - 0.8) < 1e-9


def test_y_scramble_is_near_zero_for_real_signal():
    from ivette.core.modeling import y_scramble
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(120, 4)))
    y = (X[0] * 2 + rng.normal(scale=0.3, size=120)).to_numpy()
    assert y_scramble(lambda: LinearRegression(), X, y, n_repeats=4) < 0.3


# ── Feature selection ────────────────────────────────────────────────────────

def test_select_features_variance_filter_drops_constant():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"const": [1.0] * 50, "varying": rng.normal(size=50)})
    y = X["varying"].to_numpy()
    p = FeatureSelectionParams(method="none", variance_threshold=1e-6,
                               correlation_threshold=1.0, k_best=0)
    _Xr, kept, _scores = select_features(X, y, p)
    assert "const" not in kept and "varying" in kept


def test_select_features_correlation_filter_drops_redundant():
    rng = np.random.default_rng(1)
    a = rng.normal(size=80)
    X = pd.DataFrame({"a": a, "a_copy": a * 2.0, "b": rng.normal(size=80)})
    y = (a + X["b"]).to_numpy()
    p = FeatureSelectionParams(method="none", variance_threshold=0.0,
                               correlation_threshold=0.95, k_best=0)
    _Xr, kept, _scores = select_features(X, y, p)
    assert len(kept) == 2 and "b" in kept          # one of a/a_copy dropped


def test_select_features_univariate_topk_keeps_informative():
    rng = np.random.default_rng(2)
    n = 200
    signal = rng.normal(size=n)
    X = pd.DataFrame({"signal": signal,
                      **{f"noise{i}": rng.normal(size=n) for i in range(8)}})
    y = 3.0 * signal + rng.normal(scale=0.1, size=n)
    p = FeatureSelectionParams(method="univariate", score_func="f_regression",
                               k_best=3, variance_threshold=0.0, correlation_threshold=1.0)
    _Xr, kept, scores = select_features(X, y, p)
    assert len(kept) == 3 and "signal" in kept
    assert scores                                  # relevance scores returned for plotting


def test_select_features_model_topk_catches_nonlinear():
    rng = np.random.default_rng(3)
    n = 200
    signal = rng.normal(size=n)
    X = pd.DataFrame({"signal": signal,
                      **{f"noise{i}": rng.normal(size=n) for i in range(8)}})
    y = signal ** 2 + rng.normal(scale=0.1, size=n)   # nonlinear → model-based catches it
    p = FeatureSelectionParams(method="model", k_best=3,
                               variance_threshold=0.0, correlation_threshold=1.0)
    _Xr, kept, _scores = select_features(X, y, p)
    assert len(kept) == 3 and "signal" in kept
