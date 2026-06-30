"""Integration smoke for train_target: real XGBoost fit + scaffold CV + report.

Kept small/fast but exercises the full path the new modeling primitives feed
into, including the per-target transform choice and report metadata.
"""

import numpy as np
import pandas as pd

from ivette.core.train_xgboost_emfp import train_target


def _fast_model():
    """Tiny, single-threaded XGBoost so the integration tests run in seconds.

    Single-threaded (n_jobs=1) avoids oversubscribing the CPU during scaffold CV,
    which otherwise makes these tests crawl on a busy machine.
    """
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=20, max_depth=3, learning_rate=0.3,
        n_jobs=1, tree_method="hist", random_state=42,
    )

# A spread of ring systems so scaffold grouping has >1 group to work with.
_SCAFFOLDS = [
    "O=[N+]([O-])c1ccccc1",        # nitrobenzene
    "O=[N+]([O-])c1ccccc1C",       # nitrotoluene
    "c1ccncc1",                    # pyridine
    "c1ccc2ccccc2c1",              # naphthalene
    "c1ccoc1",                     # furan
]


def _make_frame(n=80, signed=True, seed=0):
    rng = np.random.default_rng(seed)
    smiles = [_SCAFFOLDS[i % len(_SCAFFOLDS)] for i in range(n)]
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    signal = 2.0 * f1 - 1.0 * f2 + rng.normal(scale=0.2, size=n)
    target = signal if signed else 10.0 ** (signal)   # signed vs positive-wide
    return pd.DataFrame({
        "CID": [str(1000 + i) for i in range(n)],
        "SMILES": smiles,
        "feat_1": f1,
        "feat_2": f2,
        "target": target,
    })


def test_train_target_signed_uses_identity_and_scaffold_cv():
    df = _make_frame(signed=True)
    result = train_target(df, "target", ["feat_1", "feat_2"], model_factory=_fast_model)
    assert result is not None
    _model, report, _imp = result
    assert report["transform"] == "identity"      # signed target left linear
    assert report["cv_method"] == "scaffold"
    assert report["n_samples"] == len(df)
    assert "cv_r2_mean" in report


def test_train_target_positive_wide_uses_log():
    df = _make_frame(signed=False)
    _model, report, _imp = train_target(
        df, "target", ["feat_1", "feat_2"], model_factory=_fast_model)
    assert report["transform"] == "log10"


def test_build_model_uses_training_params():
    from ivette.core.train_xgboost_emfp import build_model
    from ivette.core.params import TrainingParams
    m = build_model(TrainingParams(n_estimators=37, max_depth=2, learning_rate=0.5))
    p = m.get_params()
    assert p["n_estimators"] == 37
    assert p["max_depth"] == 2
    assert p["learning_rate"] == 0.5


def test_train_target_applies_feature_selection():
    from ivette.core.params import FeatureSelectionParams
    df = _make_frame(n=80, signed=True)
    rng = np.random.default_rng(9)
    for i in range(20):
        df[f"x{i}"] = rng.normal(size=len(df))
    feats = ["feat_1", "feat_2"] + [f"x{i}" for i in range(20)]
    fsp = FeatureSelectionParams(method="univariate", k_best=5,
                                 variance_threshold=0.0, correlation_threshold=1.0)
    _m, report, _imp = train_target(df, "target", feats, model_factory=_fast_model, fsp=fsp)
    assert report["n_features"] == 5            # selection cut 22 -> 5
    assert report["fs_method"] == "univariate"


def test_train_target_honors_tp_min_samples():
    from ivette.core.params import TrainingParams
    df = _make_frame(n=40, signed=True)
    # min_samples above the row count → target is skipped (returns None).
    result = train_target(df, "target", ["feat_1", "feat_2"],
                          tp=TrainingParams(min_samples=999), model_factory=_fast_model)
    assert result is None
