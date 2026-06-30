"""Tests for the feature-selection benchmark engine."""

import numpy as np
import pandas as pd

from ivette.core.feature_benchmark import run_feature_selection_benchmark
from ivette.core.params import TrainingParams

# A few valid nitroaromatic SMILES (2 distinct Murcko scaffolds: benzene + pyridine).
_SMILES = [
    "O=[N+]([O-])c1ccccc1", "O=[N+]([O-])c1ccccc1C",
    "O=[N+]([O-])c1ccc(C)cc1", "c1ccncc1", "O=[N+]([O-])c1ccc(O)cc1",
]


def _source(n=40, seed=0):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    return pd.DataFrame({
        "CID": [str(1000 + i) for i in range(n)],
        "SMILES": [_SMILES[i % len(_SMILES)] for i in range(n)],
        "feat1": f1,
        "feat2": f2,
        "IC50_target": 10 ** (0.5 * f1 + rng.normal(scale=0.1, size=n)),  # positive, wide
    })


def _dft(n=40):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "CID": [str(1000 + i) for i in range(n)],
        "delta_gibbs_G": rng.normal(size=n),
        "anion_scf_energy": rng.normal(size=n),
    })


def _tp():
    # Small + fast for tests.
    return TrainingParams(n_estimators=15, max_depth=3, min_samples=20, nbits=64, radius=2)


def test_benchmark_runs_methods_and_blocks():
    res = run_feature_selection_benchmark(
        _source(), "IC50_target", _dft(), tp=_tp(), methods=["none", "univariate"])
    assert "error" not in res
    assert {r["block"] for r in res["results"]} == {"without_dft", "with_dft"}
    assert {r["method"] for r in res["results"]} == {"none", "univariate"}
    assert len(res["results"]) == 4
    for r in res["results"]:
        assert "cv_r2_random" in r and "cv_r2_scaffold" in r and "n_features" in r
    # univariate produced relevance scores (for the relevance plot); 'none' did not
    assert any(k.endswith(":univariate") for k in res["relevance"])


def test_benchmark_without_dft_only_has_one_block():
    res = run_feature_selection_benchmark(
        _source(), "IC50_target", None, tp=_tp(), methods=["none"])
    assert {r["block"] for r in res["results"]} == {"without_dft"}


def test_best_method_and_config_pick_highest_scaffold():
    from ivette.core.feature_benchmark import best_method, best_config
    res = {"results": [
        {"block": "with_dft", "method": "none", "cv_r2_scaffold": 0.10, "cv_r2_random": 0.5},
        {"block": "with_dft", "method": "model", "cv_r2_scaffold": 0.40, "cv_r2_random": 0.6},
        {"block": "with_dft", "method": "univariate", "cv_r2_scaffold": 0.30, "cv_r2_random": 0.55},
    ]}
    assert best_method(res) == "model"
    assert best_config(res).method == "model"


def test_benchmark_insufficient_data_returns_error():
    res = run_feature_selection_benchmark(
        _source(n=10), "IC50_target", None, tp=TrainingParams(min_samples=999), methods=["none"])
    assert "error" in res
