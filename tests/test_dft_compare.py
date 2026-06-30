"""Tests for the DFT comparison's configurable CV grouping (random vs cluster/scaffold)."""

import numpy as np
import pandas as pd

from ivette.core.dft_feature_test import compare_target_with_dft

_SMILES = ["O=[N+]([O-])c1ccccc1", "O=[N+]([O-])c1ccccc1C", "O=[N+]([O-])c1ccc(C)cc1",
           "c1ccncc1", "O=[N+]([O-])c1ccc(O)cc1", "O=[N+]([O-])c1cccc(C)c1"]


def _fast_model():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=15, max_depth=3, n_jobs=1,
                        tree_method="hist", random_state=42)


def _source(n=60, seed=0):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    return pd.DataFrame({
        "CID": [str(1000 + i) for i in range(n)],
        "SMILES": [_SMILES[i % len(_SMILES)] for i in range(n)],
        "feat1": f1,
        "IC50_target": 10 ** (0.5 * f1 + rng.normal(scale=0.1, size=n)),
    })


def _dft(n=60):
    rng = np.random.default_rng(1)
    return pd.DataFrame({"CID": [str(1000 + i) for i in range(n)],
                         "delta_gibbs_G": rng.normal(size=n)})


def test_compare_reports_random_vs_cluster_by_default():
    res = compare_target_with_dft(_source(), "IC50_target", _dft(),
                                  radius=2, nbits=64, model_factory=_fast_model)
    assert "error" not in res
    assert res["grouping"] == "cluster"
    assert "baseline_cv_r2_random" in res and "baseline_cv_r2_grouped" in res
    assert "augmented_cv_r2_grouped" in res


def test_compare_supports_scaffold_grouping():
    res = compare_target_with_dft(_source(), "IC50_target", _dft(),
                                  radius=2, nbits=64, grouping="scaffold",
                                  model_factory=_fast_model)
    assert res["grouping"] == "scaffold"
    assert "baseline_cv_r2_grouped" in res
