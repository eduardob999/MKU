"""Test the Optuna hyperparameter sweep (small/fast)."""

import numpy as np
import pandas as pd

from ivette.core.hpo import optimize_training_params
from ivette.core.params import TrainingParams

_SMILES = ["O=[N+]([O-])c1ccccc1", "O=[N+]([O-])c1ccccc1C", "O=[N+]([O-])c1ccc(C)cc1",
           "c1ccncc1", "O=[N+]([O-])c1ccc(O)cc1", "O=[N+]([O-])c1cccc(C)c1"]


def _source(n=50, seed=0):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    return pd.DataFrame({
        "CID": [str(1000 + i) for i in range(n)],
        "SMILES": [_SMILES[i % len(_SMILES)] for i in range(n)],
        "feat1": f1,
        "IC50_target": 10 ** (0.5 * f1 + rng.normal(scale=0.1, size=n)),
    })


def test_optimize_returns_best_params_within_ranges():
    base = TrainingParams(min_samples=20, nbits=64, radius=2, cv_max_folds=3)
    res = optimize_training_params(_source(), "IC50_target", None,
                                   base_tp=base, n_trials=3, grouping="cluster")
    assert "error" not in res
    assert res["grouping"] == "cluster"
    assert res["n_trials"] == 3
    assert isinstance(res["best_score"], float)

    t = res["tuned"]
    assert 2 <= t["max_depth"] <= 8
    assert 100 <= t["n_estimators"] <= 1500
    assert 5e-3 <= t["learning_rate"] <= 0.3
    assert "reg_alpha" in t and "reg_lambda" in t and "min_child_weight" in t

    # best_params is a full, reloadable TrainingParams dict carrying the tuned values
    bp = res["best_params"]
    assert bp["max_depth"] == t["max_depth"]
    assert bp["reg_alpha"] == t["reg_alpha"]
    assert bp["nbits"] == 64                      # untuned fields preserved from base


def test_optimize_insufficient_data_errors():
    base = TrainingParams(min_samples=999)
    res = optimize_training_params(_source(n=10), "IC50_target", None,
                                   base_tp=base, n_trials=2)
    assert "error" in res
