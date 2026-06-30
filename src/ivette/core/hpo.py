"""Bayesian hyperparameter optimization for the training parameters.

Uses Optuna's Tree-structured Parzen Estimator (TPE) to search the XGBoost
hyperparameters, **optimising the honest within-family CV score** (cluster by
default) so the search doesn't tune to leakage. The space deliberately includes
regularisation (reg_alpha / reg_lambda / min_child_weight) and favours shallow
trees, which is what keeps the search from over-fitting on small datasets.

It is *search*, not back-propagation: these knobs aren't differentiable w.r.t.
the CV score, so TPE proposes promising configurations, measures them by
cross-validation, and learns where to look next.
"""

from __future__ import annotations

from dataclasses import replace

from ivette.core.feature_benchmark import _build_matrices
from ivette.core.modeling import evaluate_cv, select_features
from ivette.core.params import TrainingParams, to_dict
from ivette.core.train_xgboost_emfp import build_model


def optimize_training_params(source, target, dft_df=None, *, smiles_col="SMILES",
                             base_tp=None, fsp=None, n_trials=50, grouping="cluster",
                             timeout=None, seed=42):
    """Search XGBoost hyperparameters to maximise the grouped CV R² for ``target``.

    Returns a dict with the best parameters (a TrainingParams dict, ready to save
    as a preset), the best score, and the per-trial history — or an ``error`` key.
    """
    import optuna

    base_tp = base_tp or TrainingParams()
    built = _build_matrices(source, target, dft_df, smiles_col, base_tp, grouping)
    if built is None:
        return {"error": "insufficient data or missing CID/SMILES/target columns",
                "target": target}
    base, augmented, y, groups = built
    X = augmented if augmented is not None else base
    if fsp is not None:
        X = select_features(X, y, fsp)[0]
    start_tp = replace(base_tp)   # don't mutate the caller's params

    def objective(trial):
        tp = replace(
            start_tp,
            n_estimators=trial.suggest_int("n_estimators", 100, 1500),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0),
        )
        cv = evaluate_cv(lambda: build_model(tp), X, y, groups, strategy="scaffold",
                         max_folds=base_tp.cv_max_folds, n_repeats=base_tp.cv_repeats,
                         min_reliable_samples=base_tp.min_reliable_samples)
        score = cv["cv_r2_mean"]
        return score if score is not None else -1e9

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_tp = replace(start_tp, **study.best_params)
    return {
        "target": target,
        "grouping": grouping,
        "n_samples": int(len(y)),
        "n_groups": int(len(set(groups))),
        "n_trials": len(study.trials),
        "best_score": float(study.best_value),
        "best_params": to_dict(best_tp),
        "tuned": dict(study.best_params),
        "history": [
            {"trial": t.number, "score": (float(t.value) if t.value is not None else None)}
            for t in study.trials
        ],
    }
