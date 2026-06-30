"""Bayesian hyperparameter optimization for the training parameters.

Uses Optuna's Tree-structured Parzen Estimator (TPE) to search the XGBoost
hyperparameters (plus the fingerprint size), **optimising the honest CV score
under the requested grouping** so the search doesn't tune to leakage:

* ``random``   — shuffled K-fold (optimistic; analogs may straddle folds)
* ``cluster``  — Butina within-family CV (predict new analogs of this family)
* ``scaffold`` — Bemis–Murcko CV (novel-chemotype stress test)

The search space deliberately stays in regions that make sense for the small,
analog-heavy datasets this project trains on: shallow trees, real
regularisation (reg_alpha / reg_lambda / min_child_weight), and modest
fingerprint sizes. ``nbits`` is searched as a categorical (it changes the
feature matrix, so one matrix is built per candidate size).

**Nested cross-validation for an honest estimate.** Tuning and evaluating on the
same CV is optimistic — you're reporting the score of the config you picked
*because* it scored well there. So the honest number comes from nested CV:

* an **outer** grouped K-fold splits the data (with the chosen grouping, so no
  analog leaks across the split);
* inside each outer training fold an independent Optuna study tunes via **inner**
  CV, and the winning config is scored once on the held-out outer fold;
* the mean of those outer-fold scores estimates how the *whole tune-then-train
  procedure* generalises (``nested_cv_r2``).

The hyperparameters returned to save (``best_params``) come from one **final
study over all the data** — nested CV estimates the procedure, the final study
produces the model. ``best_score`` is that final study's inner CV (optimistic by
construction; ``nested_cv_r2`` is the figure to trust). When the data is too
small/low-diversity to build a grouped outer split, nested CV is skipped and
``nested_cv_r2`` is ``None``.

It is *search*, not back-propagation: these knobs aren't differentiable w.r.t.
the CV score, so TPE proposes promising configurations, measures them by
cross-validation, and learns where to look next.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

from ivette.core.feature_benchmark import _build_matrices
from ivette.core.modeling import evaluate_cv, select_features
from ivette.core.params import TrainingParams, to_dict
from ivette.core.train_xgboost_emfp import build_model

# Sensible fingerprint sizes to search (powers of two). Smaller suits the small,
# analog-heavy datasets here; larger only helps when there's enough data to use
# the extra resolution. The configured nbits is always added to this set.
NBITS_CHOICES = (256, 512, 1024, 2048)

# Default number of outer folds for the nested-CV honest estimate.
OUTER_FOLDS = 5


def _matrix(built):
    """The feature matrix to model on (augmented with DFT when present)."""
    base, augmented, _y, _groups = built
    return augmented if augmented is not None else base


def optimize_training_params(source, target, dft_df=None, *, smiles_col="SMILES",
                             base_tp=None, fsp=None, n_trials=50, grouping="cluster",
                             timeout=None, seed=42, outer_folds=OUTER_FOLDS):
    """Search XGBoost hyperparameters + fingerprint size to maximise the grouped
    CV R² for ``target`` under ``grouping`` (``random`` / ``cluster`` /
    ``scaffold``), with a nested-CV honest estimate.

    Returns a dict with the best parameters (a TrainingParams dict, ready to save
    as a preset, from a final full-data study), the final study's inner-CV score,
    the nested-CV honest R² (mean ± std over the outer folds, or ``None`` when no
    grouped outer split was feasible), the per-outer-fold scores, and the final
    study's per-trial history — or an ``error`` key.
    """
    import optuna

    base_tp = base_tp or TrainingParams()
    grouping = grouping if grouping in ("random", "cluster", "scaffold") else "cluster"
    # Grouping for the matrix builder: random has no groups of its own, so build
    # cluster labels (only used if we later need them — random CV ignores them).
    build_grouping = "cluster" if grouping == "random" else grouping
    strategy = "random" if grouping == "random" else "scaffold"

    # nbits changes the feature matrix, so build one matrix per candidate size
    # (and always include the configured size). y/groups are identical across
    # sizes; only the fingerprint block differs.
    sizes = sorted({int(base_tp.nbits), *(int(b) for b in NBITS_CHOICES)})
    built_by_nbits = {}
    for nb in sizes:
        built = _build_matrices(source, target, dft_df, smiles_col,
                                replace(base_tp, nbits=nb), build_grouping)
        if built is not None:
            built_by_nbits[nb] = built
    if not built_by_nbits:
        return {"error": "insufficient data or missing CID/SMILES/target columns",
                "target": target}
    sizes = sorted(built_by_nbits)

    _, _, y, groups = built_by_nbits[sizes[0]]
    y = np.asarray(y)
    start_tp = replace(base_tp)   # don't mutate the caller's params

    def _suggest(trial):
        """One sampled point in the (sensible) search space, as a params dict."""
        return dict(
            nbits=trial.suggest_categorical("nbits", sizes),
            # Shallow, well-regularised, modestly-paced trees — the region that
            # generalises on small analog-heavy data rather than memorising it.
            n_estimators=trial.suggest_int("n_estimators", 100, 1200),
            max_depth=trial.suggest_int("max_depth", 2, 7),
            learning_rate=trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0),
        )

    def _study_on(train_idx, *, study_seed):
        """Run an Optuna study tuning the inner grouped CV on ``train_idx``."""
        sub_groups = None if grouping == "random" else [groups[i] for i in train_idx]

        def objective(trial):
            tp = replace(start_tp, **_suggest(trial))
            X = _matrix(built_by_nbits[tp.nbits]).iloc[train_idx].reset_index(drop=True)
            yt = y[train_idx]
            if fsp is not None:
                X = select_features(X, yt, fsp)[0]
            cv = evaluate_cv(lambda: build_model(tp), X, yt, sub_groups, strategy=strategy,
                             max_folds=base_tp.cv_max_folds, n_repeats=base_tp.cv_repeats,
                             min_reliable_samples=base_tp.min_reliable_samples)
            score = cv["cv_r2_mean"]
            return score if score is not None else -1e9

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=study_seed))
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        return study

    def _fit_score(params, train_idx, test_idx):
        """Refit ``params`` on ``train_idx`` and score the held-out ``test_idx``."""
        tp = replace(start_tp, **params)
        Xfull = _matrix(built_by_nbits[tp.nbits])
        Xtr = Xfull.iloc[train_idx].reset_index(drop=True)
        Xte = Xfull.iloc[test_idx].reset_index(drop=True)
        ytr, yte = y[train_idx], y[test_idx]
        if fsp is not None:                       # selection fit on train only
            Xtr, kept, _ = select_features(Xtr, ytr, fsp)
            Xte = Xte[kept]
        model = build_model(tp)
        model.fit(Xtr.values, ytr)
        return float(r2_score(yte, model.predict(Xte.values)))

    # ── Nested CV: outer split → tune inside each fold → score its held-out part.
    n = len(y)
    idx = np.arange(n)
    n_groups = len(set(groups))
    if not outer_folds or outer_folds < 2:
        outer = None   # nested CV skipped by request
    elif grouping == "random":
        folds = min(outer_folds, max(2, n // 10))
        outer = list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(idx))
    elif n_groups >= 2:
        folds = min(outer_folds, n_groups)
        outer = list(GroupKFold(n_splits=folds).split(idx, groups=groups))
    else:
        outer = None   # too few groups for an honest grouped split

    fold_scores = []
    if outer is not None:
        for k, (tr, te) in enumerate(outer):
            study = _study_on(tr, study_seed=seed + 1 + k)
            fold_scores.append({
                "fold": k,
                "inner_best_cv": float(study.best_value),
                "test_r2": _fit_score(study.best_params, tr, te),
                "n_test": int(len(te)),
            })
    nested = [f["test_r2"] for f in fold_scores]
    nested_cv_r2 = float(np.mean(nested)) if nested else None
    nested_cv_std = float(np.std(nested)) if nested else None

    # ── Final study over ALL the data → the config to actually save/use.
    final = _study_on(idx, study_seed=seed)
    best_tp = replace(start_tp, **final.best_params)

    return {
        "target": target,
        "grouping": grouping,
        "n_samples": int(n),
        "n_groups": int(n_groups),
        "n_trials": len(final.trials),
        "best_score": float(final.best_value),
        "nested_cv_r2": nested_cv_r2,
        "nested_cv_std": nested_cv_std,
        "outer_folds": len(fold_scores),
        "fold_scores": fold_scores,
        "best_params": to_dict(best_tp),
        "tuned": dict(final.best_params),
        "history": [
            {"trial": t.number, "score": (float(t.value) if t.value is not None else None)}
            for t in final.trials
        ],
    }
