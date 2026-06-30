"""Shared modeling foundations: per-target transforms and leakage-safe CV.

Two correctness primitives used by both the trainer
(:mod:`ivette.core.train_xgboost_emfp`) and the DFT ablation
(:mod:`ivette.core.dft_feature_test`). Centralising them keeps the two in step
and makes them unit-testable in isolation.

1. Target transform (``decide_transform`` / ``apply_transform``)
   Choose a transform **per target column from its own values** instead of
   blindly ``log10``-ing everything. Concentration/activity endpoints (IC50,
   MIC, vapour pressure…) are strictly positive and span many orders of
   magnitude, so they are modelled far better in log space. Signed physical
   properties (logP, formation enthalpies, orbital energies) must be left
   untouched — ``log10`` of a clipped negative is meaningless and silently
   turns real data into a constant.

2. Leakage-safe cross-validation (``scaffold_groups`` / ``cv_r2``)
   Cross-validate with Bemis–Murcko scaffold groups so close analogs never
   straddle the train/test split. Plain shuffled K-fold lets near-duplicate
   molecules leak across folds and inflates R², which would also bias the
   DFT-helps-or-not conclusion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold, cross_val_score

# A strictly-positive target whose values span at least this factor (≈3 decades)
# is treated as log-distributed. MW (~16×) or Tm in K (~3×) stay linear; vapour
# pressures and activities (many decades) switch to log.
LOG_DYNAMIC_RANGE = 1000.0


# ── Target transform ────────────────────────────────────────────────────────

def decide_transform(values, *, min_samples: int = 30,
                     dynamic_range: float = LOG_DYNAMIC_RANGE) -> str:
    """Return ``"log10"`` or ``"identity"`` for a target from its distribution.

    Logs only when the values are strictly positive *and* span a large dynamic
    range (robust 5th–95th percentile ratio), so signed or narrow-range targets
    are never log-transformed.
    """
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if len(v) < min_samples:
        return "identity"
    if (v <= 0).any():
        return "identity"            # any signed/zero value rules out log
    lo = float(v.quantile(0.05))
    hi = float(v.quantile(0.95))
    if lo <= 0:
        lo = float(v.min())
    if lo > 0 and hi / lo >= dynamic_range:
        return "log10"
    return "identity"


def apply_transform(values, kind: str) -> np.ndarray:
    """Apply a transform chosen by :func:`decide_transform` to target values."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if kind == "log10":
        return np.log10(v)
    return v


def inverse_transform(values, kind: str) -> np.ndarray:
    """Invert :func:`apply_transform` to recover predictions in original units."""
    a = np.asarray(values, dtype=float)
    return np.power(10.0, a) if kind == "log10" else a


# ── Scaffold-grouped cross-validation ───────────────────────────────────────

def scaffold_groups(smiles_iter) -> list[str]:
    """Bemis–Murcko scaffold SMILES per molecule, for use as CV group labels.

    Molecules sharing a scaffold get the same label and are kept on the same
    side of every split. Acyclic molecules (empty scaffold) and unparseable
    SMILES get a unique per-row label so they are treated individually rather
    than collapsing into one oversized group that would distort fold sizes.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups: list[str] = []
    for i, smi in enumerate(smiles_iter):
        label = f"__unique_{i}"
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                if scaffold:                 # empty for acyclic molecules
                    label = scaffold
        except Exception:
            pass
        groups.append(label)
    return groups


def cv_r2(model_factory, X, y, groups=None, *,
          max_folds: int = 5, min_folds: int = 3):
    """Cross-validated R², scaffold-grouped when possible.

    Returns ``(mean_r2, std_r2, n_folds, method)`` where ``method`` is
    ``"scaffold"`` (GroupKFold over ``groups``) or ``"random"`` (shuffled KFold
    fallback, used when grouping is impossible — e.g. a single scaffold).
    """
    n = len(y)
    if groups is not None:
        n_groups = len(set(groups))
        folds = min(max_folds, max(min_folds, n // 10), n_groups)
        if folds >= 2 and n_groups >= 2:
            splitter = GroupKFold(n_splits=folds)
            scores = cross_val_score(model_factory(), X, y, cv=splitter,
                                     groups=groups, scoring="r2", n_jobs=-1)
            return float(np.mean(scores)), float(np.std(scores)), folds, "scaffold"

    folds = min(max_folds, max(2, n // 10))
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model_factory(), X, y, cv=cv, scoring="r2", n_jobs=-1)
    return float(np.mean(scores)), float(np.std(scores)), folds, "random"


def _repeated_grouped_scores(model_factory, X, y, groups, folds, n_repeats):
    """Repeated scaffold CV via shuffled group→fold assignment.

    GroupKFold is deterministic, so to actually vary scaffold folds across
    repeats we shuffle which scaffolds land in which fold each time. Averaging
    over repeats tames the wild fold-to-fold variance on small datasets.
    """
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    y = np.asarray(y)
    g = np.asarray([str(v) for v in groups])
    uniq = np.array(sorted(set(g)))
    scores = []
    for rep in range(max(1, n_repeats)):
        order = np.random.default_rng(rep).permutation(len(uniq))
        fold_of = {grp: i % folds for i, grp in enumerate(uniq[order])}
        for f in range(folds):
            test = np.array([fold_of[v] == f for v in g])
            train = ~test
            if test.sum() == 0 or train.sum() < 2:
                continue
            m = model_factory()
            m.fit(Xv[train], y[train])
            scores.append(r2_score(y[test], m.predict(Xv[test])))
    return scores


def evaluate_cv(model_factory, X, y, groups=None, *, strategy="both",
                max_folds=5, min_folds=3, min_reliable_samples=50, n_repeats=1):
    """Cross-validated R² under random and/or scaffold splits.

    ``strategy`` is ``"both"``, ``"scaffold"`` or ``"random"``. Returns a dict
    with ``cv_r2_random`` / ``cv_r2_scaffold`` (either may be ``None``), the
    *primary* ``cv_r2_mean`` (the scaffold/honest score when available, else
    random), and a ``reliable`` flag — small or low-diversity targets give
    wildly unstable scaffold scores, so they're marked rather than trusted.

    The gap between the random and scaffold scores is the leakage/over-fit
    meter: large gap ⇒ the score is mostly analog memorisation.
    """
    n = len(y)
    n_groups = len(set(groups)) if groups is not None else 0
    want_random = strategy in ("both", "random")
    want_scaffold = strategy in ("both", "scaffold")
    scaffold_ok = groups is not None and n_groups >= 2

    out = {
        "n_samples": int(n), "n_scaffold_groups": int(n_groups),
        "cv_r2_random": None, "cv_r2_random_std": None,
        "cv_r2_scaffold": None, "cv_r2_scaffold_std": None,
        "cv_folds": None, "cv_strategy": strategy,
    }

    if want_scaffold and scaffold_ok:
        folds = min(max_folds, max(min_folds, n // 10), n_groups)
        if folds >= 2:
            if n_repeats and n_repeats > 1:
                s = _repeated_grouped_scores(model_factory, X, y, groups, folds, n_repeats)
            else:
                s = cross_val_score(model_factory(), X, y, cv=GroupKFold(n_splits=folds),
                                    groups=groups, scoring="r2", n_jobs=-1)
            out["cv_r2_scaffold"] = float(np.mean(s))
            out["cv_r2_scaffold_std"] = float(np.std(s))
            out["cv_folds"] = folds

    # Random is computed when asked for, or as a fallback when scaffold grouping
    # was requested but impossible (a single scaffold).
    if want_random or (want_scaffold and not scaffold_ok):
        folds = min(max_folds, max(2, n // 10))
        if n_repeats and n_repeats > 1:
            cv = RepeatedKFold(n_splits=folds, n_repeats=n_repeats, random_state=42)
        else:
            cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        s = cross_val_score(model_factory(), X, y, cv=cv, scoring="r2", n_jobs=-1)
        out["cv_r2_random"] = float(np.mean(s))
        out["cv_r2_random_std"] = float(np.std(s))
        out["cv_folds"] = out["cv_folds"] or folds

    if out["cv_r2_scaffold"] is not None:
        out["cv_r2_mean"], out["cv_r2_std"], out["cv_method"] = (
            out["cv_r2_scaffold"], out["cv_r2_scaffold_std"], "scaffold")
    else:
        out["cv_r2_mean"], out["cv_r2_std"], out["cv_method"] = (
            out["cv_r2_random"], out["cv_r2_random_std"], "random")

    out["reliable"] = (n >= min_reliable_samples and n_groups >= 5)
    out["reliability_note"] = (
        "" if out["reliable"] else f"low data (n={n}, scaffolds={n_groups})")
    return out


def cluster_groups(smiles_iter, cutoff=0.4):
    """Butina cluster label per molecule (Tanimoto on Morgan FPs).

    A middle-ground grouping between random and scaffold: compounds with
    Tanimoto similarity above ``1 - cutoff`` share a cluster. Matches "predict
    new analogs of the family I already have" better than holding out whole
    scaffolds. Invalid SMILES get a unique label.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fps = []
    for smi in smiles_iter:
        m = Chem.MolFromSmiles(str(smi))
        fps.append(gen.GetFingerprint(m) if m is not None else None)

    idx = [i for i, f in enumerate(fps) if f is not None]
    labels = [f"__invalid_{i}" for i in range(len(fps))]
    if len(idx) >= 2:
        dists = []
        for a in range(1, len(idx)):
            sims = DataStructs.BulkTanimotoSimilarity(
                fps[idx[a]], [fps[idx[b]] for b in range(a)])
            dists.extend(1.0 - s for s in sims)
        clusters = Butina.ClusterData(dists, len(idx), cutoff, isDistData=True)
        for cid, members in enumerate(clusters):
            for m in members:
                labels[idx[m]] = f"cluster_{cid}"
    elif idx:
        labels[idx[0]] = "cluster_0"
    return labels


# ── Applicability domain ─────────────────────────────────────────────────────

def fit_applicability_domain(X_train, *, k=5, percentile=95.0):
    """Define the model's applicability domain from training features.

    Uses the k-nearest-neighbour mean distance within the training set; the
    in-domain threshold is the given percentile of those distances. Returns a
    dict consumable by :func:`ad_distance` / :func:`in_domain` at predict time.
    """
    from sklearn.neighbors import NearestNeighbors
    X = np.nan_to_num(np.asarray(X_train, dtype=float))
    k = max(1, min(k, len(X) - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)   # +1: first neighbour is self
    d, _ = nn.kneighbors(X)
    mean_d = d[:, 1:].mean(axis=1)
    return {"nn": nn, "k": k, "threshold": float(np.percentile(mean_d, percentile))}


def ad_distance(ad, X_new):
    """Mean distance of each new sample to its k nearest training neighbours."""
    X = np.nan_to_num(np.asarray(X_new, dtype=float))
    d, _ = ad["nn"].kneighbors(X, n_neighbors=ad["k"])
    return d.mean(axis=1)


def in_domain(ad, X_new):
    """Boolean mask: True where a sample falls inside the applicability domain."""
    return ad_distance(ad, X_new) <= ad["threshold"]


# ── Uncertainty + sanity checks ──────────────────────────────────────────────

def cross_conformal(model_factory, X, y, *, alpha=0.2, max_folds=5):
    """Split-conformal interval half-width at confidence ``1 - alpha``.

    Out-of-fold residuals calibrate the interval, so a prediction's band is
    ``ŷ ± halfwidth`` with ~(1-alpha) coverage. Returns (halfwidth, level).
    """
    from sklearn.model_selection import KFold, cross_val_predict
    n = len(y)
    folds = min(max_folds, max(2, n // 10))
    oof = cross_val_predict(model_factory(), X, y,
                            cv=KFold(folds, shuffle=True, random_state=42))
    resid = np.abs(np.asarray(y) - np.asarray(oof))
    return float(np.quantile(resid, 1.0 - alpha)), float(1.0 - alpha)


def y_scramble(model_factory, X, y, *, n_repeats=5, max_folds=5):
    """Mean CV R² with shuffled labels — should be ≈0 if the model fits signal.

    A value well above 0 means the model can 'fit' randomised targets, i.e. it's
    chasing noise (a real risk at small n).
    """
    scores = []
    for rep in range(max(1, n_repeats)):
        y_perm = np.asarray(y)[np.random.default_rng(rep).permutation(len(y))]
        r = evaluate_cv(model_factory, X, y_perm, None, strategy="random",
                        max_folds=max_folds)
        if r["cv_r2_random"] is not None:
            scores.append(r["cv_r2_random"])
    return float(np.mean(scores)) if scores else float("nan")


# ── Feature selection ────────────────────────────────────────────────────────

def _relevance_scores(X, y, method, score_func):
    """Per-feature relevance for the chosen selector (higher = more relevant).

    Returns a dict {column: score}. NaNs are median-imputed for scoring only —
    the caller keeps the original (NaN-bearing) values for XGBoost to handle.
    """
    Xf = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    if method == "model":
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                         n_jobs=1, random_state=42, tree_method="hist")
        m.fit(Xf.values, y)
        return dict(zip(X.columns, m.feature_importances_))
    # univariate
    from sklearn.feature_selection import f_regression, mutual_info_regression
    if score_func == "f_regression":
        f, _p = f_regression(Xf.values, y)
        arr = np.nan_to_num(f)
    else:
        arr = mutual_info_regression(Xf.values, y, random_state=42)
    return dict(zip(X.columns, arr))


def select_features(X, y, params):
    """Reduce a feature matrix per the :class:`FeatureSelectionParams` ``params``.

    Pipeline: variance filter → correlation filter → a top-K selector
    (``univariate`` or ``model``). Returns ``(X_reduced, kept_columns, scores)``
    where ``scores`` maps the *post-filter* features to their relevance (for the
    comparison/relevance plots). UI-free; applied per target in training.
    """
    X = X.copy()

    # 1. drop near-constant features
    if params.variance_threshold and params.variance_threshold > 0 and X.shape[1] > 0:
        var = X.var(axis=0, numeric_only=True)
        keep = [c for c in X.columns if var.get(c, 0.0) >= params.variance_threshold]
        X = X[keep]

    # 2. drop one of each highly-correlated pair
    if (params.correlation_threshold and params.correlation_threshold < 1.0
            and X.shape[1] > 1):
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        drop = [c for c in upper.columns if (upper[c] > params.correlation_threshold).any()]
        if drop:
            X = X.drop(columns=drop)

    # 3. supervised top-K selector
    scores = {}
    if params.method in ("univariate", "model") and X.shape[1] > 0:
        scores = _relevance_scores(X, y, params.method, params.score_func)
        if params.k_best and X.shape[1] > params.k_best:
            top = sorted(scores, key=lambda c: scores.get(c, 0.0), reverse=True)[:params.k_best]
            X = X[top]

    return X, list(X.columns), scores
