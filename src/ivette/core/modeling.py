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
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

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
