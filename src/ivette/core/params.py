"""Single source of truth for every tunable parameter, per pipeline stage.

Each stage is a small dataclass whose fields carry their default *and* a
human-readable ``help`` string (and, where relevant, a ``kind`` or ``choices``
hint for the UI editor). Nothing else in the codebase should hardcode these
values: menus read defaults from here, the reusable advanced-options editor
introspects the fields, and named presets are just serialised instances.

The data here is UI-agnostic (no ``rich``/``questionary``) so the same config
can be driven by the terminal today and a web form later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

from ivette.core.download_physchem import DEFAULT_PROPERTIES


# ── Per-stage parameter groups ───────────────────────────────────────────────

@dataclass
class StructureParams:
    ring_sizes: list = field(
        default_factory=lambda: [5, 6],
        metadata={"help": "Ring sizes to enumerate", "kind": "ints"})


@dataclass
class DownloadParams:
    max_records: int = field(
        default=500, metadata={"help": "Max PubChem records per substructure query"})
    batch_size: int = field(
        default=100, metadata={"help": "CIDs per PubChem properties request"})
    sleep: float = field(
        default=0.2, metadata={"help": "Seconds to wait between PubChem requests"})
    properties: list = field(
        default_factory=lambda: list(DEFAULT_PROPERTIES),
        metadata={"help": "Physicochemical properties to fetch", "kind": "strs"})


@dataclass
class DatasetParams:
    max_compounds: int = field(
        default=0, metadata={"help": "Max compounds to process (0 = all)"})
    pubmed_max: int = field(
        default=20, metadata={"help": "Max PubMed results per compound"})
    fetch_pharma: bool = field(
        default=False, metadata={"help": "Fetch pharmacology (PubChem / ChEMBL / BindingDB)"})
    pubchem_max_aids: int = field(
        default=10, metadata={"help": "Max PubChem bioassay AIDs per compound"})
    chembl_activity_limit: int = field(
        default=100, metadata={"help": "Max ChEMBL activities per compound"})
    chembl_max_pages: int = field(
        default=5, metadata={"help": "Max ChEMBL pages per compound"})
    merge_pharma: bool = field(
        default=False, metadata={"help": "Merge pharmacology into the wide ML output"})
    wide_from_clean: bool = field(
        default=True, metadata={"help": "Build the wide output via the clean_thermo pipeline"})
    fetch_pubmed: bool = field(
        default=False, metadata={"help": "Mine PubMed (opt-in; needs IVETTE_PUBMED_API_KEY) — literature signal only"})


@dataclass
class GaussianParams:
    method: str = field(
        default="PBE0", metadata={"help": "DFT functional (e.g. B3LYP, PBE0, M062X)"})
    basis_set: str = field(
        default="6-311G", metadata={"help": "Basis set (e.g. 6-311G, 6-311G(d,p), 6-311+G(d,p))"})
    preopt_mode: str = field(
        default="auto", metadata={"help": "Pre-optimisation before DFT",
                                  "choices": ["auto", "none", "pm7", "gaussian631g"]})
    timeout: int = field(
        default=0, metadata={"help": "Per-job timeout in seconds (0 = no limit)"})
    extra_keywords: str = field(
        default="",
        metadata={"help": "Extra Gaussian route-line keywords (e.g. NoTestMO SCF=(XQC,MaxCycle=200))"})


@dataclass
class TrainingParams:
    radius: int = field(
        default=2, metadata={"help": "Morgan fingerprint radius"})
    nbits: int = field(
        default=256, metadata={"help": "Fingerprint length (bits) — smaller suits small datasets"})
    n_estimators: int = field(
        default=500, metadata={"help": "XGBoost: number of trees"})
    max_depth: int = field(
        default=5, metadata={"help": "XGBoost: maximum tree depth"})
    learning_rate: float = field(
        default=0.03, metadata={"help": "XGBoost: learning rate"})
    subsample: float = field(
        default=0.8, metadata={"help": "XGBoost: row subsample fraction"})
    colsample_bytree: float = field(
        default=0.8, metadata={"help": "XGBoost: feature subsample fraction per tree"})
    reg_alpha: float = field(
        default=0.0, metadata={"help": "XGBoost: L1 regularization (higher = simpler model)"})
    reg_lambda: float = field(
        default=1.0, metadata={"help": "XGBoost: L2 regularization (higher = simpler model)"})
    min_child_weight: float = field(
        default=1.0, metadata={"help": "XGBoost: min child weight (higher = more conservative)"})
    min_samples: int = field(
        default=30, metadata={"help": "Minimum samples required to train a target"})
    cv_max_folds: int = field(
        default=5, metadata={"help": "Maximum cross-validation folds"})
    cv_repeats: int = field(
        default=1, metadata={"help": "Repeat CV N times and average (reduces variance on small data)"})
    log_dynamic_range: float = field(
        default=1000.0,
        metadata={"help": "Positive target max/min spread above which to log-transform"})
    cv_strategy: str = field(
        default="both",
        metadata={"help": "Which CV score(s) to report (cluster = within-family middle ground)",
                  "choices": ["both", "scaffold", "random", "cluster"]})
    cluster_cutoff: float = field(
        default=0.4,
        metadata={"help": "Butina cluster CV cutoff: group compounds with Tanimoto ≥ 1-cutoff. "
                          "Lower = only near-duplicates grouped (least aggressive, closer to "
                          "random); higher = whole families held out (most aggressive)"})
    cluster_fp_radius: int = field(
        default=2,
        metadata={"help": "Morgan radius for the cluster-CV similarity fingerprint. Raise to "
                          "tell tight analogs apart (more, finer cluster groups)"})
    cluster_fp_bits: int = field(
        default=1024,
        metadata={"help": "Bit length of the cluster-CV similarity fingerprint. Raise (e.g. "
                          "2048/4096) when analogs collide to the same FP and no cutoff can "
                          "split them"})
    min_reliable_samples: int = field(
        default=50,
        metadata={"help": "Below this sample count a target's CV score is flagged unreliable"})
    conformal: bool = field(
        default=False,
        metadata={"help": "Also report a conformal prediction interval half-width"})
    conformal_alpha: float = field(
        default=0.2, metadata={"help": "Conformal miscoverage (0.2 → 80% intervals)"})
    y_scramble_runs: int = field(
        default=0, metadata={"help": "Y-scramble sanity-check repeats (0 = off; ~5 recommended)"})


@dataclass
class FeatureSelectionParams:
    method: str = field(
        default="model",
        metadata={"help": "Main selector applied after the filters",
                  "choices": ["none", "univariate", "model"]})
    k_best: int = field(
        default=50, metadata={"help": "Max features the selector keeps (0 = keep all)"})
    score_func: str = field(
        default="mutual_info",
        metadata={"help": "Univariate scoring function",
                  "choices": ["mutual_info", "f_regression"]})
    variance_threshold: float = field(
        default=0.0, metadata={"help": "Drop features with variance below this (0 = off)"})
    correlation_threshold: float = field(
        default=0.95,
        metadata={"help": "Drop one of any feature pair above this |correlation| (1 = off)"})


# ── Stage registry ───────────────────────────────────────────────────────────

# key -> (human title, dataclass). The key is also the preset-store namespace.
STAGES: dict[str, tuple[str, type]] = {
    "structures": ("Structure generation", StructureParams),
    "download": ("Compound download", DownloadParams),
    "dataset": ("Property dataset", DatasetParams),
    "gaussian": ("Gaussian / DFT", GaussianParams),
    "training": ("Model training", TrainingParams),
    "feature_selection": ("Feature selection", FeatureSelectionParams),
}


# ── (de)serialisation + introspection ────────────────────────────────────────

@dataclass
class FieldInfo:
    name: str
    value: Any
    help: str
    kind: str                       # int | float | bool | str | ints | strs
    choices: Optional[list]


def to_dict(params) -> dict:
    """Plain JSON-serialisable dict of a parameter group (for presets)."""
    return asdict(params)


def from_dict(cls, data: Optional[dict]):
    """Build a parameter group from a dict, ignoring unknown keys and filling
    any missing field from its default — so presets survive code changes."""
    valid = {f.name for f in fields(cls)}
    kept = {k: v for k, v in (data or {}).items() if k in valid}
    return cls(**kept)


def _infer_kind(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, (list, tuple)):
        return "strs"
    return "str"


def describe(params) -> "list[FieldInfo]":
    """Field-by-field view (name, current value, help, kind, choices) for the UI."""
    out = []
    for f in fields(params):
        value = getattr(params, f.name)
        kind = f.metadata.get("kind") or _infer_kind(value)
        out.append(FieldInfo(
            name=f.name,
            value=value,
            help=f.metadata.get("help", ""),
            kind=kind,
            choices=f.metadata.get("choices"),
        ))
    return out


def _fmt_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def format_defaults() -> str:
    """Plain-text dump of every stage's default parameters (UI-agnostic).

    Generated from the dataclasses, so it can never drift from the actual
    defaults. Rendered by the CLI's Configuration view and by
    ``python -m ivette.core.params`` / ``python ivette.py --show-defaults``.
    """
    lines = ["Ivette default parameters  (source of truth: ivette/core/params.py)", ""]
    for key, (title, cls) in STAGES.items():
        infos = describe(cls())
        lines.append(f"[{key}]  {title}")
        width = max((len(fi.name) for fi in infos), default=0)
        for fi in infos:
            choices = f"  (choices: {', '.join(map(str, fi.choices))})" if fi.choices else ""
            lines.append(f"  {fi.name:<{width}} = {_fmt_value(fi.value):<10}  # {fi.help}{choices}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":   # python -m ivette.core.params
    print(format_defaults())
