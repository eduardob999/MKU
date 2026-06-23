"""Persistence layer for Ivette entities (structures, compounds, runs, models, SDFs).

All metadata lives in per-entity JSON files described by :mod:`ivette.util.paths`
and is accessed through :class:`ivette.util.jsonstore.MetadataStore` instances —
replacing the five hand-written load/save/next-id triplets that used to live in
``__main__``.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from ivette.util.jsonstore import MetadataStore
from ivette.util.paths import (
    STRUCTURE_DIR,
    STRUCTURE_METADATA_FILE,
    COMPOUND_DIR,
    COMPOUND_METADATA_FILE,
    THERMO_RUN_DIR,
    THERMO_METADATA_FILE,
    MODEL_RUN_DIR,
    MODEL_METADATA_FILE,
    SDF_RUN_DIR,
    SDF_METADATA_FILE,
)

# One store per entity. Args: (file, top-level key, id prefix).
STRUCTURES = MetadataStore(STRUCTURE_METADATA_FILE, "sets", "set")
COMPOUNDS = MetadataStore(COMPOUND_METADATA_FILE, "sets", "cset")
SDFS = MetadataStore(SDF_METADATA_FILE, "sets", "sdf")
MODELS = MetadataStore(MODEL_METADATA_FILE, "models", "model")
RUNS = MetadataStore(THERMO_METADATA_FILE, "runs", "run")


def _now():
    return datetime.now().isoformat(timespec="seconds")


def ensure_storage():
    """Create every data directory and an empty metadata file where missing."""
    for store in (STRUCTURES, COMPOUNDS, SDFS, MODELS, RUNS):
        store.ensure()
    for run_dir in (THERMO_RUN_DIR, SDF_RUN_DIR, MODEL_RUN_DIR):
        run_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Structure sets
# ---------------------------------------------------------------------------

def save_structure_set(structure_set, name):
    md = STRUCTURES.load()
    set_id = STRUCTURES.next_id(md)
    filename = f"{set_id}.csv"
    df = pd.DataFrame(structure_set["structures"])
    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(STRUCTURE_DIR / filename, index=False)
    md["sets"][set_id] = {
        "name": name,
        "file": filename,
        "created": _now(),
        "generator": structure_set["metadata"]["generator"],
        "parameters": {
            "ring_sizes": structure_set["metadata"]["ring_sizes"],
            "allowed_atoms": structure_set["metadata"]["allowed_atoms"],
        },
        "structure_count": len(df),
    }
    STRUCTURES.save(md)
    return set_id


def load_structure_set(set_id):
    info = STRUCTURES.get(set_id)
    df = pd.read_csv(STRUCTURE_DIR / info["file"])
    return info, df


# ---------------------------------------------------------------------------
# Compound sets
# ---------------------------------------------------------------------------

def save_compound_set(rows, name, source_set_id, parameters):
    md = COMPOUNDS.load()
    cset_id = COMPOUNDS.next_id(md)
    filename = f"{cset_id}.csv"
    df = pd.DataFrame(rows)
    COMPOUND_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMPOUND_DIR / filename, index=False)
    md["sets"][cset_id] = {
        "name": name,
        "file": filename,
        "created": _now(),
        "source_set_id": source_set_id,
        "parameters": parameters,
        "compound_count": len(df),
    }
    COMPOUNDS.save(md)
    return cset_id, df


def load_compound_set(cset_id):
    info = COMPOUNDS.get(cset_id)
    df = pd.read_csv(COMPOUND_DIR / info["file"])
    return info, df


def compound_sets_for_structure_set(set_id):
    """List of (cset_id, info) linked to a given structure set."""
    return [
        (cset_id, info)
        for cset_id, info in COMPOUNDS.items()
        if info.get("source_set_id") == set_id
    ]


# ---------------------------------------------------------------------------
# SDF sets
# ---------------------------------------------------------------------------

def register_sdf_set(model_id, target, name, output_dir, parameters, count):
    return SDFS.register({
        "name": name,
        "model_id": model_id,
        "target": target,
        "output_dir": str(output_dir),
        "created": _now(),
        "parameters": parameters,
        "compound_count": count,
    })


def sdf_sets_for_model(model_id, target=None):
    return [
        (sdf_id, info)
        for sdf_id, info in SDFS.items()
        if info.get("model_id") == model_id
        and (target is None or info.get("target") == target)
    ]


def find_model_sdf_sets(model_id):
    return sorted(
        Path(info["output_dir"])
        for info in SDFS.records().values()
        if info.get("model_id") == model_id
    )


def count_sdfs(folder):
    return len(list(folder.glob("*.sdf")))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def register_model(thermo_run_id, name, parameters, output_dir):
    return MODELS.register({
        "name": name,
        "thermo_run_id": thermo_run_id,
        "created": _now(),
        "parameters": parameters,
        "output_dir": str(output_dir),
    })


def models_for_run(run_id):
    return [
        (model_id, info)
        for model_id, info in MODELS.items()
        if info.get("thermo_run_id") == run_id
    ]


# ---------------------------------------------------------------------------
# Thermo runs
# ---------------------------------------------------------------------------

def register_run(cset_id, name, parameters, output_dir):
    return RUNS.register({
        "name": name,
        "cset_id": cset_id,
        "created": _now(),
        "parameters": parameters,
        "output_dir": str(output_dir),
        "status": "pending",
    })


def runs_for_compound_set(cset_id):
    return [
        (run_id, info)
        for run_id, info in RUNS.items()
        if info.get("cset_id") == cset_id
    ]


def update_run_status(run_id, status):
    RUNS.update(run_id, status=status)


def load_run_info(run_id):
    return RUNS.get(run_id)
