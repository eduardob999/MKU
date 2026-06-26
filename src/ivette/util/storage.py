"""Persistence layer for Ivette entities.

Entities (coherent vocabulary):
    structure library   generated molecular structures
    compound library    PubChem compounds + physicochemical properties
    property dataset     find_thermo NIST/PubMed/pharma mining run
    model                trained regressors per target
    geometry set         3D SDF geometries for DFT

All metadata lives in per-entity JSON files (see :mod:`ivette.util.paths`)
accessed through :class:`ivette.util.jsonstore.MetadataStore` instances.
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
    DATASET_RUN_DIR,
    DATASET_METADATA_FILE,
    MODEL_RUN_DIR,
    MODEL_METADATA_FILE,
    GEOMETRY_RUN_DIR,
    GEOMETRY_METADATA_FILE,
    DFT_DESCRIPTOR_DIR,
    DFT_DESCRIPTOR_METADATA_FILE,
    LOG_DIR,
    EXPORT_DIR,
    PRESET_DIR,
)

# One store per entity. Args: (file, top-level key, id prefix).
STRUCTURES = MetadataStore(STRUCTURE_METADATA_FILE, "structures", "structure")
COMPOUNDS = MetadataStore(COMPOUND_METADATA_FILE, "compounds", "compound")
DATASETS = MetadataStore(DATASET_METADATA_FILE, "datasets", "dataset")
MODELS = MetadataStore(MODEL_METADATA_FILE, "models", "model")
GEOMETRIES = MetadataStore(GEOMETRY_METADATA_FILE, "geometries", "geometry")
DFT_DESCRIPTORS = MetadataStore(DFT_DESCRIPTOR_METADATA_FILE, "dft_descriptors", "dft")


def _now():
    return datetime.now().isoformat(timespec="seconds")


def ensure_storage():
    """Create every data directory and an empty metadata file where missing."""
    for store in (STRUCTURES, COMPOUNDS, DATASETS, MODELS, GEOMETRIES, DFT_DESCRIPTORS):
        store.ensure()
    for run_dir in (DATASET_RUN_DIR, GEOMETRY_RUN_DIR, MODEL_RUN_DIR,
                    LOG_DIR, EXPORT_DIR, PRESET_DIR):
        run_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Structure libraries
# ---------------------------------------------------------------------------

def save_structure_library(structure_set, name):
    md = STRUCTURES.load()
    structure_id = STRUCTURES.next_id(md)
    filename = f"{structure_id}.csv"
    df = pd.DataFrame(structure_set["structures"])
    STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(STRUCTURE_DIR / filename, index=False)
    md["structures"][structure_id] = {
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
    return structure_id


def load_structure_library(structure_id):
    info = STRUCTURES.get(structure_id)
    df = pd.read_csv(STRUCTURE_DIR / info["file"])
    return info, df


# ---------------------------------------------------------------------------
# Compound libraries
# ---------------------------------------------------------------------------

def save_compound_library(rows, name, structure_id, parameters):
    md = COMPOUNDS.load()
    compound_id = COMPOUNDS.next_id(md)
    filename = f"{compound_id}.csv"
    df = pd.DataFrame(rows)
    COMPOUND_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMPOUND_DIR / filename, index=False)
    md["compounds"][compound_id] = {
        "name": name,
        "file": filename,
        "created": _now(),
        "source_structure_id": structure_id,
        "parameters": parameters,
        "compound_count": len(df),
    }
    COMPOUNDS.save(md)
    return compound_id, df


def load_compound_library(compound_id):
    info = COMPOUNDS.get(compound_id)
    df = pd.read_csv(COMPOUND_DIR / info["file"])
    return info, df


def compound_libraries_for_structure(structure_id):
    """List of (compound_id, info) linked to a given structure library."""
    return [
        (compound_id, info)
        for compound_id, info in COMPOUNDS.items()
        if info.get("source_structure_id") == structure_id
    ]


# ---------------------------------------------------------------------------
# Geometry sets
# ---------------------------------------------------------------------------

def register_geometry_set(model_id, target, name, output_dir, parameters, count):
    return GEOMETRIES.register({
        "name": name,
        "model_id": model_id,
        "target": target,
        "output_dir": str(output_dir),
        "created": _now(),
        "parameters": parameters,
        "compound_count": count,
    })


def geometry_sets_for_model(model_id, target=None):
    return [
        (geometry_id, info)
        for geometry_id, info in GEOMETRIES.items()
        if info.get("model_id") == model_id
        and (target is None or info.get("target") == target)
    ]


def find_model_geometry_sets(model_id, target=None):
    """Geometry-set directories for a model, optionally scoped to a single target.

    A geometry set belongs to exactly one (model, target) pair, so passing
    ``target`` keeps each set under its own target submenu only.
    """
    return sorted(
        Path(info["output_dir"])
        for info in GEOMETRIES.records().values()
        if info.get("model_id") == model_id
        and (target is None or info.get("target") == target)
    )


def count_geometries(folder):
    return len(list(folder.glob("*.sdf")))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def register_model(dataset_id, name, parameters, output_dir):
    return MODELS.register({
        "name": name,
        "dataset_id": dataset_id,
        "created": _now(),
        "parameters": parameters,
        "output_dir": str(output_dir),
    })


def models_for_dataset(dataset_id):
    return [
        (model_id, info)
        for model_id, info in MODELS.items()
        if info.get("dataset_id") == dataset_id
    ]


# ---------------------------------------------------------------------------
# Property datasets
# ---------------------------------------------------------------------------

def register_dataset(compound_id, name, parameters, output_dir):
    return DATASETS.register({
        "name": name,
        "compound_id": compound_id,
        "created": _now(),
        "parameters": parameters,
        "output_dir": str(output_dir),
        "status": "pending",
    })


def datasets_for_compound(compound_id):
    return [
        (dataset_id, info)
        for dataset_id, info in DATASETS.items()
        if info.get("compound_id") == compound_id
    ]


def update_dataset_status(dataset_id, status):
    DATASETS.update(dataset_id, status=status)


def load_dataset_info(dataset_id):
    return DATASETS.get(dataset_id)


# ---------------------------------------------------------------------------
# DFT descriptor sets (parsed Gaussian freq properties, per model+target)
# ---------------------------------------------------------------------------

def save_dft_descriptor_set(rows, name, model_id, target, geometry_id, parameters=None):
    md = DFT_DESCRIPTORS.load()
    dft_id = DFT_DESCRIPTORS.next_id(md)
    filename = f"{dft_id}.csv"
    df = pd.DataFrame(rows)
    DFT_DESCRIPTOR_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DFT_DESCRIPTOR_DIR / filename, index=False)
    md["dft_descriptors"][dft_id] = {
        "name": name,
        "file": filename,
        "created": _now(),
        "model_id": model_id,
        "target": target,
        "geometry_id": geometry_id,
        "parameters": parameters or {},
        "compound_count": len(df),
        "property_columns": [c for c in df.columns if c != "CID"],
    }
    DFT_DESCRIPTORS.save(md)
    return dft_id, df


def load_dft_descriptor_set(dft_id):
    info = DFT_DESCRIPTORS.get(dft_id)
    df = pd.read_csv(DFT_DESCRIPTOR_DIR / info["file"])
    return info, df


def dft_descriptor_sets_for_model(model_id, target=None):
    return [
        (dft_id, info)
        for dft_id, info in DFT_DESCRIPTORS.items()
        if info.get("model_id") == model_id
        and (target is None or info.get("target") == target)
    ]


def add_dft_comparison(dft_id, result):
    """Append a CV-comparison result to a DFT descriptor set; return (id, entry)."""
    md = DFT_DESCRIPTORS.load()
    record = md["dft_descriptors"][dft_id]
    comparisons = record.setdefault("comparisons", [])
    comp_id = f"cmp_{len(comparisons) + 1:03d}"
    entry = {"id": comp_id, "created": _now(), **result}
    comparisons.append(entry)
    DFT_DESCRIPTORS.save(md)
    return comp_id, entry
