"""Canonical filesystem layout for Ivette data storage.

Single source of truth for every ``data/`` directory and metadata file.
``PROJECT_ROOT`` is the repository root (this file lives at
``src/ivette/util/paths.py`` → ``parents[3]``).

Entities (coherent vocabulary):
    Structure library   generated molecular structures      data/structures
    Compound library    PubChem compounds + physchem props  data/compounds
    Property dataset    find_thermo NIST/PubMed/pharma run   data/datasets
    Model               trained regressors per target       data/models
    Geometry set        3D SDF geometries for DFT            data/geometries
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

# Structure libraries
STRUCTURE_DIR = DATA_DIR / "structures"
STRUCTURE_METADATA_FILE = STRUCTURE_DIR / "metadata.json"

# Compound libraries
COMPOUND_DIR = DATA_DIR / "compounds"
COMPOUND_METADATA_FILE = COMPOUND_DIR / "metadata.json"

# Property datasets
DATASET_DIR = DATA_DIR / "datasets"
DATASET_RUN_DIR = DATASET_DIR / "runs"
DATASET_METADATA_FILE = DATASET_DIR / "metadata.json"

# Models
MODEL_DIR = DATA_DIR / "models"
MODEL_RUN_DIR = MODEL_DIR / "runs"
MODEL_METADATA_FILE = MODEL_DIR / "metadata.json"

# Geometry sets (3D structures + Gaussian)
GEOMETRY_DIR = DATA_DIR / "geometries"
GEOMETRY_RUN_DIR = GEOMETRY_DIR / "runs"
GEOMETRY_METADATA_FILE = GEOMETRY_DIR / "metadata.json"
GAUSSIAN_BENCHMARK_FILE = GEOMETRY_DIR / "gaussian_benchmark.json"
GAUSSIAN_BENCHMARK_RUN_DIR = GEOMETRY_DIR / "benchmark_runs"
