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

# Calculation sets (registered Gaussian runs on a geometry set; the outputs
# themselves live under each geometry set's own gaussian/ directory).
CALCULATION_DIR = DATA_DIR / "calculations"
CALCULATION_METADATA_FILE = CALCULATION_DIR / "metadata.json"

# DFT descriptor sets (per-compound properties parsed from Gaussian freq logs)
DFT_DESCRIPTOR_DIR = DATA_DIR / "dft_descriptors"
DFT_DESCRIPTOR_METADATA_FILE = DFT_DESCRIPTOR_DIR / "metadata.json"

# Cross-cutting output locations. Keeping these under data/ (which is
# .gitignored) is what keeps the repository root free of stray run artifacts.
LOG_DIR = DATA_DIR / "logs"          # application log + standalone-run timing logs
EXPORT_DIR = DATA_DIR / "exports"    # default sink for standalone CLI outputs
PRESET_DIR = DATA_DIR / "presets"    # named per-stage parameter presets (JSON)
RESUME_FILE = DATA_DIR / "resume.json"   # in-flight long-running runs (for the Resume menu)


def export_path(name: str) -> str:
    """Default output path for a standalone CLI artifact (``data/exports/<name>``).

    Used as argparse defaults so that running a sub-tool directly never drops
    files into the repository root. The interactive app always passes explicit
    paths into a run's own directory and so does not rely on this.
    """
    return str(EXPORT_DIR / name)


def log_path(name: str) -> str:
    """Default path for a log/timing artifact (``data/logs/<name>``)."""
    return str(LOG_DIR / name)
