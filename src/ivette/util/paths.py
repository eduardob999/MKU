"""Canonical filesystem layout for Ivette data storage.

Single source of truth for every ``data/`` directory and metadata file.
``PROJECT_ROOT`` is the repository root (this file lives at
``src/ivette/util/paths.py`` → ``parents[3]``).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

STRUCTURE_DIR = DATA_DIR / "structure"
STRUCTURE_METADATA_FILE = STRUCTURE_DIR / "metadata.json"

COMPOUND_DIR = DATA_DIR / "compounds"
COMPOUND_METADATA_FILE = COMPOUND_DIR / "metadata.json"

THERMO_DIR = DATA_DIR / "thermo"
THERMO_RUN_DIR = THERMO_DIR / "runs"
THERMO_METADATA_FILE = THERMO_DIR / "metadata.json"

MODEL_DIR = DATA_DIR / "models"
MODEL_RUN_DIR = MODEL_DIR / "runs"
MODEL_METADATA_FILE = MODEL_DIR / "metadata.json"

SDF_DIR = DATA_DIR / "sdfs"
SDF_RUN_DIR = SDF_DIR / "runs"
SDF_METADATA_FILE = SDF_DIR / "metadata.json"
