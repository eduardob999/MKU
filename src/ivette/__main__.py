#!/usr/bin/env python3
"""
Ivette CLI

Structure workflow:

Structure Sets
    |
    +-- set_000001.csv
    +-- set_000002.csv
    |
    +-- metadata.json

Compound Sets (downloaded from PubChem, linked to a structure set)

Compounds
    |
    +-- cset_000001.csv
    +-- cset_000002.csv
    |
    +-- metadata.json

Thermo (find_thermo runs linked to a compound set)

Thermo
    |
    +-- runs/
    |     +-- run_000001/
    |     |     report.csv
    |     |     available.csv
    |     |     parsed.csv
    |     |     pharma.csv
    |     |     wide.csv
    |     |     wide_values_only.csv
    |     |     cleaned.csv
    |     |     summary.csv
    |     |     rare.csv
    |     |     cleaning_report.csv
    |     |     timing_log.txt
    |     +-- run_000002/
    |           …
    +-- metadata.json

The CLI maintains context:
    - current mode
    - active structure set
    - active compound set
    - active run
"""


from pathlib import Path
from datetime import datetime
import json
import time
import sys

import pandas as pd

from ivette.core.generate_structures import generate_structures
from ivette.core.download_physchem import (
    get_cids_for_substructure,
    fetch_properties_for_cids,
    is_nitro_zwitterion,
    interleave_rows,
    DEFAULT_PROPERTIES,
)
from ivette.core.find_thermo import main as find_thermo_main
from ivette.core.train_xgboost_emfp import main as train_model_main
from ivette.core.train_pipeline import main as train_pipeline_main


# ============================================================
# CLI Context
# ============================================================

class IvetteContext:

    def __init__(self):

        self.mode = "Structure Sets"
        self.active_set = None
        self.active_compound_set = None
        self.active_run = None
        self.info = {}


    def clear(self):

        self.mode = "Structure Sets"
        self.active_set = None
        self.active_compound_set = None
        self.active_run = None
        self.info = {}



context = IvetteContext()



def render_header():

    print()

    print("=" * 60)

    print("IVETTE")

    print(
        f"Mode: {context.mode}"
    )


    if context.active_set:

        print(
            f"Set: {context.active_set}"
        )


    if context.active_compound_set:

        print(
            f"Compounds: {context.active_compound_set}"
        )


    if context.active_run:

        print(
            f"Run: {context.active_run}"
        )


    for key, value in context.info.items():

        print(
            f"{key}: {value}"
        )


    print("=" * 60)

    print()



# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STRUCTURE_DIR = (
    PROJECT_ROOT /
    "data" /
    "structure"
)

METADATA_FILE = (
    STRUCTURE_DIR /
    "metadata.json"
)

COMPOUND_DIR = (
    PROJECT_ROOT /
    "data" /
    "compounds"
)

COMPOUND_METADATA_FILE = (
    COMPOUND_DIR /
    "metadata.json"
)

THERMO_DIR = (
    PROJECT_ROOT /
    "data" /
    "thermo"
)

THERMO_RUN_DIR = (
    THERMO_DIR /
    "runs"
)

THERMO_METADATA_FILE = (
    THERMO_DIR /
    "metadata.json"
)

MODEL_DIR = (
    PROJECT_ROOT /
    "data" /
    "models"
)

MODEL_RUN_DIR = (
    MODEL_DIR /
    "runs"
)

MODEL_METADATA_FILE = (
    MODEL_DIR /
    "metadata.json"
)


# ============================================================
# Metadata — Structures
# ============================================================

def ensure_storage():

    STRUCTURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    if not METADATA_FILE.exists():

        with open(
            METADATA_FILE,
            "w"
        ) as f:

            json.dump(
                {"sets": {}},
                f,
                indent=4
            )


    COMPOUND_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    if not COMPOUND_METADATA_FILE.exists():

        with open(
            COMPOUND_METADATA_FILE,
            "w"
        ) as f:

            json.dump(
                {"sets": {}},
                f,
                indent=4
            )


    THERMO_RUN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    if not THERMO_METADATA_FILE.exists():

        with open(
            THERMO_METADATA_FILE,
            "w"
        ) as f:

            json.dump(
                {"runs": {}},
                f,
                indent=4
            )
    
    MODEL_RUN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    if not MODEL_METADATA_FILE.exists():

        with open(
            MODEL_METADATA_FILE,
            "w"
        ) as f:

            json.dump(
                {"models": {}},
                f,
                indent=4
            )



def load_metadata():

    with open(
        METADATA_FILE
    ) as f:

        return json.load(f)



def save_metadata(metadata):

    with open(
        METADATA_FILE,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )



def next_set_id(metadata):

    ids = metadata["sets"].keys()

    if not ids:

        return "set_000001"


    numbers = [
        int(x.split("_")[1])
        for x in ids
    ]

    return (
        f"set_{max(numbers)+1:06d}"
    )



# ============================================================
# Metadata — Compounds
# ============================================================

def load_compound_metadata():

    with open(
        COMPOUND_METADATA_FILE
    ) as f:

        return json.load(f)



def save_compound_metadata(metadata):

    with open(
        COMPOUND_METADATA_FILE,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )



def next_cset_id(metadata):

    ids = metadata["sets"].keys()

    if not ids:

        return "cset_000001"


    numbers = [
        int(x.split("_")[1])
        for x in ids
    ]

    return (
        f"cset_{max(numbers)+1:06d}"
    )


# ============================================================
# Metadata — Models
# ============================================================

def load_model_metadata():

    with open(
        MODEL_METADATA_FILE
    ) as f:

        return json.load(f)



def save_model_metadata(metadata):

    with open(
        MODEL_METADATA_FILE,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )



def next_model_id(metadata):

    ids = metadata["models"].keys()

    if not ids:

        return "model_000001"


    numbers = [
        int(x.split("_")[1])
        for x in ids
    ]

    return (
        f"model_{max(numbers)+1:06d}"
    )



def register_model(
    thermo_run_id,
    name,
    parameters,
    output_dir
):

    metadata = load_model_metadata()

    model_id = next_model_id(
        metadata
    )


    metadata["models"][model_id] = {

        "name": name,

        "thermo_run_id":
            thermo_run_id,

        "created":
            datetime.now()
            .isoformat(timespec="seconds"),

        "parameters":
            parameters,

        "output_dir":
            str(output_dir)

    }


    save_model_metadata(
        metadata
    )


    return model_id



def models_for_run(run_id):

    metadata = load_model_metadata()

    return [
        (model_id, info)
        for model_id, info
        in metadata["models"].items()
        if info.get("thermo_run_id") == run_id
    ]


def show_model(model_id):

    metadata = load_model_metadata()

    info = metadata["models"][model_id]


    context.mode = "Model"

    context.info = {

        "Status":
            "Available",

        "Created":
            info["created"].replace("T", " "),

        "Thermo Run":
            info["thermo_run_id"],

    }


    context.active_run = None

    render_header()


    print(
        f"Model: {info['name']}"
    )


    print()

    print(
        "Parameters:"
    )

    for key, value in info["parameters"].items():

        print(
            f"  {key}: {value}"
        )


    return info


def browse_model_outputs(model_id):

    metadata = load_model_metadata()

    info = metadata["models"][model_id]

    output_dir = Path(
        info["output_dir"]
    )


    show_model(model_id)


    print(
        "\nOutput files:\n"
    )


    files = sorted(
        output_dir.iterdir()
    )


    if not files:

        print(
            "  (none)"
        )

        return


    for f in files:

        size = (
            f.stat().st_size /
            1024
        )

        print(
            f"  {f.name:<40}"
            f"{size:>8.1f} KB"
        )


def load_model_report(model_id):

    metadata = load_model_metadata()

    info = metadata["models"][model_id]

    output_dir = Path(
        info["output_dir"]
    )

    report = (
        output_dir /
        "model_report.csv"
    )

    if not report.exists():

        return pd.DataFrame()


    df = pd.read_csv(
        report
    )

    if "cv_r2_mean" in df.columns:

        df = df.sort_values(
            "cv_r2_mean",
            ascending=False
        )


    return df.reset_index(drop=True)


def show_model_importance(
    model_id,
    target
):

    metadata = load_model_metadata()

    info = metadata["models"][model_id]

    output_dir = Path(
        info["output_dir"]
    )


    safe_name = (
        str(target)
        .replace("/", "_")
        .replace(":", "_")
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "_")
    )


    importance_file = (
        output_dir /
        f"{safe_name}_importance.csv"
    )


    render_header()


    if not importance_file.exists():

        print(
            "Importance file not found."
        )

        return


    df = pd.read_csv(
        importance_file
    )


    print(
        df.head(30)
        .to_string(index=False)
    )


# ============================================================
# Structure Sets
# ============================================================

def save_structure_set(
    structure_set,
    name
):

    metadata = load_metadata()

    set_id = next_set_id(
        metadata
    )


    filename = (
        f"{set_id}.csv"
    )


    df = pd.DataFrame(
        structure_set["structures"]
    )


    df.to_csv(
        STRUCTURE_DIR / filename,
        index=False
    )


    metadata["sets"][set_id] = {

        "name": name,

        "file": filename,

        "created":
            datetime.now()
            .isoformat(timespec="seconds"),

        "generator":
            structure_set["metadata"]
            ["generator"],

        "parameters": {

            "ring_sizes":
                structure_set["metadata"]
                ["ring_sizes"],

            "allowed_atoms":
                structure_set["metadata"]
                ["allowed_atoms"]

        },

        "structure_count":
            len(df)

    }


    save_metadata(
        metadata
    )


    return set_id



def load_structure_set(set_id):

    metadata = load_metadata()

    info = metadata["sets"][set_id]


    df = pd.read_csv(
        STRUCTURE_DIR /
        info["file"]
    )

    return info, df



# ============================================================
# Compound Sets
# ============================================================

def save_compound_set(
    rows,
    name,
    source_set_id,
    parameters
):

    metadata = load_compound_metadata()

    cset_id = next_cset_id(metadata)

    filename = f"{cset_id}.csv"

    df = pd.DataFrame(rows)

    df.to_csv(
        COMPOUND_DIR / filename,
        index=False
    )

    metadata["sets"][cset_id] = {

        "name": name,

        "file": filename,

        "created":
            datetime.now()
            .isoformat(timespec="seconds"),

        "source_set_id": source_set_id,

        "parameters": parameters,

        "compound_count": len(df),

    }

    save_compound_metadata(metadata)

    return cset_id, df



def load_compound_set(cset_id):

    metadata = load_compound_metadata()

    info = metadata["sets"][cset_id]

    df = pd.read_csv(
        COMPOUND_DIR /
        info["file"]
    )

    return info, df



def compound_sets_for_structure_set(set_id):
    """Return list of (cset_id, info) linked to a given structure set."""

    metadata = load_compound_metadata()

    return [
        (cset_id, info)
        for cset_id, info in metadata["sets"].items()
        if info.get("source_set_id") == set_id
    ]



# ============================================================
# Display
# ============================================================

def show_structure_set(info, set_id):

    context.mode = "Structure Set"

    context.active_set = info["name"]

    context.active_compound_set = None

    context.info = {

        "Structures":
            info["structure_count"],

        "Created":
            info["created"].replace("T", " ")

    }


    render_header()


    print(
        "Generation:"
    )


    print(
        f"  Generator: {info['generator']}"
    )


    print(
        f"  Ring sizes: "
        f"{info['parameters']['ring_sizes']}"
    )


    print(
        f"  Elements: "
        f"{info['parameters']['allowed_atoms']}"
    )



def show_compound_set(info, cset_id):

    context.mode = "Compound Set"

    context.active_compound_set = info["name"]

    context.info = {

        "Compounds":
            info["compound_count"],

        "Created":
            info["created"].replace("T", " "),

    }


    render_header()


    print("Download parameters:")

    params = info["parameters"]

    print(
        f"  Max per substructure : "
        f"{params.get('max_records', 'N/A')}"
    )

    print(
        f"  Properties           : "
        f"{', '.join(params.get('properties', []))}"
    )



def browse_structures(df):

    render_header()

    print(
        df.head(20)
        .to_string(index=False)
    )



def browse_compounds(df):

    render_header()

    print(
        df.head(20)
        .to_string(index=False)
    )



# ============================================================
# Menus
# ============================================================

def generate_structure_menu():

    context.mode = (
        "Generating Structure Set"
    )

    context.active_set = None

    context.info = {}

    render_header()


    name = input(
        "Structure set name:\n> "
    ).strip()


    if not name:

        print(
            "Cancelled."
        )

        return


    print(
        "\nGenerating..."
    )


    structure_set = generate_structures(
        ring_sizes=(5, 6)
    )


    set_id = save_structure_set(
        structure_set,
        name
    )


    context.mode = (
        "Structure Sets"
    )


    render_header()


    print(
        f"Created: {name}"
    )

    print(
        f"ID: {set_id}"
    )

    print(
        f"Structures: "
        f"{len(structure_set['structures'])}"
    )



def download_compounds_menu(set_id, df):
    """
    Prompt for download parameters and fetch compounds from PubChem
    for each SMILES in the structure set.
    """

    context.mode = "Downloading Compounds"

    context.info = {}

    render_header()


    # -- Name --------------------------------------------------------

    name = input(
        "Compound set name:\n> "
    ).strip()

    if not name:

        print("Cancelled.")

        return


    # -- SMILES column -----------------------------------------------

    smiles_candidates = [
        c for c in df.columns
        if "smiles" in c.lower()
    ]

    if not smiles_candidates:

        print(
            "Error: no SMILES column found "
            "in the structure set."
        )

        input("\nPress Enter...")

        return


    if len(smiles_candidates) == 1:

        smiles_col = smiles_candidates[0]

        print(
            f"\nUsing SMILES column: {smiles_col}"
        )

    else:

        print("\nAvailable SMILES columns:")

        for i, col in enumerate(
            smiles_candidates, 1
        ):

            print(f"  {i}. {col}")

        while True:

            raw = input(
                "Select column number: "
            ).strip()

            try:

                idx = int(raw) - 1

                smiles_col = smiles_candidates[idx]

                break

            except (ValueError, IndexError):

                print("Invalid choice.")


    smiles_list = (
        df[smiles_col]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        f"\n{len(smiles_list)} unique SMILES "
        "will be used as substructure queries."
    )


    # -- Max records -------------------------------------------------

    raw = input(
        "\nMax records per substructure [500]: "
    ).strip()

    max_records = int(raw) if raw else 500


    # -- Properties --------------------------------------------------

    print(
        f"\nDefault properties: "
        f"{', '.join(DEFAULT_PROPERTIES)}"
    )

    keep = input(
        "Use default properties? [Y/n]: "
    ).strip().lower()

    if keep in ("", "y", "yes"):

        properties = list(DEFAULT_PROPERTIES)

    else:

        print(
            "Enter property names separated "
            "by spaces:"
        )

        raw_props = input("> ").strip()

        properties = (
            raw_props.split()
            if raw_props
            else list(DEFAULT_PROPERTIES)
        )


    # -- Batch / sleep -----------------------------------------------

    raw = input(
        "\nCIDs per fetch batch [100]: "
    ).strip()

    batch_size = int(raw) if raw else 100

    raw = input(
        "Sleep between requests (s) [0.2]: "
    ).strip()

    sleep = float(raw) if raw else 0.2


    # -- Confirm -----------------------------------------------------

    render_header()

    print("Download parameters:")

    print(f"  Name          : {name}")

    print(f"  SMILES column : {smiles_col}")

    print(
        f"  Substructures : {len(smiles_list)}"
    )

    print(
        f"  Max per sub   : {max_records}"
    )

    print(
        f"  Properties    : "
        f"{', '.join(properties)}"
    )

    print(f"  Batch size    : {batch_size}")

    print(f"  Sleep         : {sleep}s")

    print()

    confirm = input(
        "Proceed? [Y/n]: "
    ).strip().lower()

    if confirm not in ("", "y", "yes"):

        print("Cancelled.")

        return


    # -- Download ----------------------------------------------------

    print("\nStarting download...\n")

    rows_by_sub = []

    for idx, smiles in enumerate(smiles_list, 1):

        print(
            f"[{idx}/{len(smiles_list)}] "
            f"Searching: {smiles}"
        )

        try:

            cids = get_cids_for_substructure(
                smiles,
                max_records=max_records
            )

        except Exception as e:

            print(
                f"  Skipping: {e}",
                file=sys.stderr
            )

            continue

        print(
            f"  Found {len(cids)} CIDs"
        )

        sub_rows = []

        for i in range(
            0, len(cids), batch_size
        ):

            batch = cids[i:i + batch_size]

            try:

                props = fetch_properties_for_cids(
                    batch,
                    properties
                )

            except Exception as e:

                print(
                    f"  Batch error: {e}",
                    file=sys.stderr
                )

                time.sleep(sleep)

                continue

            for row in props:

                smi = row.get("SMILES", "")

                if not is_nitro_zwitterion(smi):

                    continue

                row.setdefault(
                    "QuerySubstructure",
                    smiles
                )

                sub_rows.append(row)

            time.sleep(sleep)

        sub_rows.sort(
            key=lambda r: float(
                r.get("MolecularWeight", 0) or 0
            )
        )

        if sub_rows:

            rows_by_sub.append(sub_rows)

        print(
            f"  Kept {len(sub_rows)} "
            "compounds after filtering"
        )


    # -- Merge & deduplicate -----------------------------------------

    all_rows = interleave_rows(rows_by_sub)

    seen_cids: set = set()

    unique_rows = []

    for row in all_rows:

        cid = row.get("CID")

        if cid in seen_cids:

            continue

        seen_cids.add(cid)

        unique_rows.append(row)


    if not unique_rows:

        print(
            "\nNo compounds retrieved. "
            "Nothing saved."
        )

        input("\nPress Enter...")

        return


    # -- Save --------------------------------------------------------

    parameters = {
        "max_records": max_records,
        "properties": properties,
        "batch_size": batch_size,
        "sleep": sleep,
        "smiles_column": smiles_col,
    }

    cset_id, df_out = save_compound_set(
        unique_rows,
        name,
        set_id,
        parameters
    )

    render_header()

    print(f"Created: {name}")

    print(f"ID: {cset_id}")

    print(f"Compounds: {len(df_out)}")

    input("\nPress Enter...")



def generate_model_menu(run_id):

    info = load_run_info(
        run_id
    )


    context.mode = "Training Model"

    context.info = {}

    render_header()


    name = input(
        "Model name:\n> "
    ).strip()


    if not name:

        print(
            "Cancelled."
        )

        return


    metadata = load_model_metadata()

    model_id = next_model_id(
        metadata
    )


    output_dir = (
        MODEL_RUN_DIR /
        model_id
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    thermo_dir = Path(
        info["output_dir"]
    )


    # Prefer cleaned ML dataset
    input_file = (
        thermo_dir /
        "wide_clean_values_only.csv"
    )


    if not input_file.exists():

        input_file = (
            thermo_dir /
            "wide_values_only.csv"
        )


    if not input_file.exists():

        print(
            "No ML-ready thermo dataset found."
        )

        print(
            "Expected:"
        )

        print(
            "  wide_clean_values_only.csv"
        )

        print(
            "or"
        )

        print(
            "  wide_values_only.csv"
        )

        return


    parameters = {

        "radius": 2,

        "nbits": 512,

        "source_dataset":
            str(input_file)

    }


    register_model(
        run_id,
        name,
        parameters,
        output_dir
    )


    render_header()


    print(
        "Starting model pipeline...\n"
    )


    argv = [

        "--input",

        str(input_file),

        "--models-dir",

        str(output_dir),

        "--radius",

        str(parameters["radius"]),

        "--nbits",

        str(parameters["nbits"])

    ]


    try:

        train_pipeline_main(
            argv
        )


        print(
            "\nModel pipeline completed."
        )


    except Exception as exc:

        print(
            f"\nModel pipeline failed:\n{exc}"
        )


    input(
        "\nPress Enter..."
    )


def model_menu(model_id):

    df = load_model_report(
        model_id
    )


    if df.empty:

        print(
            "No models found."
        )

        input(
            "\nPress Enter..."
        )

        return


    page_size = 15

    page = 0


    while True:


        total_pages = (
            len(df) +
            page_size -
            1
        ) // page_size


        start = (
            page *
            page_size
        )

        end = start + page_size


        page_df = df.iloc[
            start:end
        ]


        show_model(
            model_id
        )


        print()

        print(
            f"Models "
            f"{start+1}-{min(end,len(df))} "
            f"of {len(df)}"
        )


        print(
            f"Page {page+1}/{total_pages}"
        )


        print()


        for i, row in page_df.iterrows():

            print(
                f"{i-start+1:2}. "
                f"R²={row['cv_r2_mean']:7.4f} | "
                f"N={int(row['n_samples']):4} | "
                f"{row['target'][:60]}"
            )


        print()

        if page > 0:

            print(
                "p. Previous page"
            )

        if page < total_pages-1:

            print(
                "n. Next page"
            )


        print(
            "0. Back"
        )


        choice = input(
            "\nSelect model: "
        ).strip().lower()



        if choice == "0":

            break


        elif choice == "n":

            if page < total_pages-1:

                page += 1


        elif choice == "p":

            if page > 0:

                page -= 1


        else:

            try:

                idx = int(choice)-1

            except ValueError:

                continue


            if 0 <= idx < len(page_df):

                row = page_df.iloc[idx]

                individual_model_menu(
                    model_id,
                    row
                )


def individual_model_menu(
    model_id,
    row
):

    while True:

        show_model(model_id)


        print()

        print(
            "Selected model:"
        )


        print(
            f"Target: {row['target']}"
        )


        print(
            f"Samples: {row['n_samples']}"
        )


        print(
            f"CV R²: {row['cv_r2_mean']:.4f}"
        )


        print(
            """
Actions:

1. Show feature importance
0. Back
"""
        )


        choice = input(
            "\nSelect option: "
        ).strip()


        if choice == "0":

            break


        elif choice == "1":

            show_model_importance(
                model_id,
                row["target"]
            )

            input(
                "\nPress Enter..."
            )


# ============================================================
# Metadata — Thermo Runs
# ============================================================

def load_thermo_metadata():

    with open(THERMO_METADATA_FILE) as f:

        return json.load(f)



def save_thermo_metadata(metadata):

    with open(
        THERMO_METADATA_FILE, "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )



def next_run_id(metadata):

    ids = metadata["runs"].keys()

    if not ids:

        return "run_000001"

    numbers = [
        int(x.split("_")[1])
        for x in ids
    ]

    return f"run_{max(numbers) + 1:06d}"



def runs_for_compound_set(cset_id):

    metadata = load_thermo_metadata()

    return [
        (run_id, info)
        for run_id, info in metadata["runs"].items()
        if info.get("cset_id") == cset_id
    ]



def register_run(
    cset_id,
    name,
    parameters,
    output_dir,
):

    metadata = load_thermo_metadata()

    run_id = next_run_id(metadata)

    metadata["runs"][run_id] = {

        "name": name,

        "cset_id": cset_id,

        "created":
            datetime.now()
            .isoformat(timespec="seconds"),

        "parameters": parameters,

        "output_dir": str(output_dir),

        "status": "pending",

    }

    save_thermo_metadata(metadata)

    return run_id



def update_run_status(run_id, status):

    metadata = load_thermo_metadata()

    metadata["runs"][run_id]["status"] = status

    save_thermo_metadata(metadata)



def load_run_info(run_id):

    return load_thermo_metadata()["runs"][run_id]



# ============================================================
# Thermo display helpers
# ============================================================

def show_run(run_id):

    info = load_run_info(run_id)

    context.mode = "Thermo Run"

    context.active_run = info["name"]

    context.info = {

        "Status":
            info["status"],

        "Created":
            info["created"].replace("T", " "),

    }

    render_header()

    return info



def browse_run_outputs(run_id):

    info = show_run(run_id)

    output_dir = Path(info["output_dir"])

    print("Output files:\n")

    output_files = (
        sorted(output_dir.glob("*.csv")) +
        sorted(output_dir.glob("*.txt"))
    )

    if not output_files:

        print("  (none found)")

    else:

        for f in output_files:

            size_kb = f.stat().st_size / 1024

            print(
                f"  {f.name:<40} "
                f"{size_kb:>8.1f} KB"
            )

    print()



def preview_run_output(run_id):
    """Pick an output CSV and print its first 20 rows."""

    import csv as _csv

    info = load_run_info(run_id)

    output_dir = Path(info["output_dir"])

    csvs = sorted(output_dir.glob("*.csv"))

    if not csvs:

        print("No CSV outputs found.")

        return

    show_run(run_id)

    print("Choose a file to preview:\n")

    for i, f in enumerate(csvs, 1):

        print(f"  {i}. {f.name}")

    print("  0. Cancel")

    raw = input("\nSelect: ").strip()

    try:

        choice = int(raw)

    except ValueError:

        return

    if choice == 0 or choice > len(csvs):

        return

    chosen = csvs[choice - 1]

    render_header()

    print(f"Preview: {chosen.name}\n")

    with open(chosen, newline="") as fh:

        reader = _csv.DictReader(fh)

        rows = [
            row
            for i, row in enumerate(reader)
            if i < 20
        ]

    if not rows:

        print("  (empty file)")

        return

    fieldnames = list(rows[0].keys())

    col_widths = {
        k: max(
            len(k),
            max(
                (len(str(r.get(k, ""))) for r in rows),
                default=0
            )
        )
        for k in fieldnames
    }

    header = "  ".join(
        k.ljust(col_widths[k])
        for k in fieldnames
    )

    print(header[:200])

    print("-" * min(len(header), 200))

    for row in rows:

        line = "  ".join(
            str(row.get(k, "")).ljust(col_widths[k])
            for k in fieldnames
        )

        print(line[:200])



# ============================================================
# Thermo parameter prompts
# ============================================================

def _ask(prompt, default, cast=str):

    while True:

        raw = input(
            f"  {prompt} [{default}]: "
        ).strip()

        if raw == "":

            return cast(default)

        try:

            return cast(raw)

        except ValueError:

            print(
                f"    ! Expected {cast.__name__}, "
                f"got: {raw!r}"
            )



def _ask_yn(prompt, default=True):

    hint = "Y/n" if default else "y/N"

    raw = input(
        f"  {prompt} [{hint}]: "
    ).strip().lower()

    if raw == "":

        return default

    return raw in ("y", "yes")



def prompt_run_parameters():

    print()

    max_compounds = _ask(
        "Max compounds to process (0 = all)",
        0,
        int
    )

    pubmed_max = _ask(
        "Max PubMed results per compound",
        20,
        int
    )

    fetch_pharma = _ask_yn(
        "Fetch pharmacology data "
        "(PubChem / ChEMBL / BindingDB)?",
        default=False
    )

    pubchem_max_aids = _ask(
        "Max PubChem bioassay AIDs per compound",
        10,
        int
    )

    chembl_activity_limit = _ask(
        "Max ChEMBL activities per compound",
        100,
        int
    )

    chembl_max_pages = _ask(
        "Max ChEMBL pages per compound",
        5,
        int
    )

    merge_pharma = _ask_yn(
        "Merge pharmacology into wide output?",
        default=False
    )

    wide_from_clean = _ask_yn(
        "Build wide output via clean_thermo pipeline?",
        default=True
    )

    return {
        "max_compounds":         max_compounds,
        "pubmed_max":            pubmed_max,
        "fetch_pharma":          fetch_pharma,
        "pubchem_max_aids":      pubchem_max_aids,
        "chembl_activity_limit": chembl_activity_limit,
        "chembl_max_pages":      chembl_max_pages,
        "merge_pharma":          merge_pharma,
        "wide_from_clean":       wide_from_clean,
    }



# ============================================================
# Thermo menus
# ============================================================

def new_run_menu(cset_id):

    context.mode = "New Thermo Run"

    context.active_run = None

    context.info = {}

    render_header()

    name = input(
        "Run name:\n> "
    ).strip()

    if not name:

        print("Cancelled.")

        return

    params = prompt_run_parameters()

    # Reserve output directory using the prospective run ID
    meta = load_thermo_metadata()

    provisional_id = next_run_id(meta)

    output_dir = THERMO_RUN_DIR / provisional_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    render_header()

    print("Run parameters:\n")

    print(f"  Name            : {name}")

    print(
        f"  Max compounds   : "
        f"{params['max_compounds'] or 'all'}"
    )

    print(
        f"  PubMed max      : "
        f"{params['pubmed_max']}"
    )

    print(
        f"  Fetch pharma    : "
        f"{params['fetch_pharma']}"
    )

    print(
        f"  Wide from clean : "
        f"{params['wide_from_clean']}"
    )

    print(
        f"  Output dir      : {output_dir}"
    )

    print()

    if not _ask_yn("Proceed?", default=True):

        print("Cancelled.")

        output_dir.rmdir()

        return

    run_id = register_run(
        cset_id,
        name,
        params,
        output_dir
    )

    update_run_status(run_id, "running")

    context.active_run = name

    render_header()

    print("Starting find_thermo...\n")

    # The compound set CSV is the input for find_thermo
    cset_info, _ = load_compound_set(cset_id)

    input_csv = str(
        COMPOUND_DIR / cset_info["file"]
    )

    argv = [
        "--input",                 input_csv,
        "--output",                str(output_dir / "report.csv"),
        "--available-output",      str(output_dir / "available.csv"),
        "--parsed-output",         str(output_dir / "parsed.csv"),
        "--ml-output",             str(output_dir / "wide.csv"),
        "--wide-output",           str(output_dir / "wide_clean.csv"),
        "--cleaned-output",        str(output_dir / "cleaned.csv"),
        "--summary-output",        str(output_dir / "summary.csv"),
        "--rare-output",           str(output_dir / "rare.csv"),
        "--cleaning-report",       str(output_dir / "cleaning_report.csv"),
        "--pharma-output",         str(output_dir / "pharma.csv"),
        "--merged-pharma-output",  str(output_dir / "merged_pharma.csv"),
        "--pubmed-max",            str(params["pubmed_max"]),
        "--pubchem-max-aids",      str(params["pubchem_max_aids"]),
        "--chembl-activity-limit", str(params["chembl_activity_limit"]),
        "--chembl-max-pages",      str(params["chembl_max_pages"]),
    ]

    if params["max_compounds"]:

        argv += ["--max", str(params["max_compounds"])]

    if params["fetch_pharma"]:

        argv.append("--fetch-pharma")

    if params["merge_pharma"]:

        argv.append("--merge-pharma")

    if params["wide_from_clean"]:

        argv.append("--wide-from-clean")

    try:

        find_thermo_main(argv)

        update_run_status(run_id, "completed")

    except Exception as exc:

        update_run_status(
            run_id,
            f"failed: {exc}"
        )

        print(
            f"\nRun failed: {exc}",
            file=sys.stderr
        )

    input("\nPress Enter...")



def run_menu(run_id):

    while True:

        show_run(run_id)


        models = models_for_run(
            run_id
        )


        print(
            "\nActions:\n"
        )


        print(
            "1. Browse output files"
        )

        print(
            "2. Preview a CSV"
        )

        print(
            "3. Train new model"
        )


        if models:

            print(
                "\nModels:\n"
            )

            for i, (model_id, info) in enumerate(
                models,
                1
            ):

                print(
                    f"{i+3}. {info['name']}"
                )

        print(
            "\n0. Back"
        )


        choice = input(
            "\nSelect option: "
        ).strip()


        if choice == "0":

            context.active_run = None

            context.info = {}

            break


        try:

            choice = int(choice)

        except ValueError:

            continue


        if choice == 1:

            browse_run_outputs(
                run_id
            )

            input(
                "\nPress Enter..."
            )


        elif choice == 2:

            preview_run_output(
                run_id
            )

            input(
                "\nPress Enter..."
            )


        elif choice == 3:

            generate_model_menu(
                run_id
            )


        elif choice > 3 and choice <= len(models)+3:

            model_id = models[
                choice-4
            ][0]

            model_menu(
                model_id
            )


def thermo_menu(cset_id):

    while True:

        runs = runs_for_compound_set(cset_id)

        context.mode = "Thermo"

        context.active_run = None

        context.info = {}

        render_header()

        print("Thermo Runs for this Compound Set:\n")

        for index, (run_id, info) in enumerate(
            runs, 1
        ):

            status = info.get("status", "?")

            print(
                f"{index}. {info['name']}"
            )

            print(
                f"   Status: {status}  "
                f"(created "
                f"{info['created'].replace('T', ' ')})\n"
            )

        new_run_option = len(runs) + 1

        print(
            f"{new_run_option}. "
            "New run"
        )

        print("0. Back")

        choice = input(
            "\nSelect option: "
        ).strip()

        if choice == "0":

            break

        try:

            choice = int(choice)

        except ValueError:

            continue

        if 1 <= choice <= len(runs):

            run_id = runs[choice - 1][0]

            run_menu(run_id)

        elif choice == new_run_option:

            new_run_menu(cset_id)



def compound_set_menu(cset_id):

    info, df = load_compound_set(cset_id)

    while True:

        show_compound_set(info, cset_id)

        print(
            """
Actions:

1. Browse compounds
2. Thermo
0. Back
"""
        )

        choice = input("Select option: ")

        if choice == "0":

            context.active_compound_set = None

            context.info = {}

            break

        elif choice == "1":

            browse_compounds(df)

            input("\nPress Enter...")

        elif choice == "2":

            thermo_menu(cset_id)



def compounds_menu(set_id):
    """
    List compound sets linked to this structure set and allow
    downloading a new one or opening an existing one.
    """

    while True:

        csets = compound_sets_for_structure_set(
            set_id
        )

        context.mode = "Compounds"

        context.active_compound_set = None

        context.info = {}

        render_header()


        print("Compound Sets for this Structure Set:\n")

        for index, (cset_id, info) in enumerate(
            csets, 1
        ):

            print(
                f"{index}. {info['name']}"
            )

            print(
                f"   {info['compound_count']} compounds  "
                f"(created {info['created'].replace('T', ' ')})\n"
            )


        download_option = len(csets) + 1

        print(
            f"{download_option}. "
            "Download new compound set"
        )

        print("0. Back")


        choice = input("\nSelect option: ")


        if choice == "0":

            break


        try:

            choice = int(choice)

        except ValueError:

            continue


        if 1 <= choice <= len(csets):

            cset_id = csets[choice - 1][0]

            compound_set_menu(cset_id)


        elif choice == download_option:

            # Load structure set df for SMILES
            _, df = load_structure_set(set_id)

            download_compounds_menu(set_id, df)



def structure_set_menu(set_id):

    info, df = load_structure_set(
        set_id
    )


    while True:

        show_structure_set(
            info,
            set_id
        )


        print(
            """
Actions:

1. Browse structures
2. Compounds
0. Back
"""
        )


        choice = input(
            "Select option: "
        )


        if choice == "0":

            context.clear()

            break


        elif choice == "1":

            browse_structures(df)

            input(
                "\nPress Enter..."
            )


        elif choice == "2":

            compounds_menu(set_id)



def main():

    ensure_storage()


    while True:

        context.mode = (
            "Structure Sets"
        )

        context.active_set = None

        context.info = {}


        metadata = load_metadata()

        sets = list(
            metadata["sets"].items()
        )


        render_header()


        print(
            "Available Structure Sets:\n"
        )


        for index, (set_id, info) in enumerate(
            sets,
            1
        ):

            print(
                f"{index}. {info['name']}"
            )

            print(
                f"   {info['structure_count']} structures\n"
            )


        generate_option = len(sets) + 1


        print(
            f"{generate_option}. "
            "Generate new structure set"
        )

        print(
            "0. Exit"
        )


        choice = input(
            "\nSelect option: "
        )


        if choice == "0":

            break


        try:

            choice = int(choice)

        except ValueError:

            continue


        if 1 <= choice <= len(sets):

            set_id = sets[
                choice - 1
            ][0]

            structure_set_menu(
                set_id
            )


        elif choice == generate_option:

            generate_structure_menu()



if __name__ == "__main__":

    main()
