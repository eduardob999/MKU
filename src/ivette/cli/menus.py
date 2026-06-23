"""Ivette interactive menus and display screens.

The CLI surface: every ``*_menu`` / ``show_*`` / ``browse_*`` / ``generate_*``
screen plus the top-level :func:`main` loop. Persistence lives in
:mod:`ivette.util.storage`; session state in :mod:`ivette.cli.context`.
"""

import csv
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from ivette.cli.context import context, render_header
from ivette.util.prompts import ask as _ask, ask_yn as _ask_yn
from ivette.util.text import slugify
from ivette.util.paths import COMPOUND_DIR, MODEL_RUN_DIR, SDF_RUN_DIR, THERMO_RUN_DIR
from ivette.util import storage
from ivette.util.storage import (
    MODELS,
    SDFS,
    STRUCTURES,
    ensure_storage,
    save_structure_set,
    load_structure_set,
    save_compound_set,
    load_compound_set,
    compound_sets_for_structure_set,
    register_sdf_set,
    find_model_sdf_sets,
    count_sdfs,
    register_model,
    models_for_run,
    register_run,
    runs_for_compound_set,
    update_run_status,
    load_run_info,
)

from ivette.core.generate_structures import generate_structures
from ivette.core.download_physchem import (
    get_cids_for_substructure,
    fetch_properties_for_cids,
    is_nitro_zwitterion,
    interleave_rows,
    DEFAULT_PROPERTIES,
)
from ivette.core.find_thermo import main as find_thermo_main
from ivette.core.train_pipeline import main as train_pipeline_main
from ivette.core.download_training_sdfs import main as download_training_sdfs_main


# ============================================================
# Model display
# ============================================================

def show_model(model_id):
    info = MODELS.get(model_id)
    context.mode = "Model"
    context.info = {
        "Status": "Available",
        "Created": info["created"].replace("T", " "),
        "Thermo Run": info["thermo_run_id"],
    }
    context.active_run = None
    render_header()
    print(f"Model: {info['name']}")
    print()
    print("Parameters:")
    for key, value in info["parameters"].items():
        print(f"  {key}: {value}")
    return info


def browse_model_outputs(model_id):
    info = MODELS.get(model_id)
    output_dir = Path(info["output_dir"])
    show_model(model_id)
    print("\nOutput files:\n")
    files = sorted(output_dir.iterdir())
    if not files:
        print("  (none)")
        return
    for f in files:
        size = f.stat().st_size / 1024
        print(f"  {f.name:<40}{size:>8.1f} KB")


def load_model_report(model_id):
    info = MODELS.get(model_id)
    report = Path(info["output_dir"]) / "model_report.csv"
    if not report.exists():
        return pd.DataFrame()
    df = pd.read_csv(report)
    if "cv_r2_mean" in df.columns:
        df = df.sort_values("cv_r2_mean", ascending=False)
    return df.reset_index(drop=True)


def show_model_importance(model_id, target):
    info = MODELS.get(model_id)
    importance_file = Path(info["output_dir"]) / f"{slugify(target)}_importance.csv"
    render_header()
    if not importance_file.exists():
        print("Importance file not found.")
        return
    df = pd.read_csv(importance_file)
    print(df.head(30).to_string(index=False))


def show_feature_importance(model_id):
    info = MODELS.get(model_id)
    importance_file = info.get("importance_file")
    if not importance_file:
        print("No feature importance file registered.")
        return
    path = Path(importance_file)
    if not path.exists():
        print(f"Importance file not found:\n{path}")
        return
    df = pd.read_csv(path)
    render_header()
    print("Feature importance:\n")
    print(df.head(30).to_string(index=False))


# ============================================================
# Structure / compound display
# ============================================================

def show_structure_set(info, set_id):
    context.mode = "Structure Set"
    context.active_set = info["name"]
    context.active_compound_set = None
    context.info = {
        "Structures": info["structure_count"],
        "Created": info["created"].replace("T", " "),
    }
    render_header()
    print("Generation:")
    print(f"  Generator: {info['generator']}")
    print(f"  Ring sizes: {info['parameters']['ring_sizes']}")
    print(f"  Elements: {info['parameters']['allowed_atoms']}")


def show_compound_set(info, cset_id):
    context.mode = "Compound Set"
    context.active_compound_set = info["name"]
    context.info = {
        "Compounds": info["compound_count"],
        "Created": info["created"].replace("T", " "),
    }
    render_header()
    print("Download parameters:")
    params = info["parameters"]
    print(f"  Max per substructure : {params.get('max_records', 'N/A')}")
    print(f"  Properties           : {', '.join(params.get('properties', []))}")


def browse_structures(df):
    render_header()
    print(df.head(20).to_string(index=False))


def browse_compounds(df):
    render_header()
    print(df.head(20).to_string(index=False))


# ============================================================
# Gaussian / SDF helpers
# ============================================================

def delete_sdf_set(sdf_id):
    """Permanently delete an SDF set (files + metadata entry)."""
    info = SDFS.get(sdf_id)
    if info is None:
        print("SDF set not found in metadata.")
        return False
    output_dir = Path(info["output_dir"])
    print("\n⚠️  WARNING: This will permanently delete:")
    print(f"  ID   : {sdf_id}")
    print(f"  Name : {info.get('name')}")
    print(f"  Path : {output_dir}")
    if input("\nType 'DELETE' to confirm: ").strip() != "DELETE":
        print("Cancelled.")
        return False
    if output_dir.exists():
        shutil.rmtree(output_dir)
    SDFS.delete(sdf_id)
    print("SDF set deleted.")
    return True


def run_gaussian_pipeline(model_id, sdf_dir, operation):
    gaussian_root = Path(sdf_dir) / "gaussian" / operation.replace(" ", "_")
    gaussian_root.mkdir(parents=True, exist_ok=True)
    checkpoint = gaussian_root / "checkpoint.json"

    print("\nGaussian pipeline")
    print(f"SDF directory: {sdf_dir}")
    print(f"Working directory: {gaussian_root}")

    # batch_run lives in the full Gaussian pipeline, which is imported lazily so
    # the rest of the CLI stays usable when that pipeline isn't available.
    try:
        from ivette.module.gaussian16_pipeline import batch_run
    except ImportError as exc:
        print(f"\nGaussian pipeline unavailable: {exc}")
        input("\nPress Enter...")
        return

    nproc = _ask("Number of CPU cores to use", 8, int)

    results = batch_run(
        sdf_dir=str(sdf_dir),
        work_dir=str(gaussian_root),
        jobs=1,
        operation=operation,
        resume=True,
        checkpoint=str(checkpoint),
        nproc=nproc,
    )

    success = sum(r.success for r in results)
    failed = len(results) - success
    print("\nGaussian finished:")
    print(f"  Successful: {success}")
    print(f"  Failed: {failed}")


# ============================================================
# Structure menus
# ============================================================

def generate_structure_menu():
    context.mode = "Generating Structure Set"
    context.active_set = None
    context.info = {}
    render_header()

    name = input("Structure set name:\n> ").strip()
    if not name:
        print("Cancelled.")
        return

    print("\nGenerating...")
    structure_set = generate_structures(ring_sizes=(5, 6))
    set_id = save_structure_set(structure_set, name)

    context.mode = "Structure Sets"
    render_header()
    print(f"Created: {name}")
    print(f"ID: {set_id}")
    print(f"Structures: {len(structure_set['structures'])}")


def download_compounds_menu(set_id, df):
    """Prompt for download parameters and fetch PubChem compounds for each
    SMILES in the structure set."""
    context.mode = "Downloading Compounds"
    context.info = {}
    render_header()

    name = input("Compound set name:\n> ").strip()
    if not name:
        print("Cancelled.")
        return

    smiles_candidates = [c for c in df.columns if "smiles" in c.lower()]
    if not smiles_candidates:
        print("Error: no SMILES column found in the structure set.")
        input("\nPress Enter...")
        return

    if len(smiles_candidates) == 1:
        smiles_col = smiles_candidates[0]
        print(f"\nUsing SMILES column: {smiles_col}")
    else:
        print("\nAvailable SMILES columns:")
        for i, col in enumerate(smiles_candidates, 1):
            print(f"  {i}. {col}")
        while True:
            raw = input("Select column number: ").strip()
            try:
                smiles_col = smiles_candidates[int(raw) - 1]
                break
            except (ValueError, IndexError):
                print("Invalid choice.")

    smiles_list = df[smiles_col].dropna().unique().tolist()
    print(f"\n{len(smiles_list)} unique SMILES will be used as substructure queries.")

    raw = input("\nMax records per substructure [500]: ").strip()
    max_records = int(raw) if raw else 500

    print(f"\nDefault properties: {', '.join(DEFAULT_PROPERTIES)}")
    keep = input("Use default properties? [Y/n]: ").strip().lower()
    if keep in ("", "y", "yes"):
        properties = list(DEFAULT_PROPERTIES)
    else:
        print("Enter property names separated by spaces:")
        raw_props = input("> ").strip()
        properties = raw_props.split() if raw_props else list(DEFAULT_PROPERTIES)

    raw = input("\nCIDs per fetch batch [100]: ").strip()
    batch_size = int(raw) if raw else 100
    raw = input("Sleep between requests (s) [0.2]: ").strip()
    sleep = float(raw) if raw else 0.2

    render_header()
    print("Download parameters:")
    print(f"  Name          : {name}")
    print(f"  SMILES column : {smiles_col}")
    print(f"  Substructures : {len(smiles_list)}")
    print(f"  Max per sub   : {max_records}")
    print(f"  Properties    : {', '.join(properties)}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Sleep         : {sleep}s")
    print()
    if input("Proceed? [Y/n]: ").strip().lower() not in ("", "y", "yes"):
        print("Cancelled.")
        return

    print("\nStarting download...\n")
    rows_by_sub = []
    for idx, smiles in enumerate(smiles_list, 1):
        print(f"[{idx}/{len(smiles_list)}] Searching: {smiles}")
        try:
            cids = get_cids_for_substructure(smiles, max_records=max_records)
        except Exception as e:
            print(f"  Skipping: {e}", file=sys.stderr)
            continue
        print(f"  Found {len(cids)} CIDs")

        sub_rows = []
        for i in range(0, len(cids), batch_size):
            batch = cids[i:i + batch_size]
            try:
                props = fetch_properties_for_cids(batch, properties)
            except Exception as e:
                print(f"  Batch error: {e}", file=sys.stderr)
                time.sleep(sleep)
                continue
            for row in props:
                if not is_nitro_zwitterion(row.get("SMILES", "")):
                    continue
                row.setdefault("QuerySubstructure", smiles)
                sub_rows.append(row)
            time.sleep(sleep)

        sub_rows.sort(key=lambda r: float(r.get("MolecularWeight", 0) or 0))
        if sub_rows:
            rows_by_sub.append(sub_rows)
        print(f"  Kept {len(sub_rows)} compounds after filtering")

    all_rows = interleave_rows(rows_by_sub)
    seen_cids = set()
    unique_rows = []
    for row in all_rows:
        cid = row.get("CID")
        if cid in seen_cids:
            continue
        seen_cids.add(cid)
        unique_rows.append(row)

    if not unique_rows:
        print("\nNo compounds retrieved. Nothing saved.")
        input("\nPress Enter...")
        return

    parameters = {
        "max_records": max_records,
        "properties": properties,
        "batch_size": batch_size,
        "sleep": sleep,
        "smiles_column": smiles_col,
    }
    cset_id, df_out = save_compound_set(unique_rows, name, set_id, parameters)

    render_header()
    print(f"Created: {name}")
    print(f"ID: {cset_id}")
    print(f"Compounds: {len(df_out)}")
    input("\nPress Enter...")


# ============================================================
# Model menus
# ============================================================

def generate_model_menu(run_id):
    info = load_run_info(run_id)
    context.mode = "Training Model"
    context.info = {}
    render_header()

    name = input("Model name:\n> ").strip()
    if not name:
        print("Cancelled.")
        return

    model_id = MODELS.next_id()
    output_dir = MODEL_RUN_DIR / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    thermo_dir = Path(info["output_dir"])
    input_file = thermo_dir / "wide_clean_values_only.csv"
    if not input_file.exists():
        input_file = thermo_dir / "wide_values_only.csv"
    if not input_file.exists():
        print("No ML-ready thermo dataset found.")
        print("Expected:")
        print("  wide_clean_values_only.csv")
        print("or")
        print("  wide_values_only.csv")
        return

    parameters = {
        "radius": 2,
        "nbits": 512,
        "source_dataset": str(input_file),
    }
    register_model(run_id, name, parameters, output_dir)

    render_header()
    print("Starting model pipeline...\n")
    argv = [
        "--input", str(input_file),
        "--models-dir", str(output_dir),
        "--radius", str(parameters["radius"]),
        "--nbits", str(parameters["nbits"]),
    ]
    try:
        train_pipeline_main(argv)
        print("\nModel pipeline completed.")
    except Exception as exc:
        print(f"\nModel pipeline failed:\n{exc}")
    input("\nPress Enter...")


def model_menu(model_id):
    report = load_model_report(model_id)
    if report.empty:
        print("No trained models found.")
        input("\nPress Enter...")
        return

    page_size = 15
    page = 0
    while True:
        start = page * page_size
        end = start + page_size
        page_df = report.iloc[start:end]

        show_model(model_id)
        print("\nModels:\n")
        for i, (_, row) in enumerate(page_df.iterrows(), 1):
            print(f"{i}. {row['target'][:70]}")
            print(f"   n={row['n_samples']} CV R²={row['cv_r2_mean']:.4f}")

        print()
        if start > 0:
            print("p. Previous page")
        if end < len(report):
            print("n. Next page")
        print("0. Back")

        choice = input("\nSelect model: ").strip()
        if choice == "0":
            break
        if choice == "n" and end < len(report):
            page += 1
            continue
        if choice == "p" and page > 0:
            page -= 1
            continue
        try:
            idx = int(choice) - 1
        except ValueError:
            continue
        if 0 <= idx < len(page_df):
            individual_model_menu(model_id, page_df.iloc[idx])


def individual_model_menu(model_id, row):
    while True:
        sdf_sets = find_model_sdf_sets(model_id)

        print("\nSelected model:")
        print(f"Target: {row['target']}")
        print(f"Samples: {row['n_samples']}")
        print(f"CV R²: {row['cv_r2_mean']:.4f}")

        print("\nSDF Sets:")
        for i, sdf_set in enumerate(sdf_sets, 1):
            print(f"  {i}. {sdf_set.name} ({count_sdfs(sdf_set)} compounds)")
        if not sdf_sets:
            print("  None generated.")

        print("\nActions:")
        generate_option = len(sdf_sets) + 1
        print(f"{generate_option}. Generate new SDF set")
        print("0. Back")

        choice = input("\nSelect SDF set or action: ").strip()
        if choice == "0":
            break
        try:
            choice_int = int(choice)
        except ValueError:
            continue
        if 1 <= choice_int <= len(sdf_sets):
            sdf_set_menu(model_id, row, sdf_sets[choice_int - 1])
        elif choice_int == generate_option:
            generate_sdf_set(model_id, row)


def generate_sdf_set(model_id, row):
    target = row["target"]
    model_info = MODELS.get(model_id)

    context.mode = "Generating SDF Set"
    context.info = {}
    render_header()

    name = input("SDF set name:\n> ").strip()
    if not name:
        return

    dataset = Path(model_info["parameters"]["source_dataset"])
    if not dataset.exists():
        print(f"Dataset missing:\n{dataset}")
        input("\nPress Enter...")
        return

    df = pd.read_csv(dataset, low_memory=False)
    if target not in df.columns:
        print("Target missing:")
        print(target)
        input("\nPress Enter...")
        return

    df_target = df[["CID", target]].dropna(subset=[target])

    sdf_id = SDFS.next_id()
    output_dir = SDF_RUN_DIR / sdf_id
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_csv = output_dir / "training.csv"
    df_target.to_csv(temp_csv, index=False)

    print("\nDownloading SDF files...")
    download_training_sdfs_main(["--input", str(temp_csv), "--output-dir", str(output_dir)])

    count = len(list(output_dir.glob("*.sdf")))
    register_sdf_set(
        model_id, target, name, output_dir,
        {"target": target, "dataset": str(dataset), "rows": len(df_target)},
        count,
    )

    print("\nSDF set created.")
    print(f"Compounds: {count}")
    input("\nPress Enter...")


def sdf_set_menu(model_id, row, sdf_dir):
    while True:
        print("\nSelected SDF set:")
        print(f"{sdf_dir.name}")
        print(f"Compounds: {count_sdfs(sdf_dir)}")
        print("""
Actions:

1. Run Gaussian optimization + frequency
2. Run Gaussian single point
3. Delete SDF set
0. Back
""")
        choice = input("\nSelect: ").strip()
        if choice == "0":
            break
        if choice == "1":
            run_gaussian_pipeline(model_id, sdf_dir, operation="opt freq")
        elif choice == "2":
            run_gaussian_pipeline(model_id, sdf_dir, operation="sp")
        elif choice == "3":
            sdf_id = next(
                (k for k, v in SDFS.items() if Path(v["output_dir"]) == sdf_dir),
                None,
            )
            if sdf_id is None:
                print("Could not resolve SDF ID.")
                continue
            if delete_sdf_set(sdf_id):
                break


# ============================================================
# Thermo run display
# ============================================================

def show_run(run_id):
    info = load_run_info(run_id)
    context.mode = "Thermo Run"
    context.active_run = info["name"]
    context.info = {
        "Status": info["status"],
        "Created": info["created"].replace("T", " "),
    }
    render_header()
    return info


def browse_run_outputs(run_id):
    info = show_run(run_id)
    output_dir = Path(info["output_dir"])
    print("Output files:\n")
    output_files = sorted(output_dir.glob("*.csv")) + sorted(output_dir.glob("*.txt"))
    if not output_files:
        print("  (none found)")
    else:
        for f in output_files:
            print(f"  {f.name:<40} {f.stat().st_size / 1024:>8.1f} KB")
    print()


def preview_run_output(run_id):
    """Pick an output CSV and print its first 20 rows."""
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

    try:
        choice = int(input("\nSelect: ").strip())
    except ValueError:
        return
    if choice == 0 or choice > len(csvs):
        return

    chosen = csvs[choice - 1]
    render_header()
    print(f"Preview: {chosen.name}\n")
    with open(chosen, newline="") as fh:
        rows = [row for i, row in enumerate(csv.DictReader(fh)) if i < 20]
    if not rows:
        print("  (empty file)")
        return

    fieldnames = list(rows[0].keys())
    col_widths = {
        k: max(len(k), max((len(str(r.get(k, ""))) for r in rows), default=0))
        for k in fieldnames
    }
    header = "  ".join(k.ljust(col_widths[k]) for k in fieldnames)
    print(header[:200])
    print("-" * min(len(header), 200))
    for row in rows:
        line = "  ".join(str(row.get(k, "")).ljust(col_widths[k]) for k in fieldnames)
        print(line[:200])


# ============================================================
# Thermo parameter prompts
# ============================================================

def prompt_run_parameters():
    print()
    max_compounds = _ask("Max compounds to process (0 = all)", 0, int)
    pubmed_max = _ask("Max PubMed results per compound", 20, int)
    fetch_pharma = _ask_yn(
        "Fetch pharmacology data (PubChem / ChEMBL / BindingDB)?", default=False
    )
    pubchem_max_aids = _ask("Max PubChem bioassay AIDs per compound", 10, int)
    chembl_activity_limit = _ask("Max ChEMBL activities per compound", 100, int)
    chembl_max_pages = _ask("Max ChEMBL pages per compound", 5, int)
    merge_pharma = _ask_yn("Merge pharmacology into wide output?", default=False)
    wide_from_clean = _ask_yn("Build wide output via clean_thermo pipeline?", default=True)
    return {
        "max_compounds": max_compounds,
        "pubmed_max": pubmed_max,
        "fetch_pharma": fetch_pharma,
        "pubchem_max_aids": pubchem_max_aids,
        "chembl_activity_limit": chembl_activity_limit,
        "chembl_max_pages": chembl_max_pages,
        "merge_pharma": merge_pharma,
        "wide_from_clean": wide_from_clean,
    }


# ============================================================
# Thermo menus
# ============================================================

def new_run_menu(cset_id):
    context.mode = "New Thermo Run"
    context.active_run = None
    context.info = {}
    render_header()

    name = input("Run name:\n> ").strip()
    if not name:
        print("Cancelled.")
        return

    params = prompt_run_parameters()

    # Reserve output directory using the prospective run ID.
    provisional_id = storage.RUNS.next_id()
    output_dir = THERMO_RUN_DIR / provisional_id
    output_dir.mkdir(parents=True, exist_ok=True)

    render_header()
    print("Run parameters:\n")
    print(f"  Name            : {name}")
    print(f"  Max compounds   : {params['max_compounds'] or 'all'}")
    print(f"  PubMed max      : {params['pubmed_max']}")
    print(f"  Fetch pharma    : {params['fetch_pharma']}")
    print(f"  Wide from clean : {params['wide_from_clean']}")
    print(f"  Output dir      : {output_dir}")
    print()
    if not _ask_yn("Proceed?", default=True):
        print("Cancelled.")
        output_dir.rmdir()
        return

    run_id = register_run(cset_id, name, params, output_dir)
    update_run_status(run_id, "running")
    context.active_run = name
    render_header()
    print("Starting find_thermo...\n")

    cset_info, _ = load_compound_set(cset_id)
    input_csv = str(COMPOUND_DIR / cset_info["file"])

    argv = [
        "--input", input_csv,
        "--output", str(output_dir / "report.csv"),
        "--available-output", str(output_dir / "available.csv"),
        "--parsed-output", str(output_dir / "parsed.csv"),
        "--ml-output", str(output_dir / "wide.csv"),
        "--wide-output", str(output_dir / "wide_clean.csv"),
        "--cleaned-output", str(output_dir / "cleaned.csv"),
        "--summary-output", str(output_dir / "summary.csv"),
        "--rare-output", str(output_dir / "rare.csv"),
        "--cleaning-report", str(output_dir / "cleaning_report.csv"),
        "--pharma-output", str(output_dir / "pharma.csv"),
        "--merged-pharma-output", str(output_dir / "merged_pharma.csv"),
        "--pubmed-max", str(params["pubmed_max"]),
        "--pubchem-max-aids", str(params["pubchem_max_aids"]),
        "--chembl-activity-limit", str(params["chembl_activity_limit"]),
        "--chembl-max-pages", str(params["chembl_max_pages"]),
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
        update_run_status(run_id, f"failed: {exc}")
        print(f"\nRun failed: {exc}", file=sys.stderr)
    input("\nPress Enter...")


def run_menu(run_id):
    while True:
        show_run(run_id)
        models = models_for_run(run_id)

        print("\nActions:\n")
        print("1. Browse output files")
        print("2. Preview a CSV")
        print("3. Train new model")
        if models:
            print("\nModels:\n")
            for i, (model_id, info) in enumerate(models, 1):
                print(f"{i + 3}. {info['name']}")
        print("\n0. Back")

        choice = input("\nSelect option: ").strip()
        if choice == "0":
            context.active_run = None
            context.info = {}
            break
        try:
            choice = int(choice)
        except ValueError:
            continue

        if choice == 1:
            browse_run_outputs(run_id)
            input("\nPress Enter...")
        elif choice == 2:
            preview_run_output(run_id)
            input("\nPress Enter...")
        elif choice == 3:
            generate_model_menu(run_id)
        elif 3 < choice <= len(models) + 3:
            model_menu(models[choice - 4][0])


def thermo_menu(cset_id):
    while True:
        runs = runs_for_compound_set(cset_id)
        context.mode = "Thermo"
        context.active_run = None
        context.info = {}
        render_header()

        print("Thermo Runs for this Compound Set:\n")
        for index, (run_id, info) in enumerate(runs, 1):
            print(f"{index}. {info['name']}")
            print(
                f"   Status: {info.get('status', '?')}  "
                f"(created {info['created'].replace('T', ' ')})\n"
            )

        new_run_option = len(runs) + 1
        print(f"{new_run_option}. New run")
        print("0. Back")

        choice = input("\nSelect option: ").strip()
        if choice == "0":
            break
        try:
            choice = int(choice)
        except ValueError:
            continue
        if 1 <= choice <= len(runs):
            run_menu(runs[choice - 1][0])
        elif choice == new_run_option:
            new_run_menu(cset_id)


# ============================================================
# Compound / structure menus
# ============================================================

def compound_set_menu(cset_id):
    info, df = load_compound_set(cset_id)
    while True:
        show_compound_set(info, cset_id)
        print("""
Actions:

1. Browse compounds
2. Thermo
0. Back
""")
        choice = input("Select option: ")
        if choice == "0":
            context.active_compound_set = None
            context.info = {}
            break
        if choice == "1":
            browse_compounds(df)
            input("\nPress Enter...")
        elif choice == "2":
            thermo_menu(cset_id)


def compounds_menu(set_id):
    """List compound sets linked to this structure set and allow downloading a
    new one or opening an existing one."""
    while True:
        csets = compound_sets_for_structure_set(set_id)
        context.mode = "Compounds"
        context.active_compound_set = None
        context.info = {}
        render_header()

        print("Compound Sets for this Structure Set:\n")
        for index, (cset_id, info) in enumerate(csets, 1):
            print(f"{index}. {info['name']}")
            print(
                f"   {info['compound_count']} compounds  "
                f"(created {info['created'].replace('T', ' ')})\n"
            )

        download_option = len(csets) + 1
        print(f"{download_option}. Download new compound set")
        print("0. Back")

        choice = input("\nSelect option: ")
        if choice == "0":
            break
        try:
            choice = int(choice)
        except ValueError:
            continue
        if 1 <= choice <= len(csets):
            compound_set_menu(csets[choice - 1][0])
        elif choice == download_option:
            _, df = load_structure_set(set_id)
            download_compounds_menu(set_id, df)


def structure_set_menu(set_id):
    info, df = load_structure_set(set_id)
    while True:
        show_structure_set(info, set_id)
        print("""
Actions:

1. Browse structures
2. Compounds
0. Back
""")
        choice = input("Select option: ")
        if choice == "0":
            context.clear()
            break
        if choice == "1":
            browse_structures(df)
            input("\nPress Enter...")
        elif choice == "2":
            compounds_menu(set_id)


def main():
    ensure_storage()
    while True:
        context.mode = "Structure Sets"
        context.active_set = None
        context.info = {}

        sets = list(STRUCTURES.items())
        render_header()
        print("Available Structure Sets:\n")
        for index, (set_id, info) in enumerate(sets, 1):
            print(f"{index}. {info['name']}")
            print(f"   {info['structure_count']} structures\n")

        generate_option = len(sets) + 1
        print(f"{generate_option}. Generate new structure set")
        print("0. Exit")

        choice = input("\nSelect option: ")
        if choice == "0":
            break
        try:
            choice = int(choice)
        except ValueError:
            continue
        if 1 <= choice <= len(sets):
            structure_set_menu(sets[choice - 1][0])
        elif choice == generate_option:
            generate_structure_menu()
