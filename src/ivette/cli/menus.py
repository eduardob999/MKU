"""Ivette interactive menus and display screens.

The CLI surface: every menu and detail screen plus the top-level :func:`main`
loop. All rendering and input goes through :mod:`ivette.cli.ui` (Rich +
questionary); persistence lives in :mod:`ivette.util.storage`; session state in
:mod:`ivette.cli.context`. The whole session runs in the alternate screen so it
stays fixed in place.
"""

import shutil
import time
from pathlib import Path

import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from ivette.cli import ui
from ivette.cli.context import context, render_header
from ivette.util import hardware
from ivette.util.paths import GAUSSIAN_BENCHMARK_RUN_DIR
from ivette.module import gaussian16_core as g16
from ivette.util.text import slugify
from ivette.util.paths import COMPOUND_DIR, MODEL_RUN_DIR, SDF_RUN_DIR, THERMO_RUN_DIR
from ivette.util.storage import (
    MODELS,
    RUNS,
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


def _df_table(df, *, title=None, max_rows=20):
    """Render the first ``max_rows`` of a DataFrame as a Rich table."""
    columns = [str(c) for c in df.columns]
    rows = [
        ["" if pd.isna(v) else v for v in record]
        for record in df.head(max_rows).itertuples(index=False, name=None)
    ]
    ui.table(columns, rows, title=title)


def _write_nitrobenzene_benchmark_sdf(path: Path) -> None:
    mol = Chem.MolFromSmiles("O=[N+]([O-])c1ccccc1")
    if mol is None:
        raise RuntimeError("Could not build nitrobenzene benchmark molecule")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        raise RuntimeError("Could not embed nitrobenzene benchmark molecule")
    AllChem.UFFOptimizeMolecule(mol)
    Chem.MolToMolFile(mol, str(path))


def _run_thread_benchmark(benchmark_sdf: Path, benchmark_dir: Path, threads: int, mem: str):
    from ivette.module import gaussian16_core as g16

    bench_dir = benchmark_dir / f"nproc_{threads}"
    bench_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    result = g16.run_compound(
        sdf_path=str(benchmark_sdf),
        work_dir=str(bench_dir),
        g16_exec="g16",
        operation="opt then freq",
        nproc=threads,
        mem=mem,
    )
    elapsed = time.perf_counter() - start
    return {
        "scenario": f"pm7-preopt + opt then freq @ {threads} threads",
        "threads": threads,
        "success": bool(result and result.success),
        "seconds": round(elapsed, 3),
        "log_path": result.log_path if result else "",
    }


def _write_benchmark_report(benchmark_dir: Path, shared_preopt_seconds: float, rows: list[dict]) -> None:
    report_rows = []
    for row in rows:
        report_rows.append({
            "scenario": row["scenario"],
            "threads": row["threads"],
            "shared_preopt_seconds": round(shared_preopt_seconds, 3),
            "run_seconds": row["seconds"],
            "total_seconds": round(shared_preopt_seconds + row["seconds"], 3),
            "success": row["success"],
            "log_path": row["log_path"],
        })
    df = pd.DataFrame(report_rows)
    df.to_csv(benchmark_dir / "benchmark_tensor.csv", index=False)


def _write_stage_report(benchmark_dir: Path, filename: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(benchmark_dir / filename, index=False)


def _run_preopt_benchmark(benchmark_dir: Path, benchmark_sdf: Path, mem: str, nproc: int):
    from ivette.module import gaussian16_pipeline as gp

    preopt_dir = benchmark_dir / "preopt_stage"
    preopt_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        {"label": "no preopt", "mode": "none"},
        {"label": "PM7 preopt", "mode": "pm7"},
        {"label": "6-31G preopt", "mode": "gjf-6-31G*"},
    ]

    dft_root = benchmark_dir / "preopt_dft"

    rows = []
    for scenario in scenarios:
        # 1. Produce the (pre)optimized geometry, timing only the preopt step.
        start = time.perf_counter()
        preopt_ok = True
        if scenario["mode"] == "none":
            sdf_for_dft = str(benchmark_sdf)
        elif scenario["mode"] == "pm7":
            sdf_for_dft = gp.gaussian_semiempirical_preopt(
                str(benchmark_sdf),
                preopt_dir / "pm7",
                "nitrobenzene",
                method="pm7",
                g16_exec="g16",
                nproc=nproc,
                mem=mem,
                cid="nitrobenzene",
            )
            preopt_ok = bool(sdf_for_dft) and Path(sdf_for_dft).exists()
        else:
            preopt_work = preopt_dir / "gaussian_631g"
            preopt_work.mkdir(parents=True, exist_ok=True)
            preopt_result = g16.run_compound(
                sdf_path=str(benchmark_sdf),
                work_dir=str(preopt_work),
                g16_exec="g16",
                basis_set="6-31G*",
                method="PBE0",
                operation="opt",
                nproc=nproc,
                mem=mem,
            )
            sdf_for_dft = str(benchmark_sdf)
            preopt_ok = bool(preopt_result and preopt_result.success)
            if preopt_ok:
                xyz_path = g16.log_to_xyz(preopt_result.log_path)
                try:
                    out_sdf = preopt_work / "nitrobenzene_631g_preopt.sdf"
                    if gp._xyz_to_sdf(xyz_path, str(out_sdf), template_sdf=str(benchmark_sdf)):
                        sdf_for_dft = str(out_sdf)
                    else:
                        preopt_ok = False
                finally:
                    Path(xyz_path).unlink(missing_ok=True)

        preopt_seconds = round(time.perf_counter() - start, 3)

        # 2. Run the SAME production DFT (opt then freq) from that geometry, so
        #    the comparison reflects how much preopt actually speeds convergence.
        #    We record wall time AND the number of geometry-optimization cycles.
        dft_seconds = opt_steps = None
        dft_ok = False
        dft_log = ""
        if preopt_ok:
            dft_dir = dft_root / scenario["mode"]
            dft_dir.mkdir(parents=True, exist_ok=True)
            dft_start = time.perf_counter()
            dft_result = g16.run_compound(
                sdf_path=sdf_for_dft,
                work_dir=str(dft_dir),
                g16_exec="g16",
                operation="opt then freq",
                nproc=nproc,
                mem=mem,
            )
            dft_seconds = round(time.perf_counter() - dft_start, 3)
            dft_ok = bool(dft_result and dft_result.success)
            dft_log = dft_result.log_path if dft_result else ""
            if dft_result and dft_result.opt_steps:
                opt_steps = dft_result.opt_steps[-1].step

        total_seconds = (
            round(preopt_seconds + dft_seconds, 3) if dft_seconds is not None else None
        )
        rows.append({
            "scenario": scenario["label"],
            "preopt_mode": scenario["mode"],
            "preopt_seconds": preopt_seconds,
            "dft_seconds": dft_seconds,
            "total_seconds": total_seconds,
            "opt_steps": opt_steps,
            "sdf_for_dft": sdf_for_dft,
            "dft_log": dft_log,
            "success": bool(preopt_ok and dft_ok),
        })

    _write_stage_report(benchmark_dir, "preopt_benchmark.csv", rows)
    return rows


def _best_preopt_row(rows):
    """Pick the preopt mode with the lowest TOTAL (preopt + DFT) wall time.

    Falls back to ``preopt_seconds`` for legacy rows that predate the DFT
    measurement, and prefers rows that actually succeeded.
    """
    def key(row):
        total = row.get("total_seconds")
        return total if total is not None else row.get("preopt_seconds", float("inf"))

    usable = [r for r in rows if r.get("success") and r.get("total_seconds") is not None]
    return min(usable or rows, key=key)


def _run_cpu_benchmark(benchmark_dir: Path, preopt_row: dict, threads: list[int], mem: str):
    run_rows = []
    for thread_count in threads:
        scenario_dir = benchmark_dir / "cpu_stage" / preopt_row["preopt_mode"] / f"nproc_{thread_count}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        result = g16.run_compound(
            sdf_path=preopt_row["sdf_for_dft"],
            work_dir=str(scenario_dir),
            g16_exec="g16",
            operation="opt then freq",
            nproc=thread_count,
            mem=mem,
        )
        elapsed = round(time.perf_counter() - start, 3)
        run_rows.append({
            "preopt_mode": preopt_row["preopt_mode"],
            "scenario": f"{preopt_row['scenario']} @ {thread_count} threads",
            "threads": thread_count,
            "run_seconds": elapsed,
            "success": bool(result and result.success),
            "log_path": result.log_path if result else "",
        })
    _write_stage_report(benchmark_dir, "cpu_benchmark.csv", run_rows)
    return run_rows


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
    ui.print(f"[heading]{info['name']}[/heading]")
    ui.table(
        [("Parameter", "muted"), ("Value", "white")],
        [(k, v) for k, v in info["parameters"].items()],
        title="Parameters",
    )
    return info


def browse_model_outputs(model_id):
    info = MODELS.get(model_id)
    output_dir = Path(info["output_dir"])
    show_model(model_id)
    files = sorted(output_dir.iterdir())
    if not files:
        ui.note("No output files.")
        return
    ui.table(
        [("File", "white"), ("Size", "muted")],
        [(f.name, f"{f.stat().st_size / 1024:.1f} KB") for f in files],
        title="Output files",
    )


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
        ui.warn("Importance file not found.")
        return
    _df_table(pd.read_csv(importance_file), title=f"Feature importance — {target}", max_rows=30)


def show_feature_importance(model_id):
    info = MODELS.get(model_id)
    importance_file = info.get("importance_file")
    if not importance_file:
        ui.warn("No feature importance file registered.")
        return
    path = Path(importance_file)
    if not path.exists():
        ui.warn(f"Importance file not found: {path}")
        return
    render_header()
    _df_table(pd.read_csv(path), title="Feature importance", max_rows=30)


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
    ui.table(
        [("Field", "muted"), ("Value", "white")],
        [
            ("Generator", info["generator"]),
            ("Ring sizes", info["parameters"]["ring_sizes"]),
            ("Elements", info["parameters"]["allowed_atoms"]),
        ],
        title="Generation",
    )


def show_compound_set(info, cset_id):
    context.mode = "Compound Set"
    context.active_compound_set = info["name"]
    context.info = {
        "Compounds": info["compound_count"],
        "Created": info["created"].replace("T", " "),
    }
    render_header()
    params = info["parameters"]
    ui.table(
        [("Field", "muted"), ("Value", "white")],
        [
            ("Max per substructure", params.get("max_records", "N/A")),
            ("Properties", ", ".join(params.get("properties", []))),
        ],
        title="Download parameters",
    )


def browse_structures(df):
    render_header()
    _df_table(df, title="Structures (first 20)", max_rows=20)


def browse_compounds(df):
    render_header()
    _df_table(df, title="Compounds (first 20)", max_rows=20)


# ============================================================
# Gaussian / SDF helpers
# ============================================================

def delete_sdf_set(sdf_id):
    """Permanently delete an SDF set (files + metadata entry)."""
    info = SDFS.get(sdf_id)
    if info is None:
        ui.warn("SDF set not found in metadata.")
        return False
    output_dir = Path(info["output_dir"])
    ui.panel(
        f"ID   : {sdf_id}\nName : {info.get('name')}\nPath : {output_dir}",
        title="⚠  Permanently delete this SDF set?",
        border_style="error",
    )
    if ui.ask_text("Type DELETE to confirm") != "DELETE":
        ui.note("Cancelled.")
        return False
    if output_dir.exists():
        shutil.rmtree(output_dir)
    SDFS.delete(sdf_id)
    ui.success("SDF set deleted.")
    return True


def run_gaussian_pipeline(model_id, sdf_dir, operation):
    gaussian_root = Path(sdf_dir) / "gaussian" / operation.replace(" ", "_")
    gaussian_root.mkdir(parents=True, exist_ok=True)
    checkpoint = gaussian_root / "checkpoint.json"

    render_header()
    ui.table(
        [("Field", "muted"), ("Value", "white")],
        [("SDF directory", sdf_dir), ("Working directory", gaussian_root),
         ("Operation", operation)],
        title="Gaussian pipeline",
    )

    # batch_run lives in the full Gaussian pipeline, which is imported lazily so
    # the rest of the CLI stays usable when that pipeline isn't available.
    try:
        from ivette.module.gaussian16_pipeline import batch_run
    except ImportError as exc:
        ui.error(f"Gaussian pipeline unavailable: {exc}")
        ui.pause()
        return

    # --- Hardware optimization (before any Gaussian calculation) ------------
    # Size cores-per-job, parallel jobs, and %mem to the detected hardware and
    # the number of molecules, so the batch runs at maximum throughput.
    n_tasks = count_sdfs(Path(sdf_dir))
    plan = hardware.recommend_gaussian_resources(n_tasks)
    ui.panel(plan.summary(), title="⚙  Hardware optimization", border_style="accent")
    ui.note("Press Enter to accept the recommended values, or override:")
    nproc = ui.ask_int("Cores per Gaussian job (%nprocshared)", plan.nproc)
    jobs = ui.ask_int("Parallel Gaussian jobs", plan.jobs)
    mem = ui.ask_text("Memory per job (%mem)", plan.mem)

    ui.rule(f"Gaussian: {operation}")
    benchmark_key = hardware.benchmark_key(
        stage="tensor",
        cores=plan.cores,
        available_mem_mb=hardware.available_memory_mb(),
        job_label="nitrobenzene",
    )
    benchmark_dir = GAUSSIAN_BENCHMARK_RUN_DIR / benchmark_key.replace(";", "_")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_sdf = benchmark_dir / "nitrobenzene.sdf"
    if not benchmark_sdf.exists():
        _write_nitrobenzene_benchmark_sdf(benchmark_sdf)

    benchmark_preopt_key = hardware.preopt_benchmark_key(
        cores=plan.cores,
        available_mem_mb=hardware.available_memory_mb(),
        fixed_threads=plan.nproc,
    )
    benchmark_cpu_key = hardware.thread_benchmark_key(
        cores=plan.cores,
        available_mem_mb=hardware.available_memory_mb(),
        preopt="winner",
    )

    preopt_rows = hardware.get_cached_benchmark_rows(benchmark_preopt_key)
    cpu_rows = hardware.get_cached_benchmark_rows(benchmark_cpu_key)

    # Discard cached preopt rows from before the DFT measurement was added — they
    # only timed the preopt step, which can't answer whether preopt helps.
    if preopt_rows is not None and not all("total_seconds" in r for r in preopt_rows):
        preopt_rows = None

    if preopt_rows is None:
        preopt_rows = _run_preopt_benchmark(benchmark_dir, benchmark_sdf, mem, plan.nproc)
        best_preopt_row = _best_preopt_row(preopt_rows)
        hardware.store_benchmark_result(benchmark_preopt_key, preopt_rows, best_preopt_row["preopt_mode"])
    else:
        best_preopt_row = _best_preopt_row(preopt_rows)

    # Show the preopt comparison so "is preopt worth it?" is answerable at a glance:
    # total wall time (preopt + DFT) and the number of DFT optimization cycles.
    comparison = []
    for row in sorted(preopt_rows, key=lambda r: (r.get("total_seconds") is None,
                                                  r.get("total_seconds") or 0)):
        steps = row.get("opt_steps")
        comparison.append((
            row["preopt_mode"] + (" ★" if row is best_preopt_row else ""),
            f"{row.get('preopt_seconds', 0):.1f}",
            f"{row['dft_seconds']:.1f}" if row.get("dft_seconds") is not None else "—",
            f"{row['total_seconds']:.1f}" if row.get("total_seconds") is not None else "—",
            steps if steps is not None else "—",
            "ok" if row.get("success") else "FAILED",
        ))
    ui.table(
        [("Preopt", "white"), ("Preopt s", "muted"), ("DFT s", "muted"),
         ("Total s", "accent"), ("Opt steps", "muted"), ("Status", "muted")],
        comparison,
        title="Preopt comparison (winner ★ = lowest total time)",
    )
    if cpu_rows is None:
        benchmark_threads = hardware.benchmark_thread_plan(
            plan.cores,
            available_mem_mb=hardware.available_memory_mb(),
            operation="opt then freq",
            job_label="nitrobenzene",
        )
        ui.panel(
            "Running nitrobenzene CPU benchmark using the preopt comparison winner.\n"
            f"Threads tested: {', '.join(str(t) for t in benchmark_threads)}",
            title="Gaussian benchmark",
            border_style="accent",
        )
        cpu_rows = _run_cpu_benchmark(benchmark_dir, best_preopt_row, benchmark_threads, mem)
        successful = [row for row in cpu_rows if row["success"]]
        best_threads = min(successful, key=lambda row: row["run_seconds"])["threads"] if successful else plan.nproc
        hardware.store_benchmark_result(benchmark_cpu_key, cpu_rows, best_threads)

    chosen_threads = hardware.get_cached_best_threads(benchmark_cpu_key) or plan.nproc
    ui.note(f"Benchmark complete. Cached preopt winner: {best_preopt_row['preopt_mode']}; CPU winner: {chosen_threads} threads")
    nproc = chosen_threads

    if best_preopt_row["preopt_mode"] == "none":
        preopt_mode = "none"
        preopt_basis_set = "6-31G*"
    elif best_preopt_row["preopt_mode"] == "pm7":
        preopt_mode = "pm7"
        preopt_basis_set = "6-31G*"
    else:
        preopt_mode = "gaussian631g"
        preopt_basis_set = "6-31G*"

    ui.panel(
        f"Preopt method  : {best_preopt_row['preopt_mode']} ({best_preopt_row['preopt_seconds']:.3f}s)\n"
        f"Cores per job  : {nproc} threads\n"
        f"Parallel jobs  : {jobs}\n"
        f"Memory per job : {mem}",
        title="Gaussian configuration (from benchmarks)",
        border_style="accent",
    )

    results = batch_run(
        sdf_dir=str(sdf_dir),
        work_dir=str(gaussian_root),
        jobs=jobs,
        operation=operation,
        resume=True,
        checkpoint=str(checkpoint),
        nproc=nproc,
        mem=mem,
        preopt_mode=preopt_mode,
        preopt_basis_set=preopt_basis_set,
    )

    success = sum(r.success for r in results)
    failed = len(results) - success
    ui.panel(
        f"[success]Successful:[/success] {success}\n[error]Failed:[/error] {failed}",
        title="Gaussian finished",
        border_style="success" if failed == 0 else "warn",
    )


# ============================================================
# Structure menus
# ============================================================

def generate_structure_menu():
    context.mode = "Generating Structure Set"
    context.active_set = None
    context.info = {}
    render_header()

    name = ui.ask_text("Structure set name")
    if not name:
        ui.warn("Cancelled.")
        ui.pause()
        return

    with ui.status("Generating structures…"):
        structure_set = generate_structures(ring_sizes=(5, 6))
    set_id = save_structure_set(structure_set, name)

    ui.success(
        f"Created '{name}'  ({set_id}) — "
        f"{len(structure_set['structures'])} structures"
    )
    ui.pause()


def download_compounds_menu(set_id, df):
    """Prompt for download parameters and fetch PubChem compounds for each
    SMILES in the structure set."""
    context.mode = "Downloading Compounds"
    context.info = {}
    render_header()

    name = ui.ask_text("Compound set name")
    if not name:
        ui.warn("Cancelled.")
        ui.pause()
        return

    smiles_candidates = [c for c in df.columns if "smiles" in c.lower()]
    if not smiles_candidates:
        ui.error("No SMILES column found in the structure set.")
        ui.pause()
        return

    if len(smiles_candidates) == 1:
        smiles_col = smiles_candidates[0]
        ui.info(f"Using SMILES column: {smiles_col}")
    else:
        smiles_col = ui.select(
            "Which SMILES column?",
            [(c, c) for c in smiles_candidates],
        )
        if smiles_col is ui.CANCEL:
            return

    smiles_list = df[smiles_col].dropna().unique().tolist()
    ui.info(f"{len(smiles_list)} unique SMILES will be used as substructure queries.")

    max_records = ui.ask_int("Max records per substructure", 500)

    if ui.confirm(f"Use default properties? ({', '.join(DEFAULT_PROPERTIES)})", default=True):
        properties = list(DEFAULT_PROPERTIES)
    else:
        raw_props = ui.ask_text("Property names (space-separated)")
        properties = raw_props.split() if raw_props else list(DEFAULT_PROPERTIES)

    batch_size = ui.ask_int("CIDs per fetch batch", 100)
    sleep = ui.ask_float("Sleep between requests (s)", 0.2)

    render_header()
    ui.table(
        [("Field", "muted"), ("Value", "white")],
        [
            ("Name", name),
            ("SMILES column", smiles_col),
            ("Substructures", len(smiles_list)),
            ("Max per substructure", max_records),
            ("Properties", ", ".join(properties)),
            ("Batch size", batch_size),
            ("Sleep", f"{sleep}s"),
        ],
        title="Download parameters",
    )
    if not ui.confirm("Proceed?", default=True):
        ui.note("Cancelled.")
        ui.pause()
        return

    rows_by_sub = []
    with ui.progress() as prog:
        task = prog.add_task("Searching PubChem…", total=len(smiles_list))
        for smiles in smiles_list:
            prog.update(task, description=str(smiles)[:38])
            try:
                cids = get_cids_for_substructure(smiles, max_records=max_records)
            except Exception as e:
                prog.console.print(f"[warn]  skip {smiles}: {e}[/warn]")
                prog.advance(task)
                continue

            sub_rows = []
            for i in range(0, len(cids), batch_size):
                batch = cids[i:i + batch_size]
                try:
                    props = fetch_properties_for_cids(batch, properties)
                except Exception as e:
                    prog.console.print(f"[warn]  batch error: {e}[/warn]")
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
            prog.update(task, description=f"{str(smiles)[:28]} → kept {len(sub_rows)}")
            prog.advance(task)

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
        ui.warn("No compounds retrieved. Nothing saved.")
        ui.pause()
        return

    parameters = {
        "max_records": max_records,
        "properties": properties,
        "batch_size": batch_size,
        "sleep": sleep,
        "smiles_column": smiles_col,
    }
    cset_id, df_out = save_compound_set(unique_rows, name, set_id, parameters)

    ui.success(f"Created '{name}'  ({cset_id}) — {len(df_out)} compounds")
    ui.pause()


# ============================================================
# Model menus
# ============================================================

def generate_model_menu(run_id):
    info = load_run_info(run_id)
    context.mode = "Training Model"
    context.info = {}
    render_header()

    name = ui.ask_text("Model name")
    if not name:
        ui.warn("Cancelled.")
        ui.pause()
        return

    model_id = MODELS.next_id()
    output_dir = MODEL_RUN_DIR / model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    thermo_dir = Path(info["output_dir"])
    input_file = thermo_dir / "wide_clean_values_only.csv"
    if not input_file.exists():
        input_file = thermo_dir / "wide_values_only.csv"
    if not input_file.exists():
        ui.error("No ML-ready thermo dataset found "
                 "(wide_clean_values_only.csv or wide_values_only.csv).")
        ui.pause()
        return

    parameters = {"radius": 2, "nbits": 512, "source_dataset": str(input_file)}
    register_model(run_id, name, parameters, output_dir)

    ui.rule("Training model")
    argv = [
        "--input", str(input_file),
        "--models-dir", str(output_dir),
        "--radius", str(parameters["radius"]),
        "--nbits", str(parameters["nbits"]),
    ]
    try:
        train_pipeline_main(argv)
        ui.success("Model pipeline completed.")
    except Exception as exc:
        ui.error(f"Model pipeline failed: {exc}")
    ui.pause()


def model_menu(model_id):
    report = load_model_report(model_id)
    if report.empty:
        ui.warn("No trained models found.")
        ui.pause()
        return

    page_size = 15
    page = 0
    while True:
        start = page * page_size
        end = start + page_size
        page_df = report.iloc[start:end]

        show_model(model_id)
        choices = []
        for pos, (_, row) in enumerate(page_df.iterrows()):
            label = (f"{row['target'][:64]}  "
                     f"(n={row['n_samples']}, CV R²={row['cv_r2_mean']:.4f})")
            choices.append((label, ("model", pos)))
        if start > 0:
            choices.append(("← Previous page", ("prev", None)))
        if end < len(report):
            choices.append(("→ Next page", ("next", None)))
        choices.append(("← Back", ("back", None)))

        choice = ui.select("Trained models", choices)
        action, pos = (("back", None) if choice is ui.CANCEL else choice)
        if action == "back":
            break
        if action == "next":
            page += 1
        elif action == "prev":
            page -= 1
        elif action == "model":
            individual_model_menu(model_id, page_df.iloc[pos])


def individual_model_menu(model_id, row):
    while True:
        sdf_sets = find_model_sdf_sets(model_id)

        render_header()
        ui.table(
            [("Field", "muted"), ("Value", "white")],
            [("Target", row["target"]), ("Samples", row["n_samples"]),
             ("CV R²", f"{row['cv_r2_mean']:.4f}")],
            title="Selected model",
        )

        choices = [
            (f"{s.name}  ({count_sdfs(s)} compounds)", ("open", i))
            for i, s in enumerate(sdf_sets)
        ]
        choices.append(("＋ Generate new SDF set", ("new", None)))
        choices.append(("← Back", ("back", None)))

        choice = ui.select("SDF sets", choices)
        action, i = (("back", None) if choice is ui.CANCEL else choice)
        if action == "back":
            break
        if action == "open":
            sdf_set_menu(model_id, row, sdf_sets[i])
        elif action == "new":
            generate_sdf_set(model_id, row)


def generate_sdf_set(model_id, row):
    target = row["target"]
    model_info = MODELS.get(model_id)

    context.mode = "Generating SDF Set"
    context.info = {}
    render_header()

    name = ui.ask_text("SDF set name")
    if not name:
        return

    dataset = Path(model_info["parameters"]["source_dataset"])
    if not dataset.exists():
        ui.error(f"Dataset missing: {dataset}")
        ui.pause()
        return

    df = pd.read_csv(dataset, low_memory=False)
    if target not in df.columns:
        ui.error(f"Target missing: {target}")
        ui.pause()
        return

    df_target = df[["CID", target]].dropna(subset=[target])

    sdf_id = SDFS.next_id()
    output_dir = SDF_RUN_DIR / sdf_id
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_csv = output_dir / "training.csv"
    df_target.to_csv(temp_csv, index=False)

    ui.rule("Downloading SDF files")
    download_training_sdfs_main(["--input", str(temp_csv), "--output-dir", str(output_dir)])

    count = len(list(output_dir.glob("*.sdf")))
    register_sdf_set(
        model_id, target, name, output_dir,
        {"target": target, "dataset": str(dataset), "rows": len(df_target)},
        count,
    )
    ui.success(f"SDF set created — {count} compounds")
    ui.pause()


def sdf_set_menu(model_id, row, sdf_dir):
    while True:
        render_header()
        ui.table(
            [("Field", "muted"), ("Value", "white")],
            [("SDF set", sdf_dir.name), ("Compounds", count_sdfs(sdf_dir))],
            title="Selected SDF set",
        )
        action = ui.select(
            "Actions",
            [
                ("Run Gaussian opt then freq", "opt"),
                ("Run Gaussian single point", "sp"),
                ("Delete SDF set", "delete"),
                ("← Back", "back"),
            ],
        )
        if action is ui.CANCEL or action == "back":
            break
        if action == "opt":
            run_gaussian_pipeline(model_id, sdf_dir, operation="opt then freq")
        elif action == "sp":
            run_gaussian_pipeline(model_id, sdf_dir, operation="sp")
        elif action == "delete":
            sdf_id = next(
                (k for k, v in SDFS.items() if Path(v["output_dir"]) == sdf_dir),
                None,
            )
            if sdf_id is None:
                ui.warn("Could not resolve SDF ID.")
                continue
            if delete_sdf_set(sdf_id):
                ui.pause()
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
    output_files = sorted(output_dir.glob("*.csv")) + sorted(output_dir.glob("*.txt"))
    if not output_files:
        ui.note("No output files found.")
        return
    ui.table(
        [("File", "white"), ("Size", "muted")],
        [(f.name, f"{f.stat().st_size / 1024:.1f} KB") for f in output_files],
        title="Output files",
    )


def preview_run_output(run_id):
    """Pick an output CSV and print its first 20 rows."""
    info = load_run_info(run_id)
    output_dir = Path(info["output_dir"])
    csvs = sorted(output_dir.glob("*.csv"))
    if not csvs:
        ui.note("No CSV outputs found.")
        return

    show_run(run_id)
    chosen = ui.select(
        "Choose a file to preview",
        [(f.name, f) for f in csvs] + [("← Cancel", None)],
    )
    if chosen is ui.CANCEL or chosen is None:
        return

    render_header()
    ui.print(f"[heading]Preview: {chosen.name}[/heading]")
    try:
        df = pd.read_csv(chosen, nrows=20)
    except Exception as exc:
        ui.error(f"Could not read file: {exc}")
        return
    if df.empty:
        ui.note("(empty file)")
        return
    _df_table(df, max_rows=20)


# ============================================================
# Thermo parameter prompts
# ============================================================

def prompt_run_parameters():
    ui.rule("Run parameters")
    return {
        "max_compounds": ui.ask_int("Max compounds to process (0 = all)", 0),
        "pubmed_max": ui.ask_int("Max PubMed results per compound", 20),
        "fetch_pharma": ui.confirm(
            "Fetch pharmacology data (PubChem / ChEMBL / BindingDB)?", default=False),
        "pubchem_max_aids": ui.ask_int("Max PubChem bioassay AIDs per compound", 10),
        "chembl_activity_limit": ui.ask_int("Max ChEMBL activities per compound", 100),
        "chembl_max_pages": ui.ask_int("Max ChEMBL pages per compound", 5),
        "merge_pharma": ui.confirm("Merge pharmacology into wide output?", default=False),
        "wide_from_clean": ui.confirm(
            "Build wide output via clean_thermo pipeline?", default=True),
    }


# ============================================================
# Thermo menus
# ============================================================

def new_run_menu(cset_id):
    context.mode = "New Thermo Run"
    context.active_run = None
    context.info = {}
    render_header()

    name = ui.ask_text("Run name")
    if not name:
        ui.warn("Cancelled.")
        ui.pause()
        return

    params = prompt_run_parameters()

    # Reserve output directory using the prospective run ID.
    provisional_id = RUNS.next_id()
    output_dir = THERMO_RUN_DIR / provisional_id
    output_dir.mkdir(parents=True, exist_ok=True)

    render_header()
    ui.table(
        [("Field", "muted"), ("Value", "white")],
        [
            ("Name", name),
            ("Max compounds", params["max_compounds"] or "all"),
            ("PubMed max", params["pubmed_max"]),
            ("Fetch pharma", params["fetch_pharma"]),
            ("Wide from clean", params["wide_from_clean"]),
            ("Output dir", output_dir),
        ],
        title="Run parameters",
    )
    if not ui.confirm("Proceed?", default=True):
        ui.note("Cancelled.")
        output_dir.rmdir()
        ui.pause()
        return

    run_id = register_run(cset_id, name, params, output_dir)
    update_run_status(run_id, "running")
    context.active_run = name

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

    ui.rule("Running find_thermo")
    try:
        find_thermo_main(argv)
        update_run_status(run_id, "completed")
        ui.success("Run completed.")
    except Exception as exc:
        update_run_status(run_id, f"failed: {exc}")
        ui.error(f"Run failed: {exc}")
    ui.pause()


def run_menu(run_id):
    while True:
        show_run(run_id)
        models = models_for_run(run_id)

        choices = [
            ("Browse output files", ("browse", None)),
            ("Preview a CSV", ("preview", None)),
            ("Train new model", ("train", None)),
        ]
        for model_id, info in models:
            choices.append((f"Model: {info['name']}", ("model", model_id)))
        choices.append(("← Back", ("back", None)))

        choice = ui.select("Thermo run", choices)
        action, payload = (("back", None) if choice is ui.CANCEL else choice)
        if action == "back":
            context.active_run = None
            context.info = {}
            break
        if action == "browse":
            browse_run_outputs(run_id)
            ui.pause()
        elif action == "preview":
            preview_run_output(run_id)
            ui.pause()
        elif action == "train":
            generate_model_menu(run_id)
        elif action == "model":
            model_menu(payload)


def thermo_menu(cset_id):
    while True:
        runs = runs_for_compound_set(cset_id)
        context.mode = "Thermo"
        context.active_run = None
        context.info = {}
        render_header()

        choices = []
        for run_id, info in runs:
            created = info["created"].replace("T", " ")
            choices.append((
                f"{info['name']}  —  {info.get('status', '?')}  (created {created})",
                ("open", run_id),
            ))
        choices.append(("＋ New run", ("new", None)))
        choices.append(("← Back", ("back", None)))

        choice = ui.select("Thermo runs for this compound set", choices)
        action, run_id = (("back", None) if choice is ui.CANCEL else choice)
        if action == "back":
            break
        if action == "open":
            run_menu(run_id)
        elif action == "new":
            new_run_menu(cset_id)


# ============================================================
# Compound / structure menus
# ============================================================

def compound_set_menu(cset_id):
    info, df = load_compound_set(cset_id)
    while True:
        show_compound_set(info, cset_id)
        action = ui.select(
            "Actions",
            [
                ("Browse compounds", "browse"),
                ("Thermo", "thermo"),
                ("← Back", "back"),
            ],
        )
        if action is ui.CANCEL or action == "back":
            context.active_compound_set = None
            context.info = {}
            break
        if action == "browse":
            browse_compounds(df)
            ui.pause()
        elif action == "thermo":
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

        choices = []
        for cset_id, info in csets:
            created = info["created"].replace("T", " ")
            choices.append((
                f"{info['name']}  ({info['compound_count']} compounds, created {created})",
                ("open", cset_id),
            ))
        choices.append(("＋ Download new compound set", ("new", None)))
        choices.append(("← Back", ("back", None)))

        choice = ui.select("Compound sets for this structure set", choices)
        action, cset_id = (("back", None) if choice is ui.CANCEL else choice)
        if action == "back":
            break
        if action == "open":
            compound_set_menu(cset_id)
        elif action == "new":
            _, df = load_structure_set(set_id)
            download_compounds_menu(set_id, df)


def structure_set_menu(set_id):
    info, df = load_structure_set(set_id)
    while True:
        show_structure_set(info, set_id)
        action = ui.select(
            "Actions",
            [
                ("Browse structures", "browse"),
                ("Compounds", "compounds"),
                ("← Back", "back"),
            ],
        )
        if action is ui.CANCEL or action == "back":
            context.clear()
            break
        if action == "browse":
            browse_structures(df)
            ui.pause()
        elif action == "compounds":
            compounds_menu(set_id)


def main():
    ensure_storage()
    with ui.fullscreen():
        while True:
            context.mode = "Structure Sets"
            context.active_set = None
            context.info = {}

            ui.clear()
            ui.banner()

            sets = list(STRUCTURES.items())
            choices = [
                (f"{info['name']}  ({info['structure_count']} structures)", ("open", set_id))
                for set_id, info in sets
            ]
            choices.append(("＋ Generate new structure set", ("new", None)))
            choices.append(("✕ Exit", ("exit", None)))

            choice = ui.select("Structure sets", choices)
            action, set_id = (("exit", None) if choice is ui.CANCEL else choice)
            if action == "exit":
                break
            if action == "open":
                structure_set_menu(set_id)
            elif action == "new":
                generate_structure_menu()
