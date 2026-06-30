"""Ivette interactive menus and display screens.

The CLI surface: every menu and detail screen plus the top-level :func:`main`
loop. All rendering and input goes through :mod:`ivette.cli.ui` (Rich +
questionary); persistence lives in :mod:`ivette.util.storage`; session state in
:mod:`ivette.cli.context`. The whole session runs in the alternate screen so it
stays fixed in place.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

from ivette.cli import ui
from ivette.cli.context import context, render_header
from ivette.util import applog
from ivette.util import hardware
from ivette.util.paths import GAUSSIAN_BENCHMARK_RUN_DIR
from ivette.module import gaussian16_core as g16
from ivette.util.text import slugify
from ivette.util.paths import COMPOUND_DIR, MODEL_RUN_DIR, GEOMETRY_RUN_DIR, DATASET_RUN_DIR
from ivette.util.storage import (
    COMPOUNDS,
    MODELS,
    DATASETS,
    GEOMETRIES,
    STRUCTURES,
    ensure_storage,
    save_structure_library,
    load_structure_library,
    save_compound_library,
    load_compound_library,
    compound_libraries_for_structure,
    register_geometry_set,
    find_model_geometry_sets,
    count_geometries,
    register_model,
    models_for_dataset,
    register_dataset,
    datasets_for_compound,
    update_dataset_status,
    load_dataset_info,
    save_dft_descriptor_set,
    load_dft_descriptor_set,
    dft_descriptor_sets_for_model,
    DFT_DESCRIPTORS,
)

from ivette.core.generate_structures import generate_structures
from ivette.core.download_physchem import (
    get_cids_for_substructure,
    fetch_properties_for_cids,
    is_nitro_zwitterion,
    interleave_rows,
)
from ivette.core.find_thermo import main as find_thermo_main
from ivette.core.train_pipeline import main as train_pipeline_main
from ivette.core.download_training_sdfs import main as download_training_sdfs_main
from ivette.core.parse_dft import parse_geometry_descriptors, parse_redox_descriptors
from ivette.core.feature_benchmark import run_feature_selection_benchmark, best_method, best_config
from ivette.core.hpo import optimize_training_params
from ivette.util import presets
from ivette.core import params as P
from ivette.cli.params_ui import configure_stage


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
        "Property Dataset": info["dataset_id"],
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

def show_structure_library(info, structure_id):
    context.mode = "Structure Library"
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


def show_compound_library(info, compound_id):
    context.mode = "Compound Library"
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
# Gaussian / geometry helpers
# ============================================================

def _has_content(path):
    """True if ``path`` exists and holds data (non-empty file or directory)."""
    p = Path(path)
    if not p.exists():
        return False
    if p.is_dir():
        return any(p.iterdir())
    return p.stat().st_size > 0


def _confirm_overwrite(path, what="output"):
    """Guard a process whose output goes to ``path``.

    Returns True to proceed. Prompts for permission only when ``path`` already
    holds data, so a process writing to a fresh location never nags the user.
    """
    if not _has_content(path):
        return True
    ui.warn(f"{what} already exists at {path}")
    return ui.confirm("Overwrite the existing data?", default=False)


def delete_geometry_set(geometry_id):
    """Permanently delete an Geometry set (files + metadata entry)."""
    info = GEOMETRIES.get(geometry_id)
    if info is None:
        ui.warn("Geometry set not found in metadata.")
        return False
    output_dir = Path(info["output_dir"])
    ui.panel(
        f"ID   : {geometry_id}\nName : {info.get('name')}\nPath : {output_dir}",
        title="⚠  Permanently delete this Geometry set?",
        border_style="error",
    )
    if ui.ask_text("Type DELETE to confirm") != "DELETE":
        ui.note("Cancelled.")
        return False
    if output_dir.exists():
        shutil.rmtree(output_dir)
    GEOMETRIES.delete(geometry_id)
    ui.success("Geometry set deleted.")
    return True


def run_gaussian_pipeline(model_id, geometry_dir, operation, *, cosmo=False, charge_states=None):
    """Run the full Gaussian pipeline (hardware sizing + benchmarking + batch).

    ``cosmo`` switches on CPCM/water solvation (rendered as
    ``scrf=(cpcm,solvent=water)`` by the core builder). ``charge_states`` is a
    list of ``(label, charge, multiplicity)`` so a single invocation can cover,
    e.g., the neutral molecule and its -1 anion. The hardware/benchmark step
    runs ONCE and is shared by every charge state; each state gets its own work
    directory and checkpoint, with independent overwrite/resume protection.
    """
    if charge_states is None:
        charge_states = [("", 0, 1)]
    geometry_dir = Path(geometry_dir)
    root_name = operation.replace(" ", "_") + ("_COSMO" if cosmo else "")
    gaussian_root = geometry_dir / "gaussian" / root_name
    gaussian_root.mkdir(parents=True, exist_ok=True)

    render_header()
    ui.table(
        [("Field", "muted"), ("Value", "white")],
        [
            ("Geometry directory", geometry_dir),
            ("Working directory", gaussian_root),
            ("Operation", operation),
            ("Solvation", "COSMO (CPCM, water)" if cosmo else "gas phase"),
            ("Charge states",
             ", ".join(f"{lbl or 'neutral'} (q={q}, mult={m})" for lbl, q, m in charge_states)),
        ],
        title="Gaussian pipeline",
    )

    # Advanced options: functional, basis set, preopt override, timeout, extra
    # route keywords (+ presets). preopt_mode "auto" defers to the benchmark.
    gp = configure_stage("gaussian")

    # Bring up the live dashboard so benchmark, convergence and batch-progress
    # plots are on screen as the run proceeds.
    _ensure_control_room()

    # The headless Gaussian service (and the pipeline it wraps) is imported
    # lazily so the rest of the CLI stays usable when that pipeline isn't available.
    try:
        from ivette.services.gaussian import run_charge_state_batches
    except ImportError as exc:
        ui.error(f"Gaussian pipeline unavailable: {exc}")
        ui.pause()
        return

    # --- Hardware optimization (before any Gaussian calculation) ------------
    # Size cores-per-job, parallel jobs, and %mem to the detected hardware and
    # the number of molecules, so the batch runs at maximum throughput.
    n_tasks = count_geometries(Path(geometry_dir))
    plan = hardware.recommend_gaussian_resources(n_tasks)
    ui.panel(plan.summary(), title="⚙  Hardware optimization", border_style="accent")
    ui.note("Press Enter to accept the recommended values, or override:")
    nproc = ui.ask_int("Cores per Gaussian job (%nprocshared)", plan.nproc)
    jobs = ui.ask_int("Parallel Gaussian jobs", plan.jobs)
    mem = ui.ask_text("Memory per job (%mem)", plan.mem)

    ui.rule(f"Gaussian: {operation}{' + COSMO' if cosmo else ''}")
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

    # Advanced override: a non-"auto" preopt choice wins over the benchmark.
    if gp.preopt_mode != "auto":
        preopt_mode = gp.preopt_mode
        ui.note(f"Preopt overridden by advanced options: {preopt_mode}")

    ui.panel(
        f"Preopt method  : {best_preopt_row['preopt_mode']} ({best_preopt_row['preopt_seconds']:.3f}s)\n"
        f"Cores per job  : {nproc} threads\n"
        f"Parallel jobs  : {jobs}\n"
        f"Memory per job : {mem}\n"
        f"Solvation      : {'COSMO (CPCM, water)' if cosmo else 'gas phase'}",
        title="Gaussian configuration (from benchmarks)",
        border_style="accent",
    )

    # Production batch — the headless service runs one batch per charge state,
    # all sharing the single benchmark above. The UI is supplied as callbacks:
    # the overwrite decision and the progress display below are the only parts
    # that know about the terminal; the orchestration itself is UI-free.
    def _decide_existing(state_name, n_existing):
        choice = ui.select(
            f"Existing {state_name} results found ({operation}"
            f"{' + COSMO' if cosmo else ''}) — {n_existing} molecule(s) already completed.",
            [
                ("Resume — keep completed molecules, run only the rest", "resume"),
                ("Restart — delete these results and recompute everything", "restart"),
                ("Skip this charge state", "skip"),
            ],
        )
        return "skip" if choice is ui.CANCEL else choice

    def _on_state_start(state):
        ui.rule(f"{operation} — {state.state_name} "
                f"(charge {state.charge}, multiplicity {state.multiplicity})")

    def _on_state_done(state):
        if state.skipped:
            ui.note(f"Skipped {state.state_name} — no data changed.")
            return
        ui.panel(
            f"[success]Successful:[/success] {state.n_success}\n"
            f"[error]Failed:[/error] {state.n_failed}",
            title=f"Gaussian finished — {state.state_name}",
            border_style="success" if state.n_failed == 0 else "warn",
        )

    run_charge_state_batches(
        geometry_dir,
        gaussian_root,
        operation=operation,
        cosmo=cosmo,
        charge_states=charge_states,
        batch_settings=dict(
            jobs=jobs, nproc=nproc, mem=mem,
            preopt_mode=preopt_mode, preopt_basis_set=preopt_basis_set,
            method=gp.method, basis_set=gp.basis_set,
            timeout=(gp.timeout or None), extra_keywords=gp.extra_keywords,
        ),
        decide_existing=_decide_existing,
        on_state_start=_on_state_start,
        on_state_done=_on_state_done,
    )
    ui.pause()


# ============================================================
# Structure menus
# ============================================================

def generate_structure_library_menu():
    context.mode = "Generating Structure Library"
    context.active_set = None
    context.info = {}
    render_header()

    name = ui.ask_text("Structure library name")
    if not name:
        ui.warn("Cancelled.")
        ui.pause()
        return

    sp = configure_stage("structures")
    with ui.status("Generating structures…"):
        structure_set = generate_structures(ring_sizes=tuple(sp.ring_sizes))
    structure_id = save_structure_library(structure_set, name)

    ui.success(
        f"Created '{name}'  ({structure_id}) — "
        f"{len(structure_set['structures'])} structures"
    )
    ui.pause()


def download_compound_library_menu(structure_id, df):
    """Prompt for download parameters and fetch PubChem compounds for each
    SMILES in the structure library."""
    context.mode = "Downloading Compounds"
    context.info = {}
    render_header()

    name = ui.ask_text("Compound library name")
    if not name:
        ui.warn("Cancelled.")
        ui.pause()
        return

    smiles_candidates = [c for c in df.columns if "smiles" in c.lower()]
    if not smiles_candidates:
        ui.error("No SMILES column found in the structure library.")
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

    # Advanced options: max records, properties, batch size, request sleep (+ presets).
    dp = configure_stage("download")
    max_records = dp.max_records
    properties = dp.properties
    batch_size = dp.batch_size
    sleep = dp.sleep

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
    compound_id, df_out = save_compound_library(unique_rows, name, structure_id, parameters)

    ui.success(f"Created '{name}'  ({compound_id}) — {len(df_out)} compounds")
    ui.pause()


# ============================================================
# Model menus
# ============================================================

def generate_model_menu(dataset_id):
    info = load_dataset_info(dataset_id)
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
    if not _confirm_overwrite(output_dir, "Model output"):
        ui.note("Cancelled.")
        ui.pause()
        return
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

    # Advanced options: fingerprint + XGBoost hyperparameters, with presets.
    tp = configure_stage("training")
    params_json = output_dir / "training_params.json"
    params_json.write_text(json.dumps(P.to_dict(tp), indent=2))

    # Optional: fold a DFT/redox descriptor set into the training features.
    dft_csv = None
    dft_sets = dft_descriptor_sets_for_model(model_id)
    if dft_sets:
        pick = ui.select(
            "Include a DFT/redox descriptor set as features?",
            [(f"{info['name']}  ({info.get('compound_count', '?')} compounds)", did)
             for did, info in dft_sets]
            + [("None — fingerprints + physchem only", None)],
        )
        if pick not in (ui.CANCEL, None):
            _, ddf = load_dft_descriptor_set(pick)
            dft_csv = output_dir / "dft_features.csv"
            ddf.to_csv(dft_csv, index=False)
            ui.info(f"Including {len([c for c in ddf.columns if c != 'CID'])} DFT/redox features "
                    f"(covering {ddf['CID'].nunique()} compounds).")

    # Optional: per-target feature selection (recommended with DFT / small data).
    fs_json = None
    if ui.confirm("Configure feature selection? (recommended for small data / many features)",
                  default=bool(dft_csv)):
        fsp = configure_stage("feature_selection")
        fs_json = output_dir / "fs_params.json"
        fs_json.write_text(json.dumps(P.to_dict(fsp), indent=2))

    parameters = {
        "radius": tp.radius, "nbits": tp.nbits,
        "training": P.to_dict(tp), "source_dataset": str(input_file),
        "dft_features": bool(dft_csv),
    }
    register_model(dataset_id, name, parameters, output_dir)

    _ensure_control_room()   # model CV-R² panel updates when training finishes
    ui.rule("Training model")
    argv = [
        "--input", str(input_file),
        "--models-dir", str(output_dir),
        "--workdir", str(output_dir),  # keep training intermediates inside the model dir
        "--params-json", str(params_json),
    ]
    if dft_csv:
        argv += ["--dft-csv", str(dft_csv)]
    if fs_json:
        argv += ["--fs-params-json", str(fs_json)]
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
        choices = [ui.section("Trained models")]
        for pos, (_, row) in enumerate(page_df.iterrows()):
            label = (f"{row['target'][:64]}  "
                     f"(n={row['n_samples']}, CV R²={row['cv_r2_mean']:.4f})")
            choices.append((label, ("model", pos)))
        choices.append(ui.section(""))
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


def _geometry_id_for(geometry_dir):
    return next((k for k, v in GEOMETRIES.items()
                 if Path(v["output_dir"]) == geometry_dir), None)


def _dft_sets_for_geometry(geometry_dir):
    gid = _geometry_id_for(geometry_dir)
    return [(d, info) for d, info in DFT_DESCRIPTORS.items()
            if info.get("geometry_id") == gid]


def _show_comparison_result(dft_id, comp):
    render_header()
    delta = comp["delta_cv_r2"]
    verdict = "[success]DFT helps[/success]" if (delta or 0) > 0 else "[warn]no CV gain[/warn]"
    ui.table(
        [("Metric", "muted"), ("Value", "white")],
        [
            ("Comparison", comp["id"]),
            ("DFT set", dft_id),
            ("Target", comp["target"]),
            ("Created", comp["created"].replace("T", " ")),
            ("Samples", comp["n_samples"]),
            ("DFT-covered compounds", comp["n_dft_covered"]),
            ("Baseline CV R\u00b2", f"{comp['baseline_cv_r2']:.4f} \u00b1 {comp['baseline_cv_std']:.3f}"),
            ("With DFT CV R\u00b2", f"{comp['augmented_cv_r2']:.4f} \u00b1 {comp['augmented_cv_std']:.3f}"),
            ("\u0394 CV R\u00b2", f"{delta:+.4f}  ({verdict})"),
            ("DFT importance share", f"{comp['dft_total_importance']:.4f}"),
        ],
        title="DFT feature value — CV comparison",
    )

    # When available, show both CV splits (random = optimistic, scaffold =
    # honest) plus a low-data flag. The random-minus-scaffold gap is the
    # leakage meter that explains a big apparent drop.
    if "baseline_cv_r2_random" in comp:
        grouping = comp.get("grouping", "scaffold")

        def _f(v):
            return "n/a" if v is None else f"{v:+.4f}"

        def _g(prefix):  # grouped value, with back-compat for old _scaffold entries
            return comp.get(f"{prefix}_grouped", comp.get(f"{prefix}_scaffold"))

        extra = [
            ("Groups", comp.get("n_groups", comp.get("n_scaffold_groups", "?"))),
            (f"Baseline R2  (random / {grouping})",
             f"{_f(comp.get('baseline_cv_r2_random'))}  /  {_f(_g('baseline_cv_r2'))}"),
            (f"With-DFT R2  (random / {grouping})",
             f"{_f(comp.get('augmented_cv_r2_random'))}  /  {_f(_g('augmented_cv_r2'))}"),
            (f"Delta R2  (random / {grouping})",
             f"{_f(comp.get('delta_cv_r2_random'))}  /  {_f(_g('delta_cv_r2'))}"),
        ]
        if comp.get("reliable") is False:
            extra.append(("Reliability",
                          f"[warn]{comp.get('reliability_note', 'low data')} - scores unstable[/warn]"))
        ui.table([("Metric", "muted"), ("Value", "white")], extra,
                 title=f"Random vs {grouping} CV  (gap = leakage)")

    top = list(comp.get("dft_importance", {}).items())[:8]
    if top:
        ui.table(
            [("DFT feature", "white"), ("Importance", "muted")],
            [(k, f"{v:.4f}") for k, v in top],
            title="Top DFT feature importances (augmented model)",
        )
    ui.pause()


def _benchmark_feature_selection(model_id, row, geometry_dir):
    """Benchmark the selection methods (before/after DFT) and open the comparison plot."""
    render_header()
    sets = _dft_sets_for_geometry(geometry_dir)
    dft_df = None
    if sets:
        _, dft_df = load_dft_descriptor_set(sets[-1][0])
    else:
        ui.note("No DFT/redox descriptor set found — benchmarking without the DFT block.")

    model_info = MODELS.get(model_id)
    params = model_info["parameters"]
    source = params.get("source_dataset")
    if not source or not Path(source).exists():
        ui.error("Model source dataset not found.")
        ui.pause()
        return
    df = pd.read_csv(source)
    try:
        dataset = DATASETS.get(model_info.get("dataset_id"))
        _, comp = load_compound_library(dataset["compound_id"])
        smiles = comp[["CID", "SMILES"]].drop_duplicates("CID").astype({"CID": str})
        df["CID"] = df["CID"].astype(str)
        df = df.merge(smiles, on="CID", how="left")
    except Exception:
        pass

    grouping = ui.select(
        "Group the cross-validation by:",
        [
            ("Cluster — predict new analogs of this family (recommended)", "cluster"),
            ("Scaffold — novel-chemotype stress test", "scaffold"),
        ],
    )
    if grouping is ui.CANCEL:
        grouping = "cluster"

    # Advanced options + presets for both the model and the selection sweep.
    tp = configure_stage("training")
    fsp = configure_stage("feature_selection")

    ui.info(f"Benchmarking '{row['target']}' — all methods × before/after DFT, "
            f"random vs {grouping} CV…")
    with ui.status("Running the sweep — this trains several small models…"):
        res = run_feature_selection_benchmark(df, row["target"], dft_df,
                                              tp=tp, fsp=fsp, grouping=grouping)
    if "error" in res:
        ui.warn(f"Cannot benchmark: {res['error']}")
        ui.pause()
        return

    # Results table — method × (with/without DFT) → kept features + dual CV.
    render_header()

    def _r(v):
        return "n/a" if v is None else f"{v:+.3f}"

    table_rows = [(r["method"], r["block"].replace("_", " "), r["n_features"],
                   _r(r.get("cv_r2_random")), _r(r.get("cv_r2_scaffold")))
                  for r in res["results"]]
    ui.table(
        [("Method", "white"), ("Block", "muted"), ("Feats", "muted"),
         ("R² random", "accent"), (f"R² {grouping}", "accent")],
        table_rows,
        title=f"Feature-selection benchmark — {row['target']}  "
              f"(n={res['n_samples']}, {grouping} groups={res['n_groups']})",
    )

    out_dir = Path(model_info["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"feature_benchmark_{slugify(row['target'])}.json"
    json_path.write_text(json.dumps(res, indent=2))

    # Offer to bake the winning method into a feature-selection preset, which
    # the training menu can then load — wiring "best method" into training.
    winner = best_method(res)
    if winner:
        ui.success(f"Best method (by {grouping} CV): {winner}")
        if ui.confirm(f"Save '{winner}' as a feature-selection preset to reuse in training?",
                      default=True):
            pname = ui.ask_text("Preset name", "best")
            if pname:
                presets.save_preset("feature_selection", pname, P.to_dict(best_config(res)))
                ui.success(f"Saved preset '{pname}'. Load it in training → feature selection.")

    ui.note("Opening comparison plot — close it to return…")
    _launch([sys.executable, str(_REPO_ROOT / "scripts" / "feature_benchmark_plot.py"),
             str(json_path)], what="feature benchmark plot")
    ui.pause()


def _optimize_training_params(model_id, row, geometry_dir):
    """Bayesian (Optuna) search over the XGBoost hyperparameters for this target."""
    render_header()
    model_info = MODELS.get(model_id)
    params = model_info["parameters"]
    source = params.get("source_dataset")
    if not source or not Path(source).exists():
        ui.error("Model source dataset not found.")
        ui.pause()
        return
    df = pd.read_csv(source)
    try:
        dataset = DATASETS.get(model_info.get("dataset_id"))
        _, comp = load_compound_library(dataset["compound_id"])
        smiles = comp[["CID", "SMILES"]].drop_duplicates("CID").astype({"CID": str})
        df["CID"] = df["CID"].astype(str)
        df = df.merge(smiles, on="CID", how="left")
    except Exception:
        pass

    # Optionally fold in the same DFT/redox features + selection you'll train on,
    # so the search optimizes the real feature matrix.
    sets = _dft_sets_for_geometry(geometry_dir)
    dft_df = None
    if sets and ui.confirm("Include the latest DFT/redox descriptor set as features?",
                           default=True):
        _, dft_df = load_dft_descriptor_set(sets[-1][0])

    grouping = ui.select(
        "Optimize the CV score grouped by:",
        [
            ("Cluster — predict new analogs of this family (recommended)", "cluster"),
            ("Scaffold — novel-chemotype stress test", "scaffold"),
        ],
    )
    if grouping is ui.CANCEL:
        grouping = "cluster"

    base_tp = configure_stage("training")
    fsp = None
    if ui.confirm("Apply a feature-selection config during the search?",
                  default=dft_df is not None):
        fsp = configure_stage("feature_selection")
    n_trials = ui.ask_int("Number of search trials (more = better, slower)", 50)

    ui.info(f"Optimizing XGBoost hyperparameters for '{row['target']}' "
            f"({grouping} CV, {n_trials} trials)…")
    with ui.status("Bayesian search (Optuna TPE) — running many quick trainings…"):
        res = optimize_training_params(df, row["target"], dft_df,
                                       base_tp=base_tp, fsp=fsp,
                                       n_trials=n_trials, grouping=grouping)
    if "error" in res:
        ui.warn(f"Cannot optimize: {res['error']}")
        ui.pause()
        return

    render_header()
    ui.table(
        [("Parameter", "accent"), ("Optimized value", "white")],
        [(k, str(v)) for k, v in res["tuned"].items()],
        title=f"Best hyperparameters — {row['target']}  "
              f"(best {grouping} CV R² = {res['best_score']:+.4f}, "
              f"n={res['n_samples']}, trials={res['n_trials']})",
    )
    if res["n_samples"] < base_tp.min_reliable_samples:
        ui.warn(f"n={res['n_samples']} is small — the 'best' config can be tuned to "
                "CV noise. Treat the improvement cautiously.")

    if ui.confirm("Save these optimized parameters as a training preset?", default=True):
        pname = ui.ask_text("Preset name", "optimized")
        if pname:
            presets.save_preset("training", pname, res["best_params"])
            ui.success(f"Saved preset '{pname}'. Load it in training → Advanced options.")
    ui.pause()


def _dft_comparisons_menu(model_id, row, geometry_dir):
    while True:
        render_header()
        sets = _dft_sets_for_geometry(geometry_dir)
        comparisons = [(d, c) for d, info in sets for c in info.get("comparisons", [])]
        if not comparisons:
            ui.note("No saved comparisons — run 'Benchmark feature selection + DFT' first.")
            ui.pause()
            return
        choices = [ui.section("Comparisons")]
        for dft_id, c in comparisons:
            label = (f"{c['id']}  ·  \u0394 CV R\u00b2={c['delta_cv_r2']:+.4f}  ·  "
                     f"{c['created'].replace('T', ' ')}")
            choices.append((label, (dft_id, c)))
        choices.append(ui.section(""))
        choices.append(("← Back", None))
        choice = ui.select("DFT comparison results", choices)
        if choice is ui.CANCEL or choice is None:
            return
        _show_comparison_result(choice[0], choice[1])


def individual_model_menu(model_id, row):
    while True:
        # Each Geometry set belongs to one (model, target); only show this target's.
        geometry_sets = find_model_geometry_sets(model_id, row["target"])

        render_header()
        ui.table(
            [("Field", "muted"), ("Value", "white")],
            [("Target", row["target"]), ("Samples", row["n_samples"]),
             ("CV R²", f"{row['cv_r2_mean']:.4f}")],
            title="Selected model",
        )

        choices = []
        if geometry_sets:
            choices.append(ui.section("Geometry sets"))
            choices += [
                (f"{s.name}  ({count_geometries(s)} compounds)", ("open", i))
                for i, s in enumerate(geometry_sets)
            ]
        choices.append(ui.section("Actions"))
        choices.append(("＋ Generate new Geometry set", ("new", None)))
        choices.append(("← Back", ("back", None)))

        choice = ui.select("Geometry sets", choices)
        action, i = (("back", None) if choice is ui.CANCEL else choice)
        if action == "back":
            break
        if action == "open":
            geometry_set_menu(model_id, row, geometry_sets[i])
        elif action == "new":
            generate_geometry_set(model_id, row)


def generate_geometry_set(model_id, row):
    target = row["target"]
    model_info = MODELS.get(model_id)

    context.mode = "Generating Geometry Set"
    context.info = {}
    render_header()

    name = ui.ask_text("Geometry set name")
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

    geometry_id = GEOMETRIES.next_id()
    output_dir = GEOMETRY_RUN_DIR / geometry_id
    if not _confirm_overwrite(output_dir, "Geometry set output"):
        ui.note("Cancelled.")
        ui.pause()
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_csv = output_dir / "training.csv"
    df_target.to_csv(temp_csv, index=False)

    ui.rule("Downloading SDF files")
    download_training_sdfs_main(["--input", str(temp_csv), "--output-dir", str(output_dir)])

    count = len(list(output_dir.glob("*.sdf")))
    register_geometry_set(
        model_id, target, name, output_dir,
        {"target": target, "dataset": str(dataset), "rows": len(df_target)},
        count,
    )
    ui.success(f"Geometry set created — {count} compounds")
    ui.pause()


def _create_dft_descriptor_set(model_id, row, geometry_dir):
    """Parse this geometry set's Gaussian freq logs into a new DFT descriptor set."""
    render_header()
    rows = parse_geometry_descriptors(geometry_dir / "gaussian")
    if not rows:
        ui.warn("No completed frequency calculations found here — "
                "run 'Gaussian opt then freq' first.")
        ui.pause()
        return
    ui.info(f"Parsed {len(rows)} compounds from frequency logs.")
    name = ui.ask_text("DFT descriptor set name", f"{row['target']} DFT")
    if not name:
        ui.warn("Cancelled.")
        ui.pause()
        return
    geometry_id = next(
        (k for k, v in GEOMETRIES.items() if Path(v["output_dir"]) == geometry_dir),
        None,
    )
    with ui.status("Saving descriptor set…"):
        dft_id, df = save_dft_descriptor_set(
            rows, name, model_id, row["target"], geometry_id,
            parameters={"source_gaussian": str(geometry_dir / "gaussian")},
        )
    n_props = len([c for c in df.columns if c != "CID"])
    ui.success(f"Created DFT descriptor set '{name}'  ({dft_id}) — "
               f"{len(df)} compounds × {n_props} properties")
    ui.pause()


def _create_redox_descriptor_set(model_id, row, geometry_dir):
    """Parse the COSMO neutral+anion runs into a redox descriptor set.

    Produces neutral_*, anion_*, and delta_* (ΔG/ΔH/ΔS of reduction, etc.)
    features for every compound completed in both charge states.
    """
    render_header()
    cosmo_root = geometry_dir / "gaussian" / "opt_then_freq_COSMO"
    if not cosmo_root.exists():
        ui.warn("No COSMO results here — run "
                "'Run opt+freq with COSMO (neutral + anion)' first.")
        ui.pause()
        return
    rows = parse_redox_descriptors(cosmo_root)
    if not rows:
        ui.warn("No compounds completed in BOTH neutral and anion states yet "
                "(a redox feature needs both).")
        ui.pause()
        return
    ui.info(f"Parsed {len(rows)} compounds with neutral / anion / Δ features.")
    name = ui.ask_text("Redox descriptor set name", f"{row['target']} redox")
    if not name:
        ui.warn("Cancelled.")
        ui.pause()
        return
    geometry_id = next(
        (k for k, v in GEOMETRIES.items() if Path(v["output_dir"]) == geometry_dir),
        None,
    )
    with ui.status("Saving descriptor set…"):
        dft_id, df = save_dft_descriptor_set(
            rows, name, model_id, row["target"], geometry_id,
            parameters={"source_cosmo": str(cosmo_root), "kind": "redox"},
        )
    n_props = len([c for c in df.columns if c != "CID"])
    ui.success(f"Created redox descriptor set '{name}'  ({dft_id}) — "
               f"{len(df)} compounds × {n_props} features")
    ui.pause()


def show_dft_descriptor_set(dft_id):
    info = DFT_DESCRIPTORS.get(dft_id)
    context.mode = "DFT Descriptor Set"
    context.info = {
        "Compounds": info["compound_count"],
        "Created": info["created"].replace("T", " "),
    }
    render_header()
    ui.table(
        [("Field", "muted"), ("Value", "white")],
        [
            ("Name", info["name"]),
            ("Model", info["model_id"]),
            ("Target", info["target"]),
            ("Geometry set", info.get("geometry_id")),
            ("Compounds", info["compound_count"]),
            ("Properties", ", ".join(info.get("property_columns", []))),
        ],
        title="DFT descriptor set",
    )
    return info


def geometry_set_menu(model_id, row, geometry_dir):
    while True:
        render_header()
        ui.table(
            [("Field", "muted"), ("Value", "white")],
            [("Geometry set", geometry_dir.name), ("Compounds", count_geometries(geometry_dir))],
            title="Selected Geometry set",
        )
        action = ui.select(
            "Actions",
            [
                ui.section("Gaussian calculations"),
                ("Run Gaussian opt then freq", "opt"),
                ("Run opt+freq with COSMO (neutral + anion)", "opt_cosmo"),
                ("Run Gaussian single point", "sp"),
                ui.section("DFT analysis"),
                ("Parse DFT descriptors (freq results)", "dft"),
                ("Parse COSMO redox descriptors (neutral / anion / Δ)", "redox"),
                ("Benchmark feature selection + DFT (random vs cluster/scaffold)", "fsbench"),
                ("Optimize training parameters (Bayesian sweep)", "hpo"),
                ("DFT comparison results (history)", "dftresults"),
                ui.section("Manage"),
                ("Delete Geometry set", "delete"),
                ("← Back", "back"),
            ],
        )
        if action is ui.CANCEL or action == "back":
            break
        if action == "opt":
            run_gaussian_pipeline(model_id, geometry_dir, operation="opt then freq")
        elif action == "opt_cosmo":
            run_gaussian_pipeline(
                model_id, geometry_dir, operation="opt then freq", cosmo=True,
                charge_states=[("neutral", 0, 1), ("anion", -1, 2)],
            )
        elif action == "sp":
            run_gaussian_pipeline(model_id, geometry_dir, operation="sp")
        elif action == "dft":
            _create_dft_descriptor_set(model_id, row, geometry_dir)
        elif action == "redox":
            _create_redox_descriptor_set(model_id, row, geometry_dir)
        elif action == "fsbench":
            _benchmark_feature_selection(model_id, row, geometry_dir)
        elif action == "hpo":
            _optimize_training_params(model_id, row, geometry_dir)
        elif action == "dftresults":
            _dft_comparisons_menu(model_id, row, geometry_dir)
        elif action == "delete":
            geometry_id = next(
                (k for k, v in GEOMETRIES.items() if Path(v["output_dir"]) == geometry_dir),
                None,
            )
            if geometry_id is None:
                ui.warn("Could not resolve geometry ID.")
                continue
            if delete_geometry_set(geometry_id):
                ui.pause()
                break


# ============================================================
# Property dataset display
# ============================================================

def show_dataset(dataset_id):
    info = load_dataset_info(dataset_id)
    context.mode = "Property Dataset"
    context.active_run = info["name"]
    context.info = {
        "Status": info["status"],
        "Created": info["created"].replace("T", " "),
    }
    render_header()
    return info


def browse_dataset_outputs(dataset_id):
    info = show_dataset(dataset_id)
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


def preview_dataset_output(dataset_id):
    """Pick an output CSV and print its first 20 rows."""
    info = load_dataset_info(dataset_id)
    output_dir = Path(info["output_dir"])
    csvs = sorted(output_dir.glob("*.csv"))
    if not csvs:
        ui.note("No CSV outputs found.")
        return

    show_dataset(dataset_id)
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
# Property dataset menus
# ============================================================

def new_dataset_menu(compound_id):
    context.mode = "New Property Dataset"
    context.active_run = None
    context.info = {}
    render_header()

    name = ui.ask_text("Run name")
    if not name:
        ui.warn("Cancelled.")
        ui.pause()
        return

    # Advanced options: compound cap, PubMed/PubChem/ChEMBL limits, pharma
    # toggles (+ presets). Returned as a plain dict for the rest of the flow.
    params = P.to_dict(configure_stage("dataset"))

    # Reserve output directory using the prospective run ID.
    provisional_id = DATASETS.next_id()
    output_dir = DATASET_RUN_DIR / provisional_id
    if not _confirm_overwrite(output_dir, "Property dataset output"):
        ui.note("Cancelled.")
        ui.pause()
        return
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

    dataset_id = register_dataset(compound_id, name, params, output_dir)
    update_dataset_status(dataset_id, "running")
    context.active_run = name

    compound_info, _ = load_compound_library(compound_id)
    input_csv = str(COMPOUND_DIR / compound_info["file"])

    argv = [
        "--input", input_csv,
        "--output", str(output_dir / "report.csv"),
        "--parsed-output", str(output_dir / "parsed.csv"),
        "--ml-output", str(output_dir / "wide.csv"),
        "--wide-output", str(output_dir / "wide_clean.csv"),
        "--cleaned-output", str(output_dir / "cleaned.csv"),
        "--summary-output", str(output_dir / "summary.csv"),
        "--rare-output", str(output_dir / "rare.csv"),
        "--sparse-output", str(output_dir / "sparse.csv"),
        "--cleaning-report", str(output_dir / "cleaning_report.csv"),
        "--pharma-output", str(output_dir / "pharma.csv"),
        "--merged-pharma-output", str(output_dir / "merged_pharma.csv"),
        "--timing-log", str(output_dir / "timing_log.txt"),
        "--pubmed-max", str(params["pubmed_max"]),
        "--pubchem-max-aids", str(params["pubchem_max_aids"]),
        "--chembl-activity-limit", str(params["chembl_activity_limit"]),
        "--chembl-max-pages", str(params["chembl_max_pages"]),
    ]
    if params["max_compounds"]:
        argv += ["--max", str(params["max_compounds"])]
    if params["fetch_pharma"]:
        argv.append("--fetch-pharma")
    if params.get("fetch_pubmed"):
        argv.append("--fetch-pubmed")
    if params["merge_pharma"]:
        argv.append("--merge-pharma")
    if params["wide_from_clean"]:
        argv.append("--wide-from-clean")

    _ensure_control_room()   # dataset-mining throughput panel goes live
    ui.rule("Running find_thermo")
    try:
        find_thermo_main(argv)
        update_dataset_status(dataset_id, "completed")
        ui.success("Run completed.")
    except Exception as exc:
        update_dataset_status(dataset_id, f"failed: {exc}")
        ui.error(f"Run failed: {exc}")
    ui.pause()


def dataset_menu(dataset_id):
    while True:
        show_dataset(dataset_id)
        models = models_for_dataset(dataset_id)

        choices = [
            ui.section("Outputs"),
            ("Browse output files", ("browse", None)),
            ("Preview a CSV", ("preview", None)),
            ui.section("Models"),
            ("Train new model", ("train", None)),
        ]
        for model_id, info in models:
            choices.append((f"Model: {info['name']}", ("model", model_id)))
        choices.append(ui.section(""))
        choices.append(("← Back", ("back", None)))

        choice = ui.select("Property dataset", choices)
        action, payload = (("back", None) if choice is ui.CANCEL else choice)
        if action == "back":
            context.active_run = None
            context.info = {}
            break
        if action == "browse":
            browse_dataset_outputs(dataset_id)
            ui.pause()
        elif action == "preview":
            preview_dataset_output(dataset_id)
            ui.pause()
        elif action == "train":
            generate_model_menu(dataset_id)
        elif action == "model":
            model_menu(payload)


def datasets_menu(compound_id):
    while True:
        runs = datasets_for_compound(compound_id)
        context.mode = "Property Datasets"
        context.active_run = None
        context.info = {}
        render_header()

        choices = []
        if runs:
            choices.append(ui.section("Property datasets"))
        for dataset_id, info in runs:
            created = info["created"].replace("T", " ")
            choices.append((
                f"{info['name']}  —  {info.get('status', '?')}  (created {created})",
                ("open", dataset_id),
            ))
        choices.append(ui.section("Actions"))
        choices.append(("＋ New dataset", ("new", None)))
        choices.append(("← Back", ("back", None)))

        choice = ui.select("Property datasets for this compound library", choices)
        action, dataset_id = (("back", None) if choice is ui.CANCEL else choice)
        if action == "back":
            break
        if action == "open":
            dataset_menu(dataset_id)
        elif action == "new":
            new_dataset_menu(compound_id)


# ============================================================
# Compound / structure menus
# ============================================================

def compound_library_menu(compound_id):
    info, df = load_compound_library(compound_id)
    while True:
        show_compound_library(info, compound_id)
        action = ui.select(
            "Actions",
            [
                ui.section("Browse"),
                ("Browse compounds", "browse"),
                ("Property datasets", "thermo"),
                ui.section(""),
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
            datasets_menu(compound_id)


def compound_libraries_menu(structure_id):
    """List compound librarys linked to this structure library and allow downloading a
    new one or opening an existing one."""
    while True:
        csets = compound_libraries_for_structure(structure_id)
        context.mode = "Compound Libraries"
        context.active_compound_set = None
        context.info = {}
        render_header()

        choices = []
        if csets:
            choices.append(ui.section("Compound libraries"))
        for compound_id, info in csets:
            created = info["created"].replace("T", " ")
            choices.append((
                f"{info['name']}  ({info['compound_count']} compounds, created {created})",
                ("open", compound_id),
            ))
        choices.append(ui.section("Actions"))
        choices.append(("＋ Download new compound library", ("new", None)))
        choices.append(("← Back", ("back", None)))

        choice = ui.select("Compound libraries for this structure library", choices)
        action, compound_id = (("back", None) if choice is ui.CANCEL else choice)
        if action == "back":
            break
        if action == "open":
            compound_library_menu(compound_id)
        elif action == "new":
            _, df = load_structure_library(structure_id)
            download_compound_library_menu(structure_id, df)


def structure_library_menu(structure_id):
    info, df = load_structure_library(structure_id)
    while True:
        show_structure_library(info, structure_id)
        action = ui.select(
            "Actions",
            [
                ui.section("Browse"),
                ("Browse structures", "browse"),
                ("Compounds", "compounds"),
                ui.section(""),
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
            compound_libraries_menu(structure_id)


# ============================================================
# Results & Reports (interactive matplotlib explorer)
# ============================================================

_SRC_DIR = Path(__file__).resolve().parents[2]      # …/src
_REPO_ROOT = Path(__file__).resolve().parents[3]    # repo root


def _launch(cmd, *, cwd=None, what="plot"):
    """Run a GUI subprocess (blocking), surfacing failures in the CLI."""
    ui.note(f"Opening {what} window — close it to return…")
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except Exception as exc:
        ui.error(f"Could not open {what}: {exc}")
        ui.pause()
        return
    if result.returncode != 0:
        ui.error(f"{what} failed:\n{(result.stderr or '').strip()[-700:]}")
        ui.pause()


def _launch_viz(kind, *ids):
    # Run from src/ so `ivette` resolves to the package (not the root ivette.py).
    _launch([sys.executable, "-m", "ivette.viz", kind, *map(str, ids)],
            cwd=str(_SRC_DIR), what="plot")


def _launch_control_room():
    _launch([sys.executable, str(_REPO_ROOT / "scripts" / "control_room.py")],
            what="control room")


# Background (non-blocking) live UI: at most one control-room window is kept
# open across operations so progress/convergence plots are always on screen
# while work runs. It refreshes off data files in a separate process, so it
# costs the pipeline nothing.
_control_room_proc = None


def _display_available() -> bool:
    """True if a GUI window can plausibly be shown.

    Avoids spawning a doomed matplotlib process on a headless box. Windows/macOS
    always have a display; on Linux we require an X/Wayland session (WSLg sets
    DISPLAY, so this is true under WSL with GUI support).
    """
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _live_ui_enabled() -> bool:
    """Auto live UI is on unless IVETTE_LIVE_UI is set to a falsey value."""
    return os.environ.get("IVETTE_LIVE_UI", "1").strip().lower() not in ("0", "false", "no", "off")


def _ensure_control_room():
    """Open the live control room in the background if not already running.

    Non-blocking and idempotent — safe to call at the start of every operation.
    No-ops when disabled, non-interactive, or no display is available.
    """
    global _control_room_proc
    if not (_live_ui_enabled() and ui._interactive() and _display_available()):
        return
    if _control_room_proc is not None and _control_room_proc.poll() is None:
        return  # one is already up
    try:
        _control_room_proc = subprocess.Popen(
            [sys.executable, str(_REPO_ROOT / "scripts" / "control_room.py")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        applog.get_logger("ui").info("live control room launched (pid=%s)", _control_room_proc.pid)
        ui.note("📊 Live control room opened in a separate window (set IVETTE_LIVE_UI=0 to disable).")
    except Exception as exc:  # never let a UI window break the actual work
        applog.get_logger("ui").warning("could not launch control room: %s", exc)
        _control_room_proc = None


def _close_control_room():
    """Terminate the background control room, if any (called on app exit)."""
    global _control_room_proc
    proc = _control_room_proc
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
    _control_room_proc = None


def _report_entities(title, store, kind, show_fn):
    """List entities of one stage; show metadata then offer the interactive plot."""
    while True:
        items = list(store.items())
        context.mode = f"Reports · {title}"
        context.active_set = context.active_compound_set = context.active_run = None
        context.info = {}
        render_header()
        if not items:
            ui.note(f"No {title.lower()} yet.")
            ui.pause()
            return
        choices = [ui.section(title)]
        choices += [(f"{info.get('name', eid)}  ({eid})", eid) for eid, info in items]
        choices.append(ui.section(""))
        choices.append(("← Back", None))
        eid = ui.select(f"Pick from {title.lower()}", choices)
        if eid is ui.CANCEL or eid is None:
            return
        show_fn(eid)
        action = ui.select("Actions", [
            ("📈 Open interactive plot", "plot"),
            ("← Back", "back"),
        ])
        if action == "plot":
            _launch_viz(kind, eid)


def reports_menu():
    while True:
        context.mode = "Results & Reports"
        context.active_set = context.active_compound_set = context.active_run = None
        context.info = {}
        render_header()
        ui.note("Browse results & metadata from every stage; open interactive plots.")
        action = ui.select("Results & Reports — choose a stage", [
            ui.section("Browse data"),
            ("Structure libraries", "structures"),
            ("Compound libraries", "compounds"),
            ("Property datasets", "thermo"),
            ("Trained models", "models"),
            ("DFT descriptor sets", "dft"),
            ui.section("Live & plots"),
            ("Gaussian benchmarks (plot)", "benchmarks"),
            ("Live control room (Gaussian monitor)", "control_room"),
            ui.section(""),
            ("← Back", "back"),
        ])
        if action is ui.CANCEL or action == "back":
            context.clear()
            return
        if action == "structures":
            _report_entities("Structure libraries", STRUCTURES, "structure_library",
                             lambda sid: show_structure_library(STRUCTURES.get(sid), sid))
        elif action == "compounds":
            _report_entities("Compound libraries", COMPOUNDS, "compound_library",
                             lambda cid: show_compound_library(COMPOUNDS.get(cid), cid))
        elif action == "thermo":
            _report_entities("Property datasets", DATASETS, "property_dataset", show_dataset)
        elif action == "models":
            _report_entities("Trained models", MODELS, "model", show_model)
        elif action == "dft":
            _report_entities("DFT descriptor sets", DFT_DESCRIPTORS, "dft_descriptor_set", show_dft_descriptor_set)
        elif action == "benchmarks":
            _launch_viz("benchmarks")
        elif action == "control_room":
            _launch_control_room()


def main():
    ensure_storage()
    applog.configure()
    log = applog.get_logger("app")
    log.info("Ivette session started")
    try:
        _run_main_loop(log)
    finally:
        _close_control_room()


def _run_main_loop(log):
    with ui.fullscreen():
        while True:
            context.mode = "Structure Libraries"
            context.active_set = None
            context.info = {}

            ui.clear()
            ui.banner()

            sets = list(STRUCTURES.items())
            choices = []
            if sets:
                choices.append(ui.section("Structure libraries"))
                choices += [
                    (f"{info['name']}  ({info['structure_count']} structures)", ("open", structure_id))
                    for structure_id, info in sets
                ]
            choices.append(ui.section("Actions"))
            choices.append(("＋ Generate new structure library", ("new", None)))
            choices.append(("📊 Results & Reports", ("reports", None)))
            choices.append(("✕ Exit", ("exit", None)))

            choice = ui.select("Structure libraries", choices)
            action, structure_id = (("exit", None) if choice is ui.CANCEL else choice)
            if action == "exit":
                log.info("Ivette session ended")
                break
            if action == "open":
                structure_library_menu(structure_id)
            elif action == "new":
                generate_structure_library_menu()
            elif action == "reports":
                reports_menu()
