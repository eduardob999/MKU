#!/usr/bin/env python3
"""
gaussian16_pipeline.py
Local Gaussian 16 pipeline on Linux: SDF → .gjf input → run g16 → parse .log output.

Usage (single compound):
    python gaussian16_pipeline.py --sdf compound.sdf --workdir ./runs

Usage (batch over a directory of SDFs):
    python gaussian16_pipeline.py --sdf-dir ./sdfs --workdir ./runs --jobs 4

# Single compound, opt + freq at PBE0/6-311G*
python gaussian16_pipeline.py --sdf 10701.sdf --workdir ./runs

# Whole SDF directory, with solvent, custom scratch dir
python gaussian16_pipeline.py \
    --sdf-dir ./sdfs \
    --workdir ./runs \
    --operation "opt freq" \
    --basis "6-311+G(d,p)" \
    --method "M062X" \
    --cosmo \
    --nproc 8 \
    --mem 16GB \
    --scratch /fast/scratch \
    --max-disk 60GB

# If g16 isn't on PATH
python gaussian16_pipeline.py --sdf 10701.sdf --g16 /opt/gaussian/g16/g16
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import shutil
import subprocess
import sys
import tempfile
import json
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Scratch management
# ──────────────────────────────────────────────────────────────────────────────

def get_scratch_dir(scratch_arg: Optional[str] = None) -> str:
    """
    Resolve the scratch directory to use, in priority order:
      1. Explicit --scratch CLI argument
      2. $GAUSS_SCRDIR environment variable
      3. /tmp as fallback
    """
    if scratch_arg:
        return scratch_arg
    return os.environ.get("GAUSS_SCRDIR", "/tmp")


def cleanup_scratch_for_cid(scratch_dir: str, cid: str) -> None:
    """
    Remove all Gaussian scratch files for a specific CID from scratch_dir.
    Called both before a job starts (clear previous crash debris) and after
    a failed/interrupted job.

    File types cleaned:
        .rwf  – read-write file (the main disk hog; can be 100s of GB)
        .skr  – scratch integral file
        .d2e  – second derivative file
        .int  – integral file
        .inp  – temporary input copy
    """
    scratch = Path(scratch_dir)
    patterns = [
        f"{cid}*.rwf",
        f"{cid}*.rwf.bak",
        f"{cid}*.skr",
        f"{cid}*.d2e",
        f"{cid}*.int",
        f"{cid}*.inp",
    ]
    removed = []
    for pattern in patterns:
        for f in scratch.glob(pattern):
            try:
                f.unlink()
                removed.append(f.name)
            except OSError as e:
                print(f"  [scratch] Warning: could not remove {f}: {e}")
    if removed:
        print(f"  [scratch] Removed {len(removed)} leftover file(s): "
              f"{', '.join(removed[:8])}{'…' if len(removed) > 8 else ''}")


def cleanup_all_scratch(scratch_dir: str) -> None:
    """
    Nuclear option: remove ALL Gaussian scratch intermediates from scratch_dir.
    Does NOT touch .chk or .log files.
    Safe to run before a fresh batch.
    """
    scratch = Path(scratch_dir)
    if not scratch.exists():
        return
    patterns = ["*.rwf", "*.rwf.bak", "*.skr", "*.d2e", "*.int"]
    total = 0
    for pattern in patterns:
        for f in scratch.glob(pattern):
            try:
                f.unlink()
                total += 1
            except OSError:
                pass
    if total:
        print(f"[scratch] Cleaned {total} stale scratch file(s) from {scratch_dir}")


def report_scratch_usage(scratch_dir: str) -> None:
    """Print a quick summary of what's currently in scratch_dir."""
    scratch = Path(scratch_dir)
    if not scratch.exists():
        return
    files = list(scratch.iterdir())
    if not files:
        print(f"[scratch] {scratch_dir} is empty.")
        return
    total = sum(f.stat().st_size for f in files if f.is_file())
    print(f"[scratch] {scratch_dir}: {len(files)} files, "
          f"{total / 1024**3:.1f} GB total")


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────────────────────────────────────

def checkpoint_path(work_dir):
    return Path(work_dir) / "checkpoint.json"


def load_checkpoint(work_dir):
    path = checkpoint_path(work_dir)
    if not path.exists():
        return {"created": datetime.now().isoformat(), "jobs": {}}
    with open(path) as f:
        return json.load(f)


def save_checkpoint(work_dir, checkpoint):
    path = checkpoint_path(work_dir)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(checkpoint, f, indent=4)
    tmp.replace(path)


def update_checkpoint(work_dir, cid, status, extra=None):
    checkpoint = load_checkpoint(work_dir)
    checkpoint["jobs"].setdefault(cid, {})
    checkpoint["jobs"][cid].update({
        "status": status,
        "updated": datetime.now().isoformat(),
    })
    if extra:
        checkpoint["jobs"][cid].update(extra)
    save_checkpoint(work_dir, checkpoint)


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ThermoData:
    temp:       Optional[float] = None   # K
    freq_scale: Optional[float] = None
    zpe:        Optional[float] = None   # kcal/mol
    te:         Optional[float] = None   # kcal/mol
    th:         Optional[float] = None   # kcal/mol
    ts:         Optional[float] = None   # cal/mol·K
    ts_trans:   Optional[float] = None
    ts_rot:     Optional[float] = None
    ts_vib:     Optional[float] = None
    cv:         Optional[float] = None   # cal/mol·K
    cv_trans:   Optional[float] = None
    cv_rot:     Optional[float] = None
    cv_vib:     Optional[float] = None


@dataclass
class OptStep:
    step:      int
    energy:    float
    delta_e:   float
    rms_force: float
    max_force: float
    rms_disp:  float
    max_disp:  float


@dataclass
class RunResult:
    cid:       str
    sdf_path:  str
    gjf_path:  str
    log_path:  str
    success:   bool
    energy:    Optional[float]      = None
    thermo:    Optional[ThermoData] = None
    opt_steps: list[OptStep]        = field(default_factory=list)
    error_msg: str                  = ""


# ──────────────────────────────────────────────────────────────────────────────
# SDF → coordinate block
# ──────────────────────────────────────────────────────────────────────────────

_ATOM_LINE = re.compile(
    r"^\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([A-Za-z]+)"
)


def sdf_to_xyz_block(sdf_path: str) -> tuple[str, int]:
    with open(sdf_path) as fh:
        lines = fh.readlines()

    coords: list[str] = []
    in_atom_block = False
    atom_count = 0

    for i, line in enumerate(lines):
        if i == 3:
            try:
                atom_count = int(line[:3].strip())
                in_atom_block = True
            except ValueError:
                pass
            continue
        if in_atom_block:
            if len(coords) >= atom_count:
                break
            m = _ATOM_LINE.match(line)
            if m:
                x, y, z, elem = m.group(1), m.group(2), m.group(3), m.group(4)
                coords.append(
                    f"  {elem:<3} {float(x):>14.8f} {float(y):>14.8f} {float(z):>14.8f}"
                )

    if not coords:
        raise ValueError(f"No atomic coordinates found in {sdf_path}")
    return "\n".join(coords), len(coords)


def read_xyz_file(xyz_file: str) -> str:
    coords: list[str] = []
    with open(xyz_file) as fh:
        for i, line in enumerate(fh):
            if i < 2:
                continue
            parts = line.split()
            if len(parts) == 4:
                elem, x, y, z = parts
                coords.append(
                    f"  {elem:<3} {float(x):>14.8f} {float(y):>14.8f} {float(z):>14.8f}"
                )
    return "\n".join(coords)


# ──────────────────────────────────────────────────────────────────────────────
# Build Gaussian 16 input (.gjf)
# ──────────────────────────────────────────────────────────────────────────────

def build_gjf(
    coord_block:    str,
    chk_path:       str,
    *,
    basis_set:      str           = "6-311G*",
    charge:         int           = 0,
    multiplicity:   int           = 1,
    method:         str           = "PBE0",
    operation:      str           = "opt freq",
    nproc:          int           = 10,
    mem:            str           = "28GB",
    title:          str           = "Gaussian 16 DFT Calculation",
    extra_keywords: str           = "NoTestMO SCF=(XQC,MaxCycle=200)",
    cosmo:          bool          = False,
    # ── Scratch / disk controls (the fix) ────────────────────────────────────
    scratch_dir:    Optional[str] = None,   # if set, explicit %RWF path
    cid:            str           = "mol",  # used to name the .rwf file
    max_disk:       Optional[str] = None,   # e.g. "60GB" → MaxDisk=60GB keyword
) -> str:
    """
    Generate a Gaussian 16 .gjf input string.

    Scratch strategy
    ----------------
    %NoSave is ALWAYS written.  This tells Gaussian to delete the .rwf
    scratch file on normal termination, which is the single most important
    fix for runaway disk usage.

    If scratch_dir is provided we also write an explicit %RWF line so that
    the scratch file goes to a known location (useful if $GAUSS_SCRDIR isn't
    set, or if you want per-job files on a fast NVMe vs your home dir).

    MaxDisk gives Gaussian a hard cap; it will abort cleanly rather than
    filling the disk.  Strongly recommended for WSL where OOM kills leave
    partial .rwf files that never get cleaned up.

    extra_keywords:
        Default is "NoTestMO SCF=(XQC,MaxCycle=200)" to avoid crashes from
        large MO coefficients (linear dependence) and improve SCF convergence.
    """
    solvent_kw  = " scrf=(cpcm,solvent=water)" if cosmo else ""
    extra       = f" {extra_keywords.strip()}" if extra_keywords.strip() else ""
    max_disk_kw = f" MaxDisk={max_disk}" if max_disk else ""

    # Always delete .rwf on normal termination
    nosave_line = "%NoSave\n"

    # Optional explicit .rwf location — useful when scratch_dir != $GAUSS_SCRDIR
    rwf_line = ""
    if scratch_dir:
        rwf_path = str(Path(scratch_dir) / f"{cid}.rwf")
        rwf_line = f"%RWF={rwf_path}\n"

    gjf = (
        f"%chk={chk_path}\n"
        f"{rwf_line}"                    # empty string if not set
        f"{nosave_line}"                 # ALWAYS present
        f"%nprocshared={nproc}\n"
        f"%mem={mem}\n"
        f"#p {method}/{basis_set} {operation}{solvent_kw}{extra}{max_disk_kw}\n"
        f"\n"
        f"{title}\n"
        f"\n"
        f"{charge} {multiplicity}\n"
        f"{coord_block}\n"
        f"\n"
    )
    return gjf


def sdf_to_gjf(
    sdf_path:    str,
    gjf_path:    str,
    chk_path:    str,
    **kwargs,
) -> str:
    coord_block, _ = sdf_to_xyz_block(sdf_path)
    gjf_content = build_gjf(coord_block, chk_path, **kwargs)
    with open(gjf_path, "w") as fh:
        fh.write(gjf_content)
    return gjf_path


def xyz_to_gjf(
    xyz_file:  str,
    gjf_path:  str,
    chk_path:  str,
    **kwargs,
) -> str:
    coord_block = read_xyz_file(xyz_file)
    gjf_content = build_gjf(coord_block, chk_path, **kwargs)
    with open(gjf_path, "w") as fh:
        fh.write(gjf_content)
    return gjf_path


# ──────────────────────────────────────────────────────────────────────────────
# Run Gaussian 16
# ──────────────────────────────────────────────────────────────────────────────

def run_gaussian(
    gjf_path:    str,
    log_path:    str,
    g16_exec:    str           = "g16",
    timeout:     Optional[int] = None,
    scratch_dir: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Run Gaussian 16 on gjf_path, writing output to log_path.

    If scratch_dir is provided, $GAUSS_SCRDIR is set for this subprocess only
    (does not affect other processes or the parent environment).
    """
    env = os.environ.copy()
    if scratch_dir:
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)
        env["GAUSS_SCRDIR"] = scratch_dir

    cmd = [g16_exec, gjf_path, log_path]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if result.returncode != 0:
            return False, result.stderr.strip() or f"Exit code {result.returncode}"
        return True, ""
    except FileNotFoundError:
        return False, f"Gaussian executable '{g16_exec}' not found in PATH"
    except subprocess.TimeoutExpired:
        return False, f"Gaussian timed out after {timeout}s"
    except Exception as exc:
        return False, str(exc)


# ──────────────────────────────────────────────────────────────────────────────
# Parse Gaussian 16 output (.log)
# ──────────────────────────────────────────────────────────────────────────────

def get_final_scf_energy(log_path: str) -> Optional[float]:
    energy = None
    pattern = re.compile(r"SCF Done:\s+E\(\S+\)\s+=\s+([-\d.]+)")
    with open(log_path) as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                energy = float(m.group(1))
    return energy


def get_geometries(
    log_path:       str,
    output_xyz_file: str,
    geometry_index: int = -1,
) -> None:
    with open(log_path) as fh:
        content = fh.read()

    block_re = re.compile(
        r"Standard orientation:.*?-{20,}.*?-{20,}\n(.*?)-{20,}",
        re.DOTALL,
    )
    row_re = re.compile(
        r"^\s+(\d+)\s+(\d+)\s+\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
        re.MULTILINE,
    )

    _Z_TO_SYM = {
        1:'H',2:'He',3:'Li',4:'Be',5:'B',6:'C',7:'N',8:'O',9:'F',10:'Ne',
        11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',18:'Ar',
        19:'K',20:'Ca',21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',26:'Fe',
        27:'Co',28:'Ni',29:'Cu',30:'Zn',31:'Ga',32:'Ge',33:'As',34:'Se',
        35:'Br',36:'Kr',37:'Rb',38:'Sr',39:'Y',40:'Zr',41:'Nb',42:'Mo',
        43:'Tc',44:'Ru',45:'Rh',46:'Pd',47:'Ag',48:'Cd',49:'In',50:'Sn',
        51:'Sb',52:'Te',53:'I',54:'Xe',55:'Cs',56:'Ba',57:'La',72:'Hf',
        73:'Ta',74:'W',75:'Re',76:'Os',77:'Ir',78:'Pt',79:'Au',80:'Hg',
        81:'Tl',82:'Pb',83:'Bi',84:'Po',85:'At',86:'Rn',
    }

    geometries: list[list[dict]] = []
    for block in block_re.finditer(content):
        rows = row_re.findall(block.group(1))
        geom = []
        for row in rows:
            center_no, atomic_no, x, y, z = row
            sym = _Z_TO_SYM.get(int(atomic_no), f"X{atomic_no}")
            geom.append({"Atom_No": int(center_no), "Atom_Tag": sym,
                         "X": float(x), "Y": float(y), "Z": float(z)})
        if geom:
            geometries.append(geom)

    if not geometries:
        print("No Standard orientation blocks found in Gaussian log.")
        return

    if geometry_index < -len(geometries) or geometry_index >= len(geometries):
        print(f"Invalid geometry_index {geometry_index} (have {len(geometries)} geometries).")
        return

    geom = geometries[geometry_index]
    with open(output_xyz_file, "w") as fh:
        fh.write(f"{len(geom)}\n")
        fh.write("Extracted from Gaussian 16 log\n")
        for atom in geom:
            fh.write(f"{atom['Atom_Tag']} {atom['X']} {atom['Y']} {atom['Z']}\n")


def log_to_xyz(log_path: str, geometry_index: int = -1) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz")
    get_geometries(log_path, tmp.name, geometry_index=geometry_index)
    return tmp.name


def get_thermo_data(log_path: str, *, thermo_data: Optional[ThermoData] = None) -> ThermoData:
    if thermo_data is None:
        thermo_data = ThermoData()

    HARTREE_TO_KCAL = 627.509474
    in_thermo_table = False
    cv_row_seen     = False

    with open(log_path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            ll   = line.lower()

            if ll.startswith("temperature") and "kelvin" in ll:
                m = re.search(r"temperature\s+([\d.]+)\s+kelvin", ll)
                if m:
                    thermo_data.temp = float(m.group(1))

            elif "scale factor for frequencies" in ll:
                m = re.search(r"scale factor for frequencies\s+=\s+([\d.]+)", ll)
                if m:
                    thermo_data.freq_scale = float(m.group(1))

            elif ll.startswith("zero-point correction="):
                m = re.search(r"=\s+([-\d.]+)", line)
                if m:
                    thermo_data.zpe = float(m.group(1)) * HARTREE_TO_KCAL

            elif ll.startswith("thermal correction to energy="):
                m = re.search(r"=\s+([-\d.]+)", line)
                if m:
                    thermo_data.te = float(m.group(1)) * HARTREE_TO_KCAL

            elif ll.startswith("thermal correction to enthalpy="):
                m = re.search(r"=\s+([-\d.]+)", line)
                if m:
                    thermo_data.th = float(m.group(1)) * HARTREE_TO_KCAL

            elif "e (thermal)" in ll and "cv" in ll:
                in_thermo_table = True

            elif in_thermo_table:
                if ll.startswith("total"):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            thermo_data.cv = float(parts[2])
                            thermo_data.ts = float(parts[3])
                            cv_row_seen = True
                        except ValueError:
                            pass

                elif ll.startswith("translational"):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            if not cv_row_seen:
                                thermo_data.ts_trans = float(parts[3])
                            else:
                                thermo_data.cv_trans = float(parts[2])
                        except ValueError:
                            pass

                elif ll.startswith("rotational"):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            if not cv_row_seen:
                                thermo_data.ts_rot = float(parts[3])
                            else:
                                thermo_data.cv_rot = float(parts[2])
                        except ValueError:
                            pass

                elif ll.startswith("vibrational"):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            if not cv_row_seen:
                                thermo_data.ts_vib = float(parts[3])
                            else:
                                thermo_data.cv_vib = float(parts[2])
                        except ValueError:
                            pass

                elif line == "":
                    in_thermo_table = False

    return thermo_data


def get_opt_steps(log_path: str, step_index: int = -1) -> Optional[OptStep]:
    steps: list[OptStep] = []
    max_force = rms_force = max_disp = rms_disp = None
    energy    = None
    step_no   = 0

    scf_re   = re.compile(r"SCF Done:\s+E\(\S+\)\s+=\s+([-\d.]+)")
    force_re = re.compile(
        r"(Maximum Force|RMS\s+Force|Maximum Displacement|RMS\s+Displacement)"
        r"\s+([-\d.]+)"
    )

    with open(log_path) as fh:
        for line in fh:
            m = scf_re.search(line)
            if m:
                energy = float(m.group(1))

            fm = force_re.search(line)
            if fm:
                label, val = fm.group(1).strip(), float(fm.group(2))
                if "Maximum Force" in label:
                    max_force = val
                elif "RMS" in label and "Force" in label:
                    rms_force = val
                elif "Maximum Displacement" in label:
                    max_disp = val
                elif "RMS" in label and "Displacement" in label:
                    rms_disp = val

                if all(v is not None for v in [max_force, rms_force, max_disp, rms_disp, energy]):
                    step_no += 1
                    delta_e = energy - steps[-1].energy if steps else 0.0
                    steps.append(OptStep(
                        step=step_no, energy=energy, delta_e=delta_e,
                        rms_force=rms_force, max_force=max_force,
                        rms_disp=rms_disp, max_disp=max_disp,
                    ))
                    max_force = rms_force = max_disp = rms_disp = None

    if not steps:
        return None
    if step_index < -len(steps) or step_index >= len(steps):
        return None
    return steps[step_index]


def check_normal_termination(log_path: str) -> bool:
    try:
        with open(log_path) as fh:
            tail = fh.read()[-4096:]
        return "Normal termination" in tail
    except OSError:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Thermochemistry helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_gibbs(energy_hartree: float, thermo: ThermoData) -> Optional[float]:
    HARTREE_TO_KJMOL = 2625.5
    KCAL_TO_KJ       = 4.184
    CAL_TO_KJ        = 4.184 / 1000.0

    if thermo.th is None or thermo.ts is None or thermo.temp is None:
        return None

    H = energy_hartree * HARTREE_TO_KJMOL + thermo.th * KCAL_TO_KJ
    S = thermo.ts * CAL_TO_KJ
    return H - thermo.temp * S


def redox_potential(
    g_red:     float,
    g_ox:      float,
    g_sol_red: float,
    g_sol_ox:  float,
) -> float:
    g_total = g_red - g_ox + (g_sol_red - g_red) - (g_sol_ox - g_ox)
    return -g_total / 96.5


# ──────────────────────────────────────────────────────────────────────────────
# Single-compound runner
# ──────────────────────────────────────────────────────────────────────────────

def run_compound(
    sdf_path:     str,
    work_dir:     str,
    g16_exec:     str           = "g16",
    basis_set:    str           = "6-311G*",
    method:       str           = "PBE0",
    operation:    str           = "opt freq",
    charge:       int           = 0,
    multiplicity: int           = 1,
    nproc:        Optional[int] = None,
    mem:          str           = "28GB",
    cosmo:        bool          = False,
    timeout:      Optional[int] = None,
    scratch_dir:  Optional[str] = None,
    max_disk:     Optional[str] = None,
) -> RunResult:
    sdf_path = str(sdf_path)
    cid      = Path(sdf_path).stem
    job_dir  = Path(work_dir) / cid
    job_dir.mkdir(parents=True, exist_ok=True)

    gjf_path = str(job_dir / f"{cid}.gjf")
    chk_path = str(job_dir / f"{cid}.chk")
    log_path = str(job_dir / f"{cid}.log")

    effective_nproc = nproc if nproc is not None else os.cpu_count() or 4

    # ── Pre-run: clear any leftover scratch from a previous crashed run ───────
    resolved_scratch = get_scratch_dir(scratch_dir)
    cleanup_scratch_for_cid(resolved_scratch, cid)

    # ── Build .gjf ────────────────────────────────────────────────────────────
    try:
        sdf_to_gjf(
            sdf_path, gjf_path, chk_path,
            basis_set=basis_set, method=method, operation=operation,
            charge=charge, multiplicity=multiplicity,
            nproc=effective_nproc, mem=mem, cosmo=cosmo,
            title=f"CID {cid}",
            scratch_dir=resolved_scratch,
            cid=cid,
            max_disk=max_disk,
        )
    except Exception as exc:
        return RunResult(
            cid=cid, sdf_path=sdf_path, gjf_path=gjf_path,
            log_path=log_path, success=False,
            error_msg=f"GJF build failed: {exc}",
        )

    # ── Run Gaussian ──────────────────────────────────────────────────────────
    ok, err = run_gaussian(
        gjf_path, log_path,
        g16_exec=g16_exec,
        timeout=timeout,
        scratch_dir=resolved_scratch,
    )

    # ── Post-run: if job failed/crashed, clean up any leftover scratch ────────
    if not ok or not check_normal_termination(log_path):
        cleanup_scratch_for_cid(resolved_scratch, cid)
        return RunResult(
            cid=cid, sdf_path=sdf_path, gjf_path=gjf_path,
            log_path=log_path, success=False,
            error_msg=err or "Abnormal termination",
        )

    # %NoSave handles deletion on normal termination, but clean up anyway
    cleanup_scratch_for_cid(resolved_scratch, cid)

    # ── Parse results ─────────────────────────────────────────────────────────
    energy    = get_final_scf_energy(log_path)
    thermo    = get_thermo_data(log_path) if "freq" in operation.lower() else None
    opt_steps = []
    if "opt" in operation.lower():
        step = get_opt_steps(log_path, step_index=-1)
        if step:
            opt_steps.append(step)

    return RunResult(
        cid=cid, sdf_path=sdf_path, gjf_path=gjf_path,
        log_path=log_path, success=True,
        energy=energy, thermo=thermo, opt_steps=opt_steps,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Batch runner
# ──────────────────────────────────────────────────────────────────────────────

def batch_run(
    sdf_dir:     str,
    work_dir:    str,
    jobs:        int           = 1,
    resume:      bool          = True,
    scratch_dir: Optional[str] = None,
    checkpoint:  Optional[str] = None,   # accepted but unused (path derived from work_dir)
    **kwargs,
) -> list[RunResult]:

    sdfs = sorted(Path(sdf_dir).glob("*.sdf"))
    if not sdfs:
        print(f"No SDF files found in {sdf_dir}")
        return []

    Path(work_dir).mkdir(parents=True, exist_ok=True)

    resolved_scratch = get_scratch_dir(scratch_dir)
    report_scratch_usage(resolved_scratch)

    checkpoint = load_checkpoint(work_dir)

    pending = []
    for sdf in sdfs:
        cid    = sdf.stem
        status = checkpoint["jobs"].get(cid, {}).get("status")
        if resume and status == "completed":
            print(f"Skipping {cid} (already completed)")
            continue
        pending.append(sdf)

    print(f"\nGaussian batch:"
          f"\n  Total     : {len(sdfs)}"
          f"\n  Done      : {len(sdfs) - len(pending)}"
          f"\n  Remaining : {len(pending)}"
          f"\n  Scratch   : {resolved_scratch}\n")

    results: list[RunResult] = []

    try:
        for i, sdf in enumerate(pending, 1):
            cid = sdf.stem
            print(f"[{i}/{len(pending)}] {cid}", flush=True)
            update_checkpoint(work_dir, cid, "running")

            try:
                result = run_compound(
                    str(sdf), work_dir,
                    scratch_dir=resolved_scratch,
                    **kwargs,
                )
            except KeyboardInterrupt:
                # Clean scratch before propagating
                cleanup_scratch_for_cid(resolved_scratch, cid)
                update_checkpoint(work_dir, cid, "interrupted")
                raise
            except Exception as exc:
                cleanup_scratch_for_cid(resolved_scratch, cid)
                result = RunResult(
                    cid=cid, sdf_path=str(sdf),
                    gjf_path="", log_path="",
                    success=False, error_msg=str(exc),
                )

            if result is None:
                result = RunResult(
                    cid=cid, sdf_path=str(sdf),
                    gjf_path="", log_path="",
                    success=False, error_msg="run_compound returned None",
                )

            results.append(result)

            if result.success:
                update_checkpoint(work_dir, cid, "completed", {"energy": result.energy})
            else:
                update_checkpoint(work_dir, cid, "failed", {"error": result.error_msg})

            _print_result(result)
            report_scratch_usage(resolved_scratch)   # show scratch state after each job

    except KeyboardInterrupt:
        print("\n\nInterrupted. Checkpoint saved. Run again to resume.")

    finally:
        _write_summary(results, Path(work_dir) / "batch_summary.tsv")

    return results


def _print_result(r: RunResult) -> None:
    if r.success:
        e_str = f"{r.energy:.6f} Ha" if r.energy is not None else "n/a"
        t_str = f"T={r.thermo.temp} K" if r.thermo and r.thermo.temp else ""
        print(f"  ✓  E={e_str}  {t_str}")
    else:
        print(f"  ✗  {r.error_msg}")


def _write_summary(results: list[RunResult], path: Path) -> None:
    import csv
    fields = ["cid", "success", "energy_ha", "temp_K", "zpe_kcal",
              "th_kcal", "ts_cal_K", "error_msg"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in results:
            if r is None:
                continue
            w.writerow({
                "cid":       r.cid,
                "success":   r.success,
                "energy_ha": r.energy if r.energy is not None else "",
                "temp_K":    r.thermo.temp if r.thermo else "",
                "zpe_kcal":  r.thermo.zpe  if r.thermo else "",
                "th_kcal":   r.thermo.th   if r.thermo else "",
                "ts_cal_K":  r.thermo.ts   if r.thermo else "",
                "error_msg": r.error_msg,
            })
    print(f"Summary written to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Gaussian 16 local pipeline (SDF → .gjf → g16 → parse)"
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sdf",      help="Single SDF file")
    src.add_argument("--sdf-dir", help="Directory of SDF files (batch mode)")

    parser.add_argument("--workdir",   default="./g16_runs", help="Working / output directory")
    parser.add_argument("--g16",       default="g16",        help="Gaussian 16 executable")
    parser.add_argument("--basis",     default="6-311G*",    help="Basis set")
    parser.add_argument("--method",    default="PBE0",       help="DFT functional")
    parser.add_argument("--operation", default="opt freq",   help="Gaussian task keywords")
    parser.add_argument("--charge",    default=0,  type=int, help="Molecular charge")
    parser.add_argument("--mult",      default=1,  type=int, help="Spin multiplicity")
    parser.add_argument("--nproc",     default=10, type=int, help="%nprocshared")
    parser.add_argument("--mem",       default="28GB",       help="%mem")
    parser.add_argument("--cosmo",     action="store_true",  help="SCRF=(CPCM,Solvent=Water)")
    parser.add_argument("--jobs",      default=1,  type=int, help="Parallel workers (batch)")
    parser.add_argument("--timeout",   default=None, type=int, help="Per-job timeout (seconds)")

    # ── Scratch controls ──────────────────────────────────────────────────────
    parser.add_argument(
        "--scratch",
        default=None,
        help=(
            "Scratch directory for Gaussian .rwf files. "
            "Overrides $GAUSS_SCRDIR. "
            "Files are deleted after each job (%NoSave + explicit cleanup). "
            "Example: --scratch /fast/nvme/g16scratch"
        ),
    )
    parser.add_argument(
        "--max-disk",
        default=None,
        dest="max_disk",
        help=(
            "Hard disk cap passed as MaxDisk=X to Gaussian. "
            "Gaussian aborts cleanly instead of filling the disk. "
            "Example: --max-disk 60GB"
        ),
    )
    parser.add_argument(
        "--clean-scratch",
        action="store_true",
        dest="clean_scratch",
        help=(
            "Wipe ALL Gaussian scratch files from the scratch directory "
            "before starting. Use after a crash left orphaned .rwf files."
        ),
    )
    parser.add_argument(
        "--extra_keywords",
        default="NoTestMO SCF=(XQC,MaxCycle=200)",
        help=(
            "Extra keywords for Gaussian route line. "
            "Default: NoTestMO SCF=(XQC,MaxCycle=200) "
            "(avoids MO coefficient crash and improves SCF convergence). "
            "Example: --extra_keywords 'NoTestMO Int=UltraFine'"
        ),
    )

    args = parser.parse_args(argv)

    resolved_scratch = get_scratch_dir(args.scratch)

    if args.clean_scratch:
        print(f"Cleaning scratch directory: {resolved_scratch}")
        cleanup_all_scratch(resolved_scratch)

    report_scratch_usage(resolved_scratch)

    kwargs = dict(
        g16_exec=args.g16,
        basis_set=args.basis,
        method=args.method,
        operation=args.operation,
        charge=args.charge,
        multiplicity=args.mult,
        nproc=args.nproc,
        mem=args.mem,
        cosmo=args.cosmo,
        timeout=args.timeout,
        scratch_dir=resolved_scratch,
        max_disk=args.max_disk,
        extra_keywords=args.extra_keywords,
    )

    if args.sdf:
        r = run_compound(args.sdf, args.workdir, **kwargs)
        _print_result(r)
        if r.thermo:
            print(f"\nThermochemistry (CID {r.cid}):")
            for f in dataclasses.fields(r.thermo):
                v = getattr(r.thermo, f.name)
                if v is not None:
                    print(f"  {f.name:12s} = {v}")
        return 0 if r.success else 1

    else:
        results = batch_run(
            args.sdf_dir, args.workdir,
            jobs=args.jobs,
            **kwargs,
        )
        return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
