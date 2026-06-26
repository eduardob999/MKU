#!/usr/bin/env python3
"""
gaussian16_pipeline.py
Local Gaussian 16 pipeline on Linux: SDF → .gjf input → run g16 → parse .log output.

Equivalent to the NWChem pipeline but targeting Gaussian 16.

Usage (single compound):
    python gaussian16_pipeline.py --sdf compound.sdf --workdir ./runs

Usage (batch over a directory of SDFs):
    python gaussian16_pipeline.py --sdf-dir ./sdfs --workdir ./runs --jobs 4

# Single compound, opt + freq at B3LYP/6-31G*
python gaussian16_pipeline.py --sdf 10701.sdf --workdir ./runs

# Whole SDF directory, single-point only, larger basis, with solvent
python gaussian16_pipeline.py \
    --sdf-dir ./sdfs \
    --workdir ./runs \
    --operation "opt freq" \
    --basis "6-311+G(d,p)" \
    --method "M062X" \
    --cosmo \
    --nproc 8 \
    --mem 16GB

# If g16 isn't on PATH
python gaussian16_pipeline.py --sdf 10701.sdf --g16 /opt/gaussian/g16/g16
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def get_scratch_dir(scratch_dir: Optional[str] = None) -> Optional[str]:
    """Resolve the Gaussian scratch directory.

    Prefers an explicit ``scratch_dir``, then ``$GAUSS_SCRDIR``; returns ``None``
    to let Gaussian use its own default when neither is set.
    """
    return scratch_dir or os.environ.get("GAUSS_SCRDIR") or None


# ──────────────────────────────────────────────────────────────────────────────
# Data containers  (mirrors the NWChem ThermoData / Step types)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ThermoData:
    temp:       Optional[float] = None   # K
    freq_scale: Optional[float] = None
    zpe:        Optional[float] = None   # kcal/mol  (Zero-point correction)
    te:         Optional[float] = None   # kcal/mol  (Thermal correction to energy)
    th:         Optional[float] = None   # kcal/mol  (Thermal correction to enthalpy)
    ts:         Optional[float] = None   # cal/mol·K (Total entropy)
    ts_trans:   Optional[float] = None
    ts_rot:     Optional[float] = None
    ts_vib:     Optional[float] = None
    cv:         Optional[float] = None   # cal/mol·K (Cv)
    cv_trans:   Optional[float] = None
    cv_rot:     Optional[float] = None
    cv_vib:     Optional[float] = None


@dataclass
class OptStep:
    step:     int
    energy:   float   # Hartree
    delta_e:  float
    rms_force: float
    max_force: float
    rms_disp: float
    max_disp: float


@dataclass
class RunResult:
    cid:        str
    sdf_path:   str
    gjf_path:   str
    log_path:   str
    success:    bool
    energy:     Optional[float]       = None   # Hartree
    thermo:     Optional[ThermoData]  = None
    opt_steps:  list[OptStep]         = field(default_factory=list)
    error_msg:  str                   = ""


# ──────────────────────────────────────────────────────────────────────────────
# SDF → XYZ coordinates (element + x, y, z only, no charge column)
# ──────────────────────────────────────────────────────────────────────────────

_ATOM_LINE = re.compile(
    r"^\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([A-Za-z]+)"
)

def sdf_to_xyz_block(sdf_path: str) -> tuple[str, int]:
    """
    Extract the atom-coordinate block from an SDF / MOL file.

    Returns
    -------
    xyz_block : str
        Newline-joined "  Element   x   y   z" lines (Gaussian molecule-spec format).
    n_atoms : int
        Number of atoms found.
    """
    with open(sdf_path) as fh:
        lines = fh.readlines()

    # Counts line is line index 3 in a V2000 MOL block: aaabbblllfffcccsssxxxrrrpppiiimmmvvvvvv
    # Robust approach: just scan for coordinate lines after the counts line.
    coords: list[str] = []
    in_atom_block = False
    atom_count = 0

    for i, line in enumerate(lines):
        if i == 3:
            # V2000 counts line: first 3 chars = num atoms
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
                coords.append(f"  {elem:<3} {float(x):>14.8f} {float(y):>14.8f} {float(z):>14.8f}")

    if not coords:
        raise ValueError(f"No atomic coordinates found in {sdf_path}")

    return "\n".join(coords), len(coords)


def read_xyz_file(xyz_file: str) -> str:
    """Read coordinates from a standard .xyz file (skip first two header lines)."""
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
    coord_block: str,
    chk_path: str,
    *,
    basis_set:    str  = "6-31G*",
    charge:       int  = 0,
    multiplicity: int  = 1,
    method:       str  = "B3LYP",
    operation:    str  = "opt freq",   # e.g. "sp", "opt", "opt freq"
    nproc:        int  = 4,
    mem:          str  = "4GB",
    title:        str  = "Gaussian 16 DFT Calculation",
    extra_keywords: str = "",          # e.g. "empiricaldispersion=gd3bj"
    cosmo:        bool = False,        # SCRF=(CPCM,Solvent=Water)
) -> str:
    """
    Generate a Gaussian 16 .gjf input string.

    Gaussian keyword line structure:
        #p method/basis operation [options]

    Parameters map to NWChem equivalents:
        method      ← method + functional  (NWChem splits these; Gaussian combines)
        operation   ← task keyword         (opt / freq / opt freq / sp)
        cosmo       ← cosmo block          (uses CPCM in Gaussian)
    """
    solvent_kw = " scrf=(cpcm,solvent=water)" if cosmo else ""
    extra = f" {extra_keywords.strip()}" if extra_keywords.strip() else ""

    gjf = (
        f"%chk={chk_path}\n"
        f"%nprocshared={nproc}\n"
        f"%mem={mem}\n"
        f"#p {method}/{basis_set} {operation}{solvent_kw}{extra}\n"
        f"\n"
        f"{title}\n"
        f"\n"
        f"{charge} {multiplicity}\n"
        f"{coord_block}\n"
        f"\n"
    )
    return gjf


def sdf_to_gjf(
    sdf_path: str,
    gjf_path: str,
    chk_path: str,
    **kwargs,
) -> str:
    """Convert an SDF file to a Gaussian .gjf input file. Returns the gjf path."""
    coord_block, _ = sdf_to_xyz_block(sdf_path)
    gjf_content = build_gjf(coord_block, chk_path, **kwargs)
    with open(gjf_path, "w") as fh:
        fh.write(gjf_content)
    return gjf_path


def xyz_to_gjf(
    xyz_file: str,
    gjf_path: str,
    chk_path: str,
    **kwargs,
) -> str:
    """Convert an .xyz file to a Gaussian .gjf input file. Returns the gjf path."""
    coord_block = read_xyz_file(xyz_file)
    gjf_content = build_gjf(coord_block, chk_path, **kwargs)
    with open(gjf_path, "w") as fh:
        fh.write(gjf_content)
    return gjf_path


# ──────────────────────────────────────────────────────────────────────────────
# Run Gaussian 16
# ──────────────────────────────────────────────────────────────────────────────

def run_gaussian(
    gjf_path: str,
    log_path: str,
    g16_exec: str = "g16",
    timeout:  Optional[int] = None,   # seconds; None = no limit
    cwd:      Optional[str] = None,
) -> tuple[bool, str]:
    """
    Run Gaussian 16 on *gjf_path*, writing output to *log_path*.

    Returns (success: bool, stderr_or_error: str).

    Gaussian writes stray files (the ``fort.7`` punch file, ``Gau-*`` temporaries)
    into its working directory. We default *cwd* to the directory holding the
    output log so those land beside the job's own files instead of polluting
    wherever the CLI was launched from (the repository root).
    """
    cmd = [g16_exec, gjf_path, log_path]
    run_cwd = cwd or os.path.dirname(os.path.abspath(log_path)) or None
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=run_cwd,
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
    """
    Extract the last SCF Done energy (Hartree) from a Gaussian log file.

    Gaussian line format:
        SCF Done:  E(RB3LYP) =  -154.062345678     A.U. after   12 cycles
    Equivalent to NWChem's:
        Total DFT energy = ...
    """
    energy = None
    pattern = re.compile(r"SCF Done:\s+E\(\S+\)\s+=\s+([-\d.]+)")
    with open(log_path) as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                energy = float(m.group(1))
    return energy


def get_geometries(
    log_path: str,
    output_xyz_file: str,
    geometry_index: int = -1,
) -> None:
    """
    Extract optimised geometries from a Gaussian log and write to an XYZ file.

    Gaussian prints geometry blocks under:
        "Standard orientation:"   (preferred — atom-centred frame)
    or  "Input orientation:"      (fallback)

    Equivalent to NWChem's "Output coordinates in angstroms" block.
    """
    with open(log_path) as fh:
        content = fh.read()

    # Each Standard orientation block ends at a dashed separator line
    block_re = re.compile(
        r"Standard orientation:.*?-{20,}.*?-{20,}\n(.*?)-{20,}",
        re.DOTALL,
    )
    # Row format: center_no  atomic_no  atomic_type  x  y  z
    row_re = re.compile(
        r"^\s+(\d+)\s+(\d+)\s+\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
        re.MULTILINE,
    )

    # Map atomic number → element symbol for the 118 common elements
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
    """
    Write the selected geometry to a temp XYZ file and return its path.
    Equivalent to NWChem's nwout_to_xyz().
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xyz")
    get_geometries(log_path, tmp.name, geometry_index=geometry_index)
    return tmp.name


def get_thermo_data(log_path: str, *, thermo_data: Optional[ThermoData] = None) -> ThermoData:
    """
    Parse thermochemical data from a Gaussian 16 frequency log.

    Gaussian thermo block (298.15 K, 1 atm) looks like:

        Zero-point correction=           0.123456 (Hartree/Particle)
        Thermal correction to Energy=    0.131234
        Thermal correction to Enthalpy=  0.132178
        Thermal correction to Gibbs Free Energy= 0.098765
        Sum of electronic and zero-point Energies= -154.123456
        ...
        Zero-point Energy=          77.5 kcal/mol
        ...
        Temperature   298.150 Kelvin.  Pressure   1.00000 Atm.
        ...
        E (Thermal)   CV            S
        KCal/Mol    Cal/Mol-Kelvin  Cal/Mol-Kelvin
        Total    XX.XXX    XX.XXX    XXX.XXX
        Electronic   0.000     0.000     0.000
        Translational  X.XXX    2.981    XX.XXX
        Rotational   X.XXX    2.981    XX.XXX
        Vibrational  X.XXX   XX.XXX    XX.XXX

    Units kept identical to the NWChem equivalents for drop-in compatibility.
    """
    if thermo_data is None:
        thermo_data = ThermoData()

    # Gaussian prints corrections in Hartree; we convert to kcal/mol (×627.509)
    # to match NWChem ThermoData units.
    HARTREE_TO_KCAL = 627.509474

    in_thermo_table = False   # True once we pass the E/CV/S header
    cv_row_seen = False       # True once we've read the "Total" CV row

    with open(log_path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            ll   = line.lower()

            # ── Temperature ───────────────────────────────────────────────
            if ll.startswith("temperature") and "kelvin" in ll:
                m = re.search(r"temperature\s+([\d.]+)\s+kelvin", ll)
                if m:
                    thermo_data.temp = float(m.group(1))

            # ── Frequency scale factor ────────────────────────────────────
            elif "scale factor for frequencies" in ll:
                m = re.search(r"scale factor for frequencies\s+=\s+([\d.]+)", ll)
                if m:
                    thermo_data.freq_scale = float(m.group(1))

            # ── ZPE (Hartree → kcal/mol) ──────────────────────────────────
            elif ll.startswith("zero-point correction="):
                m = re.search(r"=\s+([-\d.]+)", line)
                if m:
                    thermo_data.zpe = float(m.group(1)) * HARTREE_TO_KCAL

            # ── Thermal correction to Energy ──────────────────────────────
            elif ll.startswith("thermal correction to energy="):
                m = re.search(r"=\s+([-\d.]+)", line)
                if m:
                    thermo_data.te = float(m.group(1)) * HARTREE_TO_KCAL

            # ── Thermal correction to Enthalpy ────────────────────────────
            elif ll.startswith("thermal correction to enthalpy="):
                m = re.search(r"=\s+([-\d.]+)", line)
                if m:
                    thermo_data.th = float(m.group(1)) * HARTREE_TO_KCAL

            # ── Enter the E / Cv / S table ────────────────────────────────
            elif "e (thermal)" in ll and "cv" in ll:
                in_thermo_table = True

            elif in_thermo_table:
                # "Total" row gives S and Cv
                if ll.startswith("total"):
                    parts = line.split()
                    # Total  E_therm  Cv  S
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

                # Leave the table on a blank line
                elif line == "":
                    in_thermo_table = False

    return thermo_data


def get_opt_steps(log_path: str, step_index: int = -1) -> Optional[OptStep]:
    """
    Parse geometry optimisation convergence steps from a Gaussian log.

    Gaussian prints a summary table for each macro-iteration:

        Item           Value     Threshold  Converged?
        Maximum Force  0.000456  0.000450     NO
        RMS     Force  0.000123  0.000300    YES
        Maximum Displacement  0.001234  0.001800    YES
        RMS     Displacement  0.000456  0.001200    YES

    and the energy appears on the preceding "SCF Done" line.

    Equivalent to NWChem's get_step_data() (which reads @-prefixed lines).
    """
    steps: list[OptStep] = []

    max_force = rms_force = max_disp = rms_disp = None
    energy = None
    step_no = 0

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

                # All four items collected → record the step
                if all(v is not None for v in [max_force, rms_force, max_disp, rms_disp, energy]):
                    step_no += 1
                    delta_e = (
                        energy - steps[-1].energy if steps else 0.0
                    )
                    steps.append(OptStep(
                        step=step_no,
                        energy=energy,
                        delta_e=delta_e,
                        rms_force=rms_force,
                        max_force=max_force,
                        rms_disp=rms_disp,
                        max_disp=max_disp,
                    ))
                    max_force = rms_force = max_disp = rms_disp = None

    if not steps:
        return None
    if step_index < -len(steps) or step_index >= len(steps):
        return None
    return steps[step_index]


def check_normal_termination(log_path: str) -> bool:
    """Return True if Gaussian ended with 'Normal termination'."""
    try:
        with open(log_path) as fh:
            # Only need to check the last few lines
            tail = fh.read()[-4096:]
        return "Normal termination" in tail
    except OSError:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Thermochemistry helpers  (same API as the NWChem get_g / redox_potential)
# ──────────────────────────────────────────────────────────────────────────────

def compute_gibbs(energy_hartree: float, thermo: ThermoData) -> Optional[float]:
    """
    Compute G = H - T·S (kJ/mol) from Gaussian output data.

    energy_hartree : SCF Done energy in Hartree
    thermo.th      : thermal correction to enthalpy (kcal/mol, Hartree-based)
    thermo.ts      : total entropy (cal/mol·K)
    thermo.temp    : temperature (K)

    Returns G in kJ/mol, consistent with the NWChem pipeline's get_g().
    """
    HARTREE_TO_KJMOL = 2625.5
    KCAL_TO_KJ       = 4.184
    CAL_TO_KJ        = 4.184 / 1000.0

    if thermo.th is None or thermo.ts is None or thermo.temp is None:
        return None

    # Enthalpy: H = E + H_corr  (everything in kJ/mol)
    H = energy_hartree * HARTREE_TO_KJMOL + thermo.th * KCAL_TO_KJ
    S = thermo.ts * CAL_TO_KJ   # kJ/mol·K
    G = H - thermo.temp * S
    return G


def redox_potential(
    g_red:     float,   # G of reduced species (gas phase)
    g_ox:      float,   # G of oxidised species (gas phase)
    g_sol_red: float,   # G of reduced species (solution)
    g_sol_ox:  float,   # G of oxidised species (solution)
) -> float:
    """
    Compute the redox potential (V) from four Gibbs free energies (kJ/mol).
    Matches the NWChem redox_potential() formula exactly.
    """
    g_total = g_red - g_ox + (g_sol_red - g_red) - (g_sol_ox - g_ox)
    e_total = -g_total / 96.5   # 96.5 kJ/mol per eV ≈ Faraday constant
    return e_total


# ──────────────────────────────────────────────────────────────────────────────
# Single-compound runner
# ──────────────────────────────────────────────────────────────────────────────

def run_compound(
    sdf_path:    str,
    work_dir:    str,
    g16_exec:    str   = "g16",
    basis_set:   str   = "6-31G*",
    method:      str   = "B3LYP",
    operation:   str   = "opt freq",
    preopt_mode: str   = "none",
    preopt_basis_set: str = "6-31G*",
    charge:      int   = 0,
    multiplicity: int  = 1,
    nproc:       int   = 4,
    mem:         str   = "4GB",
    cosmo:       bool  = False,
    timeout:     Optional[int] = None,
) -> RunResult:
    """
    Full pipeline for one compound:
      SDF → .gjf → g16 → parse .log → RunResult
    """
    sdf_path = str(sdf_path)
    cid      = Path(sdf_path).stem
    work_dir = Path(work_dir) / cid
    work_dir.mkdir(parents=True, exist_ok=True)

    preopt_mode_lc = preopt_mode.lower().strip()
    if preopt_mode_lc != "none":
        preopt_root = work_dir / "preopt"
        preopt_root.mkdir(parents=True, exist_ok=True)
        if preopt_mode_lc == "pm7":
            from ivette.module import gaussian16_pipeline as gp

            optimized_sdf = gp.gaussian_semiempirical_preopt(
                sdf_path,
                preopt_root / "pm7",
                cid,
                method="pm7",
                g16_exec=g16_exec,
                nproc=nproc,
                mem=mem,
                charge=charge,
                multiplicity=multiplicity,
                solvent=None,
                cid=cid,
            )
            sdf_path = optimized_sdf
        elif preopt_mode_lc in {"gaussian631g", "631g", "6-31g", "6-31g*"}:
            preopt_dir = preopt_root / "gaussian631g"
            preopt_dir.mkdir(parents=True, exist_ok=True)
            preopt_gjf = str(preopt_dir / f"{cid}_preopt.gjf")
            preopt_chk = str(preopt_dir / f"{cid}_preopt.chk")
            preopt_log = str(preopt_dir / f"{cid}_preopt.log")
            preopt_out = str(preopt_dir / f"{cid}_preopt.sdf")
            try:
                sdf_to_gjf(
                    sdf_path, preopt_gjf, preopt_chk,
                    basis_set=preopt_basis_set,
                    method=method,
                    operation="opt",
                    charge=charge,
                    multiplicity=multiplicity,
                    nproc=nproc,
                    mem=mem,
                    cosmo=cosmo,
                    title=f"CID {cid} (preopt)",
                )
            except Exception as exc:
                return RunResult(
                    cid=cid, sdf_path=sdf_path, gjf_path=preopt_gjf,
                    log_path=preopt_log, success=False,
                    error_msg=f"Preopt GJF build failed: {exc}",
                )

            ok, err = run_gaussian(preopt_gjf, preopt_log, g16_exec=g16_exec, timeout=timeout)
            if not ok or not check_normal_termination(preopt_log):
                return RunResult(
                    cid=cid, sdf_path=sdf_path, gjf_path=preopt_gjf,
                    log_path=preopt_log, success=False,
                    error_msg=err or "Preopt stage aborted",
                )

            xyz_path = log_to_xyz(preopt_log)
            try:
                from ivette.module import gaussian16_pipeline as gp

                if not gp._xyz_to_sdf(xyz_path, preopt_out, template_sdf=sdf_path):
                    return RunResult(
                        cid=cid, sdf_path=sdf_path, gjf_path=preopt_gjf,
                        log_path=preopt_log, success=False,
                        error_msg="Preopt xyz→sdf conversion failed",
                    )
                sdf_path = preopt_out
            finally:
                Path(xyz_path).unlink(missing_ok=True)

    gjf_path = str(work_dir / f"{cid}.gjf")
    chk_path = str(work_dir / f"{cid}.chk")
    log_path = str(work_dir / f"{cid}.log")
    operation_lc = operation.lower()
    split_opt_freq = bool(re.search(r"\bopt\b", operation_lc)) and bool(re.search(r"\bfreq\b", operation_lc))

    if split_opt_freq:
        opt_gjf_path = str(work_dir / f"{cid}_opt.gjf")
        opt_chk_path = str(work_dir / f"{cid}_opt.chk")
        opt_log_path = str(work_dir / f"{cid}_opt.log")
        freq_gjf_path = str(work_dir / f"{cid}_freq.gjf")
        freq_chk_path = str(work_dir / f"{cid}_freq.chk")
        freq_log_path = str(work_dir / f"{cid}_freq.log")

        # Run the optimisation first, then feed its final geometry into a
        # separate frequency job so the two stages remain independently
        # inspectable on disk.
        try:
            sdf_to_gjf(
                sdf_path, opt_gjf_path, opt_chk_path,
                basis_set=basis_set, method=method, operation="opt",
                charge=charge, multiplicity=multiplicity,
                nproc=nproc, mem=mem, cosmo=cosmo,
                title=f"CID {cid} (opt)",
            )
        except Exception as exc:
            return RunResult(
                cid=cid, sdf_path=sdf_path, gjf_path=opt_gjf_path,
                log_path=opt_log_path, success=False,
                error_msg=f"GJF build failed: {exc}",
            )

        ok, err = run_gaussian(opt_gjf_path, opt_log_path, g16_exec=g16_exec, timeout=timeout)
        if not ok or not check_normal_termination(opt_log_path):
            return RunResult(
                cid=cid, sdf_path=sdf_path, gjf_path=opt_gjf_path,
                log_path=opt_log_path, success=False,
                error_msg=err or "Opt stage aborted",
            )

        opt_step = get_opt_steps(opt_log_path, step_index=-1)
        opt_steps = [opt_step] if opt_step else []
        temp_xyz = log_to_xyz(opt_log_path)
        try:
            xyz_to_gjf(
                temp_xyz, freq_gjf_path, freq_chk_path,
                basis_set=basis_set, method=method, operation="freq",
                charge=charge, multiplicity=multiplicity,
                nproc=nproc, mem=mem, cosmo=cosmo,
                title=f"CID {cid} (freq)",
            )
        except Exception as exc:
            Path(temp_xyz).unlink(missing_ok=True)
            return RunResult(
                cid=cid, sdf_path=sdf_path, gjf_path=freq_gjf_path,
                log_path=freq_log_path, success=False,
                error_msg=f"Frequency GJF build failed: {exc}",
            )
        finally:
            Path(temp_xyz).unlink(missing_ok=True)

        ok, err = run_gaussian(freq_gjf_path, freq_log_path, g16_exec=g16_exec, timeout=timeout)
        if not ok or not check_normal_termination(freq_log_path):
            return RunResult(
                cid=cid, sdf_path=sdf_path, gjf_path=freq_gjf_path,
                log_path=freq_log_path, success=False,
                error_msg=err or "Frequency stage aborted",
            )

        energy = get_final_scf_energy(freq_log_path)
        thermo = get_thermo_data(freq_log_path)

        return RunResult(
            cid=cid, sdf_path=sdf_path, gjf_path=freq_gjf_path,
            log_path=freq_log_path, success=True,
            energy=energy, thermo=thermo, opt_steps=opt_steps,
        )

    # ── Build input ───────────────────────────────────────────────────────────
    try:
        sdf_to_gjf(
            sdf_path, gjf_path, chk_path,
            basis_set=basis_set, method=method, operation=operation,
            charge=charge, multiplicity=multiplicity,
            nproc=nproc, mem=mem, cosmo=cosmo,
            title=f"CID {cid}",
        )
    except Exception as exc:
        return RunResult(
            cid=cid, sdf_path=sdf_path, gjf_path=gjf_path,
            log_path=log_path, success=False, error_msg=f"GJF build failed: {exc}",
        )

    # ── Run Gaussian ──────────────────────────────────────────────────────────
    ok, err = run_gaussian(gjf_path, log_path, g16_exec=g16_exec, timeout=timeout)
    if not ok or not check_normal_termination(log_path):
        return RunResult(
            cid=cid, sdf_path=sdf_path, gjf_path=gjf_path,
            log_path=log_path, success=False,
            error_msg=err or "Abnormal termination",
        )

    # ── Parse output ──────────────────────────────────────────────────────────
    energy    = get_final_scf_energy(log_path)
    thermo    = get_thermo_data(log_path) if "freq" in operation_lc else None
    opt_steps = []
    if "opt" in operation_lc:
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
    sdf_dir:  str,
    work_dir: str,
    jobs:     int  = 1,
    **kwargs,             # forwarded to run_compound
) -> list[RunResult]:
    """
    Run Gaussian on all SDF files in *sdf_dir*, optionally in parallel.

    NOTE: Gaussian itself may already use all cores via %nprocshared.
    Set jobs=1 (default) unless you have separate CPU allocations per job,
    otherwise CPU over-subscription will slow things down.
    """
    sdfs = sorted(Path(sdf_dir).glob("*.sdf"))
    if not sdfs:
        print(f"No SDF files found in {sdf_dir}")
        return []

    print(f"Found {len(sdfs)} SDF files. Submitting with {jobs} worker(s).")
    results: list[RunResult] = []

    if jobs <= 1:
        for i, sdf in enumerate(sdfs, 1):
            print(f"[{i}/{len(sdfs)}] {sdf.name}", flush=True)
            r = run_compound(str(sdf), work_dir, **kwargs)
            _print_result(r)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(run_compound, str(sdf), work_dir, **kwargs): sdf
                for sdf in sdfs
            }
            for i, fut in enumerate(as_completed(futures), 1):
                sdf = futures[fut]
                try:
                    r = fut.result()
                except Exception as exc:
                    r = RunResult(
                        cid=sdf.stem, sdf_path=str(sdf),
                        gjf_path="", log_path="",
                        success=False, error_msg=str(exc),
                    )
                print(f"[{i}/{len(sdfs)}] {sdf.name}", flush=True)
                _print_result(r)
                results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    ok   = sum(1 for r in results if r.success)
    fail = len(results) - ok
    print(f"\nBatch complete: {ok} succeeded, {fail} failed.")
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
            w.writerow({
                "cid":        r.cid,
                "success":    r.success,
                "energy_ha":  r.energy if r.energy is not None else "",
                "temp_K":     r.thermo.temp  if r.thermo else "",
                "zpe_kcal":   r.thermo.zpe   if r.thermo else "",
                "th_kcal":    r.thermo.th    if r.thermo else "",
                "ts_cal_K":   r.thermo.ts    if r.thermo else "",
                "error_msg":  r.error_msg,
            })
    print(f"Summary written to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Gaussian 16 local pipeline (SDF → .gjf → g16 → parse)")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sdf",     help="Single SDF file")
    src.add_argument("--sdf-dir", help="Directory of SDF files (batch mode)")

    parser.add_argument("--workdir",      default="./g16_runs",   help="Working / output directory")
    parser.add_argument("--g16",          default="g16",           help="Gaussian 16 executable name or path")
    parser.add_argument("--basis",        default="6-31G*",        help="Basis set (default: 6-31G*)")
    parser.add_argument("--method",       default="B3LYP",         help="DFT functional (default: B3LYP)")
    parser.add_argument("--operation",    default="opt freq",      help="Gaussian task keywords (default: 'opt freq')")
    parser.add_argument("--charge",       default=0,  type=int,    help="Molecular charge (default: 0)")
    parser.add_argument("--mult",         default=1,  type=int,    help="Spin multiplicity (default: 1)")
    parser.add_argument("--nproc",        default=4,  type=int,    help="%nprocshared (default: 4)")
    parser.add_argument("--mem",          default="4GB",           help="%%mem (default: 4GB)")
    parser.add_argument("--cosmo",        action="store_true",     help="Enable SCRF=(CPCM,Solvent=Water)")
    parser.add_argument("--jobs",         default=1,  type=int,    help="Parallel workers for batch (default: 1)")
    parser.add_argument("--timeout",      default=None, type=int,  help="Per-job timeout in seconds")

    args = parser.parse_args(argv)

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
        results = batch_run(args.sdf_dir, args.workdir, jobs=args.jobs, **kwargs)
        return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())