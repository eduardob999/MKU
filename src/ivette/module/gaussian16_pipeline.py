#!/usr/bin/env python3
"""
crest_gaussian_pipeline.py
──────────────────────────
Conformational search with CREST/xTB followed by Gaussian 16 DFT
optimisation + frequency calculation on the lowest-energy conformers.

Workflow
────────
  1. Flexibility check
       RDKit counts rotatable bonds.  Rigid molecules (< --flex-threshold
       rotatable bonds, default 3) skip CREST and go straight to Gaussian.

  2. GFN2-xTB pre-optimisation  (xTB)
       The SDF geometry is pre-optimised at the GFN2 level before CREST
       so the conformational search starts from a sensible structure.

  3. Conformational search  (CREST)
       CREST runs iMTD-GC (default) on the xTB-optimised geometry.
       Produces a ranked ensemble in crest_conformers.xyz.

  4. Conformer selection
       The N lowest-energy conformers within --energy-window kcal/mol of
       the global minimum are selected (default: top 5, window 3 kcal/mol).

  5. Semi-empirical pre-optimisation inside Gaussian
       Each selected conformer is pre-optimised at a cheap level of theory
       inside Gaussian before the expensive DFT run.  This moves the
       geometry close to the DFT minimum so the DFT optimiser converges
       in fewer cycles and is less likely to get stuck in a saddle point.

       Two methods are supported (--preopt-method):

         pm7   (default)
               Runs "#p PM7 opt" entirely inside Gaussian.  No extra binary
               needed.  PM7 is a good general-purpose semi-empirical
               Hamiltonian for organic/drug-like molecules.

         xtb
               Runs GFN2-xTB as a Gaussian external program via the
               xtb_external interface ("external='xtb --oniom ...' opt").
               Requires xTB on PATH and the xtb_external wrapper (bundled
               with recent xTB distributions as scripts/xtb_external).
               More accurate than PM7 for systems with heavy atoms,
               non-covalent interactions, or unusual bonding.
               Gaussian drives the optimiser; xTB computes energies
               and gradients each step via the external= interface.

       Failures degrade gracefully: if the preopt log does not contain
       "Normal termination", the CREST geometry is used unchanged and a
       warning is printed (the DFT job still runs).

       Skip with --skip-preopt.

  6. Gaussian DFT opt+freq
       Each selected conformer is submitted to gaussian16_pipeline.run_compound
       (which already splits opt and freq into separate jobs to avoid the
       PBE0-DH bug in G16 Rev C.02).

  7. Results
       A TSV summary ranks all conformers by DFT energy and reports
       thermochemistry for each.  The global-minimum conformer is identified.

Dependencies
────────────
  External binaries (must be on PATH or passed via --xtb / --crest):
    xtb   ≥ 6.5   https://github.com/grimme-lab/xtb
    crest ≥ 3.0   https://github.com/grimme-lab/crest

  Python packages:
    rdkit   (conda: conda install -c conda-forge rdkit)
    gaussian16_pipeline.py  (must be importable — put it in the same
                             directory or on PYTHONPATH)

Usage
─────
  # Single SDF, defaults (PBE0/6-311G*, top 5 conformers, 3 kcal/mol window)
  python crest_gaussian_pipeline.py --sdf molecule.sdf --workdir ./runs

  # Custom settings
  python crest_gaussian_pipeline.py \\
      --sdf molecule.sdf \\
      --workdir ./runs \\
      --method M062X --basis "6-311+G(d,p)" \\
      --n-conformers 3 \\
      --energy-window 2.0 \\
      --flex-threshold 5 \\
      --nproc 10 --mem 28GB \\
      --scratch /fast/scratch \\
      --max-disk 200GB \\
      --solvent water

  # Force CREST even for rigid molecules
  python crest_gaussian_pipeline.py --sdf molecule.sdf --force-crest

  # Skip CREST (go straight to Gaussian, same as gaussian16_pipeline.py)
  python crest_gaussian_pipeline.py --sdf molecule.sdf --skip-crest

  # Use xTB (via Gaussian external=) instead of PM7 for semi-empirical preopt
  python crest_gaussian_pipeline.py --sdf molecule.sdf --preopt-method xtb

  # Skip the semi-empirical preopt entirely (go CREST → DFT directly)
  python crest_gaussian_pipeline.py --sdf molecule.sdf --skip-preopt
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── the full Gaussian pipeline primitives (SDF→gjf→g16→parse) ────────────────
from concurrent.futures import ProcessPoolExecutor, as_completed

from ivette.module import gaussian16_core as g16
from ivette.util import jsonstore

# ── RDKit ─────────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConformerResult:
    """Result for a single conformer after Gaussian DFT."""
    rank:         int                          # 1-based rank by CREST energy
    crest_energy: Optional[float]              # kcal/mol relative to minimum
    gjf_path:     str
    log_path:     str
    success:      bool
    dft_energy:   Optional[float] = None       # Hartree (absolute SCF energy)
    thermo:       Optional[g16.ThermoData] = None
    error_msg:    str = ""


@dataclass
class PipelineResult:
    cid:              str
    sdf_path:         str
    flexible:         bool
    n_rot_bonds:      int
    crest_ran:        bool
    n_conformers_found: int
    n_conformers_calc:  int
    conformers:       list[ConformerResult] = field(default_factory=list)
    best_conformer:   Optional[ConformerResult] = None   # lowest DFT energy
    error_msg:        str = ""
    success:          bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Flexibility check
# ──────────────────────────────────────────────────────────────────────────────

def count_rotatable_bonds(sdf_path: str) -> int:
    """
    Return the number of rotatable bonds in the molecule.
    Uses RDKit's definition: non-ring, non-terminal single bonds,
    excluding bonds to hydrogen.
    Falls back to 0 (treat as rigid) if RDKit is unavailable.
    """
    if not RDKIT_AVAILABLE:
        print("  [flex] RDKit not available — assuming molecule is rigid.")
        return 0

    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None:
        print(f"  [flex] RDKit could not parse {sdf_path} — assuming rigid.")
        return 0

    n = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return n


def is_flexible(sdf_path: str, threshold: int) -> tuple[bool, int]:
    """Return (flexible, n_rotatable_bonds)."""
    n = count_rotatable_bonds(sdf_path)
    return n >= threshold, n


# ──────────────────────────────────────────────────────────────────────────────
# xTB pre-optimisation
# ──────────────────────────────────────────────────────────────────────────────

def xtb_preopt(
    sdf_path:    str,
    work_dir:    Path,
    xtb_exec:    str = "xtb",
    nproc:       int = 1,
    charge:      int = 0,
    multiplicity: int = 1,
    solvent:     Optional[str] = None,
) -> Optional[str]:
    """
    Run GFN2-xTB geometry optimisation on sdf_path.

    Returns the path to the optimised SDF file, or None on failure.

    xTB is run in a temporary subdirectory to keep its output files
    (xtbopt.log, charges, wbo, …) separate from everything else.
    The optimised geometry is converted back to SDF via RDKit so the
    downstream code can read it with the same parser.
    """
    xtb_dir = work_dir / "xtb_preopt"
    xtb_dir.mkdir(parents=True, exist_ok=True)

    # xTB wants an xyz or coord input — convert from SDF
    xyz_in = xtb_dir / "input.xyz"
    _sdf_to_xyz(sdf_path, str(xyz_in))

    uhf = multiplicity - 1   # number of unpaired electrons

    cmd = [
        xtb_exec, str(xyz_in),
        "--opt", "tight",
        "--gfn", "2",
        "--chrg", str(charge),
        "--uhf", str(uhf),
        "--parallel", str(nproc),
    ]
    if solvent:
        cmd += ["--alpb", solvent]

    log_path = xtb_dir / "xtb_preopt.log"
    print(f"  [xTB ] pre-optimising with GFN2-xTB ({nproc} threads) …")
    try:
        with open(log_path, "w") as lf:
            r = subprocess.run(
                cmd,
                cwd=str(xtb_dir),
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )
    except FileNotFoundError:
        print(f"  [xTB ] ERROR: '{xtb_exec}' not found. "
              f"Is xTB installed and on PATH?")
        return None
    except subprocess.TimeoutExpired:
        print("  [xTB ] ERROR: xTB pre-optimisation timed out after 1 h.")
        return None

    # xTB writes the optimised geometry to xtbopt.xyz in the working dir
    xtbopt_xyz = xtb_dir / "xtbopt.xyz"
    if not xtbopt_xyz.exists():
        print(f"  [xTB ] ERROR: xtbopt.xyz not found. "
              f"Check {log_path} for details.")
        return None

    # Convert optimised xyz → sdf so the rest of the pipeline can read it
    opt_sdf = xtb_dir / "xtbopt.sdf"
    if not _xyz_to_sdf(str(xtbopt_xyz), str(opt_sdf), template_sdf=sdf_path):
        print("  [xTB ] WARNING: xyz→sdf conversion failed; "
              "using original SDF for CREST input.")
        return sdf_path   # degrade gracefully

    print(f"  [xTB ] pre-optimisation complete → {xtbopt_xyz.name}")
    return str(opt_sdf)


# ──────────────────────────────────────────────────────────────────────────────
# CREST conformational search
# ──────────────────────────────────────────────────────────────────────────────

def run_crest(
    input_sdf:   str,
    work_dir:    Path,
    crest_exec:  str = "crest",
    xtb_exec:    str = "xtb",
    nproc:       int = 4,
    charge:      int = 0,
    multiplicity: int = 1,
    solvent:     Optional[str] = None,
    energy_window: float = 6.0,    # kcal/mol — passed to CREST's --ewin
) -> Optional[str]:
    """
    Run a CREST iMTD-GC conformational search.

    Returns path to the conformer ensemble XYZ file (crest_conformers.xyz),
    or None on failure.

    CREST is run in its own subdirectory.  All CREST output files
    (crest_conformers.xyz, crest_best.xyz, crest.energies, …) land there.
    """
    crest_dir = work_dir / "crest"
    crest_dir.mkdir(parents=True, exist_ok=True)

    # CREST prefers xyz input
    xyz_in = crest_dir / "input.xyz"
    _sdf_to_xyz(input_sdf, str(xyz_in))

    uhf = multiplicity - 1

    cmd = [
        crest_exec, str(xyz_in),
        "--T", str(nproc),
        "--chrg", str(charge),
        "--uhf", str(uhf),
        "--ewin", str(energy_window),   # ensemble energy window in kcal/mol
    ]
    if solvent:
        cmd += ["--alpb", solvent]

    log_path = crest_dir / "crest.log"
    print(f"  [CREST] running conformational search "
          f"({nproc} threads, ewin={energy_window} kcal/mol) …")
    try:
        with open(log_path, "w") as lf:
            r = subprocess.run(
                cmd,
                cwd=str(crest_dir),
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=86400,   # 24 h hard cap
            )
    except FileNotFoundError:
        print(f"  [CREST] ERROR: '{crest_exec}' not found. "
              f"Is CREST installed and on PATH?")
        return None
    except subprocess.TimeoutExpired:
        print("  [CREST] ERROR: CREST timed out after 24 h.")
        return None

    conformers_xyz = crest_dir / "crest_conformers.xyz"
    if not conformers_xyz.exists():
        print(f"  [CREST] ERROR: crest_conformers.xyz not produced. "
              f"Check {log_path}.")
        return None

    n = _count_conformers_in_xyz(str(conformers_xyz))
    print(f"  [CREST] search complete — {n} conformer(s) found.")
    return str(conformers_xyz)


# ──────────────────────────────────────────────────────────────────────────────
# Conformer ensemble parsing & selection
# ──────────────────────────────────────────────────────────────────────────────

def _count_conformers_in_xyz(xyz_path: str) -> int:
    """Count how many structures are in a multi-structure XYZ file."""
    count = 0
    with open(xyz_path) as fh:
        for line in fh:
            line = line.strip()
            if line.isdigit():
                count += 1
    return count


def parse_crest_ensemble(
    conformers_xyz: str,
    n_select:       int,
    energy_window:  float,    # kcal/mol relative to the lowest conformer
) -> list[tuple[int, float, str]]:
    """
    Parse a CREST multi-XYZ conformer file and select the best conformers.

    CREST orders conformers by energy (lowest first).  The comment line of
    each structure in the file is the absolute GFN2 energy in Hartree.

    Returns a list of (rank, rel_energy_kcal, xyz_block) tuples for the
    selected conformers, sorted by energy (best first).

    Selection criteria (applied in order):
      1. Only conformers within `energy_window` kcal/mol of the minimum.
      2. At most `n_select` conformers.
    """
    HARTREE_TO_KCAL = 627.509474

    conformers: list[tuple[float, str]] = []   # (energy_hartree, xyz_block)

    with open(conformers_xyz) as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if not line.isdigit():
            i += 1
            continue

        n_atoms = int(line)
        if i + 1 + n_atoms >= len(lines):
            break   # truncated file

        comment = lines[i + 1].strip()
        atom_lines = lines[i + 2 : i + 2 + n_atoms]

        # Parse energy from comment — CREST writes the energy (Hartree) there
        energy_ha = _parse_energy_from_comment(comment)

        xyz_block = f"{n_atoms}\n{comment}\n" + "".join(atom_lines)
        conformers.append((energy_ha, xyz_block))
        i += 2 + n_atoms

    if not conformers:
        return []

    # Sort by energy ascending (should already be sorted, but be safe)
    conformers.sort(key=lambda x: x[0])
    e_min = conformers[0][0]

    selected: list[tuple[int, float, str]] = []
    for rank, (e_ha, xyz_block) in enumerate(conformers, start=1):
        rel_kcal = (e_ha - e_min) * HARTREE_TO_KCAL
        if rel_kcal > energy_window:
            break   # all further conformers are outside the window
        selected.append((rank, rel_kcal, xyz_block))
        if len(selected) >= n_select:
            break

    return selected


def _parse_energy_from_comment(comment: str) -> float:
    """
    Extract a floating-point energy value from a CREST xyz comment line.
    CREST writes lines like '-10.123456789' or 'conf_1 -10.123456789 Eh'.
    Falls back to 0.0 if no number is found.
    """
    m = re.search(r"[-+]?\d+\.\d+", comment)
    return float(m.group(0)) if m else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# XYZ ↔ SDF conversion helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sdf_to_xyz(sdf_path: str, xyz_path: str) -> None:
    """
    Convert SDF to XYZ using RDKit if available, otherwise a minimal parser.
    The XYZ file includes all atoms (including H).
    """
    if RDKIT_AVAILABLE:
        mol = Chem.MolFromMolFile(sdf_path, removeHs=False, sanitize=False)
        if mol is not None:
            conf = mol.GetConformer()
            atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())]
            with open(xyz_path, "w") as fh:
                fh.write(f"{len(atoms)}\n")
                fh.write(f"Converted from {Path(sdf_path).name}\n")
                for atom in atoms:
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    sym = atom.GetSymbol()
                    fh.write(f"{sym:4s} {pos.x:14.8f} {pos.y:14.8f} {pos.z:14.8f}\n")
            return

    # Fallback: minimal SDF → XYZ parser (no RDKit)
    coord_block, n_atoms = g16.sdf_to_xyz_block(sdf_path)
    lines = coord_block.strip().split("\n")
    with open(xyz_path, "w") as fh:
        fh.write(f"{n_atoms}\n")
        fh.write(f"Converted from {Path(sdf_path).name}\n")
        for line in lines:
            parts = line.split()
            if len(parts) == 4:
                elem, x, y, z = parts
                fh.write(f"{elem:4s} {float(x):14.8f} {float(y):14.8f} {float(z):14.8f}\n")


def _xyz_to_sdf(
    xyz_path:     str,
    sdf_path:     str,
    template_sdf: Optional[str] = None,
) -> bool:
    """
    Convert an XYZ file to SDF.

    If RDKit is available and a template SDF is provided, the bond
    topology from the template is preserved (only coordinates are
    updated).  This avoids bond-order guessing, which is unreliable
    for heteroatom-rich molecules.

    Returns True on success, False on failure.
    """
    if RDKIT_AVAILABLE and template_sdf:
        try:
            template = Chem.MolFromMolFile(template_sdf, removeHs=False, sanitize=False)
            if template is None:
                raise ValueError("Template could not be parsed by RDKit")

            # Read XYZ coordinates
            new_coords = []
            with open(xyz_path) as fh:
                lines = fh.readlines()
            n_atoms = int(lines[0].strip())
            for line in lines[2 : 2 + n_atoms]:
                parts = line.split()
                new_coords.append((float(parts[1]), float(parts[2]), float(parts[3])))

            if len(new_coords) != template.GetNumAtoms():
                raise ValueError(
                    f"Atom count mismatch: XYZ={len(new_coords)}, "
                    f"template={template.GetNumAtoms()}"
                )

            # Overwrite the conformer coordinates
            from rdkit.Chem import AllChem
            mol = Chem.RWMol(template)
            conf = mol.GetConformer()
            from rdkit.Geometry import rdGeometry
            for i, (x, y, z) in enumerate(new_coords):
                conf.SetAtomPosition(i, (x, y, z))

            writer = Chem.SDWriter(sdf_path)
            writer.write(mol)
            writer.close()
            return True

        except Exception as e:
            print(f"  [conv] RDKit xyz→sdf failed ({e}); "
                  f"falling back to plain-text SDF.")

    # Fallback: write a minimal V2000 SDF with no bond table.
    # Gaussian reads coordinates directly, so a bond-table-free SDF is
    # acceptable as pipeline input (sdf_to_xyz_block only needs coords).
    try:
        with open(xyz_path) as fh:
            lines = fh.readlines()
        n_atoms = int(lines[0].strip())
        atom_lines = lines[2 : 2 + n_atoms]

        with open(sdf_path, "w") as fh:
            fh.write("\n  CREST\n\n")
            fh.write(f"{n_atoms:3d}  0  0  0  0  0  0  0  0  0999 V2000\n")
            for line in atom_lines:
                parts = line.split()
                if len(parts) < 4:
                    continue
                sym, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                fh.write(f"{x:10.4f}{y:10.4f}{z:10.4f} {sym:<3s} 0  0  0  0  0  0  0  0  0  0  0  0\n")
            fh.write("M  END\n$$$$\n")
        return True
    except Exception as e:
        print(f"  [conv] Fallback xyz→sdf also failed: {e}")
        return False


def _xyz_block_to_sdf(xyz_block: str, sdf_path: str, template_sdf: Optional[str] = None) -> bool:
    """Write an xyz_block string to a temp file, then convert to SDF."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xyz", delete=False
    ) as tmp:
        tmp.write(xyz_block)
        tmp_path = tmp.name
    try:
        return _xyz_to_sdf(tmp_path, sdf_path, template_sdf=template_sdf)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Semi-empirical pre-optimisation inside Gaussian (PM7 or xTB-external)
# ──────────────────────────────────────────────────────────────────────────────

def build_gjf_preopt_pm7(
    coord_block:  str,
    chk_path:     str,
    *,
    charge:       int           = 0,
    multiplicity: int           = 1,
    nproc:        int           = 10,
    mem:          str           = "28GB",
    scratch_dir:  Optional[str] = None,
    cid:          str           = "mol",
) -> str:
    """
    Build a Gaussian .gjf for PM7 semi-empirical geometry optimisation.

    Route line:  #p PM7 opt=(MaxCycles=500) NoTestMO

    PM7 (Stewart 2013) is a reparametrised semi-empirical Hamiltonian
    built into Gaussian 16.  No external binary is needed.  It is a good
    general-purpose method for organic/drug-like molecules and is fast
    enough that even a 50-atom molecule converges in seconds.

    The .chk written here is read back by extract_preopt_geometry_from_log
    (via formchk + the Standard orientation block) to obtain the optimised
    coordinates.  A separate scratch .rwf is used and deleted after the run
    (%NoSave) to avoid accumulating large files.
    """
    nosave_line = "%NoSave\n"
    rwf_line    = ""
    if scratch_dir:
        rwf_path = str(Path(scratch_dir) / f"{cid}_pm7preopt.rwf")
        rwf_line = f"%RWF={rwf_path}\n"

    return (
        f"%chk={chk_path}\n"
        f"{rwf_line}"
        f"{nosave_line}"
        f"%nprocshared={nproc}\n"
        f"%mem={mem}\n"
        f"#p PM7 opt=(MaxCycles=500) NoTestMO\n"
        f"\n"
        f"PM7 pre-optimisation\n"
        f"\n"
        f"{charge} {multiplicity}\n"
        f"{coord_block}\n"
        f"\n"
    )


def build_gjf_preopt_xtb_external(
    coord_block:  str,
    chk_path:     str,
    *,
    charge:       int           = 0,
    multiplicity: int           = 1,
    nproc:        int           = 10,
    mem:          str           = "28GB",
    scratch_dir:  Optional[str] = None,
    cid:          str           = "mol",
    xtb_exec:     str           = "xtb",
    solvent:      Optional[str] = None,
) -> str:
    """
    Build a Gaussian .gjf that drives xTB geometry optimisation via the
    Gaussian 'external=' interface.

    Route line:
        #p external="<xtb_cmd>" opt=(MaxCycles=500,CalcFC) NoTestMO

    How Gaussian external= works
    ────────────────────────────
    Gaussian calls the external program once per optimisation step.  It
    passes a scratch file containing the current geometry and asks for
    energy + gradient (and optionally Hessian).  The external program
    writes those quantities back.  Gaussian then runs its own GEDIIS/RFO
    optimiser using the returned data.

    The xTB binary understands the Gaussian external protocol natively
    since xTB ≥ 6.4.  The call signature Gaussian uses is:
        xtb <layer> <input_file> <output_file> <msg_file> [<fchk_file> <matel_file>]

    Required setup
    ──────────────
    • xTB ≥ 6.4 on PATH (or passed via --xtb).
    • The xTB binary must be compiled with Gaussian-interface support
      (all official binaries from Grimme group include this).
    • The Gaussian scratch directory must be writable by both processes.

    CalcFC is added so Gaussian computes a force-constant matrix at the
    start; this gives a better initial Hessian and usually speeds up
    convergence significantly for near-flat PES regions.
    """
    nosave_line = "%NoSave\n"
    rwf_line    = ""
    if scratch_dir:
        rwf_path = str(Path(scratch_dir) / f"{cid}_xtbpreopt.rwf")
        rwf_line = f"%RWF={rwf_path}\n"

    # Build the external= command string
    # --chrg and --uhf are passed via environment in xtb ≥ 6.4, but
    # including them explicitly on the command line is more robust.
    uhf     = multiplicity - 1
    xtb_cmd = f"{xtb_exec} --chrg {charge} --uhf {uhf}"
    if solvent:
        xtb_cmd += f" --alpb {solvent}"

    return (
        f"%chk={chk_path}\n"
        f"{rwf_line}"
        f"{nosave_line}"
        f"%nprocshared={nproc}\n"
        f"%mem={mem}\n"
        f'#p external="{xtb_cmd}" opt=(MaxCycles=500,CalcFC) NoTestMO\n'
        f"\n"
        f"xTB external pre-optimisation\n"
        f"\n"
        f"{charge} {multiplicity}\n"
        f"{coord_block}\n"
        f"\n"
    )


def _extract_geometry_from_log(log_path: str) -> Optional[str]:
    """
    Extract the last 'Standard orientation' Cartesian coordinate block
    from a Gaussian log file and return it as a coord_block string
    suitable for use in build_gjf / build_gjf_freq.

    Returns None if no Standard orientation block is found (e.g. the job
    crashed before printing any geometry).
    """
    # Matches the coordinate table that follows "Standard orientation:"
    # Each data row has the form:
    #   center_no  atomic_no  atom_type  x  y  z
    block_re = re.compile(
        r"Standard orientation:.*?-{20,}.*?-{20,}\n(.*?)-{20,}",
        re.DOTALL,
    )
    row_re = re.compile(
        r"^\s+\d+\s+(\d+)\s+\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
        re.MULTILINE,
    )

    _Z_TO_SYM = {
        1:'H',  2:'He', 3:'Li', 4:'Be', 5:'B',  6:'C',  7:'N',  8:'O',
        9:'F',  10:'Ne',11:'Na',12:'Mg',13:'Al',14:'Si',15:'P', 16:'S',
        17:'Cl',18:'Ar',19:'K', 20:'Ca',21:'Sc',22:'Ti',23:'V', 24:'Cr',
        25:'Mn',26:'Fe',27:'Co',28:'Ni',29:'Cu',30:'Zn',31:'Ga',32:'Ge',
        33:'As',34:'Se',35:'Br',36:'Kr',37:'Rb',38:'Sr',39:'Y', 40:'Zr',
        41:'Nb',42:'Mo',43:'Tc',44:'Ru',45:'Rh',46:'Pd',47:'Ag',48:'Cd',
        49:'In',50:'Sn',51:'Sb',52:'Te',53:'I', 54:'Xe',55:'Cs',56:'Ba',
        57:'La',72:'Hf',73:'Ta',74:'W', 75:'Re',76:'Os',77:'Ir',78:'Pt',
        79:'Au',80:'Hg',81:'Tl',82:'Pb',83:'Bi',84:'Po',85:'At',86:'Rn',
    }

    try:
        with open(log_path) as fh:
            content = fh.read()
    except OSError:
        return None

    last_block = None
    for match in block_re.finditer(content):
        rows = row_re.findall(match.group(1))
        if not rows:
            continue
        lines = []
        for atomic_no_str, x, y, z in rows:
            sym = _Z_TO_SYM.get(int(atomic_no_str), f"X{atomic_no_str}")
            lines.append(
                f"  {sym:<3} {float(x):>14.8f} {float(y):>14.8f} {float(z):>14.8f}"
            )
        last_block = "\n".join(lines)

    return last_block   # None if no blocks found


def gaussian_semiempirical_preopt(
    input_sdf:    str,
    preopt_dir:   Path,
    conf_label:   str,
    *,
    method:       str           = "pm7",    # "pm7" or "xtb"
    g16_exec:     str           = "g16",
    xtb_exec:     str           = "xtb",
    charge:       int           = 0,
    multiplicity: int           = 1,
    nproc:        int           = 10,
    mem:          str           = "28GB",
    scratch_dir:  str           = "/tmp",
    solvent:      Optional[str] = None,
    cid:          str           = "mol",
) -> str:
    """
    Run a PM7 or xTB-external geometry optimisation inside Gaussian 16 and
    return the path to an SDF containing the optimised geometry.

    On any failure (Gaussian crash, no geometry found, conversion error)
    the function logs a warning and returns `input_sdf` unchanged so the
    DFT step always has something to work with.

    Parameters
    ──────────
    input_sdf   : path to the conformer SDF coming out of CREST
    preopt_dir  : directory to write .gjf / .chk / .log into
    conf_label  : short label used in file names, e.g. "conf001"
    method      : "pm7" or "xtb"
    ...rest     : forwarded to the gjf builders / Gaussian runner

    Directory layout (all inside preopt_dir)
    ─────────────────────────────────────────
    <conf_label>_preopt.gjf   input
    <conf_label>_preopt.chk   checkpoint (deleted after geometry extraction)
    <conf_label>_preopt.log   full Gaussian output — keep for inspection
    <conf_label>_preopt.sdf   optimised geometry (returned on success)
    """
    preopt_dir.mkdir(parents=True, exist_ok=True)
    method = method.lower().strip()

    gjf_path = str(preopt_dir / f"{conf_label}_preopt.gjf")
    chk_path = str(preopt_dir / f"{conf_label}_preopt.chk")
    log_path = str(preopt_dir / f"{conf_label}_preopt.log")
    out_sdf  = str(preopt_dir / f"{conf_label}_preopt.sdf")

    tag = "pm7" if method == "pm7" else "xTB-ext"
    print(f"    [preopt/{tag}] optimising {conf_label} …")

    # ── Read input geometry ───────────────────────────────────────────────────
    try:
        coord_block, _ = g16.sdf_to_xyz_block(input_sdf)
    except Exception as exc:
        print(f"    [preopt] WARNING: could not read {input_sdf}: {exc}. "
              f"Skipping preopt.")
        return input_sdf

    # ── Build .gjf ────────────────────────────────────────────────────────────
    common_kwargs = dict(
        charge=charge,
        multiplicity=multiplicity,
        nproc=nproc,
        mem=mem,
        scratch_dir=scratch_dir,
        cid=f"{cid}_{conf_label}",
    )

    if method == "pm7":
        gjf_content = build_gjf_preopt_pm7(coord_block, chk_path, **common_kwargs)
    elif method == "xtb":
        gjf_content = build_gjf_preopt_xtb_external(
            coord_block, chk_path,
            xtb_exec=xtb_exec,
            solvent=solvent,
            **common_kwargs,
        )
    else:
        print(f"    [preopt] WARNING: unknown preopt method '{method}'. "
              f"Supported: pm7, xtb. Skipping preopt.")
        return input_sdf

    with open(gjf_path, "w") as fh:
        fh.write(gjf_content)

    # ── Run Gaussian ──────────────────────────────────────────────────────────
    ok, err = g16.run_gaussian(
        gjf_path, log_path,
        g16_exec=g16_exec,
        scratch_dir=scratch_dir,
    )

    if not ok or not g16.check_normal_termination(log_path):
        print(f"    [preopt] WARNING: Gaussian {tag} preopt did not terminate "
              f"normally ({err or 'check log'}). Using CREST geometry.")
        _cleanup_preopt_scratch(scratch_dir, cid, conf_label, method)
        return input_sdf

    # ── Extract optimised geometry from log ───────────────────────────────────
    opt_coord_block = _extract_geometry_from_log(log_path)
    if opt_coord_block is None:
        print(f"    [preopt] WARNING: no geometry found in {log_path}. "
              f"Using CREST geometry.")
        _cleanup_preopt_scratch(scratch_dir, cid, conf_label, method)
        return input_sdf

    # ── Write optimised geometry to SDF ──────────────────────────────────────
    # Build a minimal SDF from the extracted coord block.
    # We use the fallback writer directly (no RDKit bond-order guessing
    # needed — the DFT step only cares about coordinates).
    xyz_lines = []
    for line in opt_coord_block.strip().split("\n"):
        parts = line.split()
        if len(parts) == 4:
            elem, x, y, z = parts
            xyz_lines.append(
                f"{elem:4s} {float(x):14.8f} {float(y):14.8f} {float(z):14.8f}"
            )
    n_atoms = len(xyz_lines)

    # Write a temporary xyz and convert to SDF so _xyz_to_sdf can use the
    # template_sdf bond topology if RDKit is available.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xyz", delete=False
    ) as tmp_xyz:
        tmp_xyz.write(f"{n_atoms}\nGaussian {tag} preopt\n")
        for ln in xyz_lines:
            tmp_xyz.write(ln + "\n")
        tmp_xyz_path = tmp_xyz.name

    try:
        sdf_ok = _xyz_to_sdf(tmp_xyz_path, out_sdf, template_sdf=input_sdf)
    finally:
        Path(tmp_xyz_path).unlink(missing_ok=True)

    if not sdf_ok:
        print(f"    [preopt] WARNING: could not convert optimised geometry "
              f"to SDF. Using CREST geometry.")
        _cleanup_preopt_scratch(scratch_dir, cid, conf_label, method)
        return input_sdf

    # Clean up scratch; .chk no longer needed (geometry is in the SDF)
    _cleanup_preopt_scratch(scratch_dir, cid, conf_label, method)
    Path(chk_path).unlink(missing_ok=True)

    print(f"    [preopt/{tag}] done → {Path(out_sdf).name}")
    return out_sdf


def _cleanup_preopt_scratch(
    scratch_dir: str,
    cid:         str,
    conf_label:  str,
    method:      str,
) -> None:
    """Remove the .rwf scratch file written by the semi-empirical preopt."""
    suffix = "pm7preopt" if method == "pm7" else "xtbpreopt"
    rwf = Path(scratch_dir) / f"{cid}_{conf_label}_{suffix}.rwf"
    rwf.unlink(missing_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian step — wraps the existing pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_gaussian_on_conformer(
    conformer_sdf: str,
    conf_dir:      Path,
    conf_label:    str,
    *,
    g16_exec:      str,
    basis_set:     str,
    method:        str,
    operation:     str,
    charge:        int,
    multiplicity:  int,
    nproc:         int,
    mem:           str,
    cosmo:         bool,
    timeout:       Optional[int],
    scratch_dir:   str,
    max_disk:      Optional[str],
    extra_keywords: str,
) -> g16.RunResult:
    """
    Submit a single conformer SDF to the Gaussian pipeline.

    We create a per-conformer work subdirectory so each conformer gets its
    own .gjf / .chk / .log files and scratch space.
    """
    return g16.run_compound(
        sdf_path=conformer_sdf,
        work_dir=str(conf_dir),
        g16_exec=g16_exec,
        basis_set=basis_set,
        method=method,
        operation=operation,
        charge=charge,
        multiplicity=multiplicity,
        nproc=nproc,
        mem=mem,
        cosmo=cosmo,
        timeout=timeout,
        scratch_dir=scratch_dir,
        max_disk=max_disk,
        extra_keywords=extra_keywords,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    sdf_path:         str,
    work_dir:         str,
    *,
    # Flexibility / CREST control
    flex_threshold:   int           = 3,
    force_crest:      bool          = False,
    skip_crest:       bool          = False,
    n_conformers:     int           = 5,
    energy_window:    float         = 3.0,
    crest_ewin:       float         = 6.0,
    # External executables
    xtb_exec:         str           = "xtb",
    crest_exec:       str           = "crest",
    g16_exec:         str           = "g16",
    # Molecular properties
    charge:           int           = 0,
    multiplicity:     int           = 1,
    solvent:          Optional[str] = None,
    # Semi-empirical preopt (step 5)
    preopt_method:    str           = "pm7",   # "pm7" or "xtb"
    skip_preopt:      bool          = False,
    # Gaussian DFT settings
    basis_set:        str           = "6-311G*",
    method:           str           = "PBE0",
    operation:        str           = "opt freq",
    nproc:            int           = 10,
    mem:              str           = "28GB",
    cosmo:            bool          = False,
    timeout:          Optional[int] = None,
    scratch_dir:      Optional[str] = None,
    max_disk:         Optional[str] = None,
    extra_keywords:   str           = "NoTestMO SCF=(XQC,MaxCycle=200)",
) -> PipelineResult:

    sdf_path = str(Path(sdf_path).resolve())
    cid      = Path(sdf_path).stem
    job_dir  = Path(work_dir) / cid
    job_dir.mkdir(parents=True, exist_ok=True)

    resolved_scratch = g16.get_scratch_dir(scratch_dir)

    result = PipelineResult(
        cid=cid,
        sdf_path=sdf_path,
        flexible=False,
        n_rot_bonds=0,
        crest_ran=False,
        n_conformers_found=0,
        n_conformers_calc=0,
    )

    # ── 1. Flexibility check ──────────────────────────────────────────────────
    flexible, n_rot = is_flexible(sdf_path, flex_threshold)
    result.flexible    = flexible
    result.n_rot_bonds = n_rot

    do_crest = (flexible or force_crest) and not skip_crest

    if skip_crest:
        print(f"[{cid}] Skipping CREST (--skip-crest).")
    elif force_crest:
        print(f"[{cid}] Forcing CREST (--force-crest). "
              f"Rotatable bonds: {n_rot}.")
    elif flexible:
        print(f"[{cid}] Flexible molecule ({n_rot} rotatable bonds ≥ "
              f"threshold {flex_threshold}). Running CREST.")
    else:
        print(f"[{cid}] Rigid molecule ({n_rot} rotatable bonds < "
              f"threshold {flex_threshold}). Skipping CREST.")

    # ── 2-4. CREST path ───────────────────────────────────────────────────────
    if do_crest:
        # 2. GFN2-xTB pre-optimisation
        preopt_sdf = xtb_preopt(
            sdf_path, job_dir,
            xtb_exec=xtb_exec,
            nproc=nproc,
            charge=charge,
            multiplicity=multiplicity,
            solvent=solvent,
        )
        if preopt_sdf is None:
            result.error_msg = "xTB pre-optimisation failed."
            return result
        crest_input = preopt_sdf

        # 3. CREST conformational search
        conformers_xyz = run_crest(
            crest_input, job_dir,
            crest_exec=crest_exec,
            xtb_exec=xtb_exec,
            nproc=nproc,
            charge=charge,
            multiplicity=multiplicity,
            solvent=solvent,
            energy_window=crest_ewin,
        )
        if conformers_xyz is None:
            result.error_msg = "CREST conformational search failed."
            return result
        result.crest_ran = True

        # 4. Select top N conformers within energy window
        selected = parse_crest_ensemble(
            conformers_xyz,
            n_select=n_conformers,
            energy_window=energy_window,
        )
        result.n_conformers_found = _count_conformers_in_xyz(conformers_xyz)

        if not selected:
            result.error_msg = (
                "No conformers found within energy window after CREST."
            )
            return result

        print(f"  [sel ] Selected {len(selected)} conformer(s) "
              f"within {energy_window} kcal/mol of minimum.")

        # Write each selected conformer as a separate SDF for Gaussian
        conformer_sdfs: list[tuple[int, float, str]] = []  # (rank, rel_e, sdf_path)
        for rank, rel_e, xyz_block in selected:
            conf_sdf = job_dir / f"{cid}_conf{rank:03d}.sdf"
            ok = _xyz_block_to_sdf(xyz_block, str(conf_sdf), template_sdf=sdf_path)
            if not ok:
                print(f"  [sel ] WARNING: could not write SDF for conformer {rank}; skipping.")
                continue
            conformer_sdfs.append((rank, rel_e, str(conf_sdf)))

    else:
        # No CREST — treat the input SDF as the single "conformer"
        conformer_sdfs = [(1, 0.0, sdf_path)]
        result.n_conformers_found = 1

    # ── 5. Semi-empirical pre-optimisation ───────────────────────────────────
    #
    # Each conformer SDF is pre-optimised at a cheap level (PM7 or xTB-via-
    # Gaussian-external) before the expensive DFT run.  This moves the
    # geometry closer to the DFT minimum, reducing the number of DFT cycles
    # needed and lowering the risk of converging to a saddle point.
    #
    # Failures degrade gracefully: if Gaussian crashes or produces no
    # geometry, the CREST SDF is used unchanged and the DFT step runs anyway.
    if skip_preopt:
        print(f"  [preopt] Skipping semi-empirical preopt (--skip-preopt).")
        preopt_sdfs = conformer_sdfs   # pass through unchanged
    else:
        tag = preopt_method.upper()
        print(f"  [preopt] Running {tag} pre-optimisation on "
              f"{len(conformer_sdfs)} conformer(s) …")
        preopt_sdfs = []
        for rank, rel_e, conf_sdf in conformer_sdfs:
            conf_label  = f"conf{rank:03d}"
            preopt_dir  = job_dir / "preopt" / conf_label
            optimised_sdf = gaussian_semiempirical_preopt(
                conf_sdf, preopt_dir, conf_label,
                method=preopt_method,
                g16_exec=g16_exec,
                xtb_exec=xtb_exec,
                charge=charge,
                multiplicity=multiplicity,
                nproc=nproc,
                mem=mem,
                scratch_dir=resolved_scratch,
                solvent=solvent,
                cid=cid,
            )
            preopt_sdfs.append((rank, rel_e, optimised_sdf))

    # ── 6. Gaussian DFT opt+freq on each pre-optimised conformer ─────────────
    result.n_conformers_calc = len(preopt_sdfs)
    conf_results: list[ConformerResult] = []

    for i, (rank, rel_e, conf_sdf) in enumerate(preopt_sdfs, start=1):
        label = f"conf{rank:03d}"
        conf_dir = job_dir / "gaussian" / label
        conf_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{i}/{len(preopt_sdfs)}] Gaussian DFT on conformer {rank} "
              f"(ΔE_CREST = {rel_e:+.2f} kcal/mol) …")

        g16_result = run_gaussian_on_conformer(
            conf_sdf, conf_dir, label,
            g16_exec=g16_exec,
            basis_set=basis_set,
            method=method,
            operation=operation,
            charge=charge,
            multiplicity=multiplicity,
            nproc=nproc,
            mem=mem,
            cosmo=cosmo,
            timeout=timeout,
            scratch_dir=resolved_scratch,
            max_disk=max_disk,
            extra_keywords=extra_keywords,
        )

        # Extract energy — prefer freq log (has final SCF after opt converged)
        dft_e = g16_result.energy

        conf_r = ConformerResult(
            rank=rank,
            crest_energy=rel_e if do_crest else None,
            gjf_path=g16_result.gjf_path,
            log_path=g16_result.log_path,
            success=g16_result.success,
            dft_energy=dft_e,
            thermo=g16_result.thermo,
            error_msg=g16_result.error_msg,
        )
        conf_results.append(conf_r)

        if g16_result.success:
            e_str = f"{dft_e:.6f} Ha" if dft_e is not None else "n/a"
            print(f"  ✓  conformer {rank}: E(DFT) = {e_str}")
        else:
            print(f"  ✗  conformer {rank}: {g16_result.error_msg}")

    result.conformers = conf_results

    # ── 7. Find the DFT global minimum ───────────────────────────────────────
    successful = [c for c in conf_results if c.success and c.dft_energy is not None]
    if successful:
        result.best_conformer = min(successful, key=lambda c: c.dft_energy)
        result.success = True
        print(f"\n  ★  DFT global minimum: conformer {result.best_conformer.rank} "
              f"(E = {result.best_conformer.dft_energy:.6f} Ha)")
    else:
        result.error_msg = "All Gaussian calculations failed."

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Batch runner (used by the Ivette CLI)
# ──────────────────────────────────────────────────────────────────────────────

def batch_run(
    sdf_dir: str,
    work_dir: str,
    *,
    jobs: int = 1,
    operation: str = "opt freq",
    resume: bool = True,
    checkpoint: Optional[str] = None,
    nproc: int = 4,
    mem: str = "4GB",
    preopt_mode: str = "none",
    preopt_basis_set: str = "6-31G*",
    g16_exec: str = "g16",
    basis_set: str = "6-31G*",
    method: str = "B3LYP",
    charge: int = 0,
    multiplicity: int = 1,
    cosmo: bool = False,
    timeout: Optional[int] = None,
) -> "list[g16.RunResult]":
    """Run the Gaussian pipeline over every ``*.sdf`` in ``sdf_dir``.

    Each molecule goes through ``SDF → .gjf → g16 → parse`` via
    :func:`gaussian16_core.run_compound`. Progress is persisted to a JSON
    checkpoint so an interrupted batch can be resumed (completed molecules are
    skipped). With ``jobs > 1`` molecules run in parallel processes. Returns one
    ``RunResult`` per molecule executed this invocation (each has ``.success``).
    """
    sdf_dir = Path(sdf_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    sdfs = sorted(sdf_dir.glob("*.sdf"))
    if not sdfs:
        print(f"No SDF files found in {sdf_dir}")
        return []

    ckpt_path = Path(checkpoint) if checkpoint else (work_dir / "checkpoint.json")
    done = jsonstore.read_json(ckpt_path, default={}) if resume else {}

    pending = [s for s in sdfs if not done.get(s.stem, {}).get("success")]
    skipped = len(sdfs) - len(pending)
    print(
        f"Found {len(sdfs)} SDF files in {sdf_dir.name}; "
        f"{len(pending)} to run, {skipped} already complete."
    )

    kwargs = dict(
        g16_exec=g16_exec, basis_set=basis_set, method=method, operation=operation,
        preopt_mode=preopt_mode, preopt_basis_set=preopt_basis_set,
        charge=charge, multiplicity=multiplicity, nproc=nproc, mem=mem,
        cosmo=cosmo, timeout=timeout,
    )

    def _record(result):
        done[result.cid] = {
            "success": result.success,
            "log": result.log_path,
            "error": result.error_msg,
        }
        jsonstore.write_json(ckpt_path, done)

    results = []
    if jobs <= 1:
        for i, sdf in enumerate(pending, 1):
            print(f"[{i}/{len(pending)}] {sdf.name}", flush=True)
            try:
                result = g16.run_compound(str(sdf), str(work_dir), **kwargs)
            except Exception as exc:
                result = g16.RunResult(
                    cid=sdf.stem, sdf_path=str(sdf), gjf_path="",
                    log_path="", success=False, error_msg=str(exc),
                )
            results.append(result)
            _record(result)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(g16.run_compound, str(sdf), str(work_dir), **kwargs): sdf
                for sdf in pending
            }
            for i, future in enumerate(as_completed(futures), 1):
                sdf = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = g16.RunResult(
                        cid=sdf.stem, sdf_path=str(sdf), gjf_path="",
                        log_path="", success=False, error_msg=str(exc),
                    )
                print(f"[{i}/{len(pending)}] {sdf.name}", flush=True)
                results.append(result)
                _record(result)

    ok = sum(1 for r in results if r.success)
    print(f"Batch complete: {ok}/{len(results)} succeeded "
          f"({skipped} skipped from a previous run).")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Output / reporting
# ──────────────────────────────────────────────────────────────────────────────

HARTREE_TO_KCAL = 627.509474


def write_summary(result: PipelineResult, out_dir: Path) -> Path:
    """
    Write a TSV summary of all conformer DFT results, ranked by DFT energy.
    Returns the path to the summary file.
    """
    tsv_path = out_dir / f"{result.cid}_conformer_summary.tsv"

    # Sort successful conformers by DFT energy; failed ones go at the end
    successful = sorted(
        [c for c in result.conformers if c.success and c.dft_energy is not None],
        key=lambda c: c.dft_energy,
    )
    failed = [c for c in result.conformers if not c.success or c.dft_energy is None]
    ordered = successful + failed

    # Compute relative DFT energies
    e_min = successful[0].dft_energy if successful else None

    fields = [
        "dft_rank", "crest_rank", "dft_energy_Ha",
        "dft_rel_energy_kcal", "crest_rel_energy_kcal",
        "zpe_kcal", "thermal_H_kcal", "TS_cal_K",
        "temp_K", "success", "log_path", "error_msg",
    ]

    with open(tsv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for dft_rank, conf in enumerate(ordered, start=1):
            rel_dft = (
                (conf.dft_energy - e_min) * HARTREE_TO_KCAL
                if (conf.dft_energy is not None and e_min is not None)
                else ""
            )
            row = {
                "dft_rank":               dft_rank,
                "crest_rank":             conf.rank,
                "dft_energy_Ha":          conf.dft_energy if conf.dft_energy is not None else "",
                "dft_rel_energy_kcal":    f"{rel_dft:.4f}" if rel_dft != "" else "",
                "crest_rel_energy_kcal":  f"{conf.crest_energy:.4f}" if conf.crest_energy is not None else "",
                "zpe_kcal":               conf.thermo.zpe if conf.thermo else "",
                "thermal_H_kcal":         conf.thermo.th  if conf.thermo else "",
                "TS_cal_K":               conf.thermo.ts  if conf.thermo else "",
                "temp_K":                 conf.thermo.temp if conf.thermo else "",
                "success":                conf.success,
                "log_path":               conf.log_path,
                "error_msg":              conf.error_msg,
            }
            w.writerow(row)

    print(f"\n[summary] Written to {tsv_path}")
    return tsv_path


def print_result(result: PipelineResult) -> None:
    print(f"\n{'═'*60}")
    print(f"  CID         : {result.cid}")
    print(f"  Flexible    : {result.flexible} ({result.n_rot_bonds} rot. bonds)")
    print(f"  CREST ran   : {result.crest_ran}")
    if result.crest_ran:
        print(f"  Conformers  : {result.n_conformers_found} found, "
              f"{result.n_conformers_calc} submitted to Gaussian")
    print(f"  Success     : {result.success}")
    if not result.success:
        print(f"  Error       : {result.error_msg}")
    if result.best_conformer:
        bc = result.best_conformer
        print(f"  Best conf.  : rank {bc.rank}, "
              f"E = {bc.dft_energy:.6f} Ha")
        if bc.thermo:
            print(f"  Thermo (best conformer):")
            print(f"    T        = {bc.thermo.temp} K")
            print(f"    ZPE      = {bc.thermo.zpe} kcal/mol")
            print(f"    H corr   = {bc.thermo.th} kcal/mol")
            print(f"    TS       = {bc.thermo.ts} cal/mol·K")
    print(f"{'═'*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CREST conformational search + Gaussian 16 DFT pipeline.\n"
            "Flexible molecules are pre-screened with xTB/CREST before DFT."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input / output ────────────────────────────────────────────────────────
    parser.add_argument("--sdf",     required=True, help="Input SDF file")
    parser.add_argument("--workdir", default="./crest_g16_runs",
                        help="Output directory (default: ./crest_g16_runs)")

    # ── Flexibility / CREST ───────────────────────────────────────────────────
    parser.add_argument(
        "--flex-threshold", type=int, default=3, metavar="N",
        help="Min rotatable bonds to consider a molecule flexible "
             "and trigger CREST (default: 3)",
    )
    parser.add_argument(
        "--force-crest", action="store_true",
        help="Run CREST even for rigid molecules",
    )
    parser.add_argument(
        "--skip-crest", action="store_true",
        help="Skip CREST entirely; run Gaussian on the input SDF only",
    )
    parser.add_argument(
        "--n-conformers", type=int, default=5, metavar="N",
        help="Max conformers to submit to Gaussian (default: 5)",
    )
    parser.add_argument(
        "--energy-window", type=float, default=3.0, metavar="KCAL",
        help="Max CREST relative energy (kcal/mol) for conformer selection "
             "(default: 3.0)",
    )
    parser.add_argument(
        "--crest-ewin", type=float, default=6.0, metavar="KCAL",
        help="CREST --ewin value: ensemble energy window for the search "
             "(default: 6.0; larger = more conformers found)",
    )
    parser.add_argument(
        "--solvent", default=None, metavar="SOLVENT",
        help="Implicit solvent for xTB/CREST via ALPB (e.g. water, acetonitrile). "
             "Also enables SCRF=(CPCM,...) in Gaussian if --cosmo is set.",
    )

    # ── External executables ──────────────────────────────────────────────────
    parser.add_argument("--xtb",   default="xtb",   help="Path to xtb binary")
    parser.add_argument("--crest", default="crest", help="Path to crest binary")
    parser.add_argument("--g16",   default="g16",   help="Path to g16 binary")

    # ── Molecular charge / multiplicity ───────────────────────────────────────
    parser.add_argument("--charge", type=int, default=0,
                        help="Molecular charge (default: 0)")
    parser.add_argument("--mult",   type=int, default=1,
                        help="Spin multiplicity (default: 1)")

    # ── Semi-empirical preopt ─────────────────────────────────────────────────
    parser.add_argument(
        "--preopt-method",
        default="pm7",
        choices=["pm7", "xtb"],
        dest="preopt_method",
        help=(
            "Semi-empirical method for Gaussian pre-optimisation before DFT. "
            "'pm7' (default): uses Gaussian's built-in PM7 Hamiltonian — "
            "no extra binary needed, good for organic/drug-like molecules. "
            "'xtb': drives GFN2-xTB via Gaussian's external= interface — "
            "requires xTB on PATH, more accurate for heavy atoms and "
            "non-covalent interactions."
        ),
    )
    parser.add_argument(
        "--skip-preopt",
        action="store_true",
        dest="skip_preopt",
        help=(
            "Skip the semi-empirical pre-optimisation step entirely. "
            "Conformers from CREST are passed directly to DFT."
        ),
    )

    # ── Gaussian DFT settings ─────────────────────────────────────────────────
    parser.add_argument("--method",    default="PBE0",
                        help="DFT functional (default: PBE0)")
    parser.add_argument("--basis",     default="6-311G*",
                        help="Basis set (default: 6-311G*)")
    parser.add_argument("--operation", default="opt then freq",
                        help="Gaussian task keywords (default: 'opt then freq')")
    parser.add_argument("--nproc",     type=int, default=10,
                        help="%%nprocshared (default: 10)")
    parser.add_argument("--mem",       default="28GB",
                        help="%%mem (default: 28GB)")
    parser.add_argument("--cosmo",     action="store_true",
                        help="Add SCRF=(CPCM,Solvent=water) to Gaussian route")
    parser.add_argument("--timeout",   type=int, default=None,
                        help="Per-conformer Gaussian timeout in seconds")
    parser.add_argument("--scratch",   default=None,
                        help="Gaussian scratch directory (overrides $GAUSS_SCRDIR)")
    parser.add_argument("--max-disk",  default=None, dest="max_disk",
                        help="Gaussian MaxDisk cap (e.g. 200GB)")
    parser.add_argument(
        "--extra-keywords",
        default="NoTestMO SCF=(XQC,MaxCycle=200)",
        dest="extra_keywords",
        help="Extra Gaussian route keywords (default: 'NoTestMO SCF=(XQC,MaxCycle=200)')",
    )

    args = parser.parse_args(argv)

    if not Path(args.sdf).exists():
        sys.exit(f"ERROR: SDF file not found: {args.sdf}")

    if not RDKIT_AVAILABLE:
        print(
            "WARNING: RDKit is not installed. Rotatable-bond counting is "
            "unavailable; all molecules will be treated as rigid unless "
            "--force-crest is set.\n"
            "Install RDKit:  conda install -c conda-forge rdkit"
        )

    result = run_pipeline(
        sdf_path=args.sdf,
        work_dir=args.workdir,
        flex_threshold=args.flex_threshold,
        force_crest=args.force_crest,
        skip_crest=args.skip_crest,
        n_conformers=args.n_conformers,
        energy_window=args.energy_window,
        crest_ewin=args.crest_ewin,
        xtb_exec=args.xtb,
        crest_exec=args.crest,
        g16_exec=args.g16,
        charge=args.charge,
        multiplicity=args.mult,
        solvent=args.solvent,
        preopt_method=args.preopt_method,
        skip_preopt=args.skip_preopt,
        basis_set=args.basis,
        method=args.method,
        operation=args.operation,
        nproc=args.nproc,
        mem=args.mem,
        cosmo=args.cosmo,
        timeout=args.timeout,
        scratch_dir=g16.get_scratch_dir(args.scratch),
        max_disk=args.max_disk,
        extra_keywords=args.extra_keywords,
    )

    print_result(result)

    out_dir = Path(args.workdir) / result.cid
    write_summary(result, out_dir)

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())