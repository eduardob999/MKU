"""Parse DFT/thermochemistry descriptors from a geometry set's Gaussian freq logs.

Produces a per-compound (CID-indexed) table of quantum-chemical properties — the
parsed results of the frequency calculation — suitable as extra ML features.
One row per molecule that has a completed frequency calculation; the values are
left raw so a later training step can left-join them onto the existing features.
"""

from __future__ import annotations

import re
from pathlib import Path

_SCF_RE = re.compile(r"SCF Done:\s+E\(\S+\)\s+=\s+([-\d.]+)")
_FREQ_RE = re.compile(r"Frequencies --\s+([-\d.\s]+)")

# Order = column order in the output table.
DESCRIPTOR_COLUMNS = [
    "scf_energy",        # Hartree, final SCF energy
    "zpe_correction",    # Hartree, zero-point energy correction
    "thermal_corr_E",    # Hartree, thermal correction to energy
    "thermal_corr_H",    # Hartree, thermal correction to enthalpy
    "thermal_corr_G",    # Hartree, thermal correction to Gibbs free energy
    "E_plus_zpe",        # Hartree, sum of electronic and zero-point energies
    "enthalpy_H",        # Hartree, sum of electronic and thermal enthalpies
    "gibbs_G",           # Hartree, sum of electronic and thermal free energies
    "entropy_S",         # cal/mol·K, total entropy
    "temperature",       # K
    "n_imaginary",       # count of imaginary (negative) frequencies
    "lowest_freq",       # cm^-1, lowest frequency (negative => saddle point)
    "n_modes",           # number of vibrational modes found
]


def _grab(text, label):
    m = re.search(re.escape(label) + r"\s*([-\d.]+)", text)
    return float(m.group(1)) if m else None


def _frequencies(text):
    out = []
    for m in _FREQ_RE.finditer(text):
        for tok in m.group(1).split():
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out


def _entropy(text):
    # Thermochemistry table: "Total <E_thermal> <Cv> <S>"  (S is the 3rd column)
    m = re.search(r"^\s*Total\s+[\d.]+\s+[\d.]+\s+([\d.]+)", text, re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_freq_log(path):
    """Parse one ``*_freq.log`` into a descriptor dict, or None if incomplete."""
    text = Path(path).read_text(errors="replace")
    if "Normal termination" not in text:
        return None
    scf = [float(m.group(1)) for m in _SCF_RE.finditer(text)]
    freqs = _frequencies(text)
    return {
        "scf_energy": scf[-1] if scf else None,
        "zpe_correction": _grab(text, "Zero-point correction="),
        "thermal_corr_E": _grab(text, "Thermal correction to Energy="),
        "thermal_corr_H": _grab(text, "Thermal correction to Enthalpy="),
        "thermal_corr_G": _grab(text, "Thermal correction to Gibbs Free Energy="),
        "E_plus_zpe": _grab(text, "Sum of electronic and zero-point Energies="),
        "enthalpy_H": _grab(text, "Sum of electronic and thermal Enthalpies="),
        "gibbs_G": _grab(text, "Sum of electronic and thermal Free Energies="),
        "entropy_S": _entropy(text),
        "temperature": _grab(text, "Temperature"),
        "n_imaginary": sum(1 for f in freqs if f < 0),
        "lowest_freq": min(freqs) if freqs else None,
        "n_modes": len(freqs),
    }


def _descriptors_by_cid(root):
    """``{cid: {descriptor: value}}`` for every completed freq log under ``root``.

    Deduplicates by CID, keeping the most recently modified frequency log (so a
    re-run or a different operation supersedes the older result).
    """
    by_cid = {}
    for log in sorted(Path(root).rglob("*_freq.log"), key=lambda p: p.stat().st_mtime):
        cid = log.stem.replace("_freq", "")
        row = parse_freq_log(log)
        if row is None:
            continue
        by_cid[cid] = {c: row.get(c) for c in DESCRIPTOR_COLUMNS}
    return by_cid


def parse_geometry_descriptors(gaussian_root):
    """Per-CID descriptor rows for every completed freq log under ``gaussian_root``."""
    return [{"CID": cid, **row} for cid, row in _descriptors_by_cid(gaussian_root).items()]


# Descriptors for which an anion - neutral difference is physically meaningful.
# delta_gibbs_G = Gibbs energy of reduction; delta_enthalpy_H, delta_entropy_S
# similarly; delta_scf_energy ≈ adiabatic electron affinity (in solution).
REDOX_DELTA_COLUMNS = [
    "scf_energy", "zpe_correction", "thermal_corr_E", "thermal_corr_H",
    "thermal_corr_G", "E_plus_zpe", "enthalpy_H", "gibbs_G", "entropy_S",
]


def parse_redox_descriptors(cosmo_root, *, neutral_label="neutral", anion_label="anion"):
    """Per-CID redox features from a COSMO neutral+anion run.

    ``cosmo_root`` is the ``.../gaussian/opt_then_freq_COSMO`` directory holding
    ``neutral/`` and ``anion/`` subtrees. For every compound completed in *both*
    states, emit one row with ``neutral_<d>`` and ``anion_<d>`` for each
    descriptor plus ``delta_<d> = anion - neutral`` for the thermodynamic ones
    (the reduction enthalpy/entropy/Gibbs/electron-affinity terms).
    """
    root = Path(cosmo_root)
    neutral = _descriptors_by_cid(root / neutral_label)
    anion = _descriptors_by_cid(root / anion_label)

    rows = []
    for cid in sorted(set(neutral) & set(anion)):
        n, a = neutral[cid], anion[cid]
        row = {"CID": cid}
        for c in DESCRIPTOR_COLUMNS:
            row[f"neutral_{c}"] = n.get(c)
            row[f"anion_{c}"] = a.get(c)
        for c in REDOX_DELTA_COLUMNS:
            nv, av = n.get(c), a.get(c)
            row[f"delta_{c}"] = (av - nv) if (nv is not None and av is not None) else None
        rows.append(row)
    return rows
