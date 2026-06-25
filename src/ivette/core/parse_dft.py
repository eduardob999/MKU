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


def parse_geometry_descriptors(gaussian_root):
    """Per-CID descriptor rows for every completed freq log under ``gaussian_root``.

    Deduplicates by CID, keeping the most recently modified frequency log (so a
    re-run or a different operation supersedes the older result).
    """
    by_cid = {}
    logs = sorted(Path(gaussian_root).rglob("*_freq.log"),
                  key=lambda p: p.stat().st_mtime)
    for log in logs:
        cid = log.stem.replace("_freq", "")
        row = parse_freq_log(log)
        if row is None:
            continue
        by_cid[cid] = {"CID": cid, **{c: row.get(c) for c in DESCRIPTOR_COLUMNS}}
    return list(by_cid.values())
