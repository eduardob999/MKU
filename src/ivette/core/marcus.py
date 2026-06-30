"""Marcus-theory reorganization energy and activation free energy.

Built to **reuse** the optimized geometries from a COSMO neutral+anion
``opt then freq`` run — it never repeats the expensive opt/freq jobs. For each
compound finished in both states it runs only the two Marcus *cross* single
points (the non-equilibrium "vertical" energies); the two equilibrium energies
and the reduction free energy are read straight from the existing freq logs.

Notation (O = oxidized = neutral, R = reduced = anion; superscript = the geometry
the energy is evaluated at, OG = oxidized-optimized, RG = reduced-optimized):

    λ¹ = E_R^OG − E_R^RG        reduced state, relaxing from the oxidized geometry
    λ² = E_O^RG − E_O^OG        oxidized state, relaxing from the reduced geometry
    λ  = λ¹ + λ²                reorganization energy
    ΔG‡ = (λ + ΔG°_red)² / (4λ) Marcus activation free energy

``E_R^RG`` (anion freq log) and ``E_O^OG`` (neutral freq log) are free from the
prior run; ``E_R^OG`` and ``E_O^RG`` are the two single points computed here, at
the same level of theory (COSMO) so all four energies are comparable.
"""

from __future__ import annotations

from pathlib import Path

from ivette.core.parse_dft import parse_freq_log
from ivette.module import gaussian16_core as g16

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.509474

# Numeric per-CID columns suitable as ML features (everything but CID/error).
MARCUS_FEATURE_COLUMNS = [
    "reorg_energy_eV", "reorg_energy_kcal",
    "lambda1_eV", "lambda2_eV",
    "dG_reduction_eV", "activation_energy_eV", "activation_energy_kcal",
]


def _completed_freq_log(state_dir, cid):
    """The optimized-geometry freq log for ``cid`` under a charge-state dir.

    Prefers the split ``*_freq.log`` (its Standard-orientation block is the
    optimized geometry and its SCF Done is the equilibrium energy); falls back to
    a combined ``*.log``. Returns ``None`` unless the job terminated normally.
    """
    base = Path(state_dir) / cid
    for name in (f"{cid}_freq.log", f"{cid}.log", f"{cid}_opt.log"):
        log = base / name
        if log.exists() and g16.check_normal_termination(str(log)):
            return log
    return None


def available_pairs(cosmo_root, *, oxidized="neutral", reduced="anion"):
    """Sorted CIDs that completed in *both* charge states (ready for Marcus)."""
    root = Path(cosmo_root)
    ox, red = root / oxidized, root / reduced
    if not ox.exists() or not red.exists():
        return []
    common = ({p.name for p in ox.iterdir() if p.is_dir()}
              & {p.name for p in red.iterdir() if p.is_dir()})
    return [cid for cid in sorted(common)
            if _completed_freq_log(ox, cid) and _completed_freq_log(red, cid)]


def _single_point(geom_log, work_dir, cid, *, charge, multiplicity, settings):
    """Run (or resume) a COSMO single point at the geometry in ``geom_log``.

    Returns ``(energy_hartree | None, log_path, error)``. If a normally
    terminated log already exists it is parsed and returned without recomputing,
    so re-invoking the feature never repeats a single point.
    """
    work = Path(work_dir) / cid
    work.mkdir(parents=True, exist_ok=True)
    gjf = str(work / f"{cid}_sp.gjf")
    chk = str(work / f"{cid}_sp.chk")
    log = str(work / f"{cid}_sp.log")

    if Path(log).exists() and g16.check_normal_termination(log):
        return g16.get_final_scf_energy(log), log, ""

    xyz = g16.log_to_xyz(str(geom_log))   # last Standard orientation = optimized geom
    try:
        g16.xyz_to_gjf(
            xyz, gjf, chk,
            basis_set=settings["basis_set"], method=settings["method"],
            operation="sp", charge=charge, multiplicity=multiplicity,
            nproc=settings["nproc"], mem=settings["mem"], cosmo=True,
            title=f"CID {cid} Marcus SP (q={charge}, m={multiplicity})",
            extra_keywords=settings.get("extra_keywords", ""),
        )
    except Exception as exc:
        Path(xyz).unlink(missing_ok=True)
        return None, log, f"GJF build failed: {exc}"
    Path(xyz).unlink(missing_ok=True)

    ok, err = g16.run_gaussian(gjf, log, g16_exec=settings.get("g16_exec", "g16"),
                               timeout=settings.get("timeout"))
    if not ok or not g16.check_normal_termination(log):
        return None, log, err or "single point did not terminate normally"
    return g16.get_final_scf_energy(log), log, ""


def reorganization_energy(e_r_og, e_r_rg, e_o_rg, e_o_og):
    """``(λ, λ¹, λ²)`` from the four energies (same units in, same units out)."""
    if None in (e_r_og, e_r_rg, e_o_rg, e_o_og):
        return None, None, None
    lam1 = e_r_og - e_r_rg
    lam2 = e_o_rg - e_o_og
    return lam1 + lam2, lam1, lam2


def activation_energy(lam, dg_red):
    """Marcus ΔG‡ = (λ + ΔG°)² / (4λ), in the inputs' units. ``None`` if λ ≤ 0."""
    if lam is None or dg_red is None or lam <= 0:
        return None
    return (lam + dg_red) ** 2 / (4.0 * lam)


import re
from concurrent.futures import ProcessPoolExecutor, as_completed

_ROUTE_RE = re.compile(r"#\S*\s+(\S+)/(\S+)")


def cosmo_level(cosmo_root):
    """``(method, basis_set)`` parsed from a COSMO run's route line, or ``None``.

    Lets Marcus default its single points to the *same* level of theory the
    geometries were optimized at (read straight from the .gjf, so it's exact).
    Preopt inputs (PM7, no ``method/basis``) are skipped.
    """
    root = Path(cosmo_root)
    for pattern in ("*_freq.gjf", "*_opt.gjf"):
        for gjf in sorted(root.rglob(pattern)):
            if "preopt" in gjf.parts:
                continue
            for line in gjf.read_text(errors="replace").splitlines():
                if line.lstrip().startswith("#"):
                    m = _ROUTE_RE.search(line)
                    if m:
                        return m.group(1), m.group(2)
    return None


def _run_sp_pair(task):
    """Worker: the two cross single points for one compound (picklable, top-level)."""
    (cid, ox_log, red_log, red_at_ox_dir, ox_at_red_dir,
     red_q, red_m, ox_q, ox_m, settings) = task
    e_r_og, _, err1 = _single_point(ox_log, red_at_ox_dir, cid,
                                    charge=red_q, multiplicity=red_m, settings=settings)
    e_o_rg, _, err2 = _single_point(red_log, ox_at_red_dir, cid,
                                    charge=ox_q, multiplicity=ox_m, settings=settings)
    return cid, e_r_og, e_o_rg, (err1 or err2) or ""


def compute_marcus(cosmo_root, *, settings, oxidized=("neutral", 0, 1),
                   reduced=("anion", -1, 2), jobs=1, progress=None):
    """Per-CID Marcus reorganization energy + activation energy.

    ``cosmo_root`` is the ``.../gaussian/opt_then_freq_COSMO`` directory of a
    completed neutral+anion run. ``oxidized`` / ``reduced`` are
    ``(label, charge, multiplicity)`` triples matching that run's charge states.
    ``settings`` carries the level of theory + resources for the single points
    (``method``, ``basis_set``, ``nproc``, ``mem``, ``extra_keywords``,
    ``timeout``, ``g16_exec``). ``jobs`` runs that many compounds' single points
    in parallel (like the opt+freq batch). The cross single points land in a
    sibling ``marcus/`` directory and resume. ``progress(cid)`` is called per
    compound as it completes.

    Returns ``(rows, skipped)`` — one feature dict per processed CID (with an
    ``error`` key, empty on success) and the CIDs lacking a completed pair.
    """
    cosmo_root = Path(cosmo_root)
    ox_label, ox_q, ox_m = oxidized
    red_label, red_q, red_m = reduced
    ox_dir, red_dir = cosmo_root / ox_label, cosmo_root / red_label

    marcus_root = cosmo_root.parent / "marcus"
    red_at_ox_dir = marcus_root / "reduced_at_oxidized_geom"   # → E_R^OG
    ox_at_red_dir = marcus_root / "oxidized_at_reduced_geom"   # → E_O^RG

    # Equilibrium energies + reduction ΔG come for free from the existing freq
    # logs (fast, main process); only the two cross single points are computed.
    prelim, tasks, skipped = {}, [], []
    for cid in available_pairs(cosmo_root, oxidized=ox_label, reduced=red_label):
        ox_log = _completed_freq_log(ox_dir, cid)
        red_log = _completed_freq_log(red_dir, cid)
        if ox_log is None or red_log is None:
            skipped.append(cid)
            continue
        prelim[cid] = (
            g16.get_final_scf_energy(str(ox_log)),                 # E_O^OG
            g16.get_final_scf_energy(str(red_log)),                # E_R^RG
            (parse_freq_log(ox_log) or {}).get("gibbs_G"),         # G_ox
            (parse_freq_log(red_log) or {}).get("gibbs_G"),        # G_red
        )
        tasks.append((cid, str(ox_log), str(red_log), str(red_at_ox_dir),
                      str(ox_at_red_dir), red_q, red_m, ox_q, ox_m, settings))

    def _ev(x):
        return x * HARTREE_TO_EV if x is not None else None

    rows = []

    def _finish(cid, e_r_og, e_o_rg, err):
        e_o_og, e_r_rg, g_ox, g_red = prelim[cid]
        lam, lam1, lam2 = reorganization_energy(e_r_og, e_r_rg, e_o_rg, e_o_og)
        dg_red = (g_red - g_ox) if (g_red is not None and g_ox is not None) else None
        dg_act = activation_energy(lam, dg_red)
        rows.append({
            "CID": cid,
            "reorg_energy_eV": _ev(lam),
            "reorg_energy_kcal": lam * HARTREE_TO_KCAL if lam is not None else None,
            "lambda1_eV": _ev(lam1),
            "lambda2_eV": _ev(lam2),
            "dG_reduction_eV": _ev(dg_red),
            "activation_energy_eV": _ev(dg_act),
            "activation_energy_kcal": dg_act * HARTREE_TO_KCAL if dg_act is not None else None,
            "error": err,
        })
        if progress:
            progress(cid)

    if jobs and jobs > 1 and tasks:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_run_sp_pair, t) for t in tasks]
            for fut in as_completed(futures):
                _finish(*fut.result())
    else:
        for t in tasks:
            _finish(*_run_sp_pair(t))

    rows.sort(key=lambda r: str(r["CID"]))
    return rows, skipped
