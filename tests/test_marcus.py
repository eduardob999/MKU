"""Marcus reorganization energy / activation energy + pair discovery."""

import pytest

from ivette.core.marcus import (
    HARTREE_TO_EV,
    activation_energy,
    available_pairs,
    cosmo_level,
    reorganization_energy,
)

_NORMAL = "Normal termination of Gaussian 16"


def test_reorganization_energy_sums_both_contributions():
    # λ = (E_R^OG − E_R^RG) + (E_O^RG − E_O^OG)
    lam, lam1, lam2 = reorganization_energy(
        e_r_og=-10.0, e_r_rg=-10.1,    # λ¹ = +0.1
        e_o_rg=-9.0, e_o_og=-9.05,     # λ² = +0.05
    )
    assert lam1 == pytest.approx(0.1)
    assert lam2 == pytest.approx(0.05)
    assert lam == pytest.approx(0.15)


def test_reorganization_energy_missing_returns_none():
    assert reorganization_energy(None, -1.0, -1.0, -1.0) == (None, None, None)


def test_activation_energy_matches_marcus_formula():
    # ΔG‡ = (λ + ΔG°)² / (4λ); λ=0.2, ΔG°=-0.1 → (0.1)²/0.8 = 0.0125
    assert activation_energy(0.2, -0.1) == pytest.approx(0.0125)
    # At ΔG° = −λ the barrier vanishes (activationless).
    assert activation_energy(0.2, -0.2) == pytest.approx(0.0)
    # Non-positive λ is unphysical → None (avoids divide-by-zero).
    assert activation_energy(0.0, -0.1) is None
    assert activation_energy(None, -0.1) is None


def test_available_pairs_requires_both_completed_states(tmp_path):
    cosmo = tmp_path / "opt_then_freq_COSMO"
    for state in ("neutral", "anion"):
        for cid in ("100", "200"):
            d = cosmo / state / cid
            d.mkdir(parents=True)
            (d / f"{cid}_freq.log").write_text(_NORMAL)
    # 300 only finished as neutral → excluded; 200 anion log is empty → excluded.
    (cosmo / "neutral" / "300").mkdir(parents=True)
    (cosmo / "neutral" / "300" / "300_freq.log").write_text(_NORMAL)
    (cosmo / "anion" / "200" / "200_freq.log").write_text("crashed early")

    assert available_pairs(cosmo) == ["100"]


def test_available_pairs_empty_when_root_missing(tmp_path):
    assert available_pairs(tmp_path / "nope") == []


def test_cosmo_level_reads_route_line_and_skips_preopt(tmp_path):
    cosmo = tmp_path / "opt_then_freq_COSMO"
    cell = cosmo / "neutral" / "100"
    (cell / "preopt" / "pm7").mkdir(parents=True)
    # Preopt route has no method/basis "/" — must be ignored.
    (cell / "preopt" / "pm7" / "100_preopt.gjf").write_text("#p PM7 opt=(MaxCycles=500)\n")
    (cell / "100_freq.gjf").write_text(
        "%chk=100.chk\n#p PBE0/6-311G freq scrf=(cpcm,solvent=water)\n\ntitle\n")
    assert cosmo_level(cosmo) == ("PBE0", "6-311G")


def test_cosmo_level_none_when_no_route(tmp_path):
    assert cosmo_level(tmp_path / "missing") is None
