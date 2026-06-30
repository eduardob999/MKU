"""Tests for the DFT freq-log descriptor parser (ivette.core.parse_dft).

This parser turns the frequency logs you're waiting on into ML features, so its
field extraction and the dedup-by-CID behaviour are worth pinning down.
"""

import os

import pytest

from ivette.core.parse_dft import (
    parse_freq_log,
    parse_geometry_descriptors,
    parse_redox_descriptors,
)


def test_parse_freq_log_extracts_all_descriptors(tmp_path, freq_log_text):
    log = tmp_path / "12345_freq.log"
    log.write_text(freq_log_text)
    row = parse_freq_log(log)

    assert row is not None
    assert row["scf_energy"] == pytest.approx(-154.062345678)   # last SCF wins
    assert row["zpe_correction"] == pytest.approx(0.123456)
    assert row["thermal_corr_E"] == pytest.approx(0.131234)
    assert row["thermal_corr_H"] == pytest.approx(0.132178)
    assert row["thermal_corr_G"] == pytest.approx(0.098765)
    assert row["E_plus_zpe"] == pytest.approx(-154.123456)
    assert row["enthalpy_H"] == pytest.approx(-154.110000)
    assert row["gibbs_G"] == pytest.approx(-154.150000)
    assert row["entropy_S"] == pytest.approx(75.000)
    assert row["temperature"] == pytest.approx(298.150)
    assert row["n_imaginary"] == 0
    assert row["lowest_freq"] == pytest.approx(100.0)
    assert row["n_modes"] == 6


def test_parse_freq_log_counts_imaginary_modes(tmp_path, freq_log_imaginary_text):
    log = tmp_path / "999_freq.log"
    log.write_text(freq_log_imaginary_text)
    row = parse_freq_log(log)

    assert row["n_imaginary"] == 1
    assert row["lowest_freq"] == pytest.approx(-50.0)
    assert row["n_modes"] == 3


def test_parse_freq_log_returns_none_without_normal_termination(tmp_path, freq_log_text):
    # Strip the terminator → an incomplete/crashed job must not yield a row.
    crashed = freq_log_text.replace("Normal termination of Gaussian 16 at Tue Jun 25 12:00:00 2026.", "")
    log = tmp_path / "1_freq.log"
    log.write_text(crashed)
    assert parse_freq_log(log) is None


def test_parse_geometry_descriptors_dedups_by_cid_keeping_newest(tmp_path, freq_log_text):
    # Same CID computed twice in different subdirs; the newer log must win.
    old_dir = tmp_path / "run_a"
    new_dir = tmp_path / "run_b"
    old_dir.mkdir()
    new_dir.mkdir()

    old = old_dir / "555_freq.log"
    new = new_dir / "555_freq.log"
    old.write_text(freq_log_text.replace("-154.062345678", "-154.000000000"))
    new.write_text(freq_log_text.replace("-154.062345678", "-154.999999999"))

    # Force the second file to be the most-recently modified.
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))

    rows = parse_geometry_descriptors(tmp_path)
    assert len(rows) == 1
    assert rows[0]["CID"] == "555"
    assert rows[0]["scf_energy"] == pytest.approx(-154.999999999)


def test_parse_redox_descriptors_neutral_anion_delta(tmp_path, freq_log_text):
    cosmo = tmp_path / "opt_then_freq_COSMO"
    (cosmo / "neutral" / "555").mkdir(parents=True)
    (cosmo / "anion" / "555").mkdir(parents=True)
    (cosmo / "neutral" / "555" / "555_freq.log").write_text(freq_log_text)
    # Shift the anion's Gibbs free energy so the delta is nonzero and known.
    anion_text = freq_log_text.replace("-154.150000", "-154.100000")
    (cosmo / "anion" / "555" / "555_freq.log").write_text(anion_text)

    rows = parse_redox_descriptors(cosmo)
    assert len(rows) == 1
    r = rows[0]
    assert r["CID"] == "555"
    assert r["neutral_gibbs_G"] == pytest.approx(-154.150000)
    assert r["anion_gibbs_G"] == pytest.approx(-154.100000)
    # ΔG of reduction = anion - neutral
    assert r["delta_gibbs_G"] == pytest.approx(0.050000, abs=1e-6)
    # neutral/anion present for all descriptors; delta present for thermo ones
    for col in ("neutral_enthalpy_H", "anion_enthalpy_H", "delta_enthalpy_H",
                "neutral_entropy_S", "anion_entropy_S", "delta_entropy_S"):
        assert col in r


def test_parse_redox_requires_both_states(tmp_path, freq_log_text):
    cosmo = tmp_path / "opt_then_freq_COSMO"
    (cosmo / "neutral" / "555").mkdir(parents=True)
    (cosmo / "neutral" / "555" / "555_freq.log").write_text(freq_log_text)
    # No anion computed → no redox row (a delta needs both states).
    assert parse_redox_descriptors(cosmo) == []
