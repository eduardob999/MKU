"""Tests for the Gaussian 16 log parsers in ivette.module.gaussian16_core.

Covers SCF energy, thermochemistry, optimisation-step convergence, normal
termination, and the .gjf route-line builder (including the COSMO/charge path).
"""

import pytest

from ivette.module import gaussian16_core as g16


def test_get_final_scf_energy_returns_last_cycle(tmp_path, freq_log_text):
    log = tmp_path / "m_freq.log"
    log.write_text(freq_log_text)
    assert g16.get_final_scf_energy(str(log)) == pytest.approx(-154.062345678)


def test_get_thermo_data_parses_temperature_zpe_and_entropy(tmp_path, freq_log_text):
    log = tmp_path / "m_freq.log"
    log.write_text(freq_log_text)
    thermo = g16.get_thermo_data(str(log))

    assert thermo.temp == pytest.approx(298.150)
    # ZPE is converted Hartree -> kcal/mol (×627.509474) inside the parser.
    assert thermo.zpe == pytest.approx(0.123456 * 627.509474, rel=1e-6)
    assert thermo.ts == pytest.approx(75.000)   # total entropy, cal/mol·K
    assert thermo.cv == pytest.approx(20.000)


def test_get_opt_steps_returns_last_step(tmp_path, opt_log_text):
    log = tmp_path / "m_opt.log"
    log.write_text(opt_log_text)
    step = g16.get_opt_steps(str(log), step_index=-1)

    assert step is not None
    assert step.step == 2
    assert step.energy == pytest.approx(-154.062000)
    assert step.max_force == pytest.approx(0.000400)
    assert step.rms_force == pytest.approx(0.000100)
    assert step.max_disp == pytest.approx(0.001000)
    assert step.delta_e == pytest.approx(-154.062000 - (-154.050000))


def test_get_opt_steps_none_when_no_steps(tmp_path):
    log = tmp_path / "empty_opt.log"
    log.write_text("no convergence table here\n")
    assert g16.get_opt_steps(str(log)) is None


def test_check_normal_termination(tmp_path, freq_log_text):
    ok = tmp_path / "ok.log"
    bad = tmp_path / "bad.log"
    ok.write_text(freq_log_text)
    bad.write_text("Error termination via Lnk1e in /g16/l502.exe\n")
    assert g16.check_normal_termination(str(ok)) is True
    assert g16.check_normal_termination(str(bad)) is False


def test_build_gjf_cosmo_and_charge_route_line():
    gjf = g16.build_gjf(
        "C   0.0 0.0 0.0", "/tmp/job.chk",
        basis_set="6-31G*", method="PBE0", operation="opt",
        charge=-1, multiplicity=2, cosmo=True,
    )
    assert "#p PBE0/6-31G* opt scrf=(cpcm,solvent=water)" in gjf
    assert "\n-1 2\n" in gjf            # charge / multiplicity line
    assert "%chk=/tmp/job.chk" in gjf


def test_build_gjf_gas_phase_has_no_scrf():
    gjf = g16.build_gjf("C 0 0 0", "/tmp/job.chk", cosmo=False)
    assert "scrf" not in gjf.lower()
    assert "\n0 1\n" in gjf


def test_build_gjf_appends_extra_keywords():
    gjf = g16.build_gjf("C 0 0 0", "/tmp/job.chk", method="PBE0",
                        basis_set="6-31G*", operation="opt",
                        extra_keywords="NoTestMO SCF=(XQC,MaxCycle=200)")
    assert "#p PBE0/6-31G* opt NoTestMO SCF=(XQC,MaxCycle=200)" in gjf
