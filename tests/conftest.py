"""Shared Gaussian-log fixtures for parser tests.

Minimal but format-faithful snippets of real Gaussian 16 output, sized to
exercise every regex in the freq/opt parsers without shipping multi-MB logs.
"""

import pytest

# A completed frequency job: two SCF cycles (last one wins), six real modes,
# a full thermochemistry block, and Normal termination.
_FREQ_LOG = """\
 Entering Gaussian System, Link 0
 SCF Done:  E(RB3LYP) =  -154.050000000     A.U. after   10 cycles
 (geometry optimisation continues)
 SCF Done:  E(RB3LYP) =  -154.062345678     A.U. after   12 cycles

 Harmonic frequencies (cm**-1)
 Frequencies --    100.0000   200.0000   300.0000
 Red. masses --      1.0000     2.0000     3.0000
 Frequencies --    400.0000   500.0000   600.0000
 Red. masses --      1.0000     2.0000     3.0000

 Zero-point correction=                           0.123456 (Hartree/Particle)
 Thermal correction to Energy=                    0.131234
 Thermal correction to Enthalpy=                  0.132178
 Thermal correction to Gibbs Free Energy=         0.098765
 Sum of electronic and zero-point Energies=         -154.123456
 Sum of electronic and thermal Energies=            -154.118000
 Sum of electronic and thermal Enthalpies=          -154.110000
 Sum of electronic and thermal Free Energies=       -154.150000

 Temperature   298.150 Kelvin.  Pressure   1.00000 Atm.

                    E (Thermal)             CV                S
                     KCal/Mol        Cal/Mol-Kelvin    Cal/Mol-Kelvin
 Total                   55.000             20.000            75.000
 Electronic              0.000              0.000             0.000
 Translational           0.889              2.981            40.000
 Rotational              0.889              2.981            25.000
 Vibrational            53.000             14.000            10.000

 Normal termination of Gaussian 16 at Tue Jun 25 12:00:00 2026.
"""

# Same job but with one imaginary mode (negative frequency) — a saddle point.
_FREQ_LOG_IMAGINARY = """\
 SCF Done:  E(RB3LYP) =  -154.000000000     A.U. after   11 cycles
 Frequencies --    -50.0000   100.0000   200.0000
 Red. masses --      1.0000     2.0000     3.0000
 Zero-point correction=                           0.100000 (Hartree/Particle)
 Temperature   298.150 Kelvin.  Pressure   1.00000 Atm.
 Normal termination of Gaussian 16.
"""

# A two-step optimisation: step 1 not converged, step 2 converged.
_OPT_LOG = """\
 SCF Done:  E(RB3LYP) =  -154.050000     A.U. after   10 cycles
         Item               Value     Threshold  Converged?
 Maximum Force            0.010000     0.000450     NO
 RMS     Force            0.008000     0.000300     NO
 Maximum Displacement     0.040000     0.001800     NO
 RMS     Displacement     0.030000     0.001200     NO
 SCF Done:  E(RB3LYP) =  -154.062000     A.U. after   12 cycles
         Item               Value     Threshold  Converged?
 Maximum Force            0.000400     0.000450     YES
 RMS     Force            0.000100     0.000300     YES
 Maximum Displacement     0.001000     0.001800     YES
 RMS     Displacement     0.000400     0.001200     YES
 Normal termination of Gaussian 16.
"""


@pytest.fixture
def freq_log_text():
    return _FREQ_LOG


@pytest.fixture
def freq_log_imaginary_text():
    return _FREQ_LOG_IMAGINARY


@pytest.fixture
def opt_log_text():
    return _OPT_LOG
