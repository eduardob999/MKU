"""Shared regular expressions."""

import re

# Matches a thermochemistry keyword closely followed by a numeric value and a
# molar energy unit. Shared by the NIST and PubMed clients.
THERMO_REGEX = re.compile(
    r"((?:ΔH|Delta H|enthalpy|ΔS|Delta S|entropy|ΔG|Delta G|Gibbs free energy|free energy"
    r"|heat of formation|heat of combustion)[^\.\n]{0,120}?"
    r"[-+]?\d+(?:\.\d+)?\s*(?:kJ/mol|kJ mol-1|J/mol|J mol-1|kcal/mol|kcal mol-1"
    r"|cal/mol|cal mol-1|kcal per mol|kJ per mol))",
    re.IGNORECASE,
)
