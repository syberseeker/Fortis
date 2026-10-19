"""Single source of truth for framework-name canonicalization.

Shared by the report schema parser (core.reports.schema) and the mapping
validator (core.report_service) so that every alias spelling the parser
canonicalizes is also recognized by the validator. Canonical targets are the
seeded corpus framework ids (core/frameworks/__init__.py _FILES keys).
"""
from typing import Optional

FRAMEWORK_ALIASES: dict[str, str] = {
    "NIST CSF": "NIST_CSF",
    "NIST_CSF": "NIST_CSF",
    "OWASP": "OWASP_TOP10",
    "OWASP TOP 10": "OWASP_TOP10",
    "OWASP_TOP10": "OWASP_TOP10",
    "CIS CONTROLS": "CIS_CONTROLS",
    "CIS": "CIS_CONTROLS",
    "CIS_CONTROLS": "CIS_CONTROLS",
    "MITRE ATT&CK": "MITRE_ATTACK",
    "MITRE ATT&CK (ENTERPRISE)": "MITRE_ATTACK",
    "MITRE": "MITRE_ATTACK",
    "MITRE_ATTACK": "MITRE_ATTACK",
    "CIS BENCHMARKS": "CIS_BENCHMARKS",
    "CIS_BENCHMARKS": "CIS_BENCHMARKS",
    "ISO 27001": "ISO_27001",
    "ISO27001": "ISO_27001",
    "ISO/IEC 27001": "ISO_27001",
    "ISO_27001": "ISO_27001",
    "PCI DSS": "PCI_DSS",
    "PCI-DSS": "PCI_DSS",
    "PCI_DSS": "PCI_DSS",
    "SOC 2": "SOC_2",
    "SOC2": "SOC_2",
    "SOC_2": "SOC_2",
    "GDPR": "GDPR",
    "HIPAA": "HIPAA",
    "NIST 800-53": "NIST_800-53",
    "NIST SP 800-53": "NIST_800-53",
    "NIST_800-53": "NIST_800-53",
    "CWE": "CWE_TOP25",
    "CWE TOP 25": "CWE_TOP25",
    "CWE_TOP25": "CWE_TOP25",
}


def canonical_framework(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    stripped = name.strip()
    if not stripped:
        return None
    return FRAMEWORK_ALIASES.get(stripped.upper(), stripped)
