import json
import os

from ..vectorstore import seed_framework_chunks

_DIR = os.path.dirname(__file__)

_FILES = {
    "NIST_CSF": "nist_csf.json",
    "OWASP_TOP10": "owasp_top10.json",
    "CIS_CONTROLS": "cis_controls.json",
    "MITRE_ATTACK": "mitre_attack.json",
    "CIS_BENCHMARKS": "cis_benchmarks.json",
    "ISO_27001": "iso_27001.json",
    "PCI_DSS": "pci_dss.json",
    "SOC_2": "soc2_tsc.json",
    "GDPR": "gdpr.json",
    "HIPAA": "hipaa_security_rule.json",
    "NIST_800-53": "nist_800_53.json",
    "CWE_TOP25": "cwe_top25.json",
}


def seed_all_frameworks() -> None:
    for framework, filename in _FILES.items():
        path = os.path.join(_DIR, filename)
        with open(path) as f:
            entries = json.load(f)
        seed_framework_chunks(framework, entries)


FRAMEWORK_NAMES = list(_FILES.keys())
