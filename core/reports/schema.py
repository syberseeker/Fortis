from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

SeverityLevel = Literal["Critical", "High", "Medium", "Low", "Informational"]

# Every framework shipped in the seed corpus (core/frameworks/_FILES). The
# model may legally cite any of these; renderers chart framework names.
FrameworkName = str

_FRAMEWORK_ALIASES = {
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


def _normalize_framework(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    canonical = _FRAMEWORK_ALIASES.get(stripped.upper())
    if canonical:
        return canonical
    return stripped


class Finding(BaseModel):
    title: str
    severity: SeverityLevel
    description: str
    evidence: str = Field(description="Quoted or paraphrased evidence from the source document")
    source_file: Optional[str] = None
    framework: Optional[str] = None
    control_id: Optional[str] = None
    remediation: str

    @field_validator("framework", mode="before")
    @classmethod
    def _canon_framework(cls, v):
        return _normalize_framework(v)


class SecurityReport(BaseModel):
    title: str
    client_context: str = Field(description="One-paragraph summary of what was reviewed")
    scope: str
    executive_summary: str
    findings: List[Finding]
    overall_risk_rating: SeverityLevel
    recommendations_summary: List[str]
    diagram: Optional[str] = Field(default=None, description="Optional mermaid flowchart source describing the client architecture or data flow")

SEVERITY_ORDER = ["Critical", "High", "Medium", "Low", "Informational"]


def severity_counts(report: SecurityReport) -> Dict[str, int]:
    counts = {sev: 0 for sev in SEVERITY_ORDER}
    for finding in report.findings:
        if finding.severity in counts:
            counts[finding.severity] += 1
    return counts


def framework_counts(report: SecurityReport) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for finding in report.findings:
        if finding.framework:
            counts[finding.framework] = counts.get(finding.framework, 0) + 1
    return counts
