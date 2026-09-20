from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

SeverityLevel = Literal["Critical", "High", "Medium", "Low", "Informational"]
FrameworkName = Literal["NIST_CSF", "OWASP_TOP10", "CIS_CONTROLS"]


class Finding(BaseModel):
    title: str
    severity: SeverityLevel
    description: str
    evidence: str = Field(description="Quoted or paraphrased evidence from the source document")
    source_file: Optional[str] = None
    framework: Optional[FrameworkName] = None
    control_id: Optional[str] = None
    remediation: str


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
