from typing import List, Optional
from pydantic import BaseModel, Field


class Finding(BaseModel):
    title: str
    severity: str = Field(description="Critical | High | Medium | Low | Informational")
    description: str
    evidence: str = Field(description="Quoted or paraphrased evidence from the source document")
    source_file: Optional[str] = None
    framework: Optional[str] = Field(default=None, description="e.g. NIST_CSF, OWASP_TOP10, CIS_CONTROLS")
    control_id: Optional[str] = None
    remediation: str


class SecurityReport(BaseModel):
    title: str
    client_context: str = Field(description="One-paragraph summary of what was reviewed")
    scope: str
    executive_summary: str
    findings: List[Finding]
    overall_risk_rating: str
    recommendations_summary: List[str]

SEVERITY_ORDER = ["Critical", "High", "Medium", "Low", "Informational"]
