"""
title: Fortis Finding Validator
author: fortis
version: 0.1.0
description: >
  Validates and structures security findings to ensure consistent report output.
  Use this skill when generating or reviewing security findings to enforce
  the correct schema: title, severity, description, evidence, source, framework, control, remediation.
"""

from typing import Optional
from pydantic import BaseModel, Field


class FindingInput(BaseModel):
    title: str
    severity: str = Field(description="Critical | High | Medium | Low | Informational")
    description: str
    evidence: str
    source_file: Optional[str] = None
    framework: Optional[str] = None
    control_id: Optional[str] = None
    remediation: str


SEVERITY_LEVELS = ["Critical", "High", "Medium", "Low", "Informational"]

FRAMEWORK_CONTROLS = {
    "NIST CSF": [
        "ID.AM", "ID.RA", "ID.GV", "ID.SC",
        "PR.AC", "PR.DS", "PR.IP", "PR.MA", "PR.PT",
        "DE.AE", "DE.CM", "DE.DP",
        "RS.RP", "RS.CO", "RS.AN", "RS.MI", "RS.IM",
        "RC.RP", "RC.IM",
    ],
    "OWASP Top 10": [
        "A01", "A02", "A03", "A04", "A05",
        "A06", "A07", "A08", "A09", "A10",
    ],
    "CIS Controls": [
        "CIS 1", "CIS 2", "CIS 3", "CIS 4", "CIS 5",
        "CIS 6", "CIS 7", "CIS 8", "CIS 9", "CIS 10",
        "CIS 11", "CIS 12", "CIS 13", "CIS 14", "CIS 15",
        "CIS 16", "CIS 17", "CIS 18",
    ],
    "ISO 27001": [
        "A.5", "A.6", "A.7", "A.8", "A.9",
        "A.10", "A.11", "A.12", "A.13", "A.14", "A.15", "A.17", "A.18",
    ],
}


def validate_severity(severity: str) -> str:
    normalized = severity.strip().title()
    if normalized in SEVERITY_LEVELS:
        return normalized
    mappings = {
        "crit": "Critical",
        "high": "High",
        "med": "Medium",
        "medium": "Medium",
        "low": "Low",
        "info": "Informational",
        "informational": "Informational",
    }
    return mappings.get(severity.strip().lower(), "Medium")


def validate_framework(framework: Optional[str]) -> Optional[str]:
    if not framework:
        return None
    for valid in FRAMEWORK_CONTROLS:
        if valid.lower() in framework.lower():
            return valid
    return framework


def format_finding(finding: dict) -> str:
    severity = validate_severity(finding.get("severity", "Medium"))
    framework = validate_framework(finding.get("framework"))

    lines = [
        f"FINDING: {finding.get('title', 'Untitled')}",
        f"SEVERITY: {severity}",
        f"DESCRIPTION: {finding.get('description', 'No description provided')}",
        f"EVIDENCE: {finding.get('evidence', 'No evidence provided')}",
    ]

    if finding.get("source_file"):
        lines.append(f"SOURCE: {finding['source_file']}")
    if framework:
        lines.append(f"FRAMEWORK: {framework}")
    if finding.get("control_id"):
        lines.append(f"CONTROL: {finding['control_id']}")
    if finding.get("remediation"):
        lines.append(f"REMEDIATION: {finding['remediation']}")

    return "\n".join(lines)


def format_report(findings: list, executive_summary: str = "", overall_risk: str = "Medium") -> str:
    output_parts = []

    if executive_summary:
        output_parts.append("=" * 60)
        output_parts.append("EXECUTIVE SUMMARY")
        output_parts.append("=" * 60)
        output_parts.append(executive_summary)
        output_parts.append("")

    output_parts.append("=" * 60)
    output_parts.append(f"OVERALL RISK RATING: {validate_severity(overall_risk)}")
    output_parts.append("=" * 60)
    output_parts.append("")

    severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Informational": 4}
    sorted_findings = sorted(findings, key=lambda f: severity_order.get(validate_severity(f.get("severity", "Medium")), 5))

    for i, finding in enumerate(sorted_findings, 1):
        output_parts.append(f"--- Finding {i} ---")
        output_parts.append(format_finding(finding))
        output_parts.append("")

    counts = {}
    for f in findings:
        sev = validate_severity(f.get("severity", "Medium"))
        counts[sev] = counts.get(sev, 0) + 1

    output_parts.append("=" * 60)
    output_parts.append("FINDINGS SUMMARY")
    output_parts.append("=" * 60)
    for sev in SEVERITY_LEVELS:
        if sev in counts:
            output_parts.append(f"  {sev}: {counts[sev]}")
    output_parts.append(f"  Total: {len(findings)}")

    return "\n".join(output_parts)
