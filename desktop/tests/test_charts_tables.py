"""
Tests for chart rendering and table integration in reports.
"""
import os

import pytest

from core.reports.schema import SecurityReport, Finding, severity_counts, framework_counts, SEVERITY_ORDER
from core.reports.charts import render_severity_chart, render_framework_chart
from core.reports.docx_report import render_docx
from core.reports.pptx_report import render_pptx
from core.reports.pdf_report import render_pdf


@pytest.fixture
def sample_report():
    """A report with 4 findings across severities and frameworks."""
    return SecurityReport(
        title="Test Security Assessment",
        client_context="A test client with multiple findings.",
        scope="Review of configuration files and network architecture.",
        executive_summary="Multiple security issues were identified.",
        findings=[
            Finding(
                title="Critical SQL Injection",
                severity="Critical",
                description="SQL injection vulnerability in login form.",
                evidence="Unsanitized user input in query.",
                source_file="web-app.py",
                framework="OWASP_TOP10",
                control_id="A03:2021",
                remediation="Use parameterized queries.",
            ),
            Finding(
                title="High Severity Misconfiguration",
                severity="High",
                description="Default credentials on admin panel.",
                evidence="admin/admin123 found in config.",
                source_file="config.ini",
                framework="CIS_CONTROLS",
                control_id="4.1",
                remediation="Change default credentials.",
            ),
            Finding(
                title="Medium XSS Issue",
                severity="Medium",
                description="Reflected XSS in search parameter.",
                evidence="Script tag execution in response.",
                source_file="search.php",
                framework="OWASP_TOP10",
                control_id="A03:2021",
                remediation="Encode output properly.",
            ),
            Finding(
                title="Low Informational Finding",
                severity="Informational",
                description="Missing security headers.",
                evidence="No CSP header present.",
                source_file=None,
                framework=None,
                control_id=None,
                remediation="Add security headers.",
            ),
        ],
        overall_risk_rating="High",
        recommendations_summary=["Fix critical issues first", "Implement security headers"],
        diagram=None,
    )


@pytest.fixture
def report_with_diagram():
    """A report with a valid mermaid diagram in the diagram field."""
    return SecurityReport(
        title="Test Assessment with Diagram",
        client_context="Client with network diagram.",
        scope="Network architecture review.",
        executive_summary="Architecture review complete.",
        findings=[
            Finding(
                title="Network Segmentation Issue",
                severity="High",
                description="Flat network architecture.",
                evidence="All servers on same subnet.",
                source_file="network-config.txt",
                framework="NIST_CSF",
                control_id="PR.AC-5",
                remediation="Implement network segmentation.",
            ),
        ],
        overall_risk_rating="High",
        recommendations_summary=["Segment networks"],
        diagram="""flowchart LR
    Internet["Internet"] --> FW["Firewall"]
    FW --> Web["Web Server"]
    Web --> DB[("Database")]
""",
    )


@pytest.fixture
def zero_findings_report():
    """A report with no findings."""
    return SecurityReport(
        title="Clean Assessment",
        client_context="All systems secure.",
        scope="Security review.",
        executive_summary="No issues found.",
        findings=[],
        overall_risk_rating="Low",
        recommendations_summary=["Continue monitoring"],
        diagram=None,
    )


# ---- severity_counts and framework_counts --------------------------------

def test_severity_counts_all_keys_present(sample_report):
    counts = severity_counts(sample_report)
    assert set(counts.keys()) == set(SEVERITY_ORDER)
    assert counts["Critical"] == 1
    assert counts["High"] == 1
    assert counts["Medium"] == 1
    assert counts["Low"] == 0
    assert counts["Informational"] == 1


def test_framework_counts_correct(sample_report):
    counts = framework_counts(sample_report)
    assert counts["OWASP_TOP10"] == 2
    assert counts["CIS_CONTROLS"] == 1
    assert "NIST_CSF" not in counts


def test_framework_counts_excludes_none(sample_report):
    counts = framework_counts(sample_report)
    assert None not in counts


# ---- Chart rendering -----------------------------------------------------

def test_render_severity_chart_writes_png(sample_report, tmp_path):
    output_path = tmp_path / "severity.png"
    result = render_severity_chart(sample_report, str(output_path))
    
    assert result == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_framework_chart_writes_png(sample_report, tmp_path):
    output_path = tmp_path / "framework.png"
    result = render_framework_chart(sample_report, str(output_path))
    
    assert result == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_framework_chart_returns_none_for_zero_frameworks(zero_findings_report, tmp_path):
    output_path = tmp_path / "framework.png"
    result = render_framework_chart(zero_findings_report, str(output_path))
    
    assert result is None
    assert not output_path.exists()


# ---- Renderer integration with diagram field -----------------------------

def test_render_docx_with_diagram(report_with_diagram, tmp_path):
    output_path = tmp_path / "report.docx"
    result = render_docx(report_with_diagram, str(output_path))
    
    assert result == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    
    from docx import Document
    doc = Document(str(output_path))
    assert len(doc.tables) >= 1


def test_render_docx_zero_findings(zero_findings_report, tmp_path):
    output_path = tmp_path / "report.docx"
    result = render_docx(zero_findings_report, str(output_path))
    
    assert result == str(output_path)
    assert output_path.exists()
