"""Tests for the report service: analysis caching, focus trimming, and
framework-mapping validation."""
import asyncio

import pytest

from core import report_service


# ---- focus trimming -----------------------------------------------------------

def test_trim_focus_strips_report_phrasing():
    assert report_service.trim_focus_text("generate a docx report") == ""
    assert report_service.trim_focus_text("generate a pdf report please") == ""
    # conservative trim keeps grammar-safe residue; the request head is gone
    assert report_service.trim_focus_text("Please export a pptx deck of the findings").startswith("of the")
    assert report_service.trim_focus_text(
        "generate a pdf report focusing on access control"
    ) == "focusing on access control"
    assert report_service.trim_focus_text(
        "write a report as docx about the firewall rules"
    ) == "about the firewall rules"


def test_trim_focus_leaves_normal_text_alone():
    text = "review the IAM policy for privilege escalation risks"
    assert report_service.trim_focus_text(text) == text


# ---- analysis cache --------------------------------------------------------------

class _FakeReport:
    def __init__(self):
        from core.reports.schema import SecurityReport

        self._real = SecurityReport.model_validate({
            "title": "T", "client_context": "c", "scope": "s",
            "executive_summary": "e",
            "findings": [{"title": "F", "severity": "High", "description": "d",
                          "evidence": "ev", "source_file": "a.txt",
                          "framework": "NIST_CSF", "control_id": "PR.AA",
                          "remediation": "r"}],
            "overall_risk_rating": "High",
            "recommendations_summary": ["x"],
            "diagram": None,
        })

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "db_path", str(tmp_path / "cache.sqlite3"))
    report_service.ensure_cache_table()
    return tmp_path


def test_analyze_once_runs_llm_once_then_serves_from_cache(cache_env, monkeypatch):
    from core.reports.schema import SecurityReport

    calls = []

    async def fake_generate(engagement_id, focus_text=""):
        calls.append((engagement_id, focus_text))
        return SecurityReport.model_validate({
            "title": "T", "client_context": "c", "scope": "s",
            "executive_summary": "e",
            "findings": [],
            "overall_risk_rating": "Low",
            "recommendations_summary": [],
            "diagram": None,
        })

    monkeypatch.setattr(report_service, "generate_security_report", fake_generate)

    r1 = asyncio.run(report_service.analyze_once("eng-cache-1", "focus on tls"))
    r2 = asyncio.run(report_service.analyze_once("eng-cache-1", "focus on tls"))
    r3 = asyncio.run(report_service.analyze_once("eng-cache-1", "different focus"))

    assert len(calls) == 2, "second identical request must be served from cache"
    assert r1.executive_summary == r2.executive_summary


def test_cache_isolated_per_engagement(cache_env, monkeypatch):
    from core.reports.schema import SecurityReport

    calls = []

    async def fake_generate(engagement_id, focus_text=""):
        calls.append(engagement_id)
        return SecurityReport.model_validate({
            "title": "T", "client_context": "c", "scope": "s",
            "executive_summary": f"summary for {engagement_id}",
            "findings": [], "overall_risk_rating": "Low",
            "recommendations_summary": [], "diagram": None,
        })

    monkeypatch.setattr(report_service, "generate_security_report", fake_generate)

    a = asyncio.run(report_service.analyze_once("eng-A", "same focus"))
    b = asyncio.run(report_service.analyze_once("eng-B", "same focus"))
    assert len(calls) == 2
    assert a.executive_summary != b.executive_summary


def test_cache_invalidated_by_document_change(cache_env, monkeypatch):
    from core.reports.schema import SecurityReport

    calls = []

    async def fake_generate(engagement_id, focus_text=""):
        calls.append(len(calls))
        return SecurityReport.model_validate({
            "title": "T", "client_context": "c", "scope": "s",
            "executive_summary": "e", "findings": [],
            "overall_risk_rating": "Low", "recommendations_summary": [],
            "diagram": None,
        })

    monkeypatch.setattr(report_service, "generate_security_report", fake_generate)
    asyncio.run(report_service.analyze_once("eng-inv", "focus"))
    report_service.invalidate_engagement_cache("eng-inv")
    asyncio.run(report_service.analyze_once("eng-inv", "focus"))

    assert len(calls) == 2, "upload change must invalidate the cached analysis"


def test_validate_flags_unverifiable_framework(cache_env, monkeypatch):
    from core.reports.schema import SecurityReport

    class _FakeCol:
        def get(self, **kwargs):
            return {"metadatas": [
                {"framework": "NIST_CSF", "control_id": "PR.AA"},
                {"framework": "OWASP_TOP10", "control_id": "A01:2021"},
            ]}

    monkeypatch.setattr(report_service.vectorstore, "framework_collection", lambda: _FakeCol())

    report = SecurityReport.model_validate({
        "title": "T", "client_context": "c", "scope": "s",
        "executive_summary": "e",
        "findings": [
            {"title": "real", "severity": "High", "description": "d", "evidence": "e",
             "source_file": "a", "framework": "NIST_CSF", "control_id": "PR.AA",
             "remediation": "r"},
            {"title": "fabricated", "severity": "High", "description": "d", "evidence": "e",
             "source_file": "a", "framework": "NIST_CSF", "control_id": "ZZ.99",
             "remediation": "r"},
            {"title": "bogus framework", "severity": "Medium", "description": "d",
             "evidence": "e", "source_file": "a", "framework": "OWASP_TOP10",
             "control_id": "FAKE-99", "remediation": "r"},
        ],
        "overall_risk_rating": "High",
        "recommendations_summary": [],
        "diagram": None,
    })

    demoted = report_service.validate_findings_mappings(report)

    assert demoted == 2
    assert report.findings[0].control_id == "PR.AA", "verifiable mapping must stay intact"
    assert "unverified" in report.findings[1].control_id
    assert report.findings[2].framework == "OWASP_TOP10", "valid framework literal stays"
    assert "unverified" in report.findings[2].control_id


def test_validate_accepts_alias_spellings(cache_env, monkeypatch):
    from core.reports.schema import SecurityReport

    class _FakeCol:
        def get(self, **kwargs):
            return {"metadatas": [{"framework": "NIST_CSF", "control_id": "PR.AA"}]}

    monkeypatch.setattr(report_service.vectorstore, "framework_collection", lambda: _FakeCol())

    report = SecurityReport.model_validate({
        "title": "T", "client_context": "c", "scope": "s",
        "executive_summary": "e",
        "findings": [{"title": "f", "severity": "Low", "description": "d", "evidence": "e",
                      "source_file": "a", "framework": "NIST CSF", "control_id": "PR.AA",
                      "remediation": "r"}],
        "overall_risk_rating": "Low",
        "recommendations_summary": [],
        "diagram": None,
    })

    assert report_service.validate_findings_mappings(report) == 0


def test_divergent_alias_spellings_canonicalize_and_validate_consistently(cache_env, monkeypatch):
    from core import framework_aliases
    from core.reports import schema
    from core.reports.schema import SecurityReport

    assert not hasattr(schema, "_FRAMEWORK_ALIASES")
    assert not hasattr(report_service, "_FRAMEWORK_ALIASES")

    report_service.invalidate_known_framework_pairs_cache()

    class _FakeCol:
        def get(self, **kwargs):
            return {"metadatas": [
                {"framework": "MITRE_ATTACK", "control_id": "TA0043"},
                {"framework": "ISO_27001", "control_id": "A.5.1"},
            ]}

    monkeypatch.setattr(report_service.vectorstore, "framework_collection", lambda: _FakeCol())

    report = SecurityReport.model_validate({
        "title": "T", "client_context": "c", "scope": "s",
        "executive_summary": "e",
        "findings": [
            {"title": "mitre alias", "severity": "Low", "description": "d", "evidence": "e",
             "source_file": "a", "framework": "MITRE ATT&CK (Enterprise)",
             "control_id": "TA0043", "remediation": "r"},
            {"title": "iso alias", "severity": "Low", "description": "d", "evidence": "e",
             "source_file": "a", "framework": "ISO/IEC 27001",
             "control_id": "A.5.1", "remediation": "r"},
        ],
        "overall_risk_rating": "Low",
        "recommendations_summary": [],
        "diagram": None,
    })

    assert report.findings[0].framework == "MITRE_ATTACK"
    assert report.findings[1].framework == "ISO_27001"

    assert report_service.validate_findings_mappings(report) == 0
    assert report.findings[0].framework == "MITRE_ATTACK"
    assert report.findings[0].control_id == "TA0043"
    assert report.findings[1].framework == "ISO_27001"
    assert report.findings[1].control_id == "A.5.1"

    assert framework_aliases.canonical_framework("MITRE ATT&CK (Enterprise)") == "MITRE_ATTACK"
    assert framework_aliases.canonical_framework("ISO/IEC 27001") == "ISO_27001"
    assert framework_aliases.canonical_framework(None) is None
    assert framework_aliases.canonical_framework("   ") is None


def test_unknown_framework_passes_through_parser_and_is_demoted_by_validator(cache_env, monkeypatch):
    from core import framework_aliases
    from core.reports.schema import SecurityReport

    report_service.invalidate_known_framework_pairs_cache()

    class _FakeCol:
        def get(self, **kwargs):
            return {"metadatas": [{"framework": "NIST_CSF", "control_id": "PR.AA"}]}

    monkeypatch.setattr(report_service.vectorstore, "framework_collection", lambda: _FakeCol())

    report = SecurityReport.model_validate({
        "title": "T", "client_context": "c", "scope": "s",
        "executive_summary": "e",
        "findings": [{"title": "f", "severity": "Medium", "description": "d", "evidence": "e",
                      "source_file": "a", "framework": "INTERNAL_HOUSE_STANDARD",
                      "control_id": "HSE-1.2", "remediation": "r"}],
        "overall_risk_rating": "Medium",
        "recommendations_summary": [],
        "diagram": None,
    })

    assert report.findings[0].framework == "INTERNAL_HOUSE_STANDARD", \
        "unknown framework must pass through the parser verbatim"
    assert framework_aliases.canonical_framework("INTERNAL_HOUSE_STANDARD") == "INTERNAL_HOUSE_STANDARD"

    demoted = report_service.validate_findings_mappings(report)

    assert demoted == 1
    assert report.findings[0].framework == "INTERNAL_HOUSE_STANDARD (unverified)"
    assert report.findings[0].control_id == "HSE-1.2 (unverified)"
