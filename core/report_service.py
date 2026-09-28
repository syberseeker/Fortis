"""Report service: the single pipeline behind every report entry point.

Owns format/engagement validation, the stub gate, analysis-result caching,
focus-instruction trimming, framework-mapping validation, and rendering
into settings.report_dir. Routers and chat intents call into this module;
they must not re-implement any of it.

Analysis caching: a SecurityReport is keyed by (engagement_id, focus) and
stored in the engagement SQLite database. Generating DOCX, PPTX, and PDF
for the same engagement and focus therefore runs the LLM pipeline once and
renders three documents from the identical result.
"""
import hashlib
import json
import logging
import os
import re
import uuid

from . import report_intent, store, vectorstore
from .analysis import generate_security_report
from .config import settings
from .reports.docx_report import render_docx
from .reports.pdf_report import render_pdf
from .reports.pptx_report import render_pptx

logger = logging.getLogger(__name__)

RENDERERS = {"docx": render_docx, "pptx": render_pptx, "pdf": render_pdf}
for _fmt, _renderer in RENDERERS.items():
    report_intent.register_renderer(_fmt, _renderer)

# request phrasings stripped from a chat message before the residue is used
# as focus instructions ("generate a docx report focusing on X" -> "focusing on X")
_D = r"(?:docx|pptx|pdf|word|powerpoint|report|deck|presentation|write-?up|document|summary)"
_DELIVERABLE_RUN = rf"{_D}(?:\s+{_D})*"
_FOCUS_NOISE = [
    r"\b(?:please|kindly)\b",
    r"\b(?:generate|create|give|make|produce|write|export|download|build)\b",
    rf"\b(?:a|an|the)\s+(?:(?:full|complete|security|consultant(?:-|\s+)grade)\s+)?{_DELIVERABLE_RUN}\b",
    rf"\b{_DELIVERABLE_RUN}\s+(?:as|in)\s+(?:a\s+)?(?:docx|pptx|pdf|word|powerpoint)\b",
    rf"\b(?:docx|pptx|pdf|word|powerpoint)\s+{_D}\b",
    r"\bas\s+(?:a\s+)?(?:docx|pptx|pdf|word|powerpoint)\b",
    r"\bfor\s+(?:the\s+|this\s+)?engagement\b",
]

def trim_focus_text(text: str) -> str:
    """Removes report-request phrasing from a chat message so the residue can
    serve as focus instructions. Returns "" when nothing meaningful remains."""
    t = (text or "").strip()
    for pattern in _FOCUS_NOISE:
        t = re.sub(pattern, " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s{2,}", " ", t).strip(" .,;:-")
    return t.strip()


def _focus_cache_key(engagement_id: str, focus_text: str) -> str:
    normalized = re.sub(r"\s+", " ", (focus_text or "").strip()).lower()
    raw = f"{engagement_id}::{normalized}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


_SCHEMA_TABLE = """
CREATE TABLE IF NOT EXISTS report_cache (
    cache_key TEXT PRIMARY KEY,
    engagement_id TEXT NOT NULL REFERENCES engagements(id) ON DELETE CASCADE,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_report_cache_engagement
    ON report_cache(engagement_id, created_at);
"""


def ensure_cache_table() -> None:
    import sqlite3

    conn = sqlite3.connect(settings.db_path, timeout=15)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_SCHEMA_TABLE)
        conn.commit()
    finally:
        conn.close()


def _cache_get(engagement_id: str, cache_key: str):
    import sqlite3

    ensure_cache_table()
    conn = sqlite3.connect(settings.db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT report_json FROM report_cache WHERE cache_key = ? AND engagement_id = ?",
            (cache_key, engagement_id),
        ).fetchone()
        if not row:
            return None
        from .reports.schema import SecurityReport

        return SecurityReport.model_validate(json.loads(row["report_json"]))
    except Exception:
        logger.warning("Report cache read failed; regenerating analysis", exc_info=True)
        return None
    finally:
        conn.close()


def _cache_put(engagement_id: str, cache_key: str, report) -> None:
    import sqlite3

    ensure_cache_table()
    conn = sqlite3.connect(settings.db_path, timeout=15)
    try:
        payload = report.model_dump_json()
        conn.execute(
            "INSERT OR REPLACE INTO report_cache (cache_key, engagement_id, report_json, created_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (cache_key, engagement_id, payload),
        )
        conn.commit()
    except Exception:
        logger.warning("Report cache write failed (non-fatal)", exc_info=True)
    finally:
        conn.close()


def invalidate_engagement_cache(engagement_id: str) -> None:
    """Called when an engagement's documents change, so a stale analysis can
    never be reused after new material is uploaded."""
    import sqlite3

    ensure_cache_table()
    conn = sqlite3.connect(settings.db_path, timeout=15)
    try:
        conn.execute("DELETE FROM report_cache WHERE engagement_id = ?", (engagement_id,))
        conn.commit()
    except Exception:
        logger.warning("Report cache invalidation failed (non-fatal)", exc_info=True)
    finally:
        conn.close()


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
    "MITRE": "MITRE_ATTACK",
    "MITRE_ATTACK": "MITRE_ATTACK",
    "CIS BENCHMARKS": "CIS_BENCHMARKS",
    "CIS_BENCHMARKS": "CIS_BENCHMARKS",
    "ISO 27001": "ISO_27001",
    "ISO27001": "ISO_27001",
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


def _known_framework_pairs() -> set:
    pairs = set()
    try:
        col = vectorstore.framework_collection()
        data = col.get(include=["metadatas"])
        for meta in data.get("metadatas") or []:
            fw = meta.get("framework")
            cid = meta.get("control_id")
            if fw and cid:
                pairs.add((str(fw), str(cid)))
    except Exception:
        logger.warning("Framework corpus unavailable for mapping validation", exc_info=True)
    return pairs


def validate_findings_mappings(report) -> int:
    """Checks every finding's framework/control_id against the seeded corpus,
    in place. Unverifiable mappings are marked "(unverified)" on the control
    id so renderers never present a confident citation the corpus cannot back.
    Returns the number of demoted findings."""
    if not report.findings:
        return 0
    known = _known_framework_pairs()
    if not known:
        return 0
    demoted = 0
    for finding in report.findings:
        fw = (finding.framework or "").strip()
        cid = (finding.control_id or "").strip()
        if not fw and not cid:
            continue
        fw_canon = _FRAMEWORK_ALIASES.get(fw.upper(), fw) if fw else ""
        if fw_canon and (fw_canon, cid) in known:
            continue
        # framework known but control id unknown -> demote just the control id
        if fw_canon and any(f == fw_canon for f, _ in known):
            finding.control_id = f"{cid} (unverified)" if cid else "unverified"
        else:
            finding.framework = f"{finding.framework} (unverified)" if fw else finding.framework
            finding.control_id = f"{cid} (unverified)" if cid else "unverified"
        demoted += 1
    return demoted


async def analyze_once(engagement_id: str, focus_text: str = ""):
    """Runs the analysis pipeline once per (engagement, focus), serving
    repeat requests for other formats from the SQLite cache."""
    cache_key = _focus_cache_key(engagement_id, focus_text)
    cached = _cache_get(engagement_id, cache_key)
    if cached is not None:
        logger.info("Report cache hit for engagement %s", engagement_id)
        return cached
    report = await generate_security_report(engagement_id, focus_text)
    validate_findings_mappings(report)
    _cache_put(engagement_id, cache_key, report)
    return report


async def generate_report_result(
    engagement_id: str,
    fmt: str,
    focus_text: str = "",
    allow_stub: bool = False,
) -> dict:
    """Single builder shared by /report/generate, the chat report intent, and
    the orchestrator: validates format/engagement, enforces the stub gate,
    runs (or reuses) the analysis, renders into settings.report_dir, and
    returns the download metadata. Error results carry {"error", "status"}."""
    fmt = (fmt or "").lower()
    if fmt not in RENDERERS:
        return {"error": f"format must be one of {list(RENDERERS)}", "status": 400}
    engagement = store.get_engagement(engagement_id)
    if not engagement:
        return {
            "error": f"Engagement '{engagement_id}' not found. Create or select one first.",
            "status": 404,
        }
    try:
        report_intent.enforce_stub_gate(allow_stub)
        report = await analyze_once(engagement_id, focus_text)
        filename = report_intent.render_to_report_dir(report, fmt, engagement_id)
    except ValueError as e:
        message = str(e)
        is_gate = "allow_stub" in message
        return {"error": message, "status": 409 if is_gate else 400, "stub_gate": is_gate}
    except TimeoutError:
        return {
            "error": "Report analysis timed out. The model may be busy or too large "
                     "for this machine — try a smaller tier or fewer documents.",
            "status": 504,
        }
    return {
        "download_url": f"/report/download/{filename}",
        "filename": filename,
        "format": fmt,
        "findings_count": len(report.findings),
        "overall_risk_rating": report.overall_risk_rating,
        "engagement": {
            "client_name": engagement["client_name"],
            "engagement_name": engagement["name"],
        },
    }
