"""
Shared report-intent handling for every chat path (REST /chat, /chat/stream,
the orchestrator, slash commands). Detection, reply formatting, and the
placeholder-report gate live here so the direct route (/report/generate) and
the conversational path cannot drift apart.

Slash-command compatibility: the chat UI only treats "/report" as a report
request, so the UI-side detector (looksLikeReportRequest) must stay aligned
with detect_report_request below.
"""
import os
import re
import uuid
from datetime import datetime
from typing import Dict, Optional

from .config import settings

_RENDERERS = {}  # populated lazily by core.routers.report to avoid cycles

_REPORT_ACTION_RE = re.compile(
    r"\b(generate|create|give|make|produce|write|export|download|build)\b"
)
# Deliberately excludes "summary"/"document" alone: "make a summary of this
# document" is a chat request, not an export. Explicit formats, "report",
# "deck"/"presentation", or "write-up" indicate a deliverable file.
_REPORT_DELIVERABLE_RE = re.compile(
    r"\b(report|deck|presentation|docx|pptx|pdf|write-?up)\b"
)
_PPTX_HINTS = ("pptx", "powerpoint", "slide", "deck", "presentation")
_FILE_BASE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def detect_report_request(text: str, default_format: str = "docx") -> Optional[str]:
    """Returns 'docx' | 'pptx' | 'pdf' when the message asks for an exportable
    report, else None. Shared by all entry points; the UI mirrors this logic
    client-side only to decide when to show the progress bar."""
    t = (text or "").lower()
    if not (_REPORT_ACTION_RE.search(t) and _REPORT_DELIVERABLE_RE.search(t)):
        return None
    if any(k in t for k in _PPTX_HINTS):
        return "pptx"
    if "pdf" in t:
        return "pdf"
    return default_format


def register_renderer(fmt: str, renderer) -> None:
    _RENDERERS[fmt] = renderer


def render_to_report_dir(report, fmt: str, engagement_id: str) -> str:
    """Renders *report* into settings.report_dir and returns the filename."""
    renderer = _RENDERERS.get(fmt)
    if renderer is None:
        raise ValueError(f"format must be one of {sorted(_RENDERERS)}")
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    base = _FILE_BASE_RE.sub("-", (engagement_id or "report").lower())[:40].strip("-") or "report"
    filename = f"{base}_{stamp}_{uuid.uuid4().hex[:6]}.{fmt}"
    output_path = os.path.join(settings.report_dir, filename)
    renderer(report, output_path)
    return filename


def stub_gate_message() -> str:
    """Canonical placeholder-report gate message; the single source of truth
    for every report entry path (HTTP router, chat intents, orchestrator)."""
    return (
        "No model is loaded, so this report would contain placeholder (stub) data "
        "and must not be delivered to a client. Download a model via the Model dialog, "
        "or retry with allow_stub=true to acknowledge placeholder output."
    )


def chat_stub_gate_reply() -> str:
    """Chat-surface variant of the gate: no allow_stub toggle exists in a
    conversation, so the message drops the API-specific hint. No report is
    generated and nothing is written to the reports folder."""
    return (
        "⚠️ No model is loaded, so this report would contain placeholder (stub) "
        "data and must not be delivered to a client. Open the **Model** dialog "
        "to download/load a model, then ask again."
    )


def is_stub_backend() -> bool:
    try:
        import engine
        return engine.get_backend() == "stub" or (
            engine.get_backend() == "auto" and engine._resolve() == "stub"
        )
    except Exception:
        return True


def enforce_stub_gate(allow_stub: bool) -> None:
    if is_stub_backend() and not allow_stub:
        raise ValueError(stub_gate_message())


def format_report_reply(result: Dict, is_stub: bool) -> str:
    """Markdown reply shown in the chat after a report is generated."""
    eng = result.get("engagement", {})
    warning = ""
    if is_stub:
        warning = (
            "**WARNING: Placeholder report** — no model is loaded, so this report "
            "contains stub data and must not be delivered to a client.\n\n"
        )
    return (
        f"{warning}**Security report generated** for {eng.get('client_name', '')} — "
        f"{eng.get('engagement_name', '')} ({result['format'].upper()})\n\n"
        f"- Findings: {result['findings_count']}\n"
        f"- Overall risk rating: **{result['overall_risk_rating']}**\n"
        f"- [Download the report]({result['download_url']})"
    )
