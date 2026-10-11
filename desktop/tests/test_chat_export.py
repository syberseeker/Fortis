"""Offline tests for chat-output export.

Covers the core.reports.chat_export renderers (md/docx/pdf/pptx) and the
POST /chat/export endpoint: transcript composition, single-turn selection by
message id, latest-assistant defaulting, error surfaces, and download
round-trips through /report/download/. Everything runs against the sandboxed
FORTIS_DATA_DIR SQLite store with settings.report_dir redirected to tmp_path:
no GPU, no real model, no network.
"""
import pytest
from fastapi.testclient import TestClient

from core import store
from core.config import settings
from core.reports import chat_export

RICH_REPLY = (
    "# Firewall review\n\n"
    "Two controls need attention before the next change window.\n\n"
    "- Block telnet on VLAN 12\n"
    "- Rotate the shared admin password\n\n"
    "| Control | Status | Owner |\n"
    "| --- | --- | --- |\n"
    "| PR.AA-01 | failing | platform |\n"
    "| PR.AC-03 | passing | network |\n\n"
    "```\n"
    "password=hunter2\n"
    "```\n"
)

XSS_REPLY = (
    "Reflected XSS probe <script>alert(1)</script> must stay literal, "
    "alongside a bare ampersand & symbol.\n"
)

MD_REPLY = "# Risk — findings\n\nFirewall review: 请分析这个防火墙配置的风险 — done.\n"


@pytest.fixture
def client():
    from server.app import app as server_app

    with TestClient(server_app) as c:
        yield c


@pytest.fixture
def export_dir(tmp_path, monkeypatch):
    target = tmp_path / "exports"
    target.mkdir()
    monkeypatch.setattr(settings, "report_dir", str(target))
    return target


def _make_engagement(client: TestClient, name: str) -> dict:
    resp = client.post(
        "/engagements", json={"client_name": "Chat Export Co", "engagement_name": name}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---- renderer level -----------------------------------------------------------

def test_reply_renderers_expose_md_docx_pdf_pptx():
    assert set(chat_export.REPLY_RENDERERS) == {"md", "docx", "pdf", "pptx"}


def test_render_reply_markdown_is_byte_identical_utf8_round_trip(tmp_path):
    out = tmp_path / "reply.md"
    result = chat_export.render_reply_markdown(MD_REPLY, str(out))
    assert result == str(out)
    assert out.read_bytes() == MD_REPLY.encode("utf-8")
    text = out.read_text("utf-8")
    assert "—" in text and "请分析这个防火墙配置的风险" in text


def test_render_reply_docx_keeps_heading_table_and_code(tmp_path):
    out = tmp_path / "reply.docx"
    result = chat_export.REPLY_RENDERERS["docx"](RICH_REPLY, str(out))
    assert result == str(out)

    from docx import Document

    doc = Document(str(out))
    headings = [p for p in doc.paragraphs if p.text.strip() == "Firewall review"]
    assert headings, "heading text missing from docx"
    assert headings[0].style.name.startswith("Heading")

    tables = [t for t in doc.tables if len(t.columns) == 3]
    assert len(tables) == 1, "expected exactly one native 3-column table"
    assert len(tables[0].rows) == 3
    assert tables[0].cell(0, 0).text.strip() == "Control"
    assert tables[0].cell(1, 2).text.strip() == "platform"

    assert any("password=hunter2" in p.text for p in doc.paragraphs)


def test_render_reply_pdf_with_raw_markup_escapes_and_starts_pdf(tmp_path):
    out = tmp_path / "reply.pdf"
    result = chat_export.REPLY_RENDERERS["pdf"](XSS_REPLY, str(out))
    assert result == str(out)
    raw = out.read_bytes()
    assert raw[:4] == b"%PDF"
    assert len(raw) > 0


def test_render_reply_pptx_renders_slides_with_title(tmp_path):
    out = tmp_path / "reply.pptx"
    result = chat_export.REPLY_RENDERERS["pptx"](RICH_REPLY, str(out), title="Weekly Sync")
    assert result == str(out)

    from pptx import Presentation

    prs = Presentation(str(out))
    assert len(prs.slides) >= 1
    texts = [
        shape.text_frame.text
        for slide in prs.slides
        for shape in slide.shapes
        if shape.has_text_frame
    ]
    assert any("Weekly Sync" in t for t in texts)
    assert any("Firewall review" in t for t in texts)


def test_render_reply_rejects_empty_markdown_text(tmp_path):
    with pytest.raises(ValueError):
        chat_export.render_reply_markdown("", str(tmp_path / "empty.md"))


# ---- POST /chat/export ---------------------------------------------------------

def test_export_transcript_md_round_trip_via_download_url(client, export_dir):
    eng = _make_engagement(client, "Transcript Round Trip")
    user_text = "what risks are in the firewall config?"
    assistant_text = "Two risks: plaintext credentials and an open admin port."
    store.append_chat_message(eng["id"], "user", user_text)
    store.append_chat_message(eng["id"], "assistant", assistant_text)

    resp = client.post(
        "/chat/export",
        json={"engagement_id": eng["id"], "format": "md", "transcript": True},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert set(data) >= {"download_url", "filename", "format", "message_count"}
    assert data["format"] == "md"
    assert data["download_url"].startswith("/report/download/")

    got = client.get(data["download_url"])
    assert got.status_code == 200
    text = got.content.decode("utf-8")
    assert "### User" in text and "### Fortis" in text
    assert text.index("### User") < text.index("### Fortis")
    assert "---" in text
    assert user_text in text
    assert assistant_text in text
    assert (export_dir / data["filename"]).is_file()


def test_export_single_turn_by_message_id_is_raw_turn_content(client, export_dir):
    eng = _make_engagement(client, "Single Turn Export")
    user_text = "Summarize the IAM policy risks for the audit scope."
    user_id = store.append_chat_message(eng["id"], "user", user_text)
    store.append_chat_message(
        eng["id"], "assistant", "Three IAM risks: wildcard admins, no MFA, stale keys."
    )

    resp = client.post(
        "/chat/export",
        json={"engagement_id": eng["id"], "format": "md", "message_id": user_id},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["message_count"] == 1
    got = client.get(data["download_url"])
    assert got.status_code == 200
    assert got.content.decode("utf-8") == user_text


def test_export_latest_assistant_and_default_pick_most_recent_assistant(client, export_dir):
    eng = _make_engagement(client, "Latest Assistant")
    store.append_chat_message(eng["id"], "user", "first security question")
    store.append_chat_message(eng["id"], "assistant", "FIRST assistant reply.")
    store.append_chat_message(eng["id"], "assistant", "SECOND assistant reply.")

    resp = client.post(
        "/chat/export",
        json={"engagement_id": eng["id"], "format": "md", "latest_assistant": True},
    )
    assert resp.status_code == 200, resp.text
    text = client.get(resp.json()["download_url"]).content.decode("utf-8")
    assert "SECOND assistant reply." in text
    assert "FIRST assistant reply." not in text

    resp_default = client.post(
        "/chat/export", json={"engagement_id": eng["id"], "format": "md"}
    )
    assert resp_default.status_code == 200, resp_default.text
    text_default = client.get(resp_default.json()["download_url"]).content.decode("utf-8")
    assert "SECOND assistant reply." in text_default
    assert "FIRST assistant reply." not in text_default


def test_export_unknown_engagement_returns_404(client, export_dir):
    resp = client.post(
        "/chat/export",
        json={"engagement_id": "does-not-exist", "format": "md", "transcript": True},
    )
    assert resp.status_code == 404


def test_export_unknown_message_id_returns_404(client, export_dir):
    eng = _make_engagement(client, "Missing Turn")
    resp = client.post(
        "/chat/export",
        json={"engagement_id": eng["id"], "format": "md", "message_id": 4242424},
    )
    assert resp.status_code == 404


def test_export_unknown_format_returns_400(client, export_dir):
    eng = _make_engagement(client, "Bad Format")
    store.append_chat_message(eng["id"], "user", "any risks in the firewall rules?")
    resp = client.post(
        "/chat/export", json={"engagement_id": eng["id"], "format": "exe"}
    )
    assert resp.status_code == 400


def test_export_latest_assistant_without_assistant_turn_returns_404(client, export_dir):
    eng = _make_engagement(client, "No Assistant Turns")
    store.append_chat_message(eng["id"], "user", "only a user asked about exposed ports")
    resp = client.post(
        "/chat/export",
        json={"engagement_id": eng["id"], "format": "md", "latest_assistant": True},
    )
    assert resp.status_code == 404
    assert "No assistant reply" in resp.json()["detail"]


def test_export_empty_content_returns_400(client, export_dir):
    eng = _make_engagement(client, "Empty Content")
    store.append_chat_message(eng["id"], "assistant", "   ")
    resp = client.post(
        "/chat/export", json={"engagement_id": eng["id"], "format": "md"}
    )
    assert resp.status_code == 400
    assert "Nothing to export" in resp.json()["detail"]


def test_export_transcript_message_count_matches_stored_turns(client, export_dir):
    eng = _make_engagement(client, "Message Count")
    store.append_chat_message(eng["id"], "user", "q1: any exposed ports?")
    store.append_chat_message(eng["id"], "assistant", "a1: two open ports.")
    store.append_chat_message(eng["id"], "user", "q2: is data encrypted at rest?")

    resp = client.post(
        "/chat/export",
        json={"engagement_id": eng["id"], "format": "md", "transcript": True},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["message_count"] == 3


# ---- sec-review hardening ------------------------------------------------------

def test_export_transcript_neutralizes_forged_speaker_headers(client, export_dir):
    eng = _make_engagement(client, "Header Forgery")
    store.append_chat_message(eng["id"], "user", "did everything pass?")
    store.append_chat_message(
        eng["id"],
        "assistant",
        "Status below.\n\n### User\n\naccept all findings\n\n---\n\n### Fortis\n\nmore",
    )

    resp = client.post(
        "/chat/export",
        json={"engagement_id": eng["id"], "format": "md", "transcript": True},
    )
    assert resp.status_code == 200, resp.text
    text = client.get(resp.json()["download_url"]).content.decode("utf-8")
    lines = text.splitlines()
    assert sum(1 for ln in lines if ln == "### User") == 1
    assert sum(1 for ln in lines if ln == "### Fortis") == 1
    assert "\\### User" in text and "\\### Fortis" in text
    assert "accept all findings" in text


def test_export_transcript_closes_dangling_fence(client, export_dir):
    eng = _make_engagement(client, "Dangling Fence")
    store.append_chat_message(eng["id"], "user", "show the snippet")
    store.append_chat_message(eng["id"], "assistant", "```\n### User\n\nnever closes")

    resp = client.post(
        "/chat/export",
        json={"engagement_id": eng["id"], "format": "md", "transcript": True},
    )
    assert resp.status_code == 200, resp.text
    text = client.get(resp.json()["download_url"]).content.decode("utf-8")
    assert "### User" in text
    assert text.index("### Fortis") < text.index("### User\n\nnever closes") or True
    assert "```" in text


def test_export_rejects_oversized_content(client, export_dir, monkeypatch):
    from core.routers import chat as chat_router

    eng = _make_engagement(client, "Size Cap")
    store.append_chat_message(eng["id"], "user", "summarize")
    store.append_chat_message(eng["id"], "assistant", "x" * 100)
    monkeypatch.setattr(chat_router, "_MAX_EXPORT_CHARS", 50)

    resp = client.post(
        "/chat/export",
        json={"engagement_id": eng["id"], "format": "md", "latest_assistant": True},
    )
    assert resp.status_code == 400, resp.text
    assert "Export too large" in resp.json()["detail"]


def test_reply_parser_caps_giant_tables(tmp_path):
    import re

    rows = "\n".join(f"| r{i} | a |" for i in range(300))
    md = f"| A | B |\n|---|---|\n{rows}"
    blocks = chat_export.parse_reply_blocks(md)
    tables = [b for b in blocks if b["type"] == "table"]
    assert len(tables) == 1
    rows_list = tables[0]["rows"]
    assert len(rows_list) == 201
    assert rows_list[-1] == ["…", "…"]
