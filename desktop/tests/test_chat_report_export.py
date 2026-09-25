"""Regression tests for report export via the chat paths.

The chat UI sends every message (including "generate a docx report") to
/chat/stream. Report intent must be handled there so the rendered file lands
in settings.report_dir, matching what the sidebar "Open reports folder"
button opens. This file pins that contract.
"""
import io
import os
import re

import pytest
from fastapi.testclient import TestClient

from core import progress
from core.config import settings

FAKE_REPORT = {
    "title": "Stub Security Assessment",
    "client_context": "Stub context",
    "scope": "Stub scope",
    "executive_summary": "Stub finding: verify with a real model",
    "findings": [
        {
            "title": "Stub finding",
            "severity": "Medium",
            "description": "Stub finding: verify with a real model",
            "evidence": "stub",
            "source_file": None,
            "framework": "NIST_CSF",
            "control_id": "PR.PS",
            "remediation": "Load a real model and re-run.",
        }
    ],
    "overall_risk_rating": "Medium",
    "recommendations_summary": ["Load a real model and re-run."],
    "diagram": None,
}


def _make_engagement(client: TestClient, client_name: str, name: str) -> dict:
    resp = client.post(
        "/engagements", json={"client_name": client_name, "engagement_name": name}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _upload(client: TestClient, engagement_id: str, text: str, filename: str = "doc.txt") -> None:
    resp = client.post(
        "/upload",
        files={"file": (filename, io.BytesIO(text.encode()), "text/plain")},
        data={"engagement_id": engagement_id},
    )
    assert resp.status_code == 200, resp.text


def _reports_before() -> set:
    return set(os.listdir(settings.report_dir))


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    async def fake_chat(messages, temperature=0.2, json_mode=False):
        system_content = messages[0]["content"] if messages else ""
        if "ONE batch of excerpts" in system_content:
            return '{"key_points": [], "findings": []}'
        if "already reviewed a client's real document set" in system_content:
            import json
            return json.dumps(FAKE_REPORT)
        if json_mode:
            import json
            return json.dumps(FAKE_REPORT)
        return "This is a stub conversational reply from Fortis."

    async def fake_chat_stream(messages, temperature=0.2):
        text = await fake_chat(messages, temperature, json_mode=False)
        for word in text.split(" "):
            yield word + " "

    monkeypatch.setattr("core.analysis.chat", fake_chat)
    monkeypatch.setattr("core.routers.chat.chat", fake_chat)
    monkeypatch.setattr("core.routers.chat.chat_stream", fake_chat_stream)


@pytest.fixture
def client():
    from server.app import app as server_app

    with TestClient(server_app) as c:
        yield c


# ---- /chat/stream (what the chat UI actually calls) ------------------------

@pytest.mark.parametrize("fmt,suffix,check", [
    ("docx", ".docx", lambda b: b[:2] == b"PK"),
    ("pptx", ".pptx", lambda b: b[:2] == b"PK"),
    ("pdf", ".pdf", lambda b: b[:4] == b"%PDF"),
])
def test_stream_report_request_writes_file_to_report_dir(client, fmt, suffix, check):
    eng = _make_engagement(client, "Chat Export Co", f"Stream {fmt}")
    _upload(client, eng["id"], "password=hunter2\n")

    before = _reports_before()
    resp = client.post(
        "/chat/stream",
        json={"engagement_id": eng["id"], "message": f"generate a {fmt} report"},
    )
    assert resp.status_code == 200
    body = resp.text
    assert "Security report generated" in body

    new = set(os.listdir(settings.report_dir)) - before
    assert len(new) == 1, f"expected exactly one new report file, got {new}"
    filename = new.pop()
    assert filename.endswith(suffix), filename
    with open(os.path.join(settings.report_dir, filename), "rb") as f:
        assert check(f.read(4))

    m = re.search(r"\(/report/download/([^)]+)\)", body)
    assert m and m.group(1) == filename


def test_stream_report_request_format_selection(client):
    eng = _make_engagement(client, "Chat Export Co", "Format Select")
    _upload(client, eng["id"], "password=hunter2\n")
    for text, fmt in (
        ("generate a pdf report", ".pdf"),
        ("build a pptx deck report", ".pptx"),
        ("generate a report", ".docx"),
    ):
        before = _reports_before()
        resp = client.post("/chat/stream", json={"engagement_id": eng["id"], "message": text})
        assert resp.status_code == 200
        new = set(os.listdir(settings.report_dir)) - before
        assert len(new) == 1 and new.pop().endswith(fmt), (text, new)


def test_stream_report_placeholder_warning_in_stub_mode(client, monkeypatch):
    """In stub mode the chat surfaces explain the gate conversationally instead
    of erroring; no report file is written."""
    import core.report_intent as ri

    monkeypatch.setattr(ri, "is_stub_backend", lambda: True)
    eng = _make_engagement(client, "Chat Export Co", "Stub Warn")
    _upload(client, eng["id"], "password=hunter2\n")
    before = _reports_before()
    resp = client.post(
        "/chat/stream",
        json={"engagement_id": eng["id"], "message": "generate a report"},
    )
    body = resp.text
    assert resp.status_code == 200
    assert "placeholder" in body.lower()
    assert "Model" in body and "download" in body.lower()
    assert set(os.listdir(settings.report_dir)) - before == set(), "no file must be written in stub mode"


def test_stream_summary_question_is_chat_not_report(client, monkeypatch):
    """'Make a summary of this document' is a chat request; it must not render
    a report file or consume the LLM budget for a report pass."""
    import core.routers.chat as chat_router

    eng = _make_engagement(client, "Chat Export Co", "Summary Chat")
    _upload(client, eng["id"], "password=hunter2\n")
    calls = []

    async def boom(*a, **k):
        calls.append(1)
        return "This is a stub conversational reply from Fortis."

    monkeypatch.setattr(chat_router, "generate_report_result", boom)
    resp = client.post(
        "/chat/stream",
        json={"engagement_id": eng["id"], "message": "make a summary of this document"},
    )
    assert resp.status_code == 200
    assert not calls, "summary request must not hit the report pipeline"


# ---- /chat non-streaming parity -------------------------------------------

def test_chat_report_request_writes_file(client):
    eng = _make_engagement(client, "Chat Export Co", "Non Stream")
    _upload(client, eng["id"], "password=hunter2\n")
    before = _reports_before()
    resp = client.post("/chat", json={"engagement_id": eng["id"], "message": "generate a report"})
    assert resp.status_code == 200
    assert "Security report generated" in resp.json()["reply"]
    new = set(os.listdir(settings.report_dir)) - before
    assert len(new) == 1 and new.pop().endswith(".docx")


# ---- /report slash command --------------------------------------------------
def test_report_slash_command_generates_and_downloads(client):
    eng = _make_engagement(client, "Slash Co", "Slash Report")
    _upload(client, eng["id"], "password=hunter2\n")
    resp = client.post("/engagements/active", json={"user_id": "local", "engagement_id": eng["id"]})
    assert resp.status_code == 200
    try:
        before = _reports_before()
        resp = client.post("/api/command", json={"text": "/report pdf"})
        assert resp.status_code == 200
        text = resp.json()["text"]
        assert "Security report generated" in text
        new = set(os.listdir(settings.report_dir)) - before
        assert len(new) == 1 and new.pop().endswith(".pdf")
    finally:
        client.delete("/engagements/active/local")


# ---- error surfaces ----------------------------------------------------------

def test_stream_report_without_documents_is_friendly_error(client):
    eng = _make_engagement(client, "Chat Export Co", "No Docs")
    resp = client.post(
        "/chat/stream",
        json={"engagement_id": eng["id"], "message": "generate a report"},
    )
    assert resp.status_code == 400
    assert "No documents have been uploaded" in resp.text


def test_stream_report_placeholder_gate_without_allow_stub(client, monkeypatch):
    import core.report_intent as ri

    monkeypatch.setattr(ri, "is_stub_backend", lambda: True)
    eng = _make_engagement(client, "Chat Export Co", "Gate")
    _upload(client, eng["id"], "password=hunter2\n")
    resp = client.post(
        "/chat/stream",
        json={"engagement_id": eng["id"], "message": "generate a report"},
    )
    assert resp.status_code == 200
    assert "placeholder" in resp.text.lower()
    resp2 = client.post(
        "/chat",
        json={"engagement_id": eng["id"], "message": "generate a report"},
    )
    assert resp2.status_code == 200
    assert "placeholder" in resp2.json()["reply"].lower()
    before = _reports_before()
    resp3 = client.post(
        "/chat/stream",
        json={"engagement_id": eng["id"], "message": "generate a report"},
    )
    assert resp3.status_code == 200
    assert set(os.listdir(settings.report_dir)) - before == set()


def test_chat_report_engagement_not_found_404():
    from server.app import app as server_app

    with TestClient(server_app) as client:
        resp = client.post("/chat", json={"engagement_id": "nonexistent", "message": "generate a report"})
        assert resp.status_code == 404


def test_invalid_report_format_via_chat_returns_400(client):
    eng = _make_engagement(client, "Chat Export Co", "Bad Fmt")
    _upload(client, eng["id"], "password=hunter2\n")
    # detector never selects an unsupported format, but the shared builder
    # must still reject one if a caller passes it directly
    from core.routers.report import generate_report_result

    result = __import__("asyncio").run(
        generate_report_result(eng["id"], "xlsx", "")
    )
    assert "error" in result and "format must be one of" in result["error"]
    assert result["status"] == 400


# ---- detection parity ---------------------------------------------------------

def test_detect_report_request_matrix():
    from core.report_intent import detect_report_request

    assert detect_report_request("generate a report") == "docx"
    assert detect_report_request("export a PDF write-up") == "pdf"
    assert detect_report_request("make a pptx deck") == "pptx"
    assert detect_report_request("produce the powerpoint presentation") == "pptx"
    assert detect_report_request("please summarize the risks in this doc") is None
    assert detect_report_request("what is XSS?") is None
    assert detect_report_request("hello") is None
    assert detect_report_request("make a summary of this document") is None
    assert detect_report_request("") is None
    assert detect_report_request(None) is None
