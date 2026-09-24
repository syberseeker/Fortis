"""Tests for report-generation progress tracking and the stub-mode report gate."""
import io
import json

import pytest
from fastapi.testclient import TestClient

from core import progress
from core.main import app as core_app
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
    "diagram": "flowchart LR\n  A[Client] --> B[Fortis]",
}


@pytest.fixture(autouse=True)
def _reset_progress():
    progress.reset()
    yield
    progress.reset()


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


@pytest.fixture
def call_log():
    return []


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch, call_log):
    """Same canned-response routing as test_integration: map/reduce/json prompts
    get stub JSON, everything else gets the stub conversational line."""

    async def fake_chat(messages, temperature=0.2, json_mode=False):
        system_content = messages[0]["content"] if messages else ""
        call_log.append(system_content[:40])
        if "ONE batch of excerpts" in system_content:
            return json.dumps({"key_points": ["Batch discusses a configuration file"], "findings": []})
        if "already reviewed a client's real document set" in system_content:
            return json.dumps(FAKE_REPORT)
        if json_mode:
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


def test_progress_snapshot_idle_shape():
    snap = progress.snapshot()
    assert set(snap) == {"active", "stage", "done", "total", "detail"}
    assert snap["active"] is False
    assert snap["stage"] == "idle"
    assert snap["done"] == 0
    assert snap["total"] == 0
    assert snap["detail"] == ""


def test_progress_lifecycle_and_stage_transitions():
    progress.start(total=3, detail="3 batches")
    assert progress.snapshot() == {"active": True, "stage": "map", "done": 0, "total": 3, "detail": "3 batches"}
    progress.advance()
    progress.advance(detail="fizzbin")
    assert progress.snapshot()["done"] == 2
    assert progress.snapshot()["detail"] == "fizzbin"
    progress.update(done=2, stage="reduce")
    snap = progress.snapshot()
    assert snap["stage"] == "reduce" and snap["done"] == 2
    progress.finish()
    snap = progress.snapshot()
    assert snap["active"] is False and snap["done"] == 3


def test_progress_fail_marks_inactive():
    progress.start(total=2)
    progress.fail(detail="boom")
    snap = progress.snapshot()
    assert snap["active"] is False
    assert snap["detail"] == "boom"


def test_progress_endpoint_shape(client):
    resp = client.get("/api/report/progress")
    assert resp.status_code == 200
    snap = resp.json()
    assert set(snap) == {"active", "stage", "done", "total", "detail"}
    assert snap["stage"] == "idle"


def test_stub_report_blocked_without_allow_stub(monkeypatch, client):
    import core.routers.report as report_router

    monkeypatch.setattr(report_router, "_is_stub_backend", lambda: True)
    eng = _make_engagement(client, "Gate Co", "Stub Gate")
    _upload(client, eng["id"], "password=hunter2\n")
    resp = client.post("/report/generate", json={"engagement_id": eng["id"], "format": "docx"})
    assert resp.status_code == 409
    body = resp.json()
    assert "placeholder" in body["detail"].lower()
    assert "allow_stub" in body["detail"]


def test_stub_report_allowed_with_allow_stub(client, call_log):
    eng = _make_engagement(client, "Gate Co", "Stub Allowed")
    _upload(client, eng["id"], "password=hunter2\n")
    resp = client.post(
        "/report/generate",
        json={"engagement_id": eng["id"], "format": "docx", "allow_stub": True},
    )
    assert resp.status_code == 200
    assert resp.json()["findings_count"] == len(FAKE_REPORT["findings"])
    assert len(call_log) == 1


def test_gate_skipped_when_backend_not_stub(monkeypatch, client, call_log):
    eng = _make_engagement(client, "Gate Co", "Not Stub")
    _upload(client, eng["id"], "password=hunter2\n")

    import core.routers.report as report_router

    monkeypatch.setattr(report_router, "_is_stub_backend", lambda: False)
    resp = client.post("/report/generate", json={"engagement_id": eng["id"], "format": "docx"})
    assert resp.status_code == 200
    assert len(call_log) == 1


def test_map_reduce_progress_tracks_batches(client):
    eng = _make_engagement(client, "Progress Co", "Map Reduce")
    big_text = ("This configuration line describes a firewall rule. " * 40 + "\n\n") * 5
    _upload(client, eng["id"], big_text, "big.txt")

    resp = client.post(
        "/report/generate",
        json={"engagement_id": eng["id"], "format": "docx", "allow_stub": True},
    )
    assert resp.status_code == 200
    snap = progress.snapshot()
    assert snap["active"] is False
    assert snap["done"] == snap["total"] >= 2


def test_flat_report_progress_single_step(client):
    eng = _make_engagement(client, "Progress Co", "Flat")
    _upload(client, eng["id"], "password=hunter2\n")
    resp = client.post(
        "/report/generate",
        json={"engagement_id": eng["id"], "format": "docx", "allow_stub": True},
    )
    assert resp.status_code == 200
    snap = progress.snapshot()
    assert snap["done"] == 1 and snap["total"] == 1


def test_slow_backend_doubles_map_batch_tokens(monkeypatch):
    import core.analysis as analysis

    monkeypatch.setattr(settings, "engine_slow_backend", True)
    assert analysis._effective_batch_tokens() == settings.map_batch_tokens * 2
    monkeypatch.setattr(settings, "engine_slow_backend", False)
    assert analysis._effective_batch_tokens() == settings.map_batch_tokens


def test_map_reduce_plan_uses_effective_batch_size(monkeypatch):
    import core.analysis as analysis

    calls = []

    def fake_batch_chunks(chunks, max_tokens):
        calls.append(max_tokens)
        return [["chunk"]]

    monkeypatch.setattr(analysis, "_batch_chunks", fake_batch_chunks)
    monkeypatch.setattr(analysis.vectorstore, "get_document_chunks", lambda eid, fn: ["a", "b"])
    analysis._plan_map_batches("eng", ["f.txt"], analysis._effective_batch_tokens())
    assert calls == [settings.map_batch_tokens]


def test_report_failure_marks_progress_failed(client, monkeypatch):
    eng = _make_engagement(client, "Fail Co", "Progress Fail")
    _upload(client, eng["id"], "password=hunter2\n")

    import core.analysis as analysis

    def boom(*a, **k):
        raise ValueError("Analysis model returned malformed output. Try again.")

    monkeypatch.setattr(analysis, "_generate_flat", boom)
    resp = client.post(
        "/report/generate",
        json={"engagement_id": eng["id"], "format": "docx", "allow_stub": True},
    )
    assert resp.status_code == 400
    snap = progress.snapshot()
    assert snap["active"] is False
    assert snap["done"] < snap["total"] or snap["detail"] == ""
