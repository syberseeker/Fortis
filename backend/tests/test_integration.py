"""
End-to-end tests of the backend HTTP API, run entirely offline:
- app.llm.chat is monkeypatched (no real Ollama needed)
- embeddings use the hash-stub backend (no Hugging Face download needed)
- Chroma and SQLite run against temp directories (see conftest.py)

This validates the *wiring* -- routing, schema validation, engagement
scoping, map-reduce triggering, and file rendering -- not model output
quality. Model quality can only be judged against a real GPU + Ollama setup,
which this sandbox does not have; see README for the live smoke-test steps.
"""
import io
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.reports.schema import SecurityReport

FAKE_REPORT = {
    "title": "Test Engagement Security Assessment",
    "client_context": "A small test client with a handful of config files.",
    "scope": "Review of uploaded configuration files.",
    "executive_summary": "One notable finding was identified during review.",
    "findings": [
        {
            "title": "Hardcoded credential in config",
            "severity": "High",
            "description": "A credential appears to be stored in plaintext.",
            "evidence": "password=hunter2 found in uploaded file.",
            "source_file": "sample.txt",
            "framework": "NIST_CSF",
            "control_id": "PR.AA",
            "remediation": "Move the credential to a secrets manager and rotate it.",
        }
    ],
    "overall_risk_rating": "High",
    "recommendations_summary": ["Rotate exposed credentials", "Adopt a secrets manager"],
}

FAKE_MAP_RESULT = {
    "key_points": ["Batch discusses a configuration file"],
    "findings": [
        {
            "title": "Sample map-stage finding",
            "severity": "Medium",
            "description": "Something worth flagging in this batch.",
            "evidence": "excerpt text",
            "source_file": "big.txt",
        }
    ],
}


@pytest.fixture
def call_log():
    return []


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch, call_log):
    """Routes app.llm.chat calls to canned responses based on which system
    prompt is in play, and records every call for assertions."""

    async def fake_chat(messages, temperature=0.2, json_mode=False):
        system_content = messages[0]["content"] if messages else ""
        call_log.append(system_content[:40])

        if "ONE batch of excerpts" in system_content:
            return json.dumps(FAKE_MAP_RESULT)
        if "already reviewed a client's full document set" in system_content:
            return json.dumps(FAKE_REPORT)
        if json_mode:
            return json.dumps(FAKE_REPORT)
        return "This is a stub conversational reply from Fortis."

    monkeypatch.setattr("app.analysis.chat", fake_chat)
    monkeypatch.setattr("app.routers.chat.chat", fake_chat)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---- Health -----------------------------------------------------------

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "frameworks_loaded" in data
    assert set(data["frameworks_loaded"]) == {
        "NIST_CSF", "OWASP_TOP10", "CIS_CONTROLS", "MITRE_ATTACK",
        "CIS_BENCHMARKS", "ISO_27001", "PCI_DSS", "SOC_2", "GDPR",
        "HIPAA", "NIST_800-53", "CWE_TOP25",
    }


# ---- Clients & engagements, human-readable IDs -----------------------

def test_create_engagement_has_readable_slug(client):
    resp = client.post(
        "/engagements",
        json={"client_name": "Acme Corp", "engagement_name": "Q3 2026 Config Review"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "acme-corp-q3-2026-config-review"
    assert data["name"] == "Q3 2026 Config Review"
    assert data["status"] == "active"

    # GET joins in the client name and file list
    detail = client.get(f"/engagements/{data['id']}").json()
    assert detail["client_name"] == "Acme Corp"
    assert detail["files"] == []


def test_create_engagement_idempotent(client):
    r1 = client.post("/engagements", json={"client_name": "Beta LLC", "engagement_name": "Pentest Prep"})
    r2 = client.post("/engagements", json={"client_name": "Beta LLC", "engagement_name": "Pentest Prep"})
    assert r1.json()["id"] == r2.json()["id"]


def test_active_engagement_persists_per_user(client):
    eng = client.post(
        "/engagements", json={"client_name": "Gamma Inc", "engagement_name": "Firewall Audit"}
    ).json()

    set_resp = client.post(
        "/engagements/active", json={"user_id": "user-42", "engagement_id": eng["id"]}
    )
    assert set_resp.status_code == 200

    get_resp = client.get("/engagements/active/user-42")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == eng["id"]

    # A different user has no active engagement of their own
    other = client.get("/engagements/active/someone-else")
    assert other.status_code == 404


def test_activate_nonexistent_engagement_404s(client):
    resp = client.post(
        "/engagements/active", json={"user_id": "user-1", "engagement_id": "does-not-exist"}
    )
    assert resp.status_code == 404


# ---- Upload -------------------------------------------------------------

def _make_engagement(client, client_name="Upload Test Co", engagement_name="Upload Flow"):
    return client.post(
        "/engagements", json={"client_name": client_name, "engagement_name": engagement_name}
    ).json()


def test_upload_requires_valid_engagement(client):
    resp = client.post(
        "/upload",
        files={"file": ("sample.txt", io.BytesIO(b"hello world"), "text/plain")},
        data={"engagement_id": "nonexistent-engagement"},
    )
    assert resp.status_code == 404


def test_upload_and_list_files(client):
    eng = _make_engagement(client)
    content = b"admin_password=hunter2\nallow_all_inbound=true\n"
    resp = client.post(
        "/upload",
        files={"file": ("sample.txt", io.BytesIO(content), "text/plain")},
        data={"engagement_id": eng["id"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunks_indexed"] >= 1

    files_resp = client.get(f"/upload/engagement/{eng['id']}/files")
    assert files_resp.json()["files"] == ["sample.txt"]


# ---- Chat ---------------------------------------------------------------

def test_chat_requires_valid_engagement(client):
    resp = client.post(
        "/chat", json={"engagement_id": "nonexistent", "message": "hello"}
    )
    assert resp.status_code == 404


def test_chat_returns_stub_reply(client, call_log):
    eng = _make_engagement(client, "Chat Test Co", "Chat Flow")
    resp = client.post(
        "/chat", json={"engagement_id": eng["id"], "message": "what risks do you see?"}
    )
    assert resp.status_code == 200
    assert "stub conversational reply" in resp.json()["reply"]
    assert len(call_log) == 1


# ---- Report generation: flat path (small doc) ----------------------------

@pytest.mark.parametrize("fmt,magic_check", [
    ("docx", lambda b: b[:2] == b"PK"),   # docx is a zip container
    ("pptx", lambda b: b[:2] == b"PK"),   # pptx is also a zip container
    ("pdf", lambda b: b[:4] == b"%PDF"),
])
def test_report_generation_flat_path(client, call_log, fmt, magic_check):
    eng = _make_engagement(client, f"Report Co {fmt}", "Flat Report")
    client.post(
        "/upload",
        files={"file": ("config.txt", io.BytesIO(b"password=hunter2\n"), "text/plain")},
        data={"engagement_id": eng["id"]},
    )

    resp = client.post(
        "/report/generate",
        json={"engagement_id": eng["id"], "format": fmt, "focus_instructions": ""},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["findings_count"] == len(FAKE_REPORT["findings"])
    assert data["overall_risk_rating"] == FAKE_REPORT["overall_risk_rating"]

    # Small doc -> flat path -> exactly one LLM call
    assert len(call_log) == 1

    download = client.get(data["download_url"])
    assert download.status_code == 200
    assert magic_check(download.content)


def test_report_generation_requires_uploaded_documents(client):
    eng = _make_engagement(client, "Empty Co", "No Docs")
    resp = client.post(
        "/report/generate", json={"engagement_id": eng["id"], "format": "docx"}
    )
    assert resp.status_code == 400


# ---- Report generation: map-reduce path (large doc) -----------------------

def test_report_generation_triggers_map_reduce_for_large_doc(client, call_log):
    eng = _make_engagement(client, "Big Co", "Map Reduce Test")

    # HIERARCHICAL_THRESHOLD_TOKENS=200 in conftest -- this text comfortably
    # exceeds that once chunked, so the map-reduce path should trigger.
    big_text = ("This configuration line describes a firewall rule. " * 40 + "\n\n") * 5
    client.post(
        "/upload",
        files={"file": ("big.txt", io.BytesIO(big_text.encode()), "text/plain")},
        data={"engagement_id": eng["id"]},
    )

    resp = client.post(
        "/report/generate", json={"engagement_id": eng["id"], "format": "docx"}
    )
    assert resp.status_code == 200

    # Map-reduce means: at least one MAP call plus exactly one REDUCE call,
    # i.e. more than the single call the flat path would make.
    assert len(call_log) > 1

    report = SecurityReport.model_validate(FAKE_REPORT)
    assert report.title  # sanity: schema still validates
