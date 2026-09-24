"""
Stress tests for the offline suite: large-document ingestion, high-volume
engagement creation, rapid sequential chat turns, unicode/adversarial input
at the chat endpoint, and concurrent uploads via threads.

Everything here is offline: hash-stub embeddings (conftest.py), the LLM is
monkeypatched with canned responses keyed on the system prompt (kept in sync
with FLAT_SYSTEM_PROMPT / MAP_SYSTEM_PROMPT / REDUCE_SYSTEM_PROMPT wording:
"ONE batch of excerpts", "already reviewed a client's full document set"),
and Chroma/SQLite/upload/report dirs are the shared per-session temp dirs.
No GPU, no real model, no network.
"""
import io
import json
import threading

import pytest
from fastapi.testclient import TestClient

from core.main import app as core_app
from core import vectorstore

FAKE_MAP_RESULT = {
    "key_points": ["Batch discusses a configuration file"],
    "findings": [
        {
            "title": "Stress map-stage finding",
            "severity": "Medium",
            "description": "Something worth flagging in this batch.",
            "evidence": "excerpt text",
            "source_file": "stress_big.txt",
        }
    ],
}

FAKE_REPORT = {
    "title": "Stress Engagement Security Assessment",
    "client_context": "A stress-test client with generated config material.",
    "scope": "Review of uploaded configuration files.",
    "executive_summary": "One notable finding was identified during review.",
    "findings": [
        {
            "title": "Hardcoded credential in config",
            "severity": "High",
            "description": "A credential appears to be stored in plaintext.",
            "evidence": "password=hunter2 found in uploaded file.",
            "source_file": "stress_big.txt",
            "framework": "NIST_CSF",
            "control_id": "PR.AA",
            "remediation": "Move the credential to a secrets manager and rotate it.",
        }
    ],
    "overall_risk_rating": "High",
    "recommendations_summary": ["Rotate exposed credentials", "Adopt a secrets manager"],
    "diagram": None,
}


@pytest.fixture
def call_log():
    return []


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch, call_log):
    """Same canned-response routing as test_integration.py: map-stage prompts
    ("ONE batch of excerpts") get the map result, reduce-stage prompts
    ("already reviewed a client's full document set") get the full report,
    other json_mode calls get the full report, and conversational calls get a
    stub reply. Records the first 40 chars of each system prompt."""

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

    async def fake_chat_stream(messages, temperature=0.2):
        system_content = messages[0]["content"] if messages else ""
        call_log.append(system_content[:40])
        text = await fake_chat(messages, temperature, json_mode=False)
        for word in text.split(" "):
            yield word + " "

    monkeypatch.setattr("core.analysis.chat", fake_chat)
    monkeypatch.setattr("core.routers.chat.chat", fake_chat)
    monkeypatch.setattr("core.routers.chat.chat_stream", fake_chat_stream)


@pytest.fixture
def client():
    with TestClient(core_app) as c:
        yield c


def _make_engagement(client, client_name="Stress Co", engagement_name="Stress Eng"):
    resp = client.post(
        "/engagements", json={"client_name": client_name, "engagement_name": engagement_name}
    )
    assert resp.status_code == 200
    return resp.json()


# ---- Very large document ingestion ----------------------------------------

def test_one_mb_document_ingestion_and_retrieval(client):
    """A ~1MB generated text doc must ingest fully, list under its engagement,
    and be retrievable via both the dense and hybrid paths."""
    eng = _make_engagement(client, "Big Doc Co", "One MB Doc")

    line = "Firewall rule allows port 443 from admin subnet zzstressmarker.\n"
    big_text = line * (1024 * 1024 // len(line))
    assert len(big_text) >= 1024 * 1024 - len(line)

    resp = client.post(
        "/upload",
        files={"file": ("stress_big.txt", io.BytesIO(big_text.encode("utf-8")), "text/plain")},
        data={"engagement_id": eng["id"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunks_indexed"] >= 100
    assert data["characters_extracted"] >= 1024 * 1024 - len(line)

    files = client.get(f"/upload/engagement/{eng['id']}/files").json()["files"]
    assert files == ["stress_big.txt"]

    for query_fn in (vectorstore.query_user_documents, vectorstore.query_user_documents_hybrid):
        hits = query_fn(eng["id"], "firewall rule zzstressmarker", 5)
        assert len(hits) == 5
        assert all(h["metadata"]["engagement_id"] == eng["id"] for h in hits)
        assert all(h["text"] for h in hits)


def test_one_mb_document_report_generation(client, call_log):
    """The ~1MB doc far exceeds HIERARCHICAL_THRESHOLD_TOKENS=200 (conftest),
    so report generation must take the map-reduce path: >=1 map call plus one
    reduce call, and a valid downloadable report."""
    eng = _make_engagement(client, "Big Doc Co", "One MB Report")

    line = "Firewall rule allows port 443 from admin subnet zzstressmarker.\n"
    big_text = line * (1024 * 1024 // len(line))
    resp = client.post(
        "/upload",
        files={"file": ("stress_big.txt", io.BytesIO(big_text.encode("utf-8")), "text/plain")},
        data={"engagement_id": eng["id"]},
    )
    assert resp.status_code == 200

    gen = client.post(
        "/report/generate",
        json={"engagement_id": eng["id"], "format": "docx", "focus_instructions": ""},
    )
    assert gen.status_code == 200
    data = gen.json()
    assert data["findings_count"] == len(FAKE_REPORT["findings"])
    assert data["overall_risk_rating"] == FAKE_REPORT["overall_risk_rating"]
    assert len(call_log) > 1

    download = client.get(data["download_url"])
    assert download.status_code == 200
    assert download.content[:2] == b"PK"


# ---- Many engagement creation calls ----------------------------------------

def test_fifty_engagement_creations_in_loop(client):
    """50 creation calls across 5 clients must all succeed, return unique
    readable slug ids, and all show up on the list endpoint."""
    ids = []
    for i in range(50):
        client_name = f"Stress Client {i % 5}"
        resp = client.post(
            "/engagements",
            json={"client_name": client_name, "engagement_name": f"Loop Engagement {i}"},
        )
        assert resp.status_code == 200, f"creation {i} failed: {resp.text}"
        data = resp.json()
        assert data["id"]
        assert data["status"] == "active"
        ids.append(data["id"])

    assert len(set(ids)) == 50
    assert all(" " not in i for i in ids)

    listed = client.get("/engagements").json()
    listed_ids = {e["id"] for e in listed}
    assert set(ids) <= listed_ids

    for i in range(5):
        per_client = client.get("/engagements", params={"client_id": f"stress-client-{i}"}).json()
        assert len(per_client) == 10


def test_fifty_idempotent_creation_calls_return_same_id(client):
    """Repeating one identical creation 50 times must return one stable id
    every time (the UNIQUE(client_id, name) idempotency path under load)."""
    ids = set()
    for _ in range(50):
        resp = client.post(
            "/engagements",
            json={"client_name": "Idempotent Co", "engagement_name": "Same Engagement"},
        )
        assert resp.status_code == 200
        ids.add(resp.json()["id"])
    assert len(ids) == 1


# ---- Rapid sequential chat turns -------------------------------------------

def test_thirty_rapid_chat_turns_stay_correct(client, call_log):
    """30 sequential chat calls on one engagement: every response must be the
    canned stub reply, and each turn must trigger exactly one LLM call."""
    eng = _make_engagement(client, "Chat Load Co", "Rapid Turns")

    for i in range(30):
        resp = client.post(
            "/chat",
            json={"engagement_id": eng["id"], "message": f"stress turn number {i}: what risks do you see?"},
        )
        assert resp.status_code == 200, f"chat turn {i} failed: {resp.text}"
        assert resp.json()["reply"] == "This is a stub conversational reply from Fortis."

    assert len(call_log) == 30


def test_thirty_chat_turns_across_engagements_keep_scoping(client):
    """Chat history/context is scoped per engagement: alternating turns between
    two engagements must keep every reply correct and never 404/500."""
    eng_a = _make_engagement(client, "Alternate Co", "Eng A")
    eng_b = _make_engagement(client, "Alternate Co", "Eng B")

    for i in range(15):
        for eng in (eng_a, eng_b):
            resp = client.post(
                "/chat",
                json={"engagement_id": eng["id"], "message": f"turn {i} for {eng['id']}"},
            )
            assert resp.status_code == 200
            assert resp.json()["reply"] == "This is a stub conversational reply from Fortis."


# ---- Unicode / adversarial chat input ---------------------------------------

ADVERSARIAL_MESSAGES = [
    pytest.param("what about 🔥🛡️ risks ⚠️ in the config?", id="emoji"),
    pytest.param("请分析这个防火墙配置的风险", id="cjk"),
    pytest.param("日本語のテキストと firewall の監査", id="japanese"),
    pytest.param("يرجى تحليل مخاطر الأمان في الشبكة", id="rtl-arabic"),
    pytest.param(" line1\x00\x01\x02line3\x1f\x7f", id="control-chars"),
    pytest.param("🚀" * 5000, id="emoji-flood"),
    pytest.param("A" * 100_000, id="100k-single-line"),
    pytest.param("防火墙 " * 20_000, id="cjk-flood"),
    pytest.param("\x00" * 1000, id="null-flood"),
    pytest.param("​﻿\\u200b word", id="zero-width-bom"),
    pytest.param("mixed 🔀 中文 \U0001f6e1 accenté ß 漢字", id="mixed-scripts"),
]


@pytest.mark.parametrize("message", ADVERSARIAL_MESSAGES)
def test_adversarial_chat_input_does_not_500(client, message):
    """Emoji, CJK, RTL, control chars, BOM/zero-width, and very long single-line
    messages must all get a normal 200 reply, never a 500."""
    eng = _make_engagement(client, "Adversarial Co", "Unicode Test")
    resp = client.post("/chat", json={"engagement_id": eng["id"], "message": message})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "This is a stub conversational reply from Fortis."


def test_json_escaped_lone_surrogate_message_does_not_500(client):
    """A JSON body carrying a \\ud800 escape decodes to an unpaired surrogate,
    which utf-8-strict encoding rejects. The request must still get a normal
    200 reply, not a 500."""
    eng = _make_engagement(client, "Adversarial Co", "Surrogate Test")
    body = json.dumps({"engagement_id": eng["id"], "message": "PLACEHOLDER"}).encode("utf-8")
    body = body.replace(b'"message": "PLACEHOLDER"', b'"message": "lone \\ud800 surrogate"')
    resp = client.post("/chat", content=body, headers={"content-type": "application/json"})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "This is a stub conversational reply from Fortis."


def test_json_escaped_lone_surrogate_in_engagement_creation(client):
    """Same surrogate hazard on engagement creation: an unpaired surrogate in a
    client name must not 500 (it ends up in SQLite TEXT and report headers)."""
    body = json.dumps({"client_name": "PLACEHOLDER", "engagement_name": "Surrogate Eng"}).encode("utf-8")
    body = body.replace(b'"client_name": "PLACEHOLDER"', b'"client_name": "lone \\ud800 corp"')
    resp = client.post("/engagements", content=body, headers={"content-type": "application/json"})
    assert resp.status_code == 200
    assert resp.json()["id"]


def test_json_escaped_lone_surrogate_in_upload_filename(client):
    """Raw unpaired-surrogate utf-8 bytes in the multipart filename header must
    not 500 the upload: filenames flow into chroma metadata, the disk
    filename, and prompt context."""
    eng = _make_engagement(client, "Adversarial Co", "Surrogate Upload")
    body = (
        b"--BOUNDARY\r\n"
        b'Content-Disposition: form-data; name="engagement_id"\r\n\r\n' + eng["id"].encode() + b"\r\n"
        b"--BOUNDARY\r\n"
        b'Content-Disposition: form-data; name="file"; filename="lone \xed\xa0\x80.txt"\r\n'
        b"Content-Type: text/plain\r\n\r\n"
        b"firewall rule content\n"
        b"\r\n"
        b"--BOUNDARY--\r\n"
    )
    resp = client.post(
        "/upload",
        content=body,
        headers={"content-type": "multipart/form-data; boundary=BOUNDARY"},
    )
    assert resp.status_code == 200
    assert resp.json()["chunks_indexed"] >= 1
    files = client.get(f"/upload/engagement/{eng['id']}/files").json()["files"]
    assert len(files) == 1


def test_adversarial_chat_stream_does_not_500(client):
    """The streaming endpoint must tolerate the same adversarial inputs."""
    eng = _make_engagement(client, "Adversarial Co", "Stream Test")
    for message in ("🔥🛡️ stream risks?", "请分析 防火墙", "B" * 50_000):
        resp = client.post("/chat/stream", json={"engagement_id": eng["id"], "message": message})
        assert resp.status_code == 200
        assert len(resp.text) > 0


# ---- Concurrent ingestion via threads ----------------------------------------

def test_concurrent_uploads_four_threads_five_each(client):
    """4 threads x 5 uploads each, all through one shared TestClient (the same
    anyio portal serializes requests onto the app's event loop). Chroma writes
    must all land: each engagement ends with exactly its 5 files, no errors."""
    eng_ids = []
    for i in range(4):
        eng_ids.append(_make_engagement(client, "Concurrent Co", f"Concurrent Eng {i}")["id"])

    failures = []

    def worker(tid):
        try:
            for j in range(5):
                content = f"concurrent upload {tid}-{j}: firewall rule allows port {443 + j}\n"
                resp = client.post(
                    "/upload",
                    files={"file": (f"t{tid}-{j}.txt", io.BytesIO(content.encode()), "text/plain")},
                    data={"engagement_id": eng_ids[tid]},
                )
                if resp.status_code != 200:
                    failures.append((tid, j, resp.status_code, resp.text[:200]))
        except Exception as e:
            failures.append((tid, "exception", repr(e)[:200]))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert failures == []

    for tid, eng_id in enumerate(eng_ids):
        files = client.get(f"/upload/engagement/{eng_id}/files").json()["files"]
        assert sorted(files) == sorted(f"t{tid}-{j}.txt" for j in range(5))
        hits = vectorstore.query_user_documents(eng_id, "firewall rule allows port", 3)
        assert len(hits) == 3
        assert all(h["metadata"]["engagement_id"] == eng_id for h in hits)


def test_concurrent_chat_and_uploads_mixed(client, call_log):
    """Uploads and chats racing together must all succeed and never corrupt
    per-engagement scoping."""
    eng_ids = [_make_engagement(client, "Mixed Co", f"Mixed Eng {i}")["id"] for i in range(3)]
    failures = []

    def uploader(tid):
        try:
            for j in range(5):
                resp = client.post(
                    "/upload",
                    files={"file": (f"m{tid}-{j}.txt", io.BytesIO(f"mixed content {tid} {j}".encode()), "text/plain")},
                    data={"engagement_id": eng_ids[tid]},
                )
                if resp.status_code != 200:
                    failures.append(("upload", tid, j, resp.status_code))
        except Exception as e:
            failures.append(("upload-exc", tid, repr(e)[:120]))

    def chatter(tid):
        try:
            for _ in range(10):
                resp = client.post(
                    "/chat",
                    json={"engagement_id": eng_ids[tid], "message": f"mixed chat {tid}"},
                )
                if resp.status_code != 200:
                    failures.append(("chat", tid, resp.status_code))
        except Exception as e:
            failures.append(("chat-exc", tid, repr(e)[:120]))

    threads = [threading.Thread(target=uploader, args=(t,)) for t in range(3)]
    threads += [threading.Thread(target=chatter, args=(t,)) for t in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert failures == []
    for tid, eng_id in enumerate(eng_ids):
        files = client.get(f"/upload/engagement/{eng_id}/files").json()["files"]
        assert sorted(files) == sorted(f"m{tid}-{j}.txt" for j in range(5))
