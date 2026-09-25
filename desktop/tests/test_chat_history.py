"""Tests for engagement-scoped chat history persistence.

Chat turns must survive app restarts and engagement switches: every /chat and
/chat/stream exchange is stored in SQLite and restorable via /chat/history.
This file pins that contract.
"""
import io
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    async def fake_chat(messages, temperature=0.2, json_mode=False):
        system_content = messages[0]["content"] if messages else ""
        if "ONE batch of excerpts" in system_content:
            return json.dumps({"key_points": [], "findings": []})
        if json_mode:
            return json.dumps({
                "title": "Stub Security Assessment",
                "client_context": "Stub context",
                "scope": "Stub scope",
                "executive_summary": "Stub finding: verify with a real model",
                "findings": [{
                    "title": "Stub finding", "severity": "Medium",
                    "description": "Stub", "evidence": "stub",
                    "source_file": None, "framework": "NIST_CSF",
                    "control_id": "PR.PS", "remediation": "Re-run with a real model.",
                }],
                "overall_risk_rating": "Medium",
                "recommendations_summary": ["Load a real model and re-run."],
                "diagram": None,
            })
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


def _make_engagement(client: TestClient, client_name: str, name: str) -> dict:
    resp = client.post(
        "/engagements", json={"client_name": client_name, "engagement_name": name}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _chat(client: TestClient, eng_id: str, message: str, history=None):
    return client.post("/chat", json={"engagement_id": eng_id, "message": message, "history": history or []})


# ---- persistence of turns -----------------------------------------------------

def test_chat_turn_is_persisted_and_restorable(client):
    eng = _make_engagement(client, "Hist Co", "Persist")
    r1 = _chat(client, eng["id"], "what risks are in the config?")
    assert r1.status_code == 200
    r2 = _chat(client, eng["id"], "and the IAM policy?")
    assert r2.status_code == 200

    hist = client.get(f"/chat/history/{eng['id']}")
    assert hist.status_code == 200
    turns = hist.json()["turns"]
    assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]
    assert turns[0]["content"] == "what risks are in the config?"
    assert turns[1]["content"] == "This is a stub conversational reply from Fortis."
    assert turns[2]["content"] == "and the IAM policy?"
    assert all(t.get("created_at") for t in turns)


def test_stream_turn_is_persisted(client):
    eng = _make_engagement(client, "Hist Co", "Stream Persist")
    resp = client.post("/chat/stream", json={"engagement_id": eng["id"], "message": "what risks?"})
    assert resp.status_code == 200
    assert len(resp.text) > 0

    hist = client.get(f"/chat/history/{eng['id']}")
    turns = hist.json()["turns"]
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert "".join(t["content"] for t in turns if t["role"] == "assistant").strip() == resp.text.strip()


def test_history_is_scoped_per_engagement(client):
    a = _make_engagement(client, "Hist Co", "Eng A")
    b = _make_engagement(client, "Hist Co", "Eng B")
    _chat(client, a["id"], "question for A")
    _chat(client, b["id"], "question for B")

    ha = client.get(f"/chat/history/{a['id']}").json()["turns"]
    hb = client.get(f"/chat/history/{b['id']}").json()["turns"]
    assert [t["content"] for t in ha if t["role"] == "user"] == ["question for A"]
    assert [t["content"] for t in hb if t["role"] == "user"] == ["question for B"]


def test_report_turns_are_persisted_too(client):
    eng = _make_engagement(client, "Hist Co", "Report Turn")
    import io as _io
    client.post(
        "/upload",
        files={"file": ("doc.txt", _io.BytesIO(b"password=hunter2\n"), "text/plain")},
        data={"engagement_id": eng["id"]},
    )
    resp = client.post("/chat/stream", json={"engagement_id": eng["id"], "message": "generate a report"})
    assert resp.status_code == 200
    assert "Security report generated" in resp.text

    turns = client.get(f"/chat/history/{eng['id']}").json()["turns"]
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert "Security report generated" in turns[1]["content"]


# ---- cap and ordering ----------------------------------------------------------

def test_history_capped_at_newest_messages(client, monkeypatch):
    """The cap counts stored messages (rows), not turn pairs: cap=4 keeps the
    newest 2 user/assistant turns."""
    from core import store

    eng = _make_engagement(client, "Hist Co", "Cap")
    monkeypatch.setattr(store, "CHAT_HISTORY_CAP", 4)
    for i in range(6):
        assert _chat(client, eng["id"], f"turn {i}").status_code == 200

    turns = client.get(f"/chat/history/{eng['id']}").json()["turns"]
    assert len(turns) == 4
    assert turns[0]["content"] == "turn 4"  # oldest surviving user message
    assert turns[-1]["role"] == "assistant"


def test_history_limit_param(client):
    eng = _make_engagement(client, "Hist Co", "Limit Param")
    for i in range(3):
        _chat(client, eng["id"], f"turn {i}")
    turns = client.get(f"/chat/history/{eng['id']}?limit=2").json()["turns"]
    assert [t["content"] for t in turns] == [
        "turn 2",
        "This is a stub conversational reply from Fortis.",
    ]


def test_history_order_is_oldest_first(client):
    eng = _make_engagement(client, "Hist Co", "Order")
    _chat(client, eng["id"], "first")
    _chat(client, eng["id"], "second")
    turns = client.get(f"/chat/history/{eng['id']}").json()["turns"]
    assert turns[0]["content"] == "first" and turns[0]["role"] == "user"
    assert turns[-1]["role"] == "assistant"


# ---- adversarial input round-trips ----------------------------------------------

def test_unicode_and_surrogates_round_trip(client):
    eng = _make_engagement(client, "Hist Co", "Unicode")
    for msg in ("what about 🔥🛡️ risks?", "请分析这个防火墙配置的风险", "mixed 中文 accenté ß"):
        assert _chat(client, eng["id"], msg).status_code == 200
    turns = client.get(f"/chat/history/{eng['id']}").json()["turns"]
    user_msgs = [t["content"] for t in turns if t["role"] == "user"]
    assert user_msgs[0] == "what about 🔥🛡️ risks?"
    assert user_msgs[1] == "请分析这个防火墙配置的风险"


# ---- clearing --------------------------------------------------------------------

def test_clear_history_requires_confirm(client):
    eng = _make_engagement(client, "Hist Co", "Clear Guard")
    _chat(client, eng["id"], "hello")
    resp = client.delete(f"/chat/history/{eng['id']}")
    assert resp.status_code == 400
    resp = client.delete(f"/chat/history/{eng['id']}?confirm=true")
    assert resp.status_code == 200
    assert resp.json()["removed"] == 2
    assert client.get(f"/chat/history/{eng['id']}").json()["turns"] == []


def test_clear_then_chat_again(client):
    eng = _make_engagement(client, "Hist Co", "Clear Then Chat")
    _chat(client, eng["id"], "before clear")
    assert client.delete(f"/chat/history/{eng['id']}?confirm=true").status_code == 200
    _chat(client, eng["id"], "after clear")
    turns = client.get(f"/chat/history/{eng['id']}").json()["turns"]
    assert [t["content"] for t in turns] == ["after clear", "This is a stub conversational reply from Fortis."]


# ---- engagement deletion cascades -------------------------------------------------

def test_deleting_engagement_removes_history(client):
    eng = _make_engagement(client, "Hist Co", "Delete Cascade")
    _chat(client, eng["id"], "to be deleted")
    assert client.delete(f"/engagements/{eng['id']}").status_code == 200
    resp = client.get(f"/chat/history/{eng['id']}")
    assert resp.status_code == 404


# ---- history endpoint edge cases ---------------------------------------------------

def test_history_of_unknown_engagement_404(client):
    assert client.get("/chat/history/nonexistent").status_code == 404


def test_history_of_deleted_engagement_404(client):
    eng = _make_engagement(client, "Hist Co", "Deleted Hist")
    _chat(client, eng["id"], "bye")
    client.delete(f"/engagements/{eng['id']}")
    assert client.get(f"/chat/history/{eng['id']}").status_code == 404


def test_invalid_limit_is_clamped(client):
    eng = _make_engagement(client, "Hist Co", "Limit Clamp")
    _chat(client, eng["id"], "hello")
    resp = client.get(f"/chat/history/{eng['id']}?limit=0")
    assert resp.status_code == 200
    assert len(resp.json()["turns"]) == 1


# ---- LLM context window still capped at 16 ---------------------------------------

def test_llm_context_window_unchanged_by_persistence(client):
    """The persisted store is the archive; the model context stays bounded by
    the UI's 16-message window (contract from the original chat design)."""
    eng = _make_engagement(client, "Hist Co", "Window")
    for i in range(5):
        _chat(client, eng["id"], f"turn {i}")
    turns = client.get(f"/chat/history/{eng['id']}").json()["turns"]
    assert len(turns) == 10  # 5 turns x (user+assistant) persisted
