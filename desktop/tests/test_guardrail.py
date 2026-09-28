"""Tests for the cybersecurity topic guardrail now enforced in the HTTP chat
pipeline (previously it lived only on the removed orchestrator path)."""
import pytest
from fastapi.testclient import TestClient

from core import guardrail


@pytest.fixture
def client():
    from server.app import app as server_app

    with TestClient(server_app) as c:
        yield c


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    async def fake_chat(messages, temperature=0.2, json_mode=False):
        return "This is a stub conversational reply from Fortis."

    async def fake_chat_stream(messages, temperature=0.2):
        yield "This is a stub conversational reply from Fortis."

    monkeypatch.setattr("core.routers.chat.chat", fake_chat)
    monkeypatch.setattr("core.routers.chat.chat_stream", fake_chat_stream)


def test_detection_matrix():
    assert guardrail.is_cybersecurity_related("what risks do you see?")
    assert guardrail.is_cybersecurity_related("review these firewall rules")
    assert guardrail.is_cybersecurity_related("is my data encrypted?")
    assert guardrail.is_cybersecurity_related("patch management policy")
    assert guardrail.is_cybersecurity_related("generate a report")
    assert not guardrail.is_cybersecurity_related("what should I cook for dinner?")
    assert not guardrail.is_cybersecurity_related("who won the game last night?")
    assert not guardrail.is_cybersecurity_related("hello")


def test_non_latin_scripts_are_not_hard_refused():
    """The keyword list is English-only; CJK/Arabic input must not be
    keyword-refused (the persona handles those languages)."""
    assert guardrail.is_cybersecurity_related("请分析这个防火墙配置的风险")
    assert guardrail.is_cybersecurity_related("يرجى تحليل مخاطر الأمان")


def test_chat_refuses_offtopic(client, monkeypatch):
    eng_calls = []
    monkeypatch.setattr(
        "core.routers.chat.store.append_chat_message",
        lambda eid, role, content: eng_calls.append(role) or 1,
    )
    eng = client.post("/engagements", json={"client_name": "G Co", "engagement_name": "Guard"}).json()
    resp = client.post("/chat", json={"engagement_id": eng["id"], "message": "what's for dinner?"})
    assert resp.status_code == 200
    assert "cybersecurity advisory assistant" in resp.json()["reply"]
    assert eng_calls == ["user", "assistant"], "refusal turn is persisted too"


def test_stream_refuses_offtopic(client):
    eng = client.post("/engagements", json={"client_name": "G Co", "engagement_name": "Guard Stream"}).json()
    resp = client.post("/chat/stream", json={"engagement_id": eng["id"], "message": "tell me a joke"})
    assert resp.status_code == 200
    assert "cybersecurity advisory assistant" in resp.text


def test_inscope_message_skips_refusal(client):
    eng = client.post("/engagements", json={"client_name": "G Co", "engagement_name": "In Scope"}).json()
    resp = client.post("/chat", json={"engagement_id": eng["id"], "message": "what risks in my config?"})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "This is a stub conversational reply from Fortis."


def test_model_wrong_refusal_is_retried(client, monkeypatch):
    calls = []

    async def refusing_chat(messages, temperature=0.2, json_mode=False):
        full = messages[-1]["content"]
        calls.append(full[:40])
        if "Do not refuse" in full:
            return "Here is the analysis you asked for."
        return "I'm Fortis, a cybersecurity advisory assistant. I can only help with security-related topics. Please ask a cybersecurity question."

    monkeypatch.setattr("core.routers.chat.chat", refusing_chat)
    eng = client.post("/engagements", json={"client_name": "G Co", "engagement_name": "Retry"}).json()
    resp = client.post("/chat", json={"engagement_id": eng["id"], "message": "analyze my firewall config for risks"})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "Here is the analysis you asked for."
    assert len(calls) == 2, "one refusal triggers exactly one forced retry"
