"""
Tests for structured output detection, validation, and retry logic.
"""
import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient

from core.structured import (
    detect_structured_output_request,
    FORMAT_INSTRUCTIONS,
    REPAIR_INSTRUCTIONS,
    extract_mermaid_blocks,
    is_valid_mermaid,
    reply_has_valid_diagram,
    is_valid_gfm_table,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "structured")

CANNED_MERMAID = """```mermaid
flowchart LR
    A["Test"] --> B["Diagram"]
```"""


@pytest.fixture
def client():
    from server.app import app as server_app

    with TestClient(server_app) as c:
        yield c


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    async def fake_chat(messages, temperature=0.2, json_mode=False):
        system_content = messages[0]["content"] if messages else ""
        last_content = messages[-1]["content"] if messages else ""
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
                    "control_id": "PR.PS", "remediation": "Load a real model and re-run.",
                }],
                "overall_risk_rating": "Medium",
                "recommendations_summary": ["Load a real model and re-run."],
                "diagram": CANNED_MERMAID,
            })
        if "mermaid code block" in last_content and FORMAT_INSTRUCTIONS["diagram"][:40] in last_content:
            return CANNED_MERMAID
        return "This is a stub conversational reply from Fortis."

    monkeypatch.setattr("core.analysis.chat", fake_chat)
    monkeypatch.setattr("core.routers.chat.chat", fake_chat)
    monkeypatch.setattr("core.routers.chat.chat_stream", fake_chat)


@pytest.fixture
def eng_id(client):
    resp = client.post(
        "/engagements", json={"client_name": "Structured Co", "engagement_name": "Structured Eng"}
    )
    assert resp.status_code == 200
    return resp.json()["id"]


# ---- Intent detection ----------------------------------------------------

@pytest.mark.parametrize("text,expected_intent", [
    ("draw a diagram of the network", "diagram"),
    ("show the topology", "diagram"),
    ("flow chart of the attack path", "diagram"),
    ("visualize the architecture", "diagram"),
    ("compare findings in a table", "table"),
    ("give me a matrix of hosts vs vulnerabilities", "table"),
    ("bar chart of severities", "table"),
    ("what is XSS", None),
    ("summarize the report", None),
])
def test_detect_structured_output_request(text, expected_intent):
    assert detect_structured_output_request(text) == expected_intent


# ---- Mermaid validation over fixtures ------------------------------------

@pytest.mark.parametrize("filename,expected_valid", [
    ("valid_flowchart_td.mmd", True),
    ("valid_flowchart_lr.mmd", True),
    ("valid_sequence.mmd", True),
    ("invalid_shape_syntax.mmd", False),
    ("invalid_unbalanced.mmd", False),
])
def test_is_valid_mermaid_fixtures(filename, expected_valid):
    path = os.path.join(FIXTURES_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert is_valid_mermaid(content) == expected_valid


def test_reply_has_valid_diagram_with_fences():
    valid_block = """```mermaid
flowchart LR
    A["Node A"] --> B["Node B"]
```"""
    assert reply_has_valid_diagram(valid_block) is True

    invalid_block = """```mermaid
flowchart LR
    A["Node A" --> B["Node B"]
```"""
    assert reply_has_valid_diagram(invalid_block) is False

    no_block = "Just plain text without any mermaid fences"
    assert reply_has_valid_diagram(no_block) is False


def test_extract_mermaid_blocks():
    reply = """Here is the diagram:
```mermaid
flowchart TD
    A --> B
```
And some more text.
```mermaid
sequenceDiagram
    A->>B: Hello
```"""
    blocks = extract_mermaid_blocks(reply)
    assert len(blocks) == 2
    assert "flowchart TD" in blocks[0]
    assert "sequenceDiagram" in blocks[1]


# ---- GFM table validation over fixtures ----------------------------------

@pytest.mark.parametrize("filename,expected_valid", [
    ("valid_table.md", True),
    ("invalid_table.md", False),
])
def test_is_valid_gfm_table_fixtures(filename, expected_valid):
    path = os.path.join(FIXTURES_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert is_valid_gfm_table(content) == expected_valid


# ---- FORMAT_INSTRUCTIONS and REPAIR_INSTRUCTIONS checks ------------------

def test_format_instructions_diagram_mentions_types():
    diag_instr = FORMAT_INSTRUCTIONS["diagram"]
    assert "flowchart" in diag_instr.lower()
    assert "sequencediagram" in diag_instr.lower()
    assert "```mermaid" in diag_instr


def test_format_instructions_table_mentions_separator():
    table_instr = FORMAT_INSTRUCTIONS["table"]
    assert "---" in table_instr
    assert "|" in table_instr


def test_repair_instructions_diagram_mentions_constraints():
    diag_repair = REPAIR_INSTRUCTIONS["diagram"]
    assert "flowchart" in diag_repair.lower()
    assert "@{" in diag_repair


def test_repair_instructions_table_mentions_format():
    table_repair = REPAIR_INSTRUCTIONS["table"]
    assert "|" in table_repair
    assert "---" in table_repair


# ---- Retry path: bad then good -------------------------------------------

def test_run_structured_turn_retries_on_bad_mermaid(monkeypatch):
    from core.routers.chat import run_structured_turn
    
    call_count = 0
    
    async def fake_chat_bad_then_good(messages, temperature=0.2, json_mode=False):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return """Here is a bad diagram:
```mermaid
flowchart LR
    A["Unbalanced
```"""
        else:
            return """Here is the corrected diagram:
```mermaid
flowchart LR
    A["Node A"] --> B["Node B"]
```"""
    
    import core.routers.chat as chat_module
    monkeypatch.setattr(chat_module, "chat", fake_chat_bad_then_good)
    monkeypatch.setattr("core.rag.build_chat_messages", lambda eid, msg, hist, role=None: [{"role": "user", "content": msg}])
    
    async def run_test():
        return await run_structured_turn("test-eng", "draw a diagram", [], "diagram")
    
    result = asyncio.run(run_test())
    
    assert call_count == 2
    assert "flowchart LR" in result
    assert reply_has_valid_diagram(result) is True


def test_run_structured_turn_both_attempts_bad(monkeypatch):
    from core.routers.chat import run_structured_turn
    
    call_count = 0
    
    async def fake_chat_always_bad(messages, temperature=0.2, json_mode=False):
        nonlocal call_count
        call_count += 1
        return """Bad diagram both times:
```mermaid
flowchart LR
    A["Unbalanced
```"""
    
    import core.routers.chat as chat_module
    monkeypatch.setattr(chat_module, "chat", fake_chat_always_bad)
    monkeypatch.setattr("core.rag.build_chat_messages", lambda eid, msg, hist, role=None: [{"role": "user", "content": msg}])
    
    async def run_test():
        return await run_structured_turn("test-eng", "draw a diagram", [], "diagram")
    
    result = asyncio.run(run_test())
    
    assert call_count == 2
    assert "flowchart LR" in result
    assert reply_has_valid_diagram(result) is False


# ---- Chat pipeline wiring (served by /chat, what the UI calls) -----------

def test_structured_intent_wiring_via_chat(client, eng_id):
    canned_reply = """```mermaid
flowchart LR
    A["Test"] --> B["Diagram"]
```"""

    resp = client.post(
        "/chat",
        json={"engagement_id": eng_id, "message": "draw a network diagram"},
    )

    assert resp.status_code == 200
    assert canned_reply in resp.json()["reply"]


def test_report_intent_takes_priority_via_chat(client, eng_id):
    import io as _io
    client.post(
        "/upload",
        files={"file": ("doc.txt", _io.BytesIO(b"password=hunter2\n"), "text/plain")},
        data={"engagement_id": eng_id},
    )
    resp = client.post(
        "/chat",
        json={"engagement_id": eng_id, "message": "generate a pdf report"},
    )

    assert resp.status_code == 200
    reply = resp.json()["reply"]
    assert "report" in reply.lower()
    assert "```mermaid" not in reply, "report intent must win over diagram intent"


def test_plain_message_takes_normal_chat(client, eng_id):
    resp = client.post(
        "/chat",
        json={"engagement_id": eng_id, "message": "what is XSS?"},
    )

    assert resp.status_code == 200
    assert "stub conversational reply" in resp.json()["reply"].lower()
