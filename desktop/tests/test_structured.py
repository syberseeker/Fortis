"""
Tests for structured output detection, validation, and retry logic.
"""
import asyncio
import os

import pytest

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


# ---- Orchestrator wiring -------------------------------------------------

def test_orchestrator_structured_intent_wiring(monkeypatch):
    from server.orchestration import Orchestrator
    
    canned_reply = """```mermaid
flowchart LR
    A["Test"] --> B["Diagram"]
```"""
    
    async def fake_run_structured_turn(engagement_id, message, history, intent, role=None):
        return canned_reply
    
    monkeypatch.setattr("server.orchestration.run_structured_turn", fake_run_structured_turn)
    monkeypatch.setattr("server.orchestration.store.get_active_engagement", lambda uid: {
        "id": "test-eng", "client_name": "Test", "name": "Test", "status": "active"
    })
    monkeypatch.setattr("server.orchestration.vectorstore.list_engagement_files", lambda eid: [])
    
    orch = Orchestrator()
    result = orch.chat_turn("draw a network diagram", history=[])
    
    assert result["kind"] == "chat"
    assert canned_reply in result["text"]


def test_orchestrator_report_intent_takes_priority(monkeypatch):
    from server.orchestration import Orchestrator
    
    def fake_generate_report_download(self, engagement_id, fmt, focus_text):
        return {
            "download_url": "/report/download/test.docx",
            "findings_count": 1,
            "overall_risk_rating": "Medium",
            "engagement": {"client_name": "Test", "engagement_name": "Test"},
        }
    
    monkeypatch.setattr(Orchestrator, "generate_report_download", fake_generate_report_download)
    monkeypatch.setattr("server.orchestration.store.get_active_engagement", lambda uid: {
        "id": "test-eng", "client_name": "Test", "name": "Test", "status": "active"
    })
    monkeypatch.setattr("server.orchestration.vectorstore.list_engagement_files", lambda eid: [])
    
    orch = Orchestrator()
    result = orch.chat_turn("generate a pdf report", history=[])
    
    assert result["kind"] == "report"


def test_orchestrator_plain_message_takes_normal_chat(monkeypatch):
    from server.orchestration import Orchestrator
    
    async def fake_chat(messages):
        return "Normal chat response"
    
    monkeypatch.setattr("engine.chat", fake_chat)
    monkeypatch.setattr("server.orchestration.store.get_active_engagement", lambda uid: {
        "id": "test-eng", "client_name": "Test", "name": "Test", "status": "active"
    })
    monkeypatch.setattr("server.orchestration.vectorstore.list_engagement_files", lambda eid: [])
    monkeypatch.setattr("server.orchestration.rag.build_chat_messages", lambda eid, msg, hist, role=None, retrieval_query=None: [{"role": "user", "content": msg}])
    
    orch = Orchestrator()
    result = orch.chat_turn("what is XSS?", history=[])
    
    assert result["kind"] == "chat"
    assert "Normal chat response" in result["text"]
