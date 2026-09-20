"""
Offline tests for the tiered Advanced RAG upgrade, mirroring
desktop/tests/test_rag_upgrade.py against the legacy backend stack: BM25
tokenization, hybrid retrieval + RRF with hash-stub embeddings, mark-dirty
cache invalidation, cross-engagement isolation, level gating, query
condensation, retrieval_query plumbing, reranker fallback, chunk enrichment,
and the hostile-regex performance regression. No GPU, no model, no network —
see conftest.py for the environment (hash-stub embeddings, RERANK_BACKEND=none).
"""
import asyncio
import time

import pytest

from app import bm25, config, ingestion, rag, vectorstore
from app.condense import condense_query, heuristic_condense


# ---- BM25 tokenization -------------------------------------------------------

def test_tokenize_lowercases_and_splits_on_non_alphanumerics():
    assert bm25.tokenize("A.5.15 CWE-79 tls_min_version=1.2") == [
        "a", "5", "15", "cwe", "79", "tls", "min", "version", "1", "2",
    ]


# ---- mark-dirty invalidation -------------------------------------------------

def test_mark_dirty_clears_hits_after_clear_engagement():
    eng = "inv-acme-q3"
    vectorstore.add_user_document_chunks(
        eng, "audit.txt", ["The quick brown fox jumps over the logging policy."]
    )
    hits = vectorstore.query_user_documents_hybrid(eng, "fox", 5)
    assert hits, "hybrid query should hit before clear_engagement"

    vectorstore.clear_engagement(eng)
    hits_after = vectorstore.query_user_documents_hybrid(eng, "fox", 5)
    assert hits_after == []


# ---- cross-engagement isolation ---------------------------------------------

def test_hybrid_query_isolated_per_engagement():
    vectorstore.add_user_document_chunks(
        "iso-a", "secret.txt", ["This document contains zzsecretmarkerzz in its text."]
    )
    vectorstore.add_user_document_chunks(
        "iso-b", "other.txt", ["Unrelated hardening guidance about firewalls and patching."]
    )

    hits = vectorstore.query_user_documents_hybrid("iso-b", "zzsecretmarkerzz", 5)
    assert hits, "iso-B should still retrieve its own unrelated chunk"
    assert all(h["metadata"]["engagement_id"] == "iso-b" for h in hits)


# ---- RRF fusion over the public API ------------------------------------------

def test_hybrid_rescues_exact_identifier_doc():
    eng = "rrf-acme-q3"
    vectorstore.add_user_document_chunks(
        eng,
        "identifiers.txt",
        ["The logging configuration must include CWE-778 references for audit controls."],
    )
    vectorstore.add_user_document_chunks(
        eng,
        "phrases.txt",
        ["Transport security should use strong encryption protocols and careful "
         "certificate validation practices."],
    )

    top1 = vectorstore.query_user_documents_hybrid(eng, "CWE-778", 1)
    assert top1, "hybrid query on the exact identifier should return at least one hit"
    assert top1[0]["metadata"]["filename"] == "identifiers.txt"

    ordered = vectorstore.query_user_documents_hybrid(eng, "CWE-778", 5)
    assert ordered[0]["metadata"]["filename"] == "identifiers.txt"


# ---- level gating ------------------------------------------------------------

def _canned_user_hit(text="canned user text"):
    return {"text": text, "metadata": {"filename": "canned.txt", "chunk_index": 0}}


def _canned_framework_hit(text="canned framework text"):
    return {
        "text": text,
        "metadata": {"framework": "NIST_CSF", "control_id": "PR.PS"},
    }


def test_basic_level_uses_dense_path_only(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("hybrid retrieval must not run at rag_level=basic")

    monkeypatch.setattr(vectorstore, "query_user_documents_hybrid", _boom)
    monkeypatch.setattr(vectorstore, "query_frameworks_hybrid", _boom)
    monkeypatch.setattr(vectorstore, "query_user_documents", lambda *a, **k: [])
    monkeypatch.setattr(vectorstore, "query_frameworks", lambda *a, **k: [])

    monkeypatch.setattr(config.settings, "rag_level", "basic")
    block = rag.build_context_block("level-eng", "any query")

    assert "=== No uploaded documents matched this query ===" in block


def test_standard_level_calls_hybrid_and_rerank(monkeypatch):
    user_hits = [_canned_user_hit("hybrid user excerpt text")]
    framework_hits = [_canned_framework_hit("hybrid framework control text")]

    captured = {}

    def fake_user_hybrid(engagement_id, query, top_k, candidate_multiplier=4):
        captured["user"] = (engagement_id, query, top_k, candidate_multiplier)
        return list(user_hits)

    def fake_framework_hybrid(query, top_k, candidate_multiplier=4, framework=None):
        captured["framework"] = (query, top_k, candidate_multiplier)
        return list(framework_hits)

    monkeypatch.setattr(vectorstore, "query_user_documents_hybrid", fake_user_hybrid)
    monkeypatch.setattr(vectorstore, "query_frameworks_hybrid", fake_framework_hybrid)
    monkeypatch.setattr(
        "app.rerank.rerank", lambda q, hits, top_k: hits[:top_k]
    )

    monkeypatch.setattr(config.settings, "rag_level", "standard")
    block = rag.build_context_block("level-eng", "hybrid query text")

    assert captured["user"][0] == "level-eng"
    assert captured["user"][1] == "hybrid query text"
    assert captured["user"][2] == config.settings.top_k_user_doc
    assert captured["user"][3] == config.settings.rerank_candidate_multiplier
    assert captured["framework"][0] == "hybrid query text"
    assert captured["framework"][1] == config.settings.top_k_framework
    assert captured["framework"][2] == config.settings.rerank_candidate_multiplier
    assert "hybrid user excerpt text" in block
    assert "hybrid framework control text" in block


# ---- condensation ------------------------------------------------------------

def test_heuristic_condense_without_history_returns_message():
    assert heuristic_condense(None, "m") == "m"


def test_heuristic_condense_prepends_last_user_turn():
    history = [{"role": "user", "content": "what is XSS?"}]
    assert heuristic_condense(history, "how to prevent") == "what is XSS? how to prevent"


def test_heuristic_condense_with_assistant_only_history_returns_message():
    history = [{"role": "assistant", "content": "some earlier answer"}]
    assert heuristic_condense(history, "follow-up question") == "follow-up question"


def test_condense_query_standard_matches_heuristic_without_llm(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("standard tier must not call the LLM for condensation")

    import app.llm as app_llm

    monkeypatch.setattr(app_llm, "chat", _boom)
    history = [{"role": "user", "content": "what is XSS?"}]

    result = asyncio.run(condense_query(history, "how to prevent", "standard"))

    assert result == heuristic_condense(history, "how to prevent")
    assert result == "what is XSS? how to prevent"


# ---- retrieval_query passthrough ----------------------------------------------

def test_build_chat_messages_retrieval_query_drives_retrieval(monkeypatch):
    vectorstore.add_user_document_chunks(
        "rq-eng", "notes.txt", ["To debug standalone q flows, use zzretrqmarker for tracing."]
    )
    monkeypatch.setattr(config.settings, "rag_level", "basic")

    messages = rag.build_chat_messages("rq-eng", "msg", [], retrieval_query="standalone q")
    user_content = messages[-1]["content"]
    assert "standalone q" in user_content
    assert "zzretrqmarker" in user_content


def test_build_chat_messages_defaults_retrieval_query_to_message(monkeypatch):
    monkeypatch.setattr(config.settings, "rag_level", "basic")

    messages = rag.build_chat_messages("rq-eng-default", "msg", [])
    user_content = messages[-1]["content"]
    assert "=== Client question ===\nmsg" in user_content
    assert "=== Client question ===\nstandalone q" not in user_content


# ---- reranker fallback ---------------------------------------------------------

def test_rerank_falls_back_to_input_order_when_unavailable(monkeypatch):
    from app import rerank

    monkeypatch.setattr(rerank, "is_available", lambda: False)
    hits = [{"text": f"hit {i}", "metadata": {"chunk_index": i}} for i in range(5)]

    assert rerank.rerank("q", hits, 2) == hits[:2]


# ---- chunk enrichment -----------------------------------------------------------

def test_chunk_text_enriched_plain_header():
    chunks = ingestion.chunk_text_enriched("plain only", "f.txt")
    assert chunks[0].startswith("[From f.txt]\n")


def test_chunk_text_enriched_multipage_headers():
    para1 = "--- Page 1 ---\n" + "wordone " * 150
    para2 = "--- Page 2 ---\n" + "wordtwo " * 100
    para3 = "--- Page 3 ---\n" + "wordthree " * 100
    text = "\n\n".join([para1, para2, para3])

    chunks = ingestion.chunk_text_enriched(text, "doc.pdf")
    assert len(chunks) == 3
    assert "page 1" in chunks[0].split("\n")[0]
    assert "pages 2-3" in chunks[1].split("\n")[0]


def test_chunk_text_enriched_sanitizes_hostile_filename():
    chunks = ingestion.chunk_text_enriched("plain only", "we\r\nird\x00name.txt")
    header = chunks[0].split("\n")[0]
    assert header.startswith("[From ") and header.endswith("]")
    for ch in ("\r", "\n", "\x00"):
        assert ch not in header


def test_chunk_text_enriched_caps_filename_at_120_chars():
    chunks = ingestion.chunk_text_enriched("plain only", "x" * 300)
    header = chunks[0].split("\n")[0]
    assert header.startswith("[From ")
    assert header.endswith("]")
    assert len(header) <= len("[From ") + 120 + len("]")


# ---- hostile regex performance regression --------------------------------------

def test_chunk_text_enriched_handles_hostile_sheet_input_fast():
    evil = "--- Sheet: " + " ---" * 20000 + " " + "x" * 100000
    start = time.time()
    chunks = ingestion.chunk_text_enriched(evil, "evil.csv")
    elapsed = time.time() - start

    assert elapsed < 5.0
    assert chunks
    assert chunks[0].startswith("[From evil.csv]")
