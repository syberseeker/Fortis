"""
Regression locks for the performance-improvement wave, fully offline:

- Event-loop offloading: chat/upload/engagements endpoints must run blocking
  work (RAG message building, SQLite persistence, text extraction, Chroma
  file listing) via starlette.concurrency.run_in_threadpool, never on the
  event loop thread. Detected by proving asyncio.get_running_loop() fails
  inside the patched work function and that it ran on a different thread
  than the endpoint coroutine.
- TTL caches: engine.hardware.detect()/_vram_gb(), the analysis framework
  context block, known framework pairs, ensure_cache_table() DDL-once, and
  chart path caching must not redo work inside their validity window.
- Backend resolution: auto mode short-circuits to llama when a model is
  already loaded, and the models-dir scan behind current_model_path() is
  cached per TTL.
"""
import asyncio
import io
import os
import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

import engine as engine_pkg
import engine.llama_engine as llama_engine
from core import vectorstore
from core.config import settings
from core.main import app as core_app
from core.reports import charts
from core.reports.schema import Finding, SecurityReport


@pytest.fixture
def client():
    with TestClient(core_app) as c:
        yield c


def _make_engagement(client, client_name, engagement_name):
    return client.post(
        "/engagements",
        json={"client_name": client_name, "engagement_name": engagement_name},
    ).json()


def _running_loop_present():
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _canned_messages(user_message):
    return [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": user_message},
    ]


async def _fake_chat(messages, temperature=0.2, json_mode=False, max_tokens=None):
    return "Stub security answer grounded in the excerpts."


# ---- Event-loop offloading -------------------------------------------------

def test_chat_build_messages_runs_off_event_loop(client, monkeypatch):
    rec = {"loop_thread": None, "threads": [], "on_loop": []}

    async def spy_condense(history, message, level="standard"):
        rec["loop_thread"] = threading.get_ident()
        return message

    def spy_build(engagement_id, user_message, history=None, role=None, retrieval_query=None, **kwargs):
        rec["threads"].append(threading.get_ident())
        rec["on_loop"].append(_running_loop_present())
        return _canned_messages(user_message)

    monkeypatch.setattr("core.routers.chat.condense_query", spy_condense)
    monkeypatch.setattr("core.rag.build_chat_messages", spy_build)
    monkeypatch.setattr("core.routers.chat.chat", _fake_chat)

    eng = _make_engagement(client, "Perf Chat Co", "Offloop Build")
    resp = client.post(
        "/chat", json={"engagement_id": eng["id"], "message": "what risks do you see?"}
    )
    assert resp.status_code == 200
    assert resp.json()["reply"] == "Stub security answer grounded in the excerpts."
    assert rec["threads"], "build_chat_messages was never called"
    assert not any(rec["on_loop"]), "build_chat_messages ran on the event loop thread"
    assert all(t != rec["loop_thread"] for t in rec["threads"])


def test_chat_persistence_runs_off_event_loop(client, monkeypatch):
    rec = {"loop_thread": None, "roles": [], "threads": [], "on_loop": []}

    async def spy_condense(history, message, level="standard"):
        rec["loop_thread"] = threading.get_ident()
        return message

    def spy_build(engagement_id, user_message, history=None, role=None, retrieval_query=None, **kwargs):
        return _canned_messages(user_message)

    def spy_append(engagement_id, role, content, *args, **kwargs):
        rec["roles"].append(role)
        rec["threads"].append(threading.get_ident())
        rec["on_loop"].append(_running_loop_present())
        return 1

    monkeypatch.setattr("core.routers.chat.condense_query", spy_condense)
    monkeypatch.setattr("core.rag.build_chat_messages", spy_build)
    monkeypatch.setattr("core.routers.chat.chat", _fake_chat)
    monkeypatch.setattr("core.store.append_chat_message", spy_append)

    eng = _make_engagement(client, "Perf Chat Co", "Offloop Persist")
    resp = client.post(
        "/chat", json={"engagement_id": eng["id"], "message": "what risks do you see?"}
    )
    assert resp.status_code == 200
    assert rec["roles"] == ["user", "assistant"]
    assert rec["on_loop"] == [False, False], "persistence ran on the event loop thread"
    assert all(t != rec["loop_thread"] for t in rec["threads"])


def test_upload_extract_text_runs_off_event_loop(client, monkeypatch):
    rec = {"paths": [], "threads": [], "on_loop": []}
    main_thread = threading.get_ident()

    def spy_extract(path):
        rec["paths"].append(path)
        rec["threads"].append(threading.get_ident())
        rec["on_loop"].append(_running_loop_present())
        return "password=hunter2\nallow_all_inbound=true\n"

    monkeypatch.setattr("core.routers.upload.extract_text", spy_extract)

    eng = _make_engagement(client, "Perf Upload Co", "Offloop Extract")
    resp = client.post(
        "/upload",
        files={"file": ("cfg.txt", io.BytesIO(b"password=hunter2\n"), "text/plain")},
        data={"engagement_id": eng["id"]},
    )
    assert resp.status_code == 200
    assert resp.json()["chunks_indexed"] >= 1
    assert len(rec["paths"]) == 1
    assert rec["on_loop"] == [False], "extract_text ran on the event loop thread"
    assert rec["threads"][0] != main_thread


def test_engagement_detail_file_listing_runs_off_event_loop(client, monkeypatch):
    rec = []
    real_list = vectorstore.list_engagement_files

    def spy(engagement_id):
        rec.append(_running_loop_present())
        return real_list(engagement_id)

    monkeypatch.setattr("core.vectorstore.list_engagement_files", spy)

    eng = _make_engagement(client, "Perf Engagement Co", "Offloop Files")
    detail = client.get(f"/engagements/{eng['id']}")
    assert detail.status_code == 200
    assert detail.json()["files"] == []
    assert rec, "list_engagement_files was never called"
    assert not any(rec), "list_engagement_files ran on the event loop thread"


# ---- Hardware TTL cache -----------------------------------------------------

def test_detect_is_ttl_cached(monkeypatch):
    from engine import hardware

    counters = {"ram": 0, "vram": 0}

    def fake_ram():
        counters["ram"] += 1
        return 16.0

    def fake_probe():
        counters["vram"] += 1
        return 8.0

    monkeypatch.setattr(hardware, "_ram_gb", fake_ram)
    monkeypatch.setattr(hardware, "_probe_vram_gb", fake_probe)
    monkeypatch.setattr(hardware, "_HW_CACHE", None)
    monkeypatch.setattr(hardware, "_VRAM_CACHE", None)

    first = hardware.detect()
    second = hardware.detect()

    assert first is second
    assert counters == {"ram": 1, "vram": 1}
    assert first.ram_gb == 16.0
    assert first.vram_gb == 8.0
    assert first.has_nvidia is True


def test_detect_rebuilds_after_ttl_expiry(monkeypatch):
    from engine import hardware

    counters = {"ram": 0, "vram": 0}

    def fake_ram():
        counters["ram"] += 1
        return 16.0

    def fake_probe():
        counters["vram"] += 1
        return 8.0

    monkeypatch.setattr(hardware, "_ram_gb", fake_ram)
    monkeypatch.setattr(hardware, "_probe_vram_gb", fake_probe)
    monkeypatch.setattr(hardware, "_VRAM_CACHE", None)
    stale = hardware.Hardware(ram_gb=1.0, vram_gb=2.0, has_nvidia=True)
    monkeypatch.setattr(
        hardware,
        "_HW_CACHE",
        (time.monotonic() - hardware._HW_TTL_SECONDS - 1.0, stale),
    )

    fresh = hardware.detect()

    assert fresh is not stale
    assert counters["ram"] == 1
    assert counters["vram"] == 1
    assert fresh.ram_gb == 16.0


def test_vram_gb_cached_within_ttl(monkeypatch):
    from engine import hardware

    probes = []

    def fake_probe():
        probes.append(1)
        return 6.0

    monkeypatch.setattr(hardware, "_probe_vram_gb", fake_probe)
    monkeypatch.setattr(hardware, "_VRAM_CACHE", None)

    assert hardware._vram_gb() == 6.0
    assert hardware._vram_gb() == 6.0
    assert len(probes) == 1


def test_vram_cache_expires_past_ttl(monkeypatch):
    from engine import hardware

    probes = []

    def fake_probe():
        probes.append(1)
        return 6.0

    monkeypatch.setattr(hardware, "_probe_vram_gb", fake_probe)
    monkeypatch.setattr(
        hardware,
        "_VRAM_CACHE",
        (time.monotonic() - hardware._VRAM_TTL_SECONDS - 1.0, 42.0),
    )

    assert hardware._vram_gb() == 6.0
    assert len(probes) == 1


# ---- Report caches ----------------------------------------------------------

def test_framework_context_block_cache_avoids_requeries(monkeypatch):
    from core import analysis

    calls = []

    def fake_query_frameworks(query, top_k, framework=None):
        calls.append(query)
        return [
            {
                "metadata": {
                    "framework": "NIST_CSF",
                    "control_id": "PR.AA",
                    "version": "CSF 1.1",
                },
                "text": "Identify and communicate cybersecurity risk.",
            }
        ]

    monkeypatch.setattr(vectorstore, "query_frameworks", fake_query_frameworks)
    monkeypatch.setattr(analysis, "_FRAMEWORK_CONTEXT_CACHE", None)

    first = analysis._framework_context_block()
    calls_after_first = len(calls)
    second = analysis._framework_context_block()

    assert calls_after_first > 0
    assert len(calls) == calls_after_first, "second call re-queried the corpus"
    assert first == second
    assert "NIST_CSF PR.AA" in first


def test_known_framework_pairs_cached_within_ttl(monkeypatch):
    from core import report_service

    class StubCollection:
        def __init__(self):
            self.get_calls = 0

        def get(self, include=None):
            self.get_calls += 1
            return {"metadatas": [{"framework": "NIST_CSF", "control_id": "PR.AA"}]}

    stub = StubCollection()
    monkeypatch.setattr(vectorstore, "framework_collection", lambda: stub)
    monkeypatch.setattr(report_service, "_KNOWN_PAIRS_CACHE", None)

    pairs = report_service._known_framework_pairs()
    again = report_service._known_framework_pairs()

    assert pairs == {("NIST_CSF", "PR.AA")}
    assert again == pairs
    assert stub.get_calls == 1


def test_ensure_cache_table_runs_ddl_once(monkeypatch, tmp_path):
    from core import report_service

    db_path = str(tmp_path / "perf-cache.sqlite3")
    monkeypatch.setattr(settings, "db_path", db_path)
    monkeypatch.setattr(report_service, "_DDL_DONE", set())

    real_connect = sqlite3.connect
    connects = []

    def counting_connect(*args, **kwargs):
        connects.append(args[0] if args else kwargs.get("database"))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", counting_connect)

    report_service.ensure_cache_table()
    connects_after_first = len(connects)
    report_service.ensure_cache_table()

    assert connects_after_first == 1
    assert len(connects) == 1, "second ensure_cache_table() opened the database again"

    conn = real_connect(db_path)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        conn.close()
    assert "report_cache" in tables


# ---- Chart caching ----------------------------------------------------------

def test_severity_chart_cached_path_reused_and_renders_once(monkeypatch):
    unique_title = f"Perf Chart Cache {os.getpid()}-{time.monotonic_ns()}"
    report = SecurityReport(
        title=unique_title,
        client_context="Client used to lock the chart cache.",
        scope="Chart caching regression.",
        executive_summary="One finding.",
        findings=[
            Finding(
                title="Weak credential storage",
                severity="High",
                description="Plaintext credential on disk.",
                evidence="password=hunter2",
                source_file="config.txt",
                framework="NIST_CSF",
                control_id="PR.AA",
                remediation="Use a secrets manager.",
            )
        ],
        overall_risk_rating="High",
        recommendations_summary=["Adopt a secrets manager"],
        diagram=None,
    )

    calls = []

    def fake_renderer(rep, output_path):
        calls.append(output_path)
        with open(output_path, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n")
        return output_path

    monkeypatch.setattr(charts, "render_severity_chart", fake_renderer)
    cache_key = f"severity|{charts._severity_signature(report)}"
    path = None

    try:
        first = charts.render_severity_chart_cached(report)
        second = charts.render_severity_chart_cached(report)
        twin = SecurityReport.model_validate(report.model_dump())
        third = charts.render_severity_chart_cached(twin)

        assert first is not None
        path = first
        assert first == second == third
        assert len(calls) == 1, "cached chart re-rendered within the cache window"
        assert os.path.exists(first)
    finally:
        with charts._CACHE_LOCK:
            charts._CACHE_FILES.pop(cache_key, None)
        if path:
            try:
                os.remove(path)
            except OSError:
                pass


# ---- Backend resolution short-circuit and scan TTL ---------------------------

def test_auto_resolve_returns_llama_without_scan_when_model_loaded(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(engine_pkg, "_ACTIVE_BACKEND", "auto")
    monkeypatch.setitem(llama_engine._STATE, "model", sentinel)

    calls = []

    def fake_current_model_path():
        calls.append(1)
        return None

    monkeypatch.setattr(llama_engine, "current_model_path", fake_current_model_path)

    assert engine_pkg._resolve() == "llama"
    assert calls == [], "auto resolution re-scanned for a model while one is loaded"


def _clear_llama_path_caches(monkeypatch):
    for name in dir(llama_engine):
        upper = name.upper()
        if not upper.endswith("CACHE"):
            continue
        if not any(part in upper for part in ("PATH", "MODEL", "GGUF", "SCAN")):
            continue
        if callable(getattr(llama_engine, name)):
            continue
        monkeypatch.setattr(llama_engine, name, None, raising=False)


def test_current_model_path_dir_scan_cached_within_ttl(monkeypatch, tmp_path):
    import os as os_module

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "qwen.gguf").write_bytes(b"gguf")
    expected = os.path.join(str(models_dir), "qwen.gguf")

    monkeypatch.delenv("FORTIS_MODEL_PATH", raising=False)
    monkeypatch.setattr("engine.model_manager.current_selection", lambda: None)
    monkeypatch.setattr(settings, "models_dir", str(models_dir))
    _clear_llama_path_caches(monkeypatch)

    real_listdir = os_module.listdir
    scans = []

    def counting_listdir(path, *args, **kwargs):
        if str(path) == str(models_dir):
            scans.append(str(path))
        return real_listdir(path, *args, **kwargs)

    monkeypatch.setattr(os_module, "listdir", counting_listdir)

    results = [llama_engine.current_model_path() for _ in range(3)]

    assert results == [expected, expected, expected]
    assert len(scans) == 1, "models-dir scan must be served from the TTL cache"
