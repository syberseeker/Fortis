"""Regression tests for engine instrumentation: seed threading, usage
capture, and the get_last_usage contract — all without a real model.

Plain asyncio.run (no async pytest plugin), mirroring test_llma_engine_chat
conventions. Seed semantics: FORTIS_SEED set -> `seed=` kwarg reaches
create_chat_completion; unset -> the kwarg is absent entirely so older
llama-cpp wheels and existing fakes keep working.
"""
import asyncio

import pytest

import engine
import engine.llama_engine as llama_engine
from core.token_utils import count_tokens


class _KwargsLLM:
    def __init__(self, record, usage=None):
        self._record = record
        self._usage = usage

    def create_chat_completion(self, **kwargs):
        self._record.append(kwargs)
        out = {"choices": [{"message": {"content": "hello world"}}]}
        if self._usage is not None:
            out["usage"] = self._usage
        return out


class _StreamLLM:
    def __init__(self, record, usage_chunk=True, accept_stream_options=True):
        self._record = record
        self._usage_chunk = usage_chunk
        self._accept_stream_options = accept_stream_options

    def create_chat_completion(self, **kwargs):
        self._record.append(dict(kwargs))
        if "stream_options" in kwargs and not self._accept_stream_options:
            raise TypeError("unexpected keyword argument 'stream_options'")

        def gen():
            yield {"choices": [{"delta": {"content": "alpha "}}]}
            yield {"choices": [{"delta": {"content": "beta"}}]}
            if self._usage_chunk:
                yield {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}

        return gen()


@pytest.fixture
def clean_usage(monkeypatch):
    monkeypatch.setattr(llama_engine, "_LAST_USAGE", None)
    return None


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(llama_engine, "ensure_loaded", lambda: None)
    return None


def _mount(monkeypatch, llm):
    monkeypatch.setattr(llama_engine, "_STATE", {"model": llm, "path": "fake.gguf"})


def test_seed_absent_by_default(fake_model, clean_usage, monkeypatch):
    monkeypatch.setattr(llama_engine.settings, "seed", None)
    record = []
    _mount(monkeypatch, _KwargsLLM(record))
    out = asyncio.run(llama_engine.chat([{"role": "user", "content": "hi"}]))
    assert out == "hello world"
    assert "seed" not in record[0]


def test_seed_threaded_when_configured(fake_model, clean_usage, monkeypatch):
    monkeypatch.setattr(llama_engine.settings, "seed", 123)
    record = []
    _mount(monkeypatch, _KwargsLLM(record))
    asyncio.run(llama_engine.chat([{"role": "user", "content": "hi"}]))
    assert record[0].get("seed") == 123


def test_usage_from_real_response(fake_model, clean_usage, monkeypatch):
    usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    record = []
    _mount(monkeypatch, _KwargsLLM(record, usage=usage))
    asyncio.run(llama_engine.chat([{"role": "user", "content": "hi"}]))
    snap = llama_engine.get_last_usage()
    assert snap["prompt_tokens"] == 7
    assert snap["completion_tokens"] == 3
    assert snap["total_tokens"] == 10
    assert snap["estimated"] is False
    assert snap["ttft_ms"] is None
    assert snap["llm_ms"] >= 0
    assert snap["tok_per_s"] > 0


def test_usage_estimated_fallback(fake_model, clean_usage, monkeypatch):
    record = []
    _mount(monkeypatch, _KwargsLLM(record, usage=None))
    asyncio.run(llama_engine.chat([{"role": "user", "content": "hi"}]))
    snap = llama_engine.get_last_usage()
    assert snap["estimated"] is True
    assert snap["prompt_tokens"] > 0
    assert snap["completion_tokens"] == count_tokens("hello world")


def test_get_last_usage_none_before_first_call(clean_usage):
    assert llama_engine.get_last_usage() is None


def test_engine_dispatch_get_last_usage_stub(clean_usage, monkeypatch):
    monkeypatch.setattr(engine, "_ACTIVE_BACKEND", "stub")
    assert engine.get_last_usage() is None
    asyncio.run(engine.chat([{"role": "user", "content": "hello there"}]))
    snap = engine.get_last_usage()
    assert snap is not None
    assert snap["model"] == "stub"
    assert snap["estimated"] is True
    assert snap["completion_tokens"] > 0


def test_stream_usage_chunk_and_ttft(fake_model, clean_usage, monkeypatch):
    monkeypatch.setattr(llama_engine.settings, "seed", 42)
    record = []
    _mount(monkeypatch, _StreamLLM(record))

    async def consume():
        async for _token in llama_engine.chat_stream([{"role": "user", "content": "hi"}]):
            pass

    asyncio.run(consume())
    assert record[0]["stream"] is True
    assert record[0]["seed"] == 42
    assert "stream_options" in record[0]
    snap = llama_engine.get_last_usage()
    assert snap["prompt_tokens"] == 10
    assert snap["completion_tokens"] == 2
    assert snap["estimated"] is False
    assert snap["ttft_ms"] is not None


def test_stream_without_usage_chunk_estimates(fake_model, clean_usage, monkeypatch):
    record = []
    _mount(monkeypatch, _StreamLLM(record, usage_chunk=False))

    async def consume():
        async for _token in llama_engine.chat_stream([{"role": "user", "content": "hi"}]):
            pass

    asyncio.run(consume())
    snap = llama_engine.get_last_usage()
    assert snap["estimated"] is True
    assert snap["completion_tokens"] == count_tokens("alpha ") + count_tokens("beta")


def test_stream_falls_back_when_stream_options_rejected(fake_model, clean_usage, monkeypatch):
    record = []
    _mount(monkeypatch, _StreamLLM(record, accept_stream_options=False))
    asyncio.run(_consume())
    assert len(record) == 2
    assert "stream_options" not in record[1]
    snap = llama_engine.get_last_usage()
    assert snap["estimated"] is False
    assert snap["prompt_tokens"] == 10


async def _consume():
    async for _token in llama_engine.chat_stream([{"role": "user", "content": "hi"}]):
        pass
