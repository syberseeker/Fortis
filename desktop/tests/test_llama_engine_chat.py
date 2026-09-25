"""Regression tests for the llama engine's chat() max_tokens handling.

A closure-assignment bug (commit ba0d45f) made max_tokens closure-local in
run(), so every llama-backend chat() call raised UnboundLocalError on the
first read. These tests pin the default resolution and passthrough without
loading a real model and without any async pytest plugin (plain asyncio.run).
"""
import asyncio

import pytest

import engine.llama_engine as llama_engine

THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "</" + "think" + ">"


class _FakeLLM:
    def __init__(self, record: list):
        self._record = record

    def create_chat_completion(self, messages, max_tokens, temperature, stop, stream=False):
        self._record.append({"max_tokens": max_tokens, "temperature": temperature, "stream": stream})
        return {"choices": [{"message": {"content": "ok"}}]}


@pytest.fixture
def fake_model(monkeypatch):
    record = []
    monkeypatch.setattr(llama_engine, "ensure_loaded", lambda: None)
    monkeypatch.setattr(llama_engine, "_STATE", {"model": _FakeLLM(record), "path": "fake.gguf"})
    return record


def test_default_plain_chat_uses_2048(fake_model):
    asyncio.run(llama_engine.chat([{"role": "user", "content": "hi"}]))
    assert fake_model[0]["max_tokens"] == 2048


def test_default_json_mode_uses_3072(fake_model):
    asyncio.run(llama_engine.chat([{"role": "user", "content": "hi"}], json_mode=True))
    assert fake_model[0]["max_tokens"] == 3072


def test_explicit_max_tokens_is_passed_through(fake_model):
    asyncio.run(llama_engine.chat([{"role": "user", "content": "hi"}], max_tokens=512))
    assert fake_model[0]["max_tokens"] == 512


def test_explicit_max_tokens_zero_is_respected_not_defaulted(fake_model):
    asyncio.run(llama_engine.chat([{"role": "user", "content": "hi"}], max_tokens=0))
    assert fake_model[0]["max_tokens"] == 0


def test_json_mode_content_is_extracted(fake_model):
    payload = "Sure! ```json\n" + '{"ok": true}' + "\n```"

    class _JsonLLM:
        def create_chat_completion(self, messages, max_tokens, temperature, stop, stream=False):
            return {"choices": [{"message": {"content": payload}}]}

    llama_engine._STATE["model"] = _JsonLLM()
    out = asyncio.run(llama_engine.chat([{"role": "user", "content": "hi"}], json_mode=True))
    assert out == '{"ok": true}'


def test_reasoning_fences_are_stripped(fake_model):
    payload = THINK_OPEN + " pondering hard " + THINK_CLOSE + "\nThe answer is 42."

    class _ReasoningLLM:
        def create_chat_completion(self, messages, max_tokens, temperature, stop, stream=False):
            return {"choices": [{"message": {"content": payload}}]}

    llama_engine._STATE["model"] = _ReasoningLLM()
    out = asyncio.run(llama_engine.chat([{"role": "user", "content": "hi"}]))
    assert out == "The answer is 42."
