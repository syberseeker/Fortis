"""Tests for llama engine model-load diagnostics and the reduced-context
RAM fallback.

A real-world failure: Llama() raises llama-cpp's generic
'Failed to load model from file' when the true cause (a RAM allocation
failure) is suppressed by verbose=False. These tests pin the seams that
capture the real cause, surface machine context, and retry once with a
reduced context for memory-shaped failures — all without real weights.
"""
import pytest

import engine.llama_engine as le


class _FakeLLM:
    def __init__(self):
        self.warmup_calls = 0

    def create_chat_completion(self, messages, max_tokens, temperature, **kw):
        self.warmup_calls += 1
        return {"choices": [{"message": {"content": "ok"}}]}


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.setattr(le, "_STATE", {
        "model": None, "path": None, "error": None, "loading": False,
        "offload_level": None,
    })
    monkeypatch.setattr(le, "_resolved_threads", lambda: 4)
    yield


def _model_available(monkeypatch, path="fake.gguf"):
    monkeypatch.setattr(le, "current_model_path", lambda: path)


def test_ram_allocation_failure_retries_with_reduced_context(monkeypatch):
    _model_available(monkeypatch)
    constructions = []

    def fake_construct(path, n_ctx, n_gpu_layers, n_threads, n_batch):
        constructions.append({"n_ctx": n_ctx, "n_batch": n_batch, "n_gpu": n_gpu_layers})
        if n_ctx >= 8192:
            raise ValueError(
                "Failed to load model from file: fake.gguf "
                "(llama_model_load: failed to allocate compute buffer)"
            )
        return _FakeLLM()

    monkeypatch.setattr(le, "_construct_llama", fake_construct)
    monkeypatch.setattr(le, "_auto_gpu_layers", lambda p: 0)
    monkeypatch.setattr(le.settings, "n_ctx", 8192)
    monkeypatch.setattr(le.settings, "n_batch", 512)
    monkeypatch.setattr(le.settings, "n_gpu_layers", 0)
    monkeypatch.setattr(le, "_memory_context", lambda: "free RAM: 9.9 GB / 31.1 GB")

    le.ensure_loaded()

    assert len(constructions) == 2, "one reduced-context retry expected"
    assert constructions[0]["n_ctx"] == 8192
    assert constructions[1]["n_ctx"] == 4096
    assert constructions[1]["n_batch"] == 256
    assert le._STATE["model"] is not None
    assert le._STATE["error"] is None
    assert le._STATE["path"] == "fake.gguf"


def test_non_memory_failure_does_not_retry(monkeypatch):
    _model_available(monkeypatch)
    constructions = []

    def fake_construct(path, n_ctx, n_gpu_layers, n_threads, n_batch):
        constructions.append(n_ctx)
        raise ValueError("gguf_init_from_file failed: magic mismatch, not a GGUF file")

    monkeypatch.setattr(le, "_construct_llama", fake_construct)
    monkeypatch.setattr(le, "_auto_gpu_layers", lambda p: 0)
    monkeypatch.setattr(le.settings, "n_ctx", 8192)
    monkeypatch.setattr(le.settings, "n_batch", 512)
    monkeypatch.setattr(le.settings, "n_gpu_layers", 0)

    with pytest.raises(ValueError):
        le.ensure_loaded()

    assert len(constructions) == 1, "corrupt files must fail fast, no retry"
    assert "not a GGUF file" in le._STATE["error"]
    assert le._STATE["model"] is None


def test_memory_failure_below_threshold_fails_without_retry(monkeypatch):
    _model_available(monkeypatch)
    constructions = []

    def fake_construct(path, n_ctx, n_gpu_layers, n_threads, n_batch):
        constructions.append(n_ctx)
        raise ValueError("failed to allocate buffer")

    monkeypatch.setattr(le, "_construct_llama", fake_construct)
    monkeypatch.setattr(le, "_auto_gpu_layers", lambda p: 0)
    monkeypatch.setattr(le.settings, "n_ctx", 2048)
    monkeypatch.setattr(le.settings, "n_batch", 512)
    monkeypatch.setattr(le.settings, "n_gpu_layers", 0)
    monkeypatch.setattr(le, "_memory_context", lambda: "free RAM: 2.0 GB / 8.0 GB")

    with pytest.raises(ValueError):
        le.ensure_loaded()

    assert len(constructions) == 1, "no retry when context is already small"
    assert "free RAM: 2.0 GB" in le._STATE["error"], "machine context in error message"
    assert "failed to allocate buffer" in le._STATE["error"], "real cause preserved"


def test_no_model_file_sets_error(monkeypatch):
    monkeypatch.setattr(le, "current_model_path", lambda: None)
    le.ensure_loaded()
    assert le._STATE["model"] is None
    assert le._STATE["error"] == "no model file available"


def test_memory_context_reports(monkeypatch):
    ctx = le._memory_context()
    assert "RAM" in ctx
