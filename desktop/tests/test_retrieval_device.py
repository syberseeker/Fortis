"""
Offline tests of GPU device resolution for retrieval models.

When a CUDA-capable NVIDIA GPU is present, the sentence-transformers
embedding model and the cross-encoder reranker are loaded on device="cuda";
a CUDA failure at model load or first encode transparently rebuilds the
model on device="cpu" and the downgrade is remembered for the rest of the
process. No torch, no GPU, no model download: the sentence_transformers
module is faked in sys.modules (real package imported or not) and
engine.hardware.can_use_cuda is monkeypatched (see conftest.py for the
offline environment).
"""
import sys
import types

import pytest

from core import embeddings, rerank

@pytest.fixture(autouse=True)
def _fresh_device_state(monkeypatch):
    """Order-independence: reset both modules' resolved-device state and
    model caches around every test so sticky CPU / cached-constructor
    decisions cannot leak between tests."""
    monkeypatch.setattr(embeddings, "_resolved_device", "")
    monkeypatch.setattr(embeddings, "_cuda_broken", False)
    monkeypatch.setattr(rerank, "_resolved_device", "")
    monkeypatch.setattr(rerank, "_cuda_broken", False)
    monkeypatch.setattr(rerank, "_RERANK_FAILURES", 0)
    embeddings._load_model.cache_clear()
    rerank._load_cross_encoder.cache_clear()
    yield
    embeddings._load_model.cache_clear()
    rerank._load_cross_encoder.cache_clear()


@pytest.fixture
def fake_st(monkeypatch):
    """Stands in for the real sentence_transformers package so no torch
    import is ever attempted. Tests assign their own constructor doubles to
    SentenceTransformer / CrossEncoder on the returned module."""
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = None
    module.CrossEncoder = None
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


class _FakeVectorArray:
    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        return self._rows


class _StubModel:
    def __init__(self, device):
        self.device = device
        self.encode_calls = 0

    def encode(self, texts, **kwargs):
        self.encode_calls += 1
        if self.device == "cuda" and self.encode_calls == 1:
            raise RuntimeError("CUDA error: out of memory")
        return _FakeVectorArray([[0.1, 0.2, 0.3] for _ in texts])


class _StubEncoder:
    def __init__(self, device):
        self.device = device

    def predict(self, pairs):
        return [float(len(pairs) - i) for i in range(len(pairs))]


def _enable_real_backend(monkeypatch):
    """Unlocks the real loader path despite conftest's hash-stub setting."""
    monkeypatch.setattr(embeddings, "_BACKEND", "sentence-transformers")


def _cuda(monkeypatch, value):
    monkeypatch.setattr("engine.hardware.can_use_cuda", lambda: value)


def _ctor_recording(recorder, make):
    def ctor(model_name, device):
        recorder.append(device)
        return make(device)

    return ctor


def test_embedding_cuda_selected_when_available(fake_st, monkeypatch):
    _enable_real_backend(monkeypatch)
    _cuda(monkeypatch, True)
    devices = []
    fake_st.SentenceTransformer = _ctor_recording(
        devices, lambda device: _StubModel("cpu")
    )

    vectors = embeddings.embed_texts(["hello"])

    assert vectors == [[0.1, 0.2, 0.3]]
    assert devices == ["cuda"]


def test_embedding_cuda_encode_failure_falls_back_to_cpu_and_remembers(
    fake_st, monkeypatch
):
    _enable_real_backend(monkeypatch)
    _cuda(monkeypatch, True)
    devices = []
    fake_st.SentenceTransformer = _ctor_recording(
        devices, lambda device: _StubModel(device)
    )

    assert embeddings.embed_texts(["first"]) == [[0.1, 0.2, 0.3]]
    assert devices == ["cuda", "cpu"], "cpu model built after cuda encode failure"
    assert embeddings._resolved_device == "cpu"
    assert embeddings._cuda_broken is True

    assert embeddings.embed_texts(["second"]) == [[0.1, 0.2, 0.3]]
    assert devices == ["cuda", "cpu"], "second call must not re-attempt cuda"


def test_embedding_no_gpu_stays_cpu(fake_st, monkeypatch):
    _enable_real_backend(monkeypatch)
    _cuda(monkeypatch, False)
    devices = []
    fake_st.SentenceTransformer = _ctor_recording(
        devices, lambda device: _StubModel(device)
    )

    embeddings.embed_texts(["hello"])

    assert devices == ["cpu"]
    assert embeddings._resolved_device == "cpu"


def test_embedding_cuda_probe_failure_stays_cpu(fake_st, monkeypatch):
    _enable_real_backend(monkeypatch)
    monkeypatch.setattr(
        "engine.hardware.can_use_cuda",
        lambda: (_ for _ in ()).throw(Exception("hardware probe exploded")),
    )
    devices = []
    fake_st.SentenceTransformer = _ctor_recording(
        devices, lambda device: _StubModel(device)
    )

    embeddings.embed_texts(["hello"])

    assert devices == ["cpu"]


def test_embedding_cpu_load_failure_propagates(fake_st, monkeypatch):
    _enable_real_backend(monkeypatch)
    _cuda(monkeypatch, False)

    def boom(model_name, device):
        raise RuntimeError("model files missing")

    fake_st.SentenceTransformer = boom
    with pytest.raises(RuntimeError):
        embeddings.embed_texts(["hello"])
    assert embeddings._resolved_device == "cpu", "no cpu stickiness for cpu failures"


def test_rerank_cuda_selected_when_available(fake_st, monkeypatch):
    monkeypatch.setattr(rerank, "is_available", lambda: True)
    _cuda(monkeypatch, True)
    devices = []
    fake_st.CrossEncoder = _ctor_recording(devices, lambda device: _StubEncoder(device))

    hits = [{"text": f"hit {i}"} for i in range(5)]
    ranked = rerank.rerank("q", hits, 2)

    assert devices == ["cuda"]
    assert len(ranked) == 2


def test_rerank_cuda_load_failure_falls_back_to_cpu_and_remembers(
    fake_st, monkeypatch
):
    monkeypatch.setattr(rerank, "is_available", lambda: True)
    _cuda(monkeypatch, True)
    attempts = []

    def ctor(model_name, device):
        attempts.append(device)
        if device == "cuda":
            raise RuntimeError("CUDA initialization failed")
        return _StubEncoder(device)

    fake_st.CrossEncoder = ctor

    hits = [{"text": f"hit {i}"} for i in range(5)]
    assert rerank.rerank("q", hits, 2) == hits[:2]
    assert attempts == ["cuda", "cpu"]
    assert rerank._resolved_device == "cpu"
    assert rerank._cuda_broken is True

    rerank.rerank("q", hits, 2)
    assert attempts == ["cuda", "cpu"], "second call must go straight to cpu"


def test_rerank_cuda_predict_failure_demotes_to_cpu(fake_st, monkeypatch):
    monkeypatch.setattr(rerank, "is_available", lambda: True)
    _cuda(monkeypatch, True)
    attempts = []

    class _PredictBoom(_StubEncoder):
        def predict(self, pairs):
            raise RuntimeError("CUDA error: out of memory")

    def ctor(model_name, device):
        attempts.append(device)
        return _PredictBoom(device)

    fake_st.CrossEncoder = ctor

    hits = [{"text": f"hit {i}"} for i in range(5)]
    assert rerank.rerank("q", hits, 2) == hits[:2]
    assert rerank._resolved_device == "cpu"
    assert rerank._cuda_broken is True
    assert rerank._RERANK_FAILURES == 1, "predict failure still counts against the retry budget"

    fake_st.CrossEncoder = _ctor_recording(attempts, lambda device: _StubEncoder(device))
    rerank.rerank("q", hits, 2)
    assert attempts[-1] == "cpu", "rebuild after predict failure uses cpu"


def test_rerank_no_gpu_stays_cpu(fake_st, monkeypatch):
    monkeypatch.setattr(rerank, "is_available", lambda: True)
    _cuda(monkeypatch, False)
    devices = []
    fake_st.CrossEncoder = _ctor_recording(devices, lambda device: _StubEncoder(device))

    hits = [{"text": f"hit {i}"} for i in range(5)]
    rerank.rerank("q", hits, 2)

    assert devices == ["cpu"]
