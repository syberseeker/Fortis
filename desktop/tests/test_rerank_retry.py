"""
Offline tests for the bounded rerank retry logic in core/rerank.py.

The permanent kill-switch (_RERANK_BROKEN) was replaced by a cumulative
failure counter: transient failures (first-predict OOM, disk hiccup) get
up to _RERANK_MAX_FAILURES retries before reranking gives up for the
process. No GPU, no model download, no network — the cross-encoder is
monkeypatched with a stub (see conftest.py for the offline environment).
"""
import pytest

from core import rerank


def _hits(n):
    return [{"text": f"hit {i}", "metadata": {"chunk_index": i}} for i in range(n)]


class _StubEncoder:
    def __init__(self, scores=None):
        self.scores = scores
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        if self.scores is None:
            raise Exception("boom")
        return list(self.scores)


@pytest.fixture(autouse=True)
def _clean_retry_state(monkeypatch):
    """Order-independence: fresh failure counter and fresh lru_cache for
    every test, restored afterwards. Teardown must guard cache_clear:
    monkeypatch restores the stub lambda after this fixture unwinds."""
    monkeypatch.setattr(rerank, "_RERANK_FAILURES", 0)
    if hasattr(rerank._get_cross_encoder, "cache_clear"):
        rerank._get_cross_encoder.cache_clear()
    yield
    if hasattr(rerank._get_cross_encoder, "cache_clear"):
        rerank._get_cross_encoder.cache_clear()


def test_bounded_retry_short_circuits_after_two_failures(monkeypatch):
    stub = _StubEncoder()
    monkeypatch.setattr(rerank, "is_available", lambda: True)
    monkeypatch.setattr(rerank, "_get_cross_encoder", lambda: stub)

    hits = _hits(5)
    assert rerank.rerank("q", hits, 2) == hits[:2]
    assert rerank.rerank("q", hits, 2) == hits[:2]
    assert stub.calls == 2

    assert rerank.rerank("q", hits, 2) == hits[:2]
    assert stub.calls == 2, "third call must short-circuit without predict"


def test_cumulative_counter_survives_successes(monkeypatch):
    stub = _StubEncoder(scores=[0.1, 0.9, 0.5])
    monkeypatch.setattr(rerank, "is_available", lambda: True)
    monkeypatch.setattr(rerank, "_get_cross_encoder", lambda: stub)

    hits = _hits(3)
    ranked = rerank.rerank("q", hits, 2)
    assert stub.calls == 1
    assert ranked == [hits[1], hits[2]], "two highest-scored hits, in score order"

    stub.scores = None
    assert rerank.rerank("q", hits, 2) == hits[:2]
    assert stub.calls == 2
    assert rerank.rerank("q", hits, 2) == hits[:2]
    assert stub.calls == 3, "counter is cumulative: one success, then 2nd failure still retries"
    assert rerank.rerank("q", hits, 2) == hits[:2]
    assert stub.calls == 3, "short-circuit once _RERANK_MAX_FAILURES is reached"


def test_reset_retry_state_clears_failure_counter(monkeypatch):
    stub = _StubEncoder()
    monkeypatch.setattr(rerank, "is_available", lambda: True)
    monkeypatch.setattr(rerank, "_get_cross_encoder", lambda: stub)

    hits = _hits(5)
    rerank.rerank("q", hits, 2)
    rerank.rerank("q", hits, 2)
    assert stub.calls == 2
    rerank.rerank("q", hits, 2)
    assert stub.calls == 2

    rerank.reset_retry_state()
    rerank.rerank("q", hits, 2)
    assert stub.calls == 3, "after reset the model must be attempted again"


def test_rerank_unavailable_never_touches_stub(monkeypatch):
    stub = _StubEncoder()
    monkeypatch.setattr(rerank, "is_available", lambda: False)
    monkeypatch.setattr(rerank, "_get_cross_encoder", lambda: stub)

    hits = _hits(5)
    assert rerank.rerank("q", hits, 2) == hits[:2]
    rerank.rerank("q", hits, 2)
    assert stub.calls == 0
