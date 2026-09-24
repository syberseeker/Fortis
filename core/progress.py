"""Thread-safe progress tracking for report generation.

Report analysis can issue many sequential LLM calls (map-reduce), which on
CPU-only machines takes minutes. This module lets the analysis pipeline
publish where it is so the UI can poll instead of appearing hung.
"""
import threading
from typing import Dict

_LOCK = threading.Lock()
_STATE: Dict = {
    "active": False,
    "stage": "idle",
    "done": 0,
    "total": 0,
    "detail": "",
}

_STAGES = ("idle", "map", "reduce", "render")


def start(total: int, detail: str = "") -> None:
    with _LOCK:
        _STATE.update(
            active=True,
            stage="map",
            done=0,
            total=max(1, int(total)),
            detail=detail,
        )


def update(done: int, detail: str = "", stage: str = "") -> None:
    with _LOCK:
        _STATE["done"] = max(0, int(done))
        if detail:
            _STATE["detail"] = detail
        if stage in _STAGES:
            _STATE["stage"] = stage


def advance(detail: str = "") -> None:
    with _LOCK:
        _STATE["done"] = min(_STATE["total"], _STATE["done"] + 1) if _STATE["total"] else _STATE["done"]
        if detail:
            _STATE["detail"] = detail


def finish(detail: str = "") -> None:
    with _LOCK:
        _STATE.update(active=False, done=_STATE["total"], stage="idle", detail=detail or _STATE["detail"])


def fail(detail: str = "") -> None:
    with _LOCK:
        _STATE.update(active=False, stage="idle", detail=detail or _STATE["detail"])


def reset() -> None:
    with _LOCK:
        _STATE.update(active=False, stage="idle", done=0, total=0, detail="")


def snapshot() -> Dict:
    with _LOCK:
        return dict(_STATE)
