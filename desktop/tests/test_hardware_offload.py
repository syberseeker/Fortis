"""
Offline tests for GPU offload decisions, the OOM retry ladder, and CPU
thread/batch tuning. Everything is mocked: no GPU, no nvidia-smi call, and
no llama_cpp import is required (a fake llama_cpp module is injected into
sys.modules where a load is exercised, mirroring the mocking style of
test_integration.py).
"""
import os
import sys
import types

import pytest

from core.config import settings
from engine.hardware import Hardware


@pytest.fixture
def engine_state(monkeypatch, tmp_path):
    """Resets llama_engine._STATE and points current_model_path at a small
    fake GGUF so load tests never touch the real model directory."""
    import engine.llama_engine as le
    saved = dict(le._STATE)
    le._STATE.update(model=None, path=None, error=None, offload_level=None)
    fake_gguf = tmp_path / "fake-model.gguf"
    fake_gguf.write_bytes(b"\0" * (8 * 1024 * 1024))
    monkeypatch.setattr(le, "current_model_path", lambda: str(fake_gguf))
    yield le
    le._STATE.clear()
    le._STATE.update(saved)


@pytest.fixture
def fake_llama(monkeypatch):
    """Installs a fake llama_cpp module; returns (constructed_kwargs, state)
    where state["fails"] controls how many Llama() constructions raise."""
    import engine.llama_engine as le
    constructed = []
    state = {"fails": 0}

    class FakeLlama:
        def __init__(self, **kwargs):
            constructed.append(kwargs)
            if state["fails"] > 0:
                state["fails"] -= 1
                raise RuntimeError("llama_decode: CUDA error: out of memory")

        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": "ok"}}]}

    fake = types.ModuleType("llama_cpp")
    fake.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    return constructed, state


def _cuda_env(monkeypatch, free_gb):
    """Makes can_use_cuda() true with a controlled free VRAM value."""
    monkeypatch.setattr("engine.hardware.can_use_cuda", lambda: True)
    monkeypatch.setattr("engine.hardware.free_vram_gb", lambda: free_gb)


# ---- free_vram_gb -----------------------------------------------------------

def test_free_vram_gb_returns_zero_without_nvidia(monkeypatch):
    from engine import hardware
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)

    def _no_subprocess(*args, **kwargs):
        raise AssertionError("nvidia-smi must not be spawned when absent")

    monkeypatch.setattr(hardware.subprocess, "run", _no_subprocess)
    assert hardware.free_vram_gb() == 0.0


def test_free_vram_gb_parses_nvidia_smi_fallback(monkeypatch):
    from engine import hardware
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "C:/fake/nvidia-smi.exe")

    class FakeCompleted:
        stdout = " 6144\n"

    def fake_run(cmd, **kwargs):
        assert any("memory.free" in part for part in cmd)
        assert kwargs.get("timeout") == 5
        return FakeCompleted()

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)
    assert hardware.free_vram_gb() == pytest.approx(6.0)


# ---- _auto_gpu_layers decision ---------------------------------------------

def test_auto_gpu_layers_explicit_override_wins(engine_state, monkeypatch, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 1024)
    monkeypatch.setattr(settings, "n_gpu_layers", 24)
    monkeypatch.setattr("engine.hardware.can_use_cuda", lambda: False)
    assert engine_state._auto_gpu_layers(str(model)) == 24


def test_auto_gpu_layers_no_cuda_is_zero(engine_state, monkeypatch, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 1024)
    monkeypatch.setattr(settings, "n_gpu_layers", -1)
    monkeypatch.setattr(settings, "n_ctx", 8192)
    monkeypatch.setattr("engine.hardware.can_use_cuda", lambda: False)
    assert engine_state._auto_gpu_layers(str(model)) == 0


def test_auto_gpu_layers_full_when_vram_covers_model_and_kv(engine_state, monkeypatch, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\0" * (8 * 1024 * 1024))
    monkeypatch.setattr(settings, "n_gpu_layers", -1)
    monkeypatch.setattr(settings, "n_ctx", 8192)
    size_gb = model.stat().st_size / (1024 ** 3)
    kv_gb = (8192 / 8192.0) * size_gb * 0.35
    _cuda_env(monkeypatch, size_gb + kv_gb + 0.5)
    assert engine_state._auto_gpu_layers(str(model)) == 999


def test_auto_gpu_layers_fractional_when_vram_partial(engine_state, monkeypatch, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\0" * (8 * 1024 * 1024))
    monkeypatch.setattr(settings, "n_gpu_layers", -1)
    monkeypatch.setattr(settings, "n_ctx", 8192)
    size_gb = model.stat().st_size / (1024 ** 3)
    need = size_gb * (1 + 0.35)
    _cuda_env(monkeypatch, need * 0.5)
    result = engine_state._auto_gpu_layers(str(model))
    assert 0 < result < 999
    assert result == int(999 * 0.5)


def test_auto_gpu_layers_fraction_clamped_to_95_percent(engine_state, monkeypatch, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\0" * (8 * 1024 * 1024))
    monkeypatch.setattr(settings, "n_gpu_layers", -1)
    monkeypatch.setattr(settings, "n_ctx", 8192)
    size_gb = model.stat().st_size / (1024 ** 3)
    need = size_gb * (1 + 0.35)
    _cuda_env(monkeypatch, need * 10)
    result = engine_state._auto_gpu_layers(str(model))
    assert result == int(999 * 0.95)
    assert result < 999


def test_auto_gpu_layers_kv_estimate_scales_with_n_ctx(engine_state, monkeypatch, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\0" * (8 * 1024 * 1024))
    monkeypatch.setattr(settings, "n_gpu_layers", -1)
    size_gb = model.stat().st_size / (1024 ** 3)

    monkeypatch.setattr(settings, "n_ctx", 8192)
    _cuda_env(monkeypatch, size_gb * 1.35 * 0.99)
    small_ctx = engine_state._auto_gpu_layers(str(model))

    monkeypatch.setattr(settings, "n_ctx", 12288)
    kv_12k = (12288 / 8192.0) * size_gb * 0.35
    _cuda_env(monkeypatch, size_gb + kv_12k + 0.5)
    assert engine_state._auto_gpu_layers(str(model)) == 999

    _cuda_env(monkeypatch, (size_gb + kv_12k) * 0.5)
    assert 0 < engine_state._auto_gpu_layers(str(model)) < 999

    monkeypatch.setattr(settings, "n_ctx", 8192)
    _cuda_env(monkeypatch, (size_gb + size_gb * 0.35) * 0.99 + 0.49)
    assert 0 < small_ctx < 999


# ---- _oom_ladder ------------------------------------------------------------

def test_oom_ladder_decreases_to_cpu():
    import engine.llama_engine as le
    assert le._oom_ladder(999) == [749, 499, 0]
    assert le._oom_ladder(1) == [0]
    assert le._oom_ladder(2) == [1, 0]
    assert le._oom_ladder(0) == []
    ladder = le._oom_ladder(999)
    seq = [999] + ladder
    assert all(a > b for a, b in zip(seq, ladder))


# ---- _load_sync retry ladder ------------------------------------------------

def _cuda_full_offload(monkeypatch):
    monkeypatch.setattr(settings, "n_gpu_layers", -1)
    monkeypatch.setattr(settings, "n_ctx", 8192)
    _cuda_env(monkeypatch, 100.0)


def test_load_sync_ladder_retries_then_succeeds(engine_state, fake_llama, monkeypatch):
    constructed, state = fake_llama
    state["fails"] = 2
    _cuda_full_offload(monkeypatch)
    engine_state._load_sync()
    assert [k["n_gpu_layers"] for k in constructed] == [999, 749, 499]
    assert engine_state._STATE["model"] is not None
    assert engine_state._STATE["offload_level"] == 499
    assert engine_state._STATE["error"] is None


def test_load_sync_ladder_ends_cpu_when_all_gpu_attempts_fail(engine_state, fake_llama, monkeypatch):
    constructed, state = fake_llama
    state["fails"] = 3
    _cuda_full_offload(monkeypatch)
    engine_state._load_sync()
    assert [k["n_gpu_layers"] for k in constructed] == [999, 749, 499, 0]
    assert engine_state._STATE["offload_level"] == 0
    assert engine_state._STATE["model"] is not None


def test_load_sync_single_attempt_when_cpu_only(engine_state, fake_llama, monkeypatch):
    constructed, state = fake_llama
    state["fails"] = 1
    monkeypatch.setattr(settings, "n_gpu_layers", -1)
    monkeypatch.setattr(settings, "n_ctx", 8192)
    monkeypatch.setattr("engine.hardware.can_use_cuda", lambda: False)
    with pytest.raises(RuntimeError, match="out of memory"):
        engine_state._load_sync()
    assert [k["n_gpu_layers"] for k in constructed] == [0]
    assert engine_state._STATE["model"] is None
    assert engine_state._STATE["error"] is not None


def test_load_sync_cpu_failure_is_single_attempt_via_ladder(engine_state, fake_llama, monkeypatch):
    constructed, state = fake_llama
    state["fails"] = 1
    monkeypatch.setattr(settings, "n_gpu_layers", 0)
    with pytest.raises(RuntimeError):
        engine_state._load_sync()
    assert [k["n_gpu_layers"] for k in constructed] == [0]


def test_load_sync_reuses_known_good_offload_level(engine_state, fake_llama, monkeypatch):
    constructed, state = fake_llama
    state["fails"] = 2
    _cuda_full_offload(monkeypatch)
    engine_state._load_sync()
    assert [k["n_gpu_layers"] for k in constructed] == [999, 749, 499]

    engine_state._STATE["model"] = None
    constructed.clear()
    state["fails"] = 0
    engine_state._load_sync()
    assert [k["n_gpu_layers"] for k in constructed] == [499]
    assert engine_state._STATE["offload_level"] == 499


# ---- thread / batch tuning --------------------------------------------------

def test_load_sync_passes_threads_and_batch(engine_state, fake_llama, monkeypatch):
    constructed, _ = fake_llama
    monkeypatch.setattr(settings, "n_batch", 777)
    monkeypatch.setattr(settings, "n_threads", 5)
    monkeypatch.setattr(settings, "n_gpu_layers", 0)
    monkeypatch.setattr(settings, "n_ctx", 8192)
    engine_state._load_sync()
    kwargs = constructed[0]
    assert kwargs["n_threads"] == 5
    assert kwargs["n_batch"] == 777
    assert kwargs["n_ctx"] == 8192
    assert kwargs["n_gpu_layers"] == 0


def test_physical_core_estimate_honors_explicit_threads(monkeypatch):
    from engine.hardware import physical_core_estimate
    monkeypatch.setattr(settings, "n_threads", 6)
    assert physical_core_estimate() == 6


def test_physical_core_estimate_halves_high_logical_counts(monkeypatch):
    import os as os_module
    from engine import hardware
    monkeypatch.setattr(settings, "n_threads", -1)
    monkeypatch.setattr(os_module, "cpu_count", lambda: 16)
    assert hardware.physical_core_estimate() == 8
    monkeypatch.setattr(os_module, "cpu_count", lambda: 32)
    assert hardware.physical_core_estimate() == 16


def test_physical_core_estimate_uses_logical_when_small(monkeypatch):
    import os as os_module
    from engine import hardware
    monkeypatch.setattr(settings, "n_threads", -1)
    monkeypatch.setattr(os_module, "cpu_count", lambda: 8)
    assert hardware.physical_core_estimate() == 8
    monkeypatch.setattr(os_module, "cpu_count", lambda: 4)
    assert hardware.physical_core_estimate() == 4


def test_physical_core_estimate_minimum_one(monkeypatch):
    import os as os_module
    from engine import hardware
    monkeypatch.setattr(settings, "n_threads", -1)
    monkeypatch.setattr(os_module, "cpu_count", lambda: None)
    assert hardware.physical_core_estimate() == 1
    monkeypatch.setattr(os_module, "cpu_count", lambda: 0)
    assert hardware.physical_core_estimate() == 1


def test_load_sync_auto_threads_uses_hardware_estimate(engine_state, fake_llama, monkeypatch):
    constructed, _ = fake_llama
    import engine.hardware as hardware
    monkeypatch.setattr(settings, "n_threads", -1)
    monkeypatch.setattr(hardware, "physical_core_estimate", lambda: 7)
    monkeypatch.setattr(settings, "n_gpu_layers", 0)
    engine_state._load_sync()
    assert constructed[0]["n_threads"] == 7


# ---- hardware_status / backend_wheel contract -------------------------------

def test_hardware_status_unknown_before_first_load(engine_state):
    status = engine_state.hardware_status(Hardware(ram_gb=16.0, vram_gb=6.0, has_nvidia=True))
    assert set(status) == {
        "vram_gb", "ram_gb", "has_nvidia", "offload", "offload_layers", "backend_wheel",
    }
    assert status["offload"] == "unknown"
    assert status["vram_gb"] == 6.0
    assert status["ram_gb"] == 16.0
    assert status["has_nvidia"] is True


def test_hardware_status_offload_levels(engine_state):
    hw = Hardware(ram_gb=16.0, vram_gb=6.0, has_nvidia=True)
    for level, expected in ((999, "full"), (400, "partial"), (0, "cpu"), (None, "unknown")):
        engine_state._STATE["offload_level"] = level
        assert engine_state.hardware_status(hw)["offload"] == expected
        assert engine_state.hardware_status(hw)["offload_layers"] == level


def test_hardware_status_reports_cpu_when_cuda_load_collapsed(engine_state, monkeypatch):
    monkeypatch.setattr("engine.hardware.can_use_cuda", lambda: True)
    engine_state._STATE["offload_level"] = 0
    status = engine_state.hardware_status(Hardware(ram_gb=16.0, vram_gb=6.0, has_nvidia=True))
    assert status["offload"] == "cpu"


def test_backend_wheel_variants(monkeypatch):
    import engine.llama_engine as le
    fake = types.ModuleType("llama_cpp")

    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    monkeypatch.setattr("engine.hardware.cuda_marker_exists", lambda: True)
    assert le._backend_wheel() == "cuda"

    monkeypatch.setattr("engine.hardware.cuda_marker_exists", lambda: False)
    assert le._backend_wheel() == "cpu"

    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    monkeypatch.setattr("engine.hardware.cuda_marker_exists", lambda: True)
    assert le._backend_wheel() == "none"


# ---- model_status without llama_cpp ------------------------------------------

def test_model_status_works_without_llama_cpp(engine_state, monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    assert engine_state.model_status() is None
    engine_state._STATE["error"] = "boom"
    assert "boom" in engine_state.model_status()


# ---- /api/model/status contract ----------------------------------------------

def test_model_status_endpoint_exposes_hardware_fields():
    from fastapi.testclient import TestClient
    from server.app import app as server_app

    with TestClient(server_app) as client:
        resp = client.get("/api/model/status")
    assert resp.status_code == 200
    data = resp.json()
    hw = data["hardware"]
    for key in ("ram_gb", "vram_gb", "has_nvidia", "can_cuda", "offload",
                "offload_layers", "backend_wheel"):
        assert key in hw
    for key in ("suggested_tier", "suggested_tiers", "tiers", "active_model",
                "loaded", "load_error", "warning"):
        assert key in data
    assert hw["offload"] in ("full", "partial", "cpu", "unknown")
    assert hw["backend_wheel"] in ("cuda", "cpu", "none")
