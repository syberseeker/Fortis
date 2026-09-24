"""
Hardware detection for model tier selection.

Detects, best-effort and without hard dependencies:
- NVIDIA VRAM, total and free (pynvml if available, else nvidia-smi CLI)
- system RAM
- logical core count (hyperthreading-adjusted estimate for worker threads)
- CUDA availability for llama.cpp (presence of an NVIDIA driver is treated
  as the proxy; whether the installed wheel actually has CUDA is checked by
  the setup script at install time and recorded to a marker file)
"""
import ctypes
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict


@dataclass
class Hardware:
    ram_gb: float
    vram_gb: float          # 0.0 if no dedicated NVIDIA GPU detected
    has_nvidia: bool


def cuda_marker_exists() -> bool:
    """True when the setup script recorded a verified CUDA llama-cpp wheel."""
    marker = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".llama-cuda-ok")
    return os.path.exists(marker)


def can_use_cuda() -> bool:
    """True when the llama-cpp wheel is believed to have CUDA support: an
    NVIDIA GPU is present AND the install-time marker was written by the
    setup script (which verified the CUDA wheel imported)."""
    return _vram_gb() > 0.0 and cuda_marker_exists()


def free_vram_gb() -> float:
    """Free (not total) VRAM on GPU 0, used to size partial layer offload.
    0.0 when no NVIDIA GPU is present or the query fails."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml.nvmlDeviceGetMemoryInfo(h).free / (1024 ** 3)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            mb = int(out.stdout.strip().splitlines()[0])
            return mb / 1024.0
        except Exception:
            pass
    return 0.0


def physical_core_estimate() -> int:
    """Best-effort physical core count for llama.cpp worker threads: the
    logical count, halved when >= 16 logical CPUs (hyperthreading SMT
    assumption). settings.n_threads > 0 overrides the estimate."""
    try:
        from core.config import settings
        if settings.n_threads > 0:
            return int(settings.n_threads)
    except Exception:
        pass
    logical = os.cpu_count() or 0
    if logical >= 16:
        logical //= 2
    return max(1, logical)


def _ram_gb() -> float:
    try:
        if platform.system() == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 8.0  # conservative default


def _vram_gb() -> float:
    # 1) pynvml if installed
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 ** 3)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    # 2) nvidia-smi CLI
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            mb = int(out.stdout.strip().splitlines()[0])
            return mb / 1024.0
        except Exception:
            pass
    return 0.0


def _has_nvidia() -> bool:
    return _vram_gb() > 0.0


def detect() -> Hardware:
    return Hardware(ram_gb=_ram_gb(), vram_gb=_vram_gb(), has_nvidia=_has_nvidia())


def suggest_tiers(hw: Hardware) -> Dict[str, str]:
    """Picks model tier ids for each role based on hardware.
    
    Matrix:
    - vram_gb >= 5.5  -> {"chat": "8b-llama", "grc": "8b-llama", "code": "7b-coder", "osint": "7b-r1", "cti": "8b-llama"}
    - vram_gb >= 3.5  -> {"chat": "4b", "grc": "4b", "code": "3b-coder", "osint": "4b", "cti": "4b"}
    - vram_gb >= 1.8 or ram_gb >= 16 -> {"chat": "2b", "grc": "3b-llama", "code": "3b-coder", "osint": "2b", "cti": "2b"}
    - ram_gb >= 8 -> {"chat": "2b", "grc": "2b", "code": "1.5b-coder", "osint": "2b", "cti": "2b"}
    - else -> {"chat": "0.8b", "grc": "0.8b", "code": "1.5b-coder", "osint": "0.8b", "cti": "0.8b"}
    """
    from engine.model_manager import TIERS
    
    def _fallback(tier_id: str) -> str:
        return tier_id if tier_id in TIERS else "4b"
    
    if hw.vram_gb >= 5.5:
        return {
            "chat": _fallback("8b-llama"),
            "grc": _fallback("8b-llama"),
            "code": _fallback("7b-coder"),
            "osint": _fallback("7b-r1"),
            "cti": _fallback("8b-llama"),
        }
    if hw.vram_gb >= 3.5:
        return {
            "chat": _fallback("4b"),
            "grc": _fallback("4b"),
            "code": _fallback("3b-coder"),
            "osint": _fallback("4b"),
            "cti": _fallback("4b"),
        }
    if hw.vram_gb >= 1.8 or hw.ram_gb >= 16:
        return {
            "chat": _fallback("2b"),
            "grc": _fallback("3b-llama"),
            "code": _fallback("3b-coder"),
            "osint": _fallback("2b"),
            "cti": _fallback("2b"),
        }
    if hw.ram_gb >= 8:
        return {
            "chat": _fallback("2b"),
            "grc": _fallback("2b"),
            "code": _fallback("1.5b-coder"),
            "osint": _fallback("2b"),
            "cti": _fallback("2b"),
        }
    return {
        "chat": _fallback("0.8b"),
        "grc": _fallback("0.8b"),
        "code": _fallback("1.5b-coder"),
        "osint": _fallback("0.8b"),
        "cti": _fallback("0.8b"),
    }


def suggest_tier(hw: Hardware) -> str:
    """Picks a model tier id for chat role. Delegates to suggest_tiers()."""
    return suggest_tiers(hw)["chat"]
