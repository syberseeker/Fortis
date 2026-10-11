import atexit
import hashlib
import logging
import os
import shutil
import tempfile
import threading
from typing import Optional

from .schema import SecurityReport, severity_counts, framework_counts

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "fortis-chart-cache", f"pid-{os.getpid()}")
_CACHE_LOCK = threading.Lock()
_CACHE_FILES: dict = {}


def _cache_file_path(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()
    return os.path.join(_CACHE_DIR, f"{digest}.png")


def _cleanup_cache_files() -> None:
    with _CACHE_LOCK:
        paths = list(_CACHE_FILES.values())
        _CACHE_FILES.clear()
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        shutil.rmtree(_CACHE_DIR)
    except OSError:
        pass


atexit.register(_cleanup_cache_files)


def _cached_chart_path(report: SecurityReport, kind: str, signature: str, renderer) -> Optional[str]:
    key = f"{kind}|{signature}"
    with _CACHE_LOCK:
        existing = _CACHE_FILES.get(key)
    if existing is not None and os.path.exists(existing):
        return existing

    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = _cache_file_path(key)
    digest = os.path.splitext(os.path.basename(path))[0]
    tmp_path = os.path.join(_CACHE_DIR, f"{digest}.{os.getpid()}.{threading.get_ident()}.tmp.png")
    written = renderer(report, tmp_path)
    if not written:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return None
    try:
        os.replace(tmp_path, path)
    except OSError:
        path = tmp_path
    with _CACHE_LOCK:
        _CACHE_FILES[key] = path
    return path


def _severity_signature(report: SecurityReport) -> str:
    counts = severity_counts(report)
    return f"{report.title}|" + "|".join(f"{sev}={counts[sev]}" for sev in counts)


def _framework_signature(report: SecurityReport) -> str:
    counts = framework_counts(report)
    return f"{report.title}|" + "|".join(f"{fw}={counts[fw]}" for fw in sorted(counts, key=str))


def render_severity_chart_cached(report: SecurityReport) -> Optional[str]:
    return _cached_chart_path(
        report, "severity", _severity_signature(report), render_severity_chart
    )


def render_framework_chart_cached(report: SecurityReport) -> Optional[str]:
    return _cached_chart_path(
        report, "framework", _framework_signature(report), render_framework_chart
    )


def render_severity_chart(report: SecurityReport, output_path: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = severity_counts(report)
        severities = list(counts.keys())
        values = list(counts.values())
        colors = ["#8B0000", "#C0392B", "#E67E22", "#218738", "#5D6D7E"]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(severities, values, color=colors)
        ax.set_title("Findings by Severity")
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return output_path
    except Exception as e:
        logger.warning("Failed to render severity chart: %s", e)
        return None
    finally:
        if "fig" in locals():
            plt.close(fig)


def render_framework_chart(report: SecurityReport, output_path: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = framework_counts(report)
        if not counts or all(v == 0 for v in counts.values()):
            return None

        frameworks = list(counts.keys())
        values = list(counts.values())

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(frameworks, values, color="#0B1F3A")
        ax.set_title("Findings by Framework")
        ax.set_xlabel("Count")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return output_path
    except Exception as e:
        logger.warning("Failed to render framework chart: %s", e)
        return None
    finally:
        if "fig" in locals():
            plt.close(fig)
