"""
UI smoke tests: the shipped chat UI (server/static/index.html) must keep the
key element ids, classes and JS hooks that the desktop app and its offline
verification rely on. These are static parse checks — no browser needed.
"""
import os
import re

import pytest

_STATIC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "server", "static", "index.html",
)


@pytest.fixture(scope="module")
def index_html() -> str:
    with open(_STATIC, "r", encoding="utf-8") as f:
        return f.read()


def _script_body(html: str) -> str:
    m = re.search(r"<script>(.*)</script>", html, re.DOTALL)
    assert m, "index.html must contain an inline <script> block"
    return m.group(1)


def test_core_element_ids_present(index_html):
    for el_id in ("chat", "chat-inner", "composer", "input", "btn-send",
                  "status-line", "status-dot", "model-btn", "engagements",
                  "sidebar", "dlg-model", "tier-list", "hw-line",
                  "dl-progress", "dl-status", "file-input"):
        assert f'id="{el_id}"' in index_html, f"missing required id #{el_id}"


def test_report_progress_elements_present(index_html):
    for el_id in ("report-progress", "report-progress-stage", "report-progress-count",
                  "report-progress-bar", "report-progress-detail"):
        assert f'id="{el_id}"' in index_html, f"missing required id #{el_id}"
    assert 'role="progressbar"' in index_html
    assert 'aria-live="polite"' in index_html


def test_report_progress_poll_contract(index_html):
    script = _script_body(index_html)
    for needle in ("/api/report/progress", "startReportProgressPoll",
                   "pollReportProgress", "showReportProgress", "finishReportProgress"):
        assert needle in script, f"report-progress JS missing: {needle}"
    assert "Analyzing document batches" in script
    assert "Merging findings" in script
    assert "Building document" in script
    assert "Finishing up" in script
    for fn in ("startReportProgressPoll", "pollReportProgress", "showReportProgress"):
        assert re.search(rf"function {fn}\(", script), f"{fn} must stay a top-level function"


def test_report_progress_poll_interval_is_800ms(index_html):
    script = _script_body(index_html)
    m = re.search(r"setTimeout\(\s*(?:function\s*\(\)\s*\{|(?:\(\)\s*=>)\s*\{)[^}]*pollReportProgress\(gen\);[^}]*\}\s*,\s*(\d+)\s*\)", script, re.DOTALL)
    assert m, "poll loop must reschedule pollReportProgress via setTimeout"
    assert m.group(1) == "800", "poll interval must stay 800ms"


def test_report_progress_hides_after_1500ms(index_html):
    script = _script_body(index_html)
    m = re.search(r"finishReportProgress\(gen\)\s*\{[^}]*?setTimeout\(\s*(?:function\s*\(\)\s*\{|(?:\(\)\s*=>)\s*\{)[^}]*?style\.display\s*=\s*'none'[^}]*?\}\s*,\s*(\d+)\s*\)", script, re.DOTALL)
    assert m, "finishReportProgress must schedule the hide"
    assert m.group(1) == "1500", "hide delay must stay ~1.5s"


def test_hardware_status_line_present(index_html):
    assert 'id="hw-status-line"' in index_html
    script = _script_body(index_html)
    assert "renderHardwareStatusLine" in script
    assert re.search(r"function renderHardwareStatusLine\(", script), \
        "renderHardwareStatusLine must stay a top-level function"
    assert "GPU: all layers offloaded (CUDA)" in script
    assert "Running on CPU" in script
    assert "re-run setup to enable GPU acceleration" in script


def test_cpu_mode_progress_hint(index_html):
    script = _script_body(index_html)
    assert "CPU mode — this can take several minutes" in script
    assert "offload === 'cpu'" in script


def test_no_unsafe_model_output_insertion(index_html):
    script = _script_body(index_html)
    assert ".outerHTML" not in script, "model output must not be inserted via outerHTML"
    for m in re.finditer(r"bubble\.innerHTML\s*=", script):
        after = script[m.start():m.start() + 200]
        assert "renderMd(" in after or "esc(" in after, \
            "assistant bubble innerHTML must come from renderMd/esc (sanitized markdown)"
    assert "esc(raw)" in script, "mermaid fallback must escape the raw source"
