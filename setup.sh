#!/usr/bin/env bash
# Fortis Desktop setup (macOS / Linux)
# Usage:  bash setup.sh
set -euo pipefail

echo ""
echo "  Fortis - Cybersecurity Advisor : setup"
echo "  ----------------------------------------"

# 1. Python check
PY=""
for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
        v="$("$c" --version 2>&1 | grep -oE 'Python 3\.[0-9]+' || true)"
        if [ -n "$v" ]; then PY="$c"; break; fi
    fi
done
if [ -z "$PY" ]; then
    echo "  [X] Python 3 not found. Install python3 (https://www.python.org/downloads/)"
    echo "      or via brew: brew install python"
    exit 1
fi
echo "  [ok] $("$PY" --version)"

# 2. venv
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
    echo "  [..] Creating virtual environment (one-time)..."
    "$PY" -m venv .venv
fi
echo "  [ok] Virtual environment ready"
VPY=".venv/bin/python"

# 3. dependencies
if ! "$VPY" -c "import fastapi, chromadb, fitz, docx, pptx, openpyxl, reportlab, huggingface_hub, uvicorn, multipart, rank_bm25, matplotlib, webview, pytest" >/dev/null 2>&1; then
    echo "  [..] Installing dependencies (first run only, ~5 min)..."
    "$VPY" -m pip install --disable-pip-version-check -r requirements-dev.txt || {
        echo "  [X] Dependency install failed - check internet connection"
        exit 1
    }
    echo "  [ok] Dependencies installed"
else
    echo "  [ok] Dependencies present"
fi

# 4. llama-cpp-python
#    Prebuilt wheels only, from the llama.cpp wheel indexes (PyPI has no
#    wheels for this package - just the sdist). macOS on Apple Silicon and
#    Linux get wheels from the CPU index; NVIDIA machines try CUDA first.
MARKER="$(dirname "$0")/.llama-cuda-ok"
if ! "$VPY" -c "import llama_cpp" >/dev/null 2>&1; then
    echo "  [..] Installing the local AI engine (llama.cpp)..."
    OK=0
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        echo "       NVIDIA GPU detected - trying the CUDA wheel first..."
        if "$VPY" -m pip install --disable-pip-version-check --quiet --only-binary :all: \
            llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
            && "$VPY" -c "import llama_cpp" >/dev/null 2>&1; then
            OK=1
        else
            echo "       CUDA wheel failed to install or import - cleaning up and trying the CPU wheel..."
            "$VPY" -m pip uninstall --disable-pip-version-check --yes llama_cpp_python >/dev/null 2>&1 || true
        fi
    fi
    if [ "$OK" = "0" ]; then
        echo "       Trying the prebuilt CPU wheel (llama.cpp CPU index)..."
        if "$VPY" -m pip install --disable-pip-version-check --quiet --only-binary :all: \
            llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
            && "$VPY" -c "import llama_cpp" >/dev/null 2>&1; then
            OK=1
        else
            echo "       CPU wheel unavailable - cleaning up..."
            "$VPY" -m pip uninstall --disable-pip-version-check --yes llama_cpp_python >/dev/null 2>&1 || true
        fi
    fi
    if [ "$OK" = "1" ]; then
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 \
            && "$VPY" -c "import ctypes; ctypes.CDLL('libcuda.so.1')" >/dev/null 2>&1; then
            touch "$MARKER"
        fi
    else
        echo "  [X] No prebuilt llama-cpp-python wheel for this system."
        echo "      Options:"
        echo "        1. Install Visual Studio Build Tools / build-essential + CMake and rerun (builds from source)."
        echo "        2. Check https://abetlen.github.io/llama-cpp-python/whl/ for available wheels."
        exit 1
    fi
else
    echo "  [ok] AI engine present"
    [ -f "$MARKER" ] || touch "$MARKER"
fi

# 5. offline self-test
echo "  [..] Running offline sanity tests..."
EMBEDDING_BACKEND=hash-stub "$VPY" -m pytest desktop/tests/ -q --no-header | tail -n 15 || {
    echo "  [X] Self-test failed - see output above"
    exit 1
}
echo "  [ok] Self-tests passed"

echo ""
echo "  Setup complete. Starting Fortis..."
echo "  (First chat message downloads the AI model - about 3.4 GB, one time.)"
echo ""
exec "$VPY" desktop/app.py
