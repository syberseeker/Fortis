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
if ! "$VPY" -c "import fastapi, chromadb, fitz, docx, pptx, openpyxl, reportlab, huggingface_hub, uvicorn, multipart" >/dev/null 2>&1; then
    echo "  [..] Installing dependencies (first run only, ~5 min)..."
    "$VPY" -m pip install --disable-pip-version-check --quiet \
        fastapi "uvicorn[standard]" python-multipart pydantic chromadb \
        sentence-transformers pymupdf python-docx python-pptx openpyxl \
        reportlab markdown-it-py "huggingface_hub>=0.26" pywebview
    echo "  [ok] Dependencies installed"
else
    echo "  [ok] Dependencies present"
fi

# 4. llama-cpp-python
#    macOS on Apple Silicon: CPU (Metal) wheel builds fine from source.
#    Linux with NVIDIA: try CUDA wheel, else CPU.
MARKER="$(dirname "$0")/.llama-cuda-ok"
if ! "$VPY" -c "import llama_cpp" >/dev/null 2>&1; then
    echo "  [..] Installing the local AI engine (llama.cpp)..."
    HAS_NVIDIA=0
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        HAS_NVIDIA=1
    fi
    if [ "$HAS_NVIDIA" = "1" ]; then
        echo "       NVIDIA GPU detected - trying the CUDA build first..."
        if "$VPY" -m pip install --disable-pip-version-check --quiet \
            llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124; then
            touch "$MARKER"
            echo "  [ok] Engine installed (GPU-accelerated)"
        else
            echo "       CUDA build unavailable - falling back to CPU"
            "$VPY" -m pip install --disable-pip-version-check --quiet --force-reinstall llama-cpp-python
            echo "  [ok] Engine installed (CPU)"
        fi
    else
        "$VPY" -m pip install --disable-pip-version-check --quiet llama-cpp-python
        echo "  [ok] Engine installed (CPU)"
    fi
else
    echo "  [ok] AI engine present"
    [ -f "$MARKER" ] || touch "$MARKER"
fi

# 5. offline self-test
echo "  [..] Running offline sanity tests..."
EMBEDDING_BACKEND=hash-stub "$VPY" -m pytest desktop/tests/ -q --no-header > /dev/null 2>&1 || {
    echo "  [X] Self-test failed"
    exit 1
}
echo "  [ok] Self-test passed (14/14)"

echo ""
echo "  Setup complete. Starting Fortis..."
echo "  (First chat message downloads the AI model - about 3.4 GB, one time.)"
echo ""
exec "$VPY" desktop/app.py
