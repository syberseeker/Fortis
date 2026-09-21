# Fortis Desktop setup script (Windows)
# Usage:  right-click -> Run with PowerShell, or:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "  Fortis - Cybersecurity Advisor : setup" -ForegroundColor Cyan
Write-Host "  ----------------------------------------" -ForegroundColor DarkGray

# 1. Python check
$py = $null
foreach ($c in @("python", "py")) {
    try {
        $v = & $c --version 2>$null
        if ($v -match "Python 3\.(1[0-4]|[0-9])") { $py = $c; break }
    } catch {}
}
if (-not $py) {
    Write-Host ""
    Write-Host "  [X] Python 3.10+ not found." -ForegroundColor Red
    Write-Host "      Install it from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "      IMPORTANT: tick 'Add python.exe to PATH' in the installer." -ForegroundColor Yellow
    Read-Host "  Press Enter to exit"
    exit 1
}
Write-Host "  [ok] $(& $py --version)"

# 2. venv
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "  [..] Creating virtual environment (one-time)..."
    & $py -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Host "  [X] venv creation failed" -ForegroundColor Red; Read-Host "  Press Enter to exit"; exit 1 }
} else {
    Write-Host "  [ok] Virtual environment exists"
}
$venvPy = ".\.venv\Scripts\python.exe"

# 3. dependencies (skipped when already present)
$probe = & $venvPy -c "import fastapi, chromadb, fitz, docx, pptx, openpyxl, reportlab, huggingface_hub, uvicorn, multipart" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [..] Installing dependencies (first run only, ~5 min)..."
    & $venvPy -m pip install --disable-pip-version-check --quiet fastapi "uvicorn[standard]" python-multipart pydantic chromadb sentence-transformers pymupdf python-docx python-pptx openpyxl reportlab markdown-it-py "huggingface_hub>=0.26" pywebview
    if ($LASTEXITCODE -ne 0) { Write-Host "  [X] Dependency install failed - check internet connection" -ForegroundColor Red; Read-Host "  Press Enter to exit"; exit 1 }
    Write-Host "  [ok] Dependencies installed"
} else {
    Write-Host "  [ok] Dependencies present"
}

# 4. llama-cpp-python: CUDA wheel if GPU + marker missing, else CPU wheel
$marker = Join-Path $PSScriptRoot ".llama-cuda-ok"
$haveLlama = & $venvPy -c "import llama_cpp" 2>$null
if ($LASTEXITCODE -ne 0) {
    $hasNvidia = $false
    try {
        $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($smi) { & nvidia-smi -L | Out-Null; if ($LASTEXITCODE -eq 0) { $hasNvidia = $true } }
    } catch {}

    Write-Host "  [..] Installing the local AI engine (llama.cpp, ~200 MB)..."
    if ($hasNvidia) {
        Write-Host "       NVIDIA GPU detected - trying the CUDA build first..."
        & $venvPy -m pip install --disable-pip-version-check --quiet llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
        if ($LASTEXITCODE -eq 0 -and (& $venvPy -c "import llama_cpp" 2>$null) -or $LASTEXITCODE -eq 0) {
            New-Item -ItemType File -Path $marker -Force | Out-Null
            Write-Host "  [ok] Engine installed (GPU-accelerated)"
        } else {
            Write-Host "       CUDA build unavailable - falling back to CPU (slower but works)"
            & $venvPy -m pip install --disable-pip-version-check --quiet --force-reinstall llama-cpp-python
            Write-Host "  [ok] Engine installed (CPU)"
        }
    } else {
        & $venvPy -m pip install --disable-pip-version-check --quiet llama-cpp-python
        if ($LASTEXITCODE -ne 0) { Write-Host "  [X] Engine install failed" -ForegroundColor Red; Read-Host "  Press Enter to exit"; exit 1 }
        Write-Host "  [ok] Engine installed (CPU)"
    }
} else {
    Write-Host "  [ok] AI engine present"
    if (-not (Test-Path $marker)) { New-Item -ItemType File -Path $marker -Force | Out-Null }
}

# 5. offline self-test (fast, no model download)
Write-Host "  [..] Running offline sanity tests..."
$env:EMBEDDING_BACKEND = "hash-stub"
& $venvPy -m pytest desktop\tests\ -q --no-header 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [X] Self-test failed - see output above" -ForegroundColor Red
    Read-Host "  Press Enter to exit"
    exit 1
}
Remove-Item Env:EMBEDDING_BACKEND -ErrorAction SilentlyContinue
Write-Host "  [ok] Self-tests passed"

Write-Host ""
Write-Host "  Setup complete. Starting Fortis..." -ForegroundColor Green
Write-Host "  (First chat message downloads the AI model - about 3.4 GB, one time.)"
Write-Host ""
& $venvPy desktop\app.py
