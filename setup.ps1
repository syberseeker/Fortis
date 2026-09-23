# Fortis Desktop setup script (Windows)
# Usage:  right-click -> Run with PowerShell, or:  powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"

function Invoke-Native {
    param([scriptblock]$Command)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Command } finally { $ErrorActionPreference = $prev }
}

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
$probe = Invoke-Native { & $venvPy -c "import fastapi, chromadb, fitz, docx, pptx, openpyxl, reportlab, huggingface_hub, uvicorn, multipart, rank_bm25, matplotlib, webview, pytest" 2>$null }
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [..] Installing dependencies (first run only, ~5 min)..."
    & $venvPy -m pip install --disable-pip-version-check -r requirements-dev.txt
    if ($LASTEXITCODE -ne 0) { Write-Host "  [X] Dependency install failed - check internet connection" -ForegroundColor Red; Read-Host "  Press Enter to exit"; exit 1 }
    Write-Host "  [ok] Dependencies installed"
} else {
    Write-Host "  [ok] Dependencies present"
}

# 4. llama-cpp-python: prebuilt CUDA wheel if GPU, else prebuilt CPU wheel
#    (never builds from source - that needs MSVC/CMake which we can't assume)
$marker = Join-Path $PSScriptRoot ".llama-cuda-ok"
$haveLlama = Invoke-Native { & $venvPy -c "import llama_cpp" 2>$null }
if ($LASTEXITCODE -ne 0) {
    $hasNvidia = $false
    try {
        $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($smi) { & nvidia-smi -L | Out-Null; if ($LASTEXITCODE -eq 0) { $hasNvidia = $true } }
    } catch {}

    Write-Host "  [..] Installing the local AI engine (llama.cpp, ~200 MB)..."
    $ok = $false
    if ($hasNvidia) {
        Write-Host "       NVIDIA GPU detected - trying the CUDA wheel first..."
        & $venvPy -m pip install --disable-pip-version-check --quiet --only-binary :all: llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
        if ($LASTEXITCODE -eq 0) { Invoke-Native { & $venvPy -c "import llama_cpp" 2>$null }; if ($LASTEXITCODE -eq 0) { $ok = $true } }
        if ($ok) {
            New-Item -ItemType File -Path $marker -Force | Out-Null
            Write-Host "  [ok] Engine installed (GPU-accelerated)"
        } else {
            Write-Host "       CUDA wheel unavailable - trying the CPU wheel (slower but works)"
        }
    }
    if (-not $ok) {
        & $venvPy -m pip install --disable-pip-version-check --quiet --only-binary :all: llama-cpp-python
        if ($LASTEXITCODE -ne 0) {
            Write-Host "       CPU wheel not on PyPI for this Python - trying the llama.cpp CPU index..."
            & $venvPy -m pip install --disable-pip-version-check --quiet --only-binary :all: llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
        }
        if ($LASTEXITCODE -eq 0) { Invoke-Native { & $venvPy -c "import llama_cpp" 2>$null }; if ($LASTEXITCODE -eq 0) { $ok = $true } }
        if ($ok) {
            Write-Host "  [ok] Engine installed (CPU)"
        } else {
            $hasMsvc = $false
            try {
                $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
                if (Test-Path $vswhere) {
                    $vs = Invoke-Native { & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null }
                    if ($vs) { $hasMsvc = $true }
                }
                if (-not $hasMsvc -and (Get-Command cl.exe -ErrorAction SilentlyContinue)) { $hasMsvc = $true }
            } catch {}
            if ($hasMsvc) {
                Write-Host "       No prebuilt wheel - compiling from source with the detected MSVC toolchain (needs CMake+nmake)..."
                & $venvPy -m pip install --disable-pip-version-check --quiet llama-cpp-python
                if ($LASTEXITCODE -eq 0) { Invoke-Native { & $venvPy -c "import llama_cpp" 2>$null }; if ($LASTEXITCODE -eq 0) { $ok = $true } }
            }
            if ($ok) {
                Write-Host "  [ok] Engine installed (CPU, built from source)"
            } else {
                Write-Host "  [X] No prebuilt llama-cpp-python wheel for $(& $py --version) on this system." -ForegroundColor Red
                Write-Host "      Options:" -ForegroundColor Yellow
                Write-Host "        1. Install Python 3.13 (has prebuilt wheels) and rerun setup." -ForegroundColor Yellow
                Write-Host "        2. Install Visual Studio Build Tools (Desktop development with C++) and rerun." -ForegroundColor Yellow
                Write-Host "        3. Use the community Windows wheel collection (cp314) from: llama-cpp-wheels on Hugging Face." -ForegroundColor Yellow
                Read-Host "  Press Enter to exit"
                exit 1
            }
        }
    }
} else {
    Write-Host "  [ok] AI engine present"
    if (-not (Test-Path $marker)) { New-Item -ItemType File -Path $marker -Force | Out-Null }
}

# 5. offline self-test (fast, no model download)
Write-Host "  [..] Running offline sanity tests..."
$env:EMBEDDING_BACKEND = "hash-stub"
Invoke-Native { & $venvPy -m pytest desktop\tests\ -q --no-header 2>&1 } | Select-Object -Last 15
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
