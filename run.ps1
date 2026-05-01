# ============================================================================
# HypothesisLoop — one-shot setup + launch (Windows / PowerShell)
# ============================================================================
# Usage:    .\run.ps1
#
# Idempotent. First run: creates .venv, installs deps, launches Streamlit.
# Every subsequent run: skips setup, launches Streamlit.
#
# Bypass execution policy if needed:
#   powershell -ExecutionPolicy Bypass -File .\run.ps1
# ============================================================================

$ErrorActionPreference = "Stop"

function Write-Step($msg) {
    Write-Host "" -ForegroundColor Cyan
    Write-Host ">> $msg" -ForegroundColor Cyan
}

function Write-Info($msg) {
    Write-Host "   $msg" -ForegroundColor DarkGray
}

# Move to script's own directory so this works regardless of where it's invoked.
Set-Location -Path $PSScriptRoot

# --- 1. Check Python ---
Write-Step "Checking Python"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 'python' not on PATH. Install Python 3.11+ from https://www.python.org/downloads/ and re-run." -ForegroundColor Red
    exit 1
}
$pyVer = & python --version
Write-Info "$pyVer"

# --- 2. Create venv if missing ---
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Step "Creating .venv (one-time)"
    & python -m venv .venv
} else {
    Write-Info ".venv already exists — reusing"
}

# --- 3. Activate venv ---
Write-Step "Activating .venv"
& ".\.venv\Scripts\Activate.ps1"

# --- 4. Install / update deps if needed ---
# Use a sentinel file so we don't re-pip-install on every launch.
$sentinel = ".venv\.hl_install_marker"
$needInstall = $true
if (Test-Path $sentinel) {
    # If pyproject is newer than the sentinel, deps may have changed.
    $sentMtime = (Get-Item $sentinel).LastWriteTime
    $pyprojMtime = (Get-Item "pyproject.toml").LastWriteTime
    if ($sentMtime -gt $pyprojMtime) {
        $needInstall = $false
    }
}

if ($needInstall) {
    Write-Step "Installing dependencies (one-time, ~5 min — AutoGluon is the slow one)"
    & python -m pip install --upgrade pip --quiet
    & python -m pip install -e .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: pip install -e . failed. See output above." -ForegroundColor Red
        exit 1
    }
    Set-Content -Path $sentinel -Value (Get-Date).ToString() -Encoding utf8
} else {
    Write-Info "Dependencies up to date — skipping install"
}

# --- 5. Check .env ---
if (-not (Test-Path ".env")) {
    Write-Host "" -ForegroundColor Yellow
    Write-Host "ERROR: .env file is missing at the repo root." -ForegroundColor Red
    Write-Host ""
    Write-Host "Create a file named exactly '.env' (no extension) here:" -ForegroundColor Yellow
    Write-Host "   $((Resolve-Path .).Path)\.env" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "with these contents:" -ForegroundColor Yellow
    Write-Host @"

OPENAI_API_KEY=sk-proj-...
KIMI_API_KEY=sk-...
MOONSHOT_BASE_URL=https://api.moonshot.ai/v1
DEFAULT_CHAT_PROVIDER=moonshot
DEFAULT_CHAT_MODEL_OPENAI=gpt-4o-mini
DEFAULT_CHAT_MODEL_MOONSHOT=kimi-k2.6
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

"@ -ForegroundColor DarkGray
    Write-Host "Then re-run .\run.ps1" -ForegroundColor Yellow
    exit 1
}
Write-Info ".env found"

# --- 6. Build RAG index if missing (cheap, ~5s, no-op when already there) ---
Write-Step "Checking RAG index"
& python scripts/ensure_rag_index.py

# --- 7. Launch Streamlit ---
Write-Step "Launching Streamlit at http://localhost:8501"
Write-Info "Press Ctrl+C in this terminal to stop. Re-run .\run.ps1 anytime."
Write-Host ""
& python -m streamlit run hypothesisloop/ui/streamlit_app.py
