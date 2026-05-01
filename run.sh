#!/usr/bin/env bash
# ============================================================================
# HypothesisLoop — one-shot setup + launch (Mac / Linux / WSL)
# ============================================================================
# Usage:    ./run.sh
# (chmod +x run.sh once if needed)
#
# Idempotent. First run: creates .venv, installs deps, launches Streamlit.
# Every subsequent run: skips setup, launches Streamlit.
# ============================================================================

set -e

cd "$(dirname "$0")"

step() { echo ""; echo ">> $1"; }
info() { echo "   $1"; }

# --- 1. Check Python ---
step "Checking Python"
if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
    echo "ERROR: 'python' / 'python3' not on PATH. Install Python 3.11+ from https://www.python.org/downloads/ and re-run." >&2
    exit 1
fi
PY=$(command -v python3 || command -v python)
info "$($PY --version)"

# --- 2. Create venv if missing ---
if [ ! -f .venv/bin/python ]; then
    step "Creating .venv (one-time)"
    $PY -m venv .venv
else
    info ".venv already exists — reusing"
fi

# --- 3. Activate ---
step "Activating .venv"
# shellcheck disable=SC1091
source .venv/bin/activate

# --- 4. Install / update deps if needed ---
SENTINEL=".venv/.hl_install_marker"
NEED_INSTALL=1
if [ -f "$SENTINEL" ] && [ "$SENTINEL" -nt "pyproject.toml" ]; then
    NEED_INSTALL=0
fi

if [ "$NEED_INSTALL" -eq 1 ]; then
    step "Installing dependencies (one-time, ~5 min — AutoGluon is the slow one)"
    python -m pip install --upgrade pip --quiet
    python -m pip install -e .
    date > "$SENTINEL"
else
    info "Dependencies up to date — skipping install"
fi

# --- 5. Check .env ---
if [ ! -f .env ]; then
    cat <<EOF >&2

ERROR: .env file is missing at the repo root.

Create a file named exactly '.env' here:
    $(pwd)/.env

with these contents:

    OPENAI_API_KEY=sk-proj-...
    KIMI_API_KEY=sk-...
    MOONSHOT_BASE_URL=https://api.moonshot.ai/v1
    DEFAULT_CHAT_PROVIDER=moonshot
    DEFAULT_CHAT_MODEL_OPENAI=gpt-4o-mini
    DEFAULT_CHAT_MODEL_MOONSHOT=kimi-k2.6
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com

Then re-run ./run.sh
EOF
    exit 1
fi
info ".env found"

# --- 6. Build RAG index if missing (cheap, ~5s, no-op when already there) ---
step "Checking RAG index"
python scripts/ensure_rag_index.py

# --- 7. Launch ---
step "Launching Streamlit at http://localhost:8501"
info "Press Ctrl+C to stop. Re-run ./run.sh anytime."
echo ""
python -m streamlit run hypothesisloop/ui/streamlit_app.py
