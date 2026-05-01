# HypothesisLoop — Hugging Face Spaces deployment
# Reference: https://huggingface.co/docs/hub/spaces-sdks-docker
#
# Build:    docker build -t hypothesisloop .
# Run:      docker run -p 7860:7860 --env-file .env hypothesisloop
#
# On HF Spaces, port 7860 is the default and gets exposed via the Space's URL.
# Secrets (OPENAI_API_KEY, KIMI_API_KEY, LANGFUSE_*) come in as env vars
# injected by HF — never bake them into the image.

FROM python:3.11-slim

# System deps for AutoGluon (LightGBM/XGBoost/CatBoost native code) and FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires a non-root user
RUN useradd --create-home --uid 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR $HOME/app

# Copy the project (uses .dockerignore to skip archive/, .venv/, reports/, etc.)
COPY --chown=user . $HOME/app

# Install. pyproject.toml is the dep source-of-truth.
RUN pip install --user --upgrade pip setuptools wheel \
    && pip install --user -e .

# Streamlit on port 7860 (HF default), bind 0.0.0.0, no XSRF (HF reverse proxy).
# At container start, ensure the FAISS index is present (it's gitignored because
# binary blobs trip HF's pre-receive hook; cheap to rebuild on first boot).
EXPOSE 7860

CMD ["sh", "-c", "python scripts/ensure_rag_index.py && python -m streamlit run hypothesisloop/ui/streamlit_app.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false --browser.gatherUsageStats=false"]
