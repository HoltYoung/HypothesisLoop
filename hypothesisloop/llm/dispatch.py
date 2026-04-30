"""Multi-provider LLM dispatch.

Salvaged from `archive/builds/build4_rag_router_agent.py:86-123`.

Default model is Moonshot Kimi K2.6 (`moonshot-v1-128k`); fallback is
OpenAI `gpt-4o-mini`. Embeddings always use OpenAI `text-embedding-3-small`.

Provider is auto-resolved from the model name prefix:
    moonshot-* → Moonshot base_url (api.moonshot.ai/v1) + KIMI/MOONSHOT key
    gpt-*       → OpenAI base_url + OPENAI_API_KEY
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# CRITICAL: override=True — the user's shell may have a stale OPENAI_API_KEY
# that masks the value in .env. See feedback memory: feedback_env_override.
load_dotenv(override=True)


HL_MODEL_DEFAULT = "moonshot-v1-128k"          # Kimi K2.6
HL_MODEL_FALLBACK = "gpt-4o-mini"
HL_EMBED_MODEL = "text-embedding-3-small"      # always OpenAI


def _resolve_provider(model: str) -> str:
    name = model.lower()
    if name.startswith("moonshot") or name.startswith("kimi"):
        return "moonshot"
    if name.startswith("gpt") or name.startswith("o1") or name.startswith("o3") or name.startswith("o4"):
        return "openai"
    # Fall back to env override or openai
    return (os.environ.get("DEFAULT_CHAT_PROVIDER") or "openai").lower()


def _configure_llm(provider: str) -> Dict[str, Any]:
    """Return kwargs to pass to ChatOpenAI for the given provider."""
    if provider == "moonshot":
        key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
        if not key:
            raise RuntimeError(
                "provider=moonshot but neither KIMI_API_KEY nor MOONSHOT_API_KEY is set in .env"
            )
        base = os.environ.get("MOONSHOT_BASE_URL") or "https://api.moonshot.ai/v1"
        return {"api_key": key, "base_url": base}
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("provider=openai but OPENAI_API_KEY is not set in .env")
        return {"api_key": key}
    raise ValueError(f"Unknown provider: {provider!r}. Use 'openai' or 'moonshot'.")


def get_llm(model: Optional[str] = None, temperature: float = 0.7) -> ChatOpenAI:
    """Return a ChatOpenAI client wired for the chosen provider."""
    model = model or os.environ.get("HL_MODEL") or HL_MODEL_DEFAULT
    provider = _resolve_provider(model)
    kwargs = _configure_llm(provider)
    return ChatOpenAI(model=model, temperature=temperature, **kwargs)
