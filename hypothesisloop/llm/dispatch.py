"""Multi-provider LLM dispatch.

Salvaged from ``archive/builds/build4_rag_router_agent.py:86-123``, extended
in Phase 9 with runtime provider overrides (UI-supplied API key + base URL)
and a per-call cost tracker hook.

Default model is Moonshot Kimi K2.6 (``moonshot-v1-128k``); fallback is
OpenAI ``gpt-4o-mini``. Embeddings always use OpenAI ``text-embedding-3-small``.

Provider is auto-resolved from the model-name prefix:
    moonshot-* / kimi-*  -> Moonshot base_url (api.moonshot.ai/v1) + KIMI/MOONSHOT key
    gpt-* / o[134]-*     -> OpenAI base_url + OPENAI_API_KEY
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, TYPE_CHECKING

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:  # pragma: no cover
    from hypothesisloop.llm.cost_tracker import CostTracker

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
    return (os.environ.get("DEFAULT_CHAT_PROVIDER") or "openai").lower()


def _configure_llm(
    provider: str,
    *,
    api_key_override: Optional[str] = None,
    api_base_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Return kwargs for ``ChatOpenAI``. Overrides take precedence over env."""
    if provider == "moonshot":
        key = api_key_override or os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
        if not key:
            raise RuntimeError(
                "provider=moonshot but no API key supplied (KIMI_API_KEY / MOONSHOT_API_KEY / explicit override)"
            )
        base = api_base_override or os.environ.get("MOONSHOT_BASE_URL") or "https://api.moonshot.ai/v1"
        return {"api_key": key, "base_url": base}
    if provider == "openai":
        key = api_key_override or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "provider=openai but no API key supplied (OPENAI_API_KEY / explicit override)"
            )
        kwargs: Dict[str, Any] = {"api_key": key}
        if api_base_override:
            kwargs["base_url"] = api_base_override
        return kwargs
    raise ValueError(f"Unknown provider: {provider!r}. Use 'openai' or 'moonshot'.")


class _TrackerCallback(BaseCallbackHandler):
    """LangChain ``BaseCallbackHandler`` that records every LLM call.

    Implemented at the callback layer (not via attribute monkey-patching)
    so it propagates through ``llm.bind(...)`` chains, which a Pydantic v2
    ``ChatOpenAI`` instance otherwise blocks.
    """

    def __init__(self, model: str, tracker: "CostTracker"):
        super().__init__()
        self.model = model
        self.tracker = tracker

    def on_llm_end(self, response, **kwargs):  # langchain LLMResult
        try:
            usage = None
            # LLMResult.llm_output sometimes carries token_usage; otherwise
            # AIMessage.generations[0][0].message.usage_metadata is populated.
            if getattr(response, "llm_output", None):
                usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")
            if usage is None:
                gens = getattr(response, "generations", None)
                if gens and gens[0]:
                    msg = getattr(gens[0][0], "message", None)
                    usage = (
                        getattr(msg, "usage_metadata", None)
                        if msg is not None
                        else None
                    )
            if usage is not None:
                self.tracker.record(self.model, usage)
        except Exception:  # pragma: no cover — defensive
            pass

    # No-op overrides so the base class's defaults stay quiet.


def get_llm(
    model: Optional[str] = None,
    *,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    tracker: "Optional[CostTracker]" = None,
) -> ChatOpenAI:
    """Return a ``ChatOpenAI`` client wired for the resolved provider.

    ``api_key`` / ``api_base`` override the env-resolved values when supplied
    (used by the Streamlit UI's runtime config). ``tracker`` is an optional
    :class:`hypothesisloop.llm.cost_tracker.CostTracker`; when present, every
    ``.invoke`` call appends a usage record to it.
    """
    model = model or os.environ.get("HL_MODEL") or HL_MODEL_DEFAULT
    provider = _resolve_provider(model)
    kwargs = _configure_llm(
        provider, api_key_override=api_key, api_base_override=api_base
    )
    if tracker is not None:
        kwargs["callbacks"] = [_TrackerCallback(model=model, tracker=tracker)]
    return ChatOpenAI(model=model, temperature=temperature, **kwargs)
