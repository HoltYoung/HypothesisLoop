"""Per-call LLM usage and cost accumulation.

Source of truth: the API response's ``usage`` field. NOT Langfuse's session
rollup — that's a separate (lagging) view useful for the post-run report,
but unsuitable for the live sidebar metric.

Wire-up: ``hypothesisloop.llm.dispatch.get_llm`` accepts an optional
``tracker: CostTracker`` and records every ``.invoke`` response into it. The
Streamlit app instantiates one tracker per session in ``st.session_state``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


# Pricing per 1M tokens, USD. Update when providers change rates.
RATES: dict[str, dict[str, float]] = {
    "moonshot-v1-128k":   {"input": 0.95, "output": 4.00, "cache_hit": 0.16},
    "moonshot-v1-32k":    {"input": 0.95, "output": 4.00, "cache_hit": 0.16},
    "moonshot-v1-8k":     {"input": 0.95, "output": 4.00, "cache_hit": 0.16},
    "gpt-4o-mini":        {"input": 0.15, "output": 0.60, "cache_hit": 0.075},
    "gpt-4o":             {"input": 2.50, "output": 10.00, "cache_hit": 1.25},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0, "cache_hit": 0.0},
}

_DEFAULT_RATE = {"input": 0.0, "output": 0.0, "cache_hit": 0.0}


@dataclass
class UsageRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cache_tokens: int = 0
    cost_usd: float = 0.0


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _read_usage_field(usage: Any, *names: str) -> int:
    """Pull ``usage.<name>`` (or ``usage[name]``) for the first hit."""
    if usage is None:
        return 0
    for name in names:
        if hasattr(usage, name):
            return _coerce_int(getattr(usage, name))
        if isinstance(usage, dict) and name in usage:
            return _coerce_int(usage[name])
    return 0


def _read_cache_field(usage: Any) -> int:
    """LangChain emits cache hits in different fields per provider; try the common shapes."""
    direct = _read_usage_field(
        usage,
        "cache_read_input_tokens",
        "cache_hit_tokens",
        "prompt_cache_hit_tokens",
    )
    if direct:
        return direct
    # Some SDKs nest cache details inside input_token_details / cache_creation
    details = getattr(usage, "input_token_details", None) or (
        usage.get("input_token_details") if isinstance(usage, dict) else None
    )
    if details:
        return _coerce_int(
            getattr(details, "cache_read", None)
            or (details.get("cache_read") if isinstance(details, dict) else None)
        )
    return 0


@dataclass
class CostTracker:
    """Thread-safe per-session usage + cost accumulator."""

    records: list[UsageRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ---- recording -----------------------------------------------------
    def record(self, model: str, usage: Any) -> UsageRecord:
        """Append a record for one LLM call.

        ``usage`` is the response's usage field, which can be:
          - LangChain's ``response.usage_metadata`` dict (input_tokens/output_tokens/total_tokens),
          - the raw OpenAI/Moonshot dict with prompt_tokens/completion_tokens,
          - a Pydantic-ish object exposing the same attributes.
        """
        # LangChain uses input_tokens/output_tokens; OpenAI uses prompt_tokens/completion_tokens.
        input_tokens = _read_usage_field(usage, "input_tokens", "prompt_tokens")
        output_tokens = _read_usage_field(usage, "output_tokens", "completion_tokens")
        cache_tokens = _read_cache_field(usage)

        rate = RATES.get(model, _DEFAULT_RATE)
        # Cache hits are billed at the lower cache_hit rate; subtract them
        # from the regular input tokens before pricing the rest.
        billable_input = max(0, input_tokens - cache_tokens)
        cost = (
            (billable_input * rate["input"])
            + (cache_tokens * rate["cache_hit"])
            + (output_tokens * rate["output"])
        ) / 1_000_000.0

        rec = UsageRecord(
            model=model,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            cache_tokens=int(cache_tokens),
            cost_usd=float(cost),
        )
        with self._lock:
            self.records.append(rec)
        return rec

    # ---- aggregates ----------------------------------------------------
    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(r.input_tokens + r.output_tokens for r in self.records)

    @property
    def total_input_tokens(self) -> int:
        with self._lock:
            return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        with self._lock:
            return sum(r.output_tokens for r in self.records)

    @property
    def total_cost_usd(self) -> float:
        with self._lock:
            return sum(r.cost_usd for r in self.records)

    @property
    def total_calls(self) -> int:
        with self._lock:
            return len(self.records)

    def by_model(self) -> dict[str, dict]:
        """Per-model breakdown for the report's run-metadata table."""
        out: dict[str, dict] = {}
        with self._lock:
            for r in self.records:
                bucket = out.setdefault(
                    r.model,
                    {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_tokens": 0, "cost_usd": 0.0},
                )
                bucket["calls"] += 1
                bucket["input_tokens"] += r.input_tokens
                bucket["output_tokens"] += r.output_tokens
                bucket["cache_tokens"] += r.cache_tokens
                bucket["cost_usd"] += r.cost_usd
        return out


__all__ = ["CostTracker", "UsageRecord", "RATES"]
