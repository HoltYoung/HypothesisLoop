"""Langfuse tracing plumbing.

Phase 0: minimal — exposes `observe`, `start_session`, `tag_span` (placeholder),
and the RD-Agent-style trace tag constants from SPEC §6.6. Span-tagging
helpers will get fleshed out in later phases when there are real spans
to log.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# CRITICAL: override=True — see feedback_env_override memory.
load_dotenv(override=True)

# Re-export observe so the rest of the package does:
#   from hypothesisloop.trace import observe
try:
    from langfuse import observe  # type: ignore
except ImportError:  # pragma: no cover
    # Fallback no-op decorator so imports never fail; surfaces via tag_span warnings.
    def observe(*dargs, **dkwargs):  # type: ignore
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]

        def _decorator(func):
            return func

        return _decorator


# ---------------------------------------------------------------------------
# Trace schema constants (SPEC §6.6 — RD-Agent-style tags, verbatim).
# ---------------------------------------------------------------------------
RESEARCH_HYPOTHESIS = "research.hypothesis"
EVOLVING_CODES = "evolving.codes"
EVOLVING_FEEDBACKS = "evolving.feedbacks"
FEEDBACK_HYPOTHESIS = "feedback.hypothesis_feedback"
LOOP_ITERATION = "loop.iteration"
SAFETY_BIAS_FLAG = "safety.bias_flag"


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def _new_session_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"hl-{ts}-{uuid.uuid4().hex[:4]}"


def start_session(session_id: Optional[str] = None) -> str:
    """Initialize a Langfuse client + register a session_id for this run.

    Returns the resolved session_id (caller-supplied or freshly minted).
    The Langfuse client is constructed via `langfuse.Langfuse()`, which
    reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from env.
    """
    session_id = session_id or _new_session_id()

    try:
        from langfuse import Langfuse  # type: ignore

        # Instantiating warms the global client used by `@observe`.
        Langfuse()
    except ImportError:
        # Allow phase-0 tests to run without langfuse installed.
        pass

    # Stash on env so nested @observe calls can pick it up if needed.
    os.environ["HL_SESSION_ID"] = session_id
    return session_id


def get_session_usage(session_id: str) -> Dict[str, Any]:
    """Best-effort rollup of token usage / cost / wall-time for a Langfuse session.

    Returns::

        {
          "total_tokens": int,
          "input_tokens": int,
          "output_tokens": int,
          "total_cost_usd": float,
          "wall_time_s": float,
          "trace_count": int,
        }

    Failure modes (Langfuse offline, SDK API drift, unknown session) never
    raise — the report still has to render. Instead returns the same shape with
    zeros + an ``"_error"`` key carrying the exception message.
    """
    out: Dict[str, Any] = {
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost_usd": 0.0,
        "wall_time_s": 0.0,
        "trace_count": 0,
    }
    try:
        from langfuse import Langfuse  # type: ignore

        client = Langfuse()
        # Try several SDK shapes; v3 has moved this around.
        traces = None
        try:
            resp = client.api.trace.list(session_id=session_id)
            traces = getattr(resp, "data", None) or getattr(resp, "items", None) or resp
        except Exception:
            try:
                traces = list(client.fetch_traces(session_id=session_id).data)  # type: ignore[attr-defined]
            except Exception:
                traces = None

        if not traces:
            return out

        starts: list = []
        ends: list = []
        for t in traces:
            usage = getattr(t, "usage", None) or {}
            if isinstance(usage, dict):
                out["input_tokens"] += int(usage.get("input", 0) or usage.get("input_tokens", 0) or 0)
                out["output_tokens"] += int(usage.get("output", 0) or usage.get("output_tokens", 0) or 0)
                out["total_tokens"] += int(usage.get("total", 0) or usage.get("total_tokens", 0) or 0)
            else:
                out["input_tokens"] += int(getattr(usage, "input", 0) or 0)
                out["output_tokens"] += int(getattr(usage, "output", 0) or 0)
                out["total_tokens"] += int(getattr(usage, "total", 0) or 0)
            cost = getattr(t, "totalCost", None) or getattr(t, "total_cost", None) or 0.0
            try:
                out["total_cost_usd"] += float(cost or 0.0)
            except Exception:
                pass
            ts_start = getattr(t, "timestamp", None) or getattr(t, "createdAt", None) or getattr(t, "created_at", None)
            ts_end = getattr(t, "endTime", None) or getattr(t, "end_time", None) or ts_start
            if ts_start is not None:
                starts.append(ts_start)
            if ts_end is not None:
                ends.append(ts_end)

        out["trace_count"] = len(list(traces))
        if starts and ends:
            try:
                out["wall_time_s"] = max(_seconds_since_epoch(e) for e in ends) - min(
                    _seconds_since_epoch(s) for s in starts
                )
            except Exception:
                out["wall_time_s"] = 0.0
    except Exception as e:
        out["_error"] = f"{type(e).__name__}: {e}"
    return out


def _seconds_since_epoch(ts) -> float:
    """Coerce a datetime-or-string-or-number timestamp into POSIX seconds."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if hasattr(ts, "timestamp"):
        try:
            return float(ts.timestamp())
        except Exception:
            pass
    # Fallback: ISO string parsing
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def tag_span(name: str, payload: Dict[str, Any]) -> None:
    """Tag the current Langfuse span with `name` and structured `payload`.

    Phase 0 placeholder — real implementation lands in later phases when we
    have nested spans to write into. For now, this is a no-op so callers
    can wire in tag calls without depending on partial spans.

    TODO(phase 2+): use `langfuse.get_current_observation()` (or the
    decorator-context equivalent) to set `metadata`/`tags` on the active span.
    """
    return None
