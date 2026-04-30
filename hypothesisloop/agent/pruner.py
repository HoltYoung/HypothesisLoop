"""Context-window hygiene — slim the trace into prompt-ready views.

The pruner is **deterministic and pure**: same trace in, same view out, no
mutation. It returns lists of dicts shaped for ``hypothesize.j2``:

- :meth:`Pruner.prior_hypotheses_view` — accepted hypotheses (with feedback).
  Only the most recent ``keep_full_attempts_back`` iterations keep their
  ``code_snippet``; older iterations get ``code_snippet=None`` (the full code
  is still on disk and in Langfuse — we just don't pay for it on every prompt).
- :meth:`Pruner.rejected_view` — novelty-rejected hypotheses.

Token estimation uses ``tiktoken`` with ``cl100k_base`` (close enough for both
OpenAI and Kimi for budgeting purposes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import tiktoken

from hypothesisloop.agent.state import DAGTrace, Experiment


_REASON_TRUNCATE = 120
_CODE_TRUNCATE = 1500
_NOVELTY_REJECTION_REASON = "embedding similarity above novelty gate"


@dataclass
class PrunerConfig:
    max_tokens: int = 50_000
    keep_full_attempts_back: int = 2  # how many most-recent iterations keep code
    encoding_name: str = "cl100k_base"


def _truncate(text: str, n: int) -> str:
    if not text:
        return text
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"


def _code_snippet(experiment: Optional[Experiment]) -> Optional[str]:
    """Code from the final successful attempt, or ``None`` if there isn't one."""
    if experiment is None or not experiment.succeeded:
        return None
    if not experiment.attempts:
        return None
    last = experiment.attempts[-1]
    return _truncate(last.code, _CODE_TRUNCATE) if last.code else None


def _final_metrics(experiment: Optional[Experiment]) -> Optional[dict]:
    if experiment is None or not experiment.attempts:
        return None
    return dict(experiment.attempts[-1].metrics or {})


class Pruner:
    """Returns slim views over a :class:`DAGTrace` without mutating it."""

    def __init__(self, config: PrunerConfig | None = None):
        self.config = config or PrunerConfig()
        self._enc = tiktoken.get_encoding(self.config.encoding_name)

    # ---- public views ----------------------------------------------------
    def prior_hypotheses_view(self, trace: DAGTrace) -> list[dict]:
        """Slim list of accepted hypotheses for ``prior_hypotheses`` in the prompt.

        Only nodes with a non-None ``feedback`` are surfaced — without
        feedback we have no decision to report, so the entry would be useless.
        """
        all_hyps = trace.all_hypotheses()

        # Iterations of nodes that have feedback (= eligible for inclusion).
        eligible = []
        for hyp in all_hyps:
            node = trace.get(hyp.id)
            if node.feedback is None:
                continue
            eligible.append((hyp, node))

        if not eligible:
            return []

        # Determine the iteration cutoff: only the top-K most recent iterations
        # keep code_snippet. We take the last K *eligible* iterations, not the
        # last K hypothesis ids overall — eligible == "we have something to
        # show," so this is the natural unit.
        keep_n = max(0, self.config.keep_full_attempts_back)
        keep_ids = {hyp.id for hyp, _ in eligible[-keep_n:]} if keep_n > 0 else set()

        view: list[dict] = []
        for hyp, node in eligible:
            keep_code = hyp.id in keep_ids
            view.append(
                {
                    "statement": hyp.statement,
                    "decision": node.feedback.decision,
                    "reason_short": _truncate(node.feedback.reason, _REASON_TRUNCATE),
                    "metrics": _final_metrics(node.experiment),
                    "code_snippet": _code_snippet(node.experiment) if keep_code else None,
                    "re_explore": bool(hyp.re_explore),
                }
            )
        return view

    def rejected_view(self, trace: DAGTrace) -> list[dict]:
        """Slim list of novelty-rejected hypotheses for ``rejected_hypotheses``."""
        return [
            {
                "statement": h.statement,
                "rejection_reason": _NOVELTY_REJECTION_REASON,
            }
            for h in trace.novelty_rejected
        ]

    # ---- token estimation -----------------------------------------------
    def estimate_tokens(self, trace: DAGTrace) -> int:
        """Token count of the projected prompt body (priors + rejections)."""
        priors = self.prior_hypotheses_view(trace)
        rejected = self.rejected_view(trace)
        rendered = self._render_for_count(priors, rejected)
        return len(self._enc.encode(rendered))

    # ---- internals -------------------------------------------------------
    def _render_for_count(
        self, priors: list[dict], rejected: list[dict]
    ) -> str:
        parts: list[str] = []
        for h in priors:
            re_tag = " · re-explored" if h["re_explore"] else ""
            parts.append(
                f"- [{h['decision']}{re_tag}] {h['statement']} — {h['reason_short']}"
            )
            if h["metrics"]:
                parts.append(f"  metrics: {h['metrics']}")
            if h["code_snippet"]:
                parts.append(f"  code:\n{h['code_snippet']}")
        for r in rejected:
            parts.append(f"- (rejected) {r['statement']} — {r['rejection_reason']}")
        return "\n".join(parts)


__all__ = ["PrunerConfig", "Pruner"]
