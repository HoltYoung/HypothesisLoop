"""Novelty detection — three-layer gate per SPEC §6.5 and DESIGN_DECISIONS row 3.

Layer 1 — prompt injection of prior hypotheses — happens in
``hypothesize.j2`` and is not this module's job.

Layer 2 — embedding cosine gate at ``base_threshold`` (default 0.85) — and
Layer 3 — soft-decay loosening to ``decayed_threshold`` (default 0.92)
after ``consecutive_rejections_for_decay`` consecutive rejections — both
live here. The accepted hypothesis that escapes the decay regime is tagged
``re_explore=True``; the counter resets on any acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from hypothesisloop.agent.state import DAGTrace, Hypothesis
from hypothesisloop.llm.embed import cosine_similarity


@dataclass
class NoveltyConfig:
    base_threshold: float = 0.85
    decayed_threshold: float = 0.92
    consecutive_rejections_for_decay: int = 3


class NoveltyChecker:
    """Stateful novelty filter. Compatible with the loop's ``NoveltyFn`` shape."""

    def __init__(
        self,
        *,
        embed_fn: Callable[[str], list[float]],
        config: NoveltyConfig | None = None,
    ):
        self.embed_fn = embed_fn
        self.config = config or NoveltyConfig()
        self._consecutive_rejections: int = 0

    @property
    def consecutive_rejections(self) -> int:
        """Read-only view for tests / observability."""
        return self._consecutive_rejections

    def __call__(self, hypothesis: Hypothesis, trace: DAGTrace) -> bool:
        # 1. Always embed — both branches benefit (priors / future comparisons).
        if not hypothesis.embedding:
            hypothesis.embedding = list(self.embed_fn(hypothesis.statement))

        # 2. Pool of priors = accepted (in DAG) ∪ previously novelty-rejected.
        priors: list[Hypothesis] = list(trace.all_hypotheses()) + list(trace.novelty_rejected)

        # 3. First hypothesis ever — accept unconditionally.
        if not priors:
            self._consecutive_rejections = 0
            return True

        # 4. Compute max cosine similarity vs priors that have an embedding.
        max_sim = self._max_similarity(hypothesis, priors)

        # 5. Pick the threshold: decayed iff we've hit the consecutive-rejection bar.
        in_decay = (
            self._consecutive_rejections >= self.config.consecutive_rejections_for_decay
        )
        threshold = (
            self.config.decayed_threshold if in_decay else self.config.base_threshold
        )

        # Sam's audit fix #11: hard escalation. If we've rejected 5+ in a row
        # the loop is starving — better to accept a near-duplicate (with
        # re_explore=True) than to burn the whole iteration budget on
        # rejections. The agent will still see prior_hypotheses in the
        # rendered prompt and try to differentiate.
        if self._consecutive_rejections >= 5:
            hypothesis.re_explore = True
            self._consecutive_rejections = 0
            return True

        # 6. Apply gate. Strict ``<`` — at exactly threshold, reject.
        if max_sim < threshold:
            if in_decay:
                hypothesis.re_explore = True
            self._consecutive_rejections = 0
            return True
        else:
            self._consecutive_rejections += 1
            return False

    def _max_similarity(
        self, hypothesis: Hypothesis, priors: list[Hypothesis]
    ) -> float:
        best = -1.0
        for prior in priors:
            if not prior.embedding:
                continue  # older trace files may lack embeddings — skip cleanly
            sim = cosine_similarity(hypothesis.embedding, prior.embedding)
            if sim > best:
                best = sim
        return best


__all__ = ["NoveltyConfig", "NoveltyChecker"]
