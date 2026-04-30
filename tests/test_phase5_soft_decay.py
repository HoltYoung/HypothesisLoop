"""Phase 5 acceptance — soft-decay regime triggers re_explore tagging.

After 3 consecutive rejections, the gate loosens from 0.85 to 0.92. The first
hypothesis that escapes the loosened gate is tagged ``re_explore=True``, and
the counter resets — so the *next* acceptance under the base threshold is
no longer flagged.
"""

from __future__ import annotations

import math

from hypothesisloop.agent.novelty import NoveltyChecker, NoveltyConfig
from hypothesisloop.agent.state import (
    DAGTrace,
    Hypothesis,
    HypothesisFeedback,
    new_hypothesis_id,
)


def _vector_at_cosine(c: float) -> list[float]:
    """Unit vector with cosine ``c`` against ``[1, 0, 0]``."""
    s = math.sqrt(max(0.0, 1.0 - c * c))
    return [c, s, 0.0]


class _StubEmbed:
    def __init__(self):
        self.next_vector: list[float] = [1.0, 0.0, 0.0]

    def __call__(self, _text: str) -> list[float]:
        return list(self.next_vector)


def _hyp(statement: str) -> Hypothesis:
    return Hypothesis(
        id=new_hypothesis_id(),
        parent_id=None,
        iteration=1,
        statement=statement,
        null="not " + statement,
        test_type="custom",
        target_columns=["x"],
        expected_outcome="?",
        concise_reason="r",
        concise_observation="o",
        concise_justification="j",
        concise_knowledge="k",
    )


def test_soft_decay_triggers_re_explore_after_3_rejections():
    trace = DAGTrace(session_id="soft-decay", dataset_path="d.csv", question="?")

    # Seed the trace with one accepted hypothesis whose embedding is the basis
    # axis. All cosine math is then `next_vector @ [1,0,0]` = `next_vector[0]`.
    seed = _hyp("seed accepted hypothesis")
    seed.embedding = [1.0, 0.0, 0.0]
    trace.add_node(seed)
    trace.update_feedback(
        seed.id,
        HypothesisFeedback(
            hypothesis_id=seed.id,
            decision="confirmed",
            reason="canned",
            observations="canned",
            novel_subhypotheses=[],
            confidence=0.9,
        ),
    )

    stub = _StubEmbed()
    checker = NoveltyChecker(
        embed_fn=stub,
        config=NoveltyConfig(
            base_threshold=0.85,
            decayed_threshold=0.92,
            consecutive_rejections_for_decay=3,
        ),
    )

    # 3 rejections at cosine ~0.99 (well above both thresholds).
    # We deliberately do NOT add these to trace.novelty_rejected here — the
    # checker's internal counter is what drives soft-decay, and including
    # near-duplicate vectors in the priors pool would muddy the cosine math
    # for the next vector. The "loop appends rejected → priors pool grows"
    # interaction is covered separately by test_loop tests.
    stub.next_vector = _vector_at_cosine(0.99)
    for i in range(3):
        h = _hyp(f"near-duplicate #{i}")
        h.embedding = []
        assert checker(h, trace) is False, f"iteration {i} should reject"
    assert checker.consecutive_rejections == 3

    # Decay regime is now active. A hypothesis at cosine 0.86 — above base
    # (0.85) but below decayed (0.92) — should be ACCEPTED with re_explore=True.
    stub.next_vector = _vector_at_cosine(0.86)
    revisit = _hyp("re-explored angle on the seed area")
    revisit.embedding = []
    assert checker(revisit, trace) is True
    assert revisit.re_explore is True
    assert checker.consecutive_rejections == 0, "counter must reset on acceptance"
    trace.add_node(revisit)
    trace.update_feedback(
        revisit.id,
        HypothesisFeedback(
            hypothesis_id=revisit.id,
            decision="rejected",
            reason="follow-up didn't pan out",
            observations="...",
            novel_subhypotheses=[],
            confidence=0.4,
        ),
    )

    # A new, well-distinct hypothesis under the BASE regime — should accept
    # without the re_explore tag.
    stub.next_vector = _vector_at_cosine(0.20)
    distinct = _hyp("entirely different region")
    distinct.embedding = []
    assert checker(distinct, trace) is True
    assert distinct.re_explore is False
    assert checker.consecutive_rejections == 0
