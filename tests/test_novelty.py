"""Phase 5 novelty-checker unit tests (no network — stub embed_fn)."""

from __future__ import annotations

import math
from typing import Optional

import pytest

from hypothesisloop.agent.novelty import NoveltyChecker, NoveltyConfig
from hypothesisloop.agent.state import (
    DAGTrace,
    Hypothesis,
    HypothesisFeedback,
    new_hypothesis_id,
)


# ---------------------------------------------------------------------------
# stubs / helpers
# ---------------------------------------------------------------------------
class StubEmbed:
    """Returns whatever vector the test set last. Always returns a fresh list."""

    def __init__(self, default: Optional[list[float]] = None):
        self.next_vector: list[float] = list(default or [1.0, 0.0, 0.0])
        self.calls: int = 0

    def __call__(self, _text: str) -> list[float]:
        self.calls += 1
        return list(self.next_vector)


def _make_hypothesis(statement: str = "an example claim") -> Hypothesis:
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


def _seed_accepted(trace: DAGTrace, *, embedding: list[float], statement: str = "seed") -> Hypothesis:
    """Add an accepted node to the trace so it appears in priors."""
    h = _make_hypothesis(statement)
    h.embedding = list(embedding)
    trace.add_node(h)
    trace.update_feedback(
        h.id,
        HypothesisFeedback(
            hypothesis_id=h.id,
            decision="confirmed",
            reason="canned",
            observations="canned",
            novel_subhypotheses=[],
            confidence=0.9,
        ),
    )
    return h


def _vector_at_cosine(c: float) -> list[float]:
    """Return a unit vector with cosine ``c`` against ``[1, 0, 0]``."""
    s = math.sqrt(max(0.0, 1.0 - c * c))
    return [c, s, 0.0]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_novelty_first_hypothesis_always_accepted():
    trace = DAGTrace(session_id="s1", dataset_path="d.csv", question="?")
    stub = StubEmbed(_vector_at_cosine(0.99))
    checker = NoveltyChecker(embed_fn=stub)

    h = _make_hypothesis("first ever")
    assert checker(h, trace) is True
    assert checker.consecutive_rejections == 0


def test_novelty_rejects_duplicate():
    trace = DAGTrace(session_id="s2", dataset_path="d.csv", question="?")
    _seed_accepted(trace, embedding=[1.0, 0.0, 0.0], statement="seed")

    stub = StubEmbed([1.0, 0.0, 0.0])  # cosine=1.0 with prior
    checker = NoveltyChecker(embed_fn=stub)

    h = _make_hypothesis("a near-duplicate of seed")
    assert checker(h, trace) is False
    assert checker.consecutive_rejections == 1


def test_novelty_accepts_distinct():
    trace = DAGTrace(session_id="s3", dataset_path="d.csv", question="?")
    _seed_accepted(trace, embedding=[1.0, 0.0, 0.0])

    stub = StubEmbed(_vector_at_cosine(0.5))  # distinct
    checker = NoveltyChecker(embed_fn=stub)

    h = _make_hypothesis("a distinct claim")
    assert checker(h, trace) is True
    assert checker.consecutive_rejections == 0


def test_novelty_at_threshold_boundary():
    """The gate is strict ``<`` — at exactly ``base_threshold``, reject."""
    trace = DAGTrace(session_id="s4", dataset_path="d.csv", question="?")
    _seed_accepted(trace, embedding=[1.0, 0.0, 0.0])

    stub = StubEmbed(_vector_at_cosine(0.85))
    checker = NoveltyChecker(embed_fn=stub, config=NoveltyConfig(base_threshold=0.85))

    assert checker(_make_hypothesis("at-threshold"), trace) is False


def test_novelty_populates_embedding():
    trace = DAGTrace(session_id="s5", dataset_path="d.csv", question="?")
    stub = StubEmbed(_vector_at_cosine(0.10))
    checker = NoveltyChecker(embed_fn=stub)

    h = _make_hypothesis("with empty embedding")
    assert h.embedding == []
    checker(h, trace)
    assert h.embedding == _vector_at_cosine(0.10)


def test_novelty_consecutive_rejections_increment_counter():
    trace = DAGTrace(session_id="s6", dataset_path="d.csv", question="?")
    _seed_accepted(trace, embedding=[1.0, 0.0, 0.0])

    stub = StubEmbed(_vector_at_cosine(0.99))
    checker = NoveltyChecker(embed_fn=stub)

    for _ in range(3):
        assert checker(_make_hypothesis(), trace) is False
    assert checker.consecutive_rejections == 3
