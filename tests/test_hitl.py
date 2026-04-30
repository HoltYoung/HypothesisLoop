"""Phase 6 HITL prompt + summary tests."""

from __future__ import annotations

import io

from hypothesisloop.agent.state import (
    DAGTrace,
    Hypothesis,
    HypothesisFeedback,
    new_hypothesis_id,
)
from hypothesisloop.ui.hitl import hitl_prompt, print_run_summary


def _make_node(trace: DAGTrace, statement: str, *, decision: str = "confirmed", re_explore: bool = False):
    h = Hypothesis(
        id=new_hypothesis_id(),
        parent_id=None,
        iteration=trace.iteration_count() + 1,
        statement=statement,
        null="not " + statement,
        test_type="correlation",
        target_columns=["a"],
        expected_outcome="e",
        concise_reason="r",
        concise_observation="o",
        concise_justification="j",
        concise_knowledge="k",
        re_explore=re_explore,
    )
    node = trace.add_node(h)
    fb = HypothesisFeedback(
        hypothesis_id=h.id,
        decision=decision,
        reason="r" * 200,
        observations="o",
        novel_subhypotheses=[],
        confidence=0.7,
    )
    trace.update_feedback(h.id, fb)
    return node


def _trace() -> DAGTrace:
    return DAGTrace(session_id="t1", dataset_path="data/adult.csv", question="Q?")


# ---------------------------------------------------------------------------
# hitl_prompt
# ---------------------------------------------------------------------------
def test_hitl_continue_default():
    trace = _trace()
    node = _make_node(trace, "stmt 1")
    out = io.StringIO()
    result = hitl_prompt(node, stream_in=io.StringIO("\n"), stream_out=out)
    assert result == {"action": "continue"}


def test_hitl_continue_explicit_c():
    trace = _trace()
    node = _make_node(trace, "stmt 1")
    out = io.StringIO()
    result = hitl_prompt(node, stream_in=io.StringIO("c\n"), stream_out=out)
    assert result == {"action": "continue"}


def test_hitl_stop():
    trace = _trace()
    node = _make_node(trace, "stmt 1")
    out = io.StringIO()
    result = hitl_prompt(node, stream_in=io.StringIO("s\n"), stream_out=out)
    assert result == {"action": "stop"}


def test_hitl_redirect():
    trace = _trace()
    node = _make_node(trace, "stmt 1")
    out = io.StringIO()
    result = hitl_prompt(
        node,
        stream_in=io.StringIO("r test new direction\n"),
        stream_out=out,
    )
    assert result == {"action": "redirect", "hypothesis": "test new direction"}


def test_hitl_invalid_then_valid():
    trace = _trace()
    node = _make_node(trace, "stmt 1")
    out = io.StringIO()
    result = hitl_prompt(
        node,
        stream_in=io.StringIO("garbage\nq\n\n"),  # 2 invalid lines, then empty -> continue
        stream_out=out,
    )
    assert result == {"action": "continue"}
    rendered = out.getvalue()
    assert "invalid input 'garbage'" in rendered
    assert "invalid input 'q'" in rendered


def test_hitl_eof_treated_as_stop():
    """Closed stdin (EOF) should not loop — treat as ``stop``."""
    trace = _trace()
    node = _make_node(trace, "stmt 1")
    out = io.StringIO()
    result = hitl_prompt(node, stream_in=io.StringIO(""), stream_out=out)
    assert result == {"action": "stop"}


# ---------------------------------------------------------------------------
# print_run_summary
# ---------------------------------------------------------------------------
def test_print_run_summary_renders():
    trace = _trace()
    _make_node(trace, "iter 1 statement")
    _make_node(trace, "iter 2 statement", decision="rejected")

    out = io.StringIO()
    print_run_summary(trace, stream_out=out)
    rendered = out.getvalue()

    assert "HypothesisLoop run: t1" in rendered
    assert "iterations   : 2" in rendered
    assert "iter  1" in rendered and "iter  2" in rendered
    assert "iter 1 statement" in rendered
    assert "iter 2 statement" in rendered
    assert "[confirmed" in rendered and "[rejected" in rendered


def test_print_run_summary_with_novelty_rejections():
    trace = _trace()
    _make_node(trace, "accepted one")
    rejected = Hypothesis(
        id=new_hypothesis_id(),
        parent_id=None,
        iteration=2,
        statement="rejected duplicate",
        null="x",
        test_type="correlation",
        target_columns=["a"],
        expected_outcome="e",
        concise_reason="r",
        concise_observation="o",
        concise_justification="j",
        concise_knowledge="k",
    )
    trace.add_novelty_rejection(rejected)

    out = io.StringIO()
    print_run_summary(trace, stream_out=out)
    rendered = out.getvalue()

    assert "novelty rej. : 1" in rendered
    assert "rejected as duplicates (1)" in rendered
    assert "rejected duplicate" in rendered


def test_print_run_summary_with_re_explore_flag():
    trace = _trace()
    _make_node(trace, "first try")
    _make_node(trace, "after soft-decay", re_explore=True)

    out = io.StringIO()
    print_run_summary(trace, stream_out=out)
    rendered = out.getvalue()

    assert "re-explored" in rendered
