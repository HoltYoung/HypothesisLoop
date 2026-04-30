"""Phase 2 loop tests — pure orchestration, mocked step fns."""

from __future__ import annotations

from typing import Optional

import pytest

from hypothesisloop.agent.loop import run_loop
from hypothesisloop.agent.scheduler import LinearScheduler
from hypothesisloop.agent.state import (
    DAGTrace,
    Experiment,
    ExperimentAttempt,
    Hypothesis,
    HypothesisFeedback,
    TraceNode,
    new_hypothesis_id,
)


# ---------------------------------------------------------------------------
# mock step fn factory
# ---------------------------------------------------------------------------
def _make_attempt(idx: int = 0) -> ExperimentAttempt:
    return ExperimentAttempt(
        attempt_idx=idx,
        code="print('mock')",
        exit_code=0,
        stdout="mock\n",
        stderr="",
        figures=[],
        metrics={},
        blocked_reason=None,
        duration_s=0.01,
        timed_out=False,
        oom_killed=False,
    )


def make_mock_steps(scheduler: LinearScheduler):
    """Returns (hypothesize_fn, experiment_fn, evaluate_fn).

    hypothesize_fn auto-increments iteration, links to parent if any, and
    prefers an injected redirect text when present.
    """
    counter = {"iter": 0}

    def hypothesize_fn(trace: DAGTrace, parent: Optional[TraceNode]) -> Hypothesis:
        counter["iter"] += 1
        injected = scheduler.consume_injection()
        statement = injected if injected is not None else f"mock hypothesis #{counter['iter']}"
        return Hypothesis(
            id=new_hypothesis_id(),
            parent_id=(parent.id if parent is not None else None),
            iteration=counter["iter"],
            statement=statement,
            null=f"not: {statement}",
            test_type="correlation",
            target_columns=["a", "b"],
            expected_outcome="effect",
            concise_reason="mock",
            concise_observation="mock",
            concise_justification="mock",
            concise_knowledge="mock",
        )

    def experiment_fn(hyp: Hypothesis) -> Experiment:
        return Experiment(hypothesis_id=hyp.id, attempts=[_make_attempt()], succeeded=True)

    def evaluate_fn(hyp: Hypothesis, exp: Experiment) -> HypothesisFeedback:
        return HypothesisFeedback(
            hypothesis_id=hyp.id,
            decision="confirmed",
            reason="mocked confirmation",
            observations="mocked observation",
            novel_subhypotheses=[],
            confidence=0.9,
        )

    return hypothesize_fn, experiment_fn, evaluate_fn


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_dry_run_5_iters():
    trace = DAGTrace(session_id="s1", dataset_path="d.csv", question="?")
    sched = LinearScheduler()
    hf, ef, vf = make_mock_steps(sched)

    out = run_loop(
        trace=trace,
        scheduler=sched,
        hypothesize_fn=hf,
        experiment_fn=ef,
        evaluate_fn=vf,
        max_iters=5,
    )

    assert out is trace
    hyps = out.all_hypotheses()
    assert len(hyps) == 5
    assert [h.iteration for h in hyps] == [1, 2, 3, 4, 5]
    for h in hyps:
        node = out.get(h.id)
        assert node.experiment is not None and node.experiment.succeeded
        assert node.feedback is not None and node.feedback.decision == "confirmed"
    # Linear scheduler links each new node to the previous one as parent.
    assert hyps[0].parent_id is None
    assert all(hyps[i].parent_id == hyps[i - 1].id for i in range(1, 5))


def test_dry_run_with_novelty_rejection():
    trace = DAGTrace(session_id="s2", dataset_path="d.csv", question="?")
    sched = LinearScheduler()
    hf, ef, vf = make_mock_steps(sched)

    seen = {"count": 0}

    def novelty_fn(h: Hypothesis, t: DAGTrace) -> bool:
        seen["count"] += 1
        return seen["count"] != 2  # reject the 2nd hypothesis only

    out = run_loop(
        trace=trace,
        scheduler=sched,
        hypothesize_fn=hf,
        experiment_fn=ef,
        evaluate_fn=vf,
        novelty_fn=novelty_fn,
        max_iters=5,
    )

    # 5 max_iters, 1 rejected, 4 added.
    assert len(out.all_hypotheses()) == 4


def test_dry_run_hitl_stop():
    trace = DAGTrace(session_id="s3", dataset_path="d.csv", question="?")
    sched = LinearScheduler()
    hf, ef, vf = make_mock_steps(sched)

    seen = {"count": 0}

    def hitl_fn(node: TraceNode) -> dict:
        seen["count"] += 1
        if seen["count"] >= 2:
            return {"action": "stop"}
        return {"action": "continue"}

    out = run_loop(
        trace=trace,
        scheduler=sched,
        hypothesize_fn=hf,
        experiment_fn=ef,
        evaluate_fn=vf,
        hitl_fn=hitl_fn,
        max_iters=5,
    )

    assert len(out.all_hypotheses()) == 2


def test_dry_run_hitl_redirect():
    trace = DAGTrace(session_id="s4", dataset_path="d.csv", question="?")
    sched = LinearScheduler()
    hf, ef, vf = make_mock_steps(sched)

    seen = {"count": 0}
    redirect_text = "test new direction"

    def hitl_fn(node: TraceNode) -> dict:
        seen["count"] += 1
        if seen["count"] == 1:
            return {"action": "redirect", "hypothesis": redirect_text}
        return {"action": "continue"}

    out = run_loop(
        trace=trace,
        scheduler=sched,
        hypothesize_fn=hf,
        experiment_fn=ef,
        evaluate_fn=vf,
        hitl_fn=hitl_fn,
        max_iters=3,
    )

    hyps = out.all_hypotheses()
    assert len(hyps) == 3
    # Iter 1 happens BEFORE hitl fires; iter 2's hypothesize_fn pulls the
    # injection. So statement on iter 2 should equal the redirect text.
    assert hyps[1].statement == redirect_text
    # Injection was consumed exactly once.
    assert sched.consume_injection() is None


def test_loop_returns_trace_with_session_metadata():
    trace = DAGTrace(
        session_id="hl-session-XYZ",
        dataset_path="data/adult.csv",
        question="does education affect income?",
        schema_summary="32k rows, 14 cols",
    )
    sched = LinearScheduler()
    hf, ef, vf = make_mock_steps(sched)

    out = run_loop(
        trace=trace,
        scheduler=sched,
        hypothesize_fn=hf,
        experiment_fn=ef,
        evaluate_fn=vf,
        max_iters=5,
    )

    assert out.session_id == "hl-session-XYZ"
    assert out.dataset_path == "data/adult.csv"
    assert out.question == "does education affect income?"
    assert out.schema_summary == "32k rows, 14 cols"
    assert len(out.all_hypotheses()) == 5
