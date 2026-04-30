"""Phase 5 pruner unit tests."""

from __future__ import annotations

from copy import deepcopy

import pytest

from hypothesisloop.agent.pruner import Pruner, PrunerConfig
from hypothesisloop.agent.state import (
    DAGTrace,
    Experiment,
    ExperimentAttempt,
    Hypothesis,
    HypothesisFeedback,
    new_hypothesis_id,
)


def _hyp(iteration: int, statement: str, *, parent_id: str | None = None, re_explore: bool = False) -> Hypothesis:
    return Hypothesis(
        id=new_hypothesis_id(),
        parent_id=parent_id,
        iteration=iteration,
        statement=statement,
        null="not " + statement,
        test_type="custom",
        target_columns=["x"],
        expected_outcome="?",
        concise_reason="r",
        concise_observation="o",
        concise_justification="j",
        concise_knowledge="k",
        re_explore=re_explore,
    )


def _experiment(hyp_id: str, *, code: str, metrics: dict, succeeded: bool = True) -> Experiment:
    attempt = ExperimentAttempt(
        attempt_idx=0,
        code=code,
        exit_code=0 if succeeded else 1,
        stdout="x" * 600,
        stderr="" if succeeded else "boom",
        figures=[],
        metrics=metrics,
        blocked_reason=None,
        duration_s=0.5,
        timed_out=False,
        oom_killed=False,
    )
    return Experiment(hypothesis_id=hyp_id, attempts=[attempt], succeeded=succeeded)


def _feedback(hyp_id: str, *, decision: str = "confirmed", reason: str | None = None) -> HypothesisFeedback:
    return HypothesisFeedback(
        hypothesis_id=hyp_id,
        decision=decision,
        reason=reason or "p<0.05; effect size moderate; this needs follow-up next round",
        observations="r=0.4, p=0.04",
        novel_subhypotheses=[],
        confidence=0.7,
    )


def _build_5_iter_trace() -> DAGTrace:
    trace = DAGTrace(session_id="prune-5", dataset_path="d.csv", question="Q?")
    parent_id = None
    for i in range(1, 6):
        h = _hyp(i, f"hypothesis #{i}", parent_id=parent_id)
        trace.add_node(h)
        trace.update_experiment(
            h.id,
            _experiment(
                h.id,
                code=("# code line\n" * 30) + f"# attempt {i}\n",
                metrics={"p_value": 0.001 * i, "effect_size": 0.1 * i, "n": 1000},
            ),
        )
        trace.update_feedback(h.id, _feedback(h.id))
        parent_id = h.id
    return trace


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_pruner_view_shape():
    trace = DAGTrace(session_id="s1", dataset_path="d.csv", question="?")
    h0 = _hyp(1, "h0")
    trace.add_node(h0)
    trace.update_experiment(h0.id, _experiment(h0.id, code="print(1)\n", metrics={"x": 1}))
    trace.update_feedback(h0.id, _feedback(h0.id))

    h1 = _hyp(2, "h1", parent_id=h0.id)
    trace.add_node(h1)  # no feedback — should NOT appear in view

    h2 = _hyp(3, "h2", parent_id=h0.id, re_explore=True)
    trace.add_node(h2)
    trace.update_experiment(h2.id, _experiment(h2.id, code="print(2)\n", metrics={"y": 2}))
    trace.update_feedback(h2.id, _feedback(h2.id, decision="rejected"))

    view = Pruner().prior_hypotheses_view(trace)
    assert len(view) == 2  # h1 lacks feedback, excluded
    statements = [v["statement"] for v in view]
    assert statements == ["h0", "h2"]
    for entry in view:
        assert set(entry.keys()) == {
            "statement", "decision", "reason_short", "metrics",
            "code_snippet", "re_explore",
        }
        assert isinstance(entry["statement"], str)
        assert isinstance(entry["decision"], str)
        assert isinstance(entry["reason_short"], str)
        assert isinstance(entry["metrics"], dict)
        assert isinstance(entry["re_explore"], bool)
    assert view[1]["re_explore"] is True
    assert view[0]["re_explore"] is False


def test_pruner_drops_old_code_snippets():
    trace = _build_5_iter_trace()
    pruner = Pruner(PrunerConfig(keep_full_attempts_back=2))
    view = pruner.prior_hypotheses_view(trace)

    assert len(view) == 5
    # Iterations 1, 2, 3 → no code; iterations 4, 5 → keep code.
    code_kept = [v["code_snippet"] is not None for v in view]
    assert code_kept == [False, False, False, True, True]


def test_pruner_estimate_tokens_under_target():
    trace = _build_5_iter_trace()
    pruner = Pruner()
    n_tokens = pruner.estimate_tokens(trace)
    assert n_tokens > 0
    assert n_tokens < 50_000


def test_pruner_rejected_view():
    trace = DAGTrace(session_id="rej", dataset_path="d.csv", question="?")
    r1 = _hyp(1, "rejected one")
    r2 = _hyp(2, "rejected two")
    trace.add_novelty_rejection(r1)
    trace.add_novelty_rejection(r2)

    view = Pruner().rejected_view(trace)
    assert len(view) == 2
    for entry in view:
        assert set(entry.keys()) == {"statement", "rejection_reason"}
        assert isinstance(entry["rejection_reason"], str) and entry["rejection_reason"]
    assert [e["statement"] for e in view] == ["rejected one", "rejected two"]


def test_pruner_does_not_mutate_trace():
    trace = _build_5_iter_trace()
    trace.add_novelty_rejection(_hyp(99, "rejected ghost"))
    snapshot = deepcopy(trace.to_dict())

    pruner = Pruner()
    _ = pruner.prior_hypotheses_view(trace)
    _ = pruner.rejected_view(trace)
    _ = pruner.estimate_tokens(trace)

    assert trace.to_dict() == snapshot, "pruner must not mutate the trace"
