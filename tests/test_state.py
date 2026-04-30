"""Phase 2 state tests — Hypothesis/Experiment/Feedback dataclasses and DAGTrace."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from hypothesisloop.agent.state import (
    DAGTrace,
    Experiment,
    ExperimentAttempt,
    Hypothesis,
    HypothesisFeedback,
    new_hypothesis_id,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_hypothesis(
    *,
    iteration: int = 1,
    parent_id: str | None = None,
    statement: str = "x correlates with y",
) -> Hypothesis:
    return Hypothesis(
        id=new_hypothesis_id(),
        parent_id=parent_id,
        iteration=iteration,
        statement=statement,
        null="x does not correlate with y",
        test_type="correlation",
        target_columns=["x", "y"],
        expected_outcome="positive correlation, p<0.05",
        concise_reason="initial profiling motivated this",
        concise_observation="both columns are numeric",
        concise_justification="cheap and informative first test",
        concise_knowledge="will tell us if x meaningfully relates to y",
    )


def make_experiment(hypothesis_id: str, *, succeeded: bool = True) -> Experiment:
    attempt = ExperimentAttempt(
        attempt_idx=0,
        code="print('ok')",
        exit_code=0 if succeeded else 1,
        stdout="ok\n" if succeeded else "",
        stderr="" if succeeded else "Traceback...\n",
        figures=[],
        metrics={"p_value": 0.04} if succeeded else {},
        blocked_reason=None,
        duration_s=0.12,
        timed_out=False,
        oom_killed=False,
    )
    return Experiment(hypothesis_id=hypothesis_id, attempts=[attempt], succeeded=succeeded)


def make_feedback(hypothesis_id: str) -> HypothesisFeedback:
    return HypothesisFeedback(
        hypothesis_id=hypothesis_id,
        decision="confirmed",
        reason="p < 0.05, effect size 0.4",
        observations="r=0.4, p=0.04",
        novel_subhypotheses=["does z mediate the relationship?"],
        confidence=0.78,
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_hypothesis_dataclass_roundtrip():
    h = make_hypothesis()
    d = asdict(h)
    h2 = Hypothesis(**d)
    assert h == h2


def test_dag_trace_add_and_get():
    """Topology: root (h0) -> {h1, h2}; h1 -> h3."""
    trace = DAGTrace(session_id="s1", dataset_path="data/adult.csv", question="?")
    h0 = make_hypothesis(iteration=1, parent_id=None)
    trace.add_node(h0)
    h1 = make_hypothesis(iteration=2, parent_id=h0.id)
    trace.add_node(h1)
    h2 = make_hypothesis(iteration=3, parent_id=h0.id)
    trace.add_node(h2)
    h3 = make_hypothesis(iteration=4, parent_id=h1.id)
    trace.add_node(h3)

    assert trace.get(h0.id).hypothesis == h0
    assert trace.get(h3.id).parent_id == h1.id

    leaf_ids = {n.id for n in trace.leaves()}
    assert leaf_ids == {h2.id, h3.id}, "leaves are nodes with no children"

    ancestors_h3 = trace.ancestors(h3.id)
    assert [n.id for n in ancestors_h3] == [h0.id, h1.id]

    assert trace.latest().id == h3.id
    assert trace.iteration_count() == 4
    assert [h.id for h in trace.all_hypotheses()] == [h0.id, h1.id, h2.id, h3.id]


def test_dag_trace_update_experiment_and_feedback():
    trace = DAGTrace(session_id="s2", dataset_path="d.csv", question="?")
    h = make_hypothesis()
    trace.add_node(h)

    exp = make_experiment(h.id)
    trace.update_experiment(h.id, exp)
    assert trace.get(h.id).experiment == exp
    assert trace.get(h.id).experiment.final_attempt.exit_code == 0

    fb = make_feedback(h.id)
    trace.update_feedback(h.id, fb)
    assert trace.get(h.id).feedback == fb
    assert trace.get(h.id).feedback.confidence == 0.78


def test_dag_trace_mark_stale_excludes_from_leaves():
    trace = DAGTrace(session_id="s3", dataset_path="d.csv", question="?")
    h0 = make_hypothesis(iteration=1, parent_id=None)
    trace.add_node(h0)
    h1 = make_hypothesis(iteration=2, parent_id=h0.id)
    trace.add_node(h1)
    h2 = make_hypothesis(iteration=3, parent_id=h0.id)
    trace.add_node(h2)

    # Before mark: leaves are h1 and h2 (h0 has children).
    assert {n.id for n in trace.leaves()} == {h1.id, h2.id}

    trace.mark_stale(h1.id)
    assert trace.get(h1.id).stale is True
    assert {n.id for n in trace.leaves()} == {h2.id}, "stale node must be excluded from leaves()"


def test_dag_trace_save_load_roundtrip(tmp_path: Path):
    trace = DAGTrace(
        session_id="s4",
        dataset_path="data/adult.csv",
        question="does education affect income?",
        schema_summary="14 columns, 32k rows",
    )
    h0 = make_hypothesis(iteration=1, parent_id=None, statement="education > income")
    trace.add_node(h0)
    trace.update_experiment(h0.id, make_experiment(h0.id))
    trace.update_feedback(h0.id, make_feedback(h0.id))

    h1 = make_hypothesis(iteration=2, parent_id=h0.id, statement="hours > income")
    trace.add_node(h1)
    # h1 stays without experiment/feedback to exercise the Optional-None path.

    h2 = make_hypothesis(iteration=3, parent_id=h0.id, statement="age > income")
    trace.add_node(h2)
    trace.mark_stale(h2.id)

    out = tmp_path / "trace.json"
    trace.save(out)
    loaded = DAGTrace.load(out)

    assert loaded.session_id == trace.session_id
    assert loaded.dataset_path == trace.dataset_path
    assert loaded.question == trace.question
    assert loaded.schema_summary == trace.schema_summary
    assert loaded.created_at == trace.created_at
    assert [h.id for h in loaded.all_hypotheses()] == [h0.id, h1.id, h2.id]

    # Deep equality across nodes (dataclass __eq__ recurses into Hypothesis/Experiment/Feedback).
    for nid in (h0.id, h1.id, h2.id):
        assert loaded.get(nid) == trace.get(nid), f"node {nid} did not round-trip equal"

    # Children/leaves topology survives.
    assert {n.id for n in loaded.leaves()} == {n.id for n in trace.leaves()}
    assert loaded.get(h2.id).stale is True
    assert loaded.get(h1.id).experiment is None
    assert loaded.get(h0.id).experiment is not None


def test_add_duplicate_id_raises():
    trace = DAGTrace(session_id="s5", dataset_path="d.csv", question="?")
    h = make_hypothesis()
    trace.add_node(h)
    h_dup = make_hypothesis()
    h_dup.id = h.id  # force collision
    with pytest.raises(ValueError):
        trace.add_node(h_dup)


def test_update_unknown_node_raises():
    trace = DAGTrace(session_id="s6", dataset_path="d.csv", question="?")
    h = make_hypothesis()
    with pytest.raises(KeyError):
        trace.update_experiment("does-not-exist", make_experiment(h.id))
    with pytest.raises(KeyError):
        trace.update_feedback("does-not-exist", make_feedback(h.id))
    with pytest.raises(KeyError):
        trace.mark_stale("does-not-exist")
    with pytest.raises(KeyError):
        trace.get("does-not-exist")
