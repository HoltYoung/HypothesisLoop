"""Phase 4 acceptance — retry mechanic verified through run_loop.

No live LLM. Hypothesize and evaluate are mocked; the experiment step uses
``_StubLLM`` returning broken code first, fixed code second. Demonstrates that
the run_loop integration honors the retry contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from hypothesisloop.agent.loop import run_loop
from hypothesisloop.agent.scheduler import LinearScheduler
from hypothesisloop.agent.state import (
    DAGTrace,
    Experiment,
    Hypothesis,
    HypothesisFeedback,
    TraceNode,
    new_hypothesis_id,
)
from hypothesisloop.steps.experiment import ExperimentStep


class _StubLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    def invoke(self, _prompt):
        if not self._responses:
            raise AssertionError("StubLLM out of responses")

        class _Resp:
            content = self._responses.pop(0)

        return _Resp()


_DATASET = Path("data/adult.csv").resolve()


def test_phase4_retry_triggers_with_planted_error(tmp_path: Path):
    trace = DAGTrace(
        session_id="phase4-retry",
        dataset_path=str(_DATASET),
        question="Q?",
    )

    fixed_hyp_id = new_hypothesis_id()

    def hypothesize_fn(_t: DAGTrace, parent: Optional[TraceNode]) -> Hypothesis:
        return Hypothesis(
            id=fixed_hyp_id,
            parent_id=(parent.id if parent is not None else None),
            iteration=1,
            statement="age has a non-trivial mean",
            null="age has zero mean",
            test_type="custom",
            target_columns=["age"],
            expected_outcome="mean > 0",
            concise_reason="r",
            concise_observation="o",
            concise_justification="j",
            concise_knowledge="k",
        )

    broken = "x = df['nonexistent_column'].mean()\nprint(x)\n"
    fixed = (
        "x = df['age'].mean()\n"
        "print(f'mean_age={x:.4f}')\n"
        "hl_emit('mean_age', float(x))\n"
        "hl_emit('n', int(len(df)))\n"
    )
    experiment_step = ExperimentStep(
        llm=_StubLLM([broken, fixed]),
        session_root=tmp_path / "session",
        dataset_path=_DATASET,
        schema_summary="age: numeric",
        max_retries=3,
        timeout_s=30,
        ram_mb=1024,
        seed=42,
    )

    def evaluate_fn(h: Hypothesis, _exp: Experiment) -> HypothesisFeedback:
        return HypothesisFeedback(
            hypothesis_id=h.id,
            decision="confirmed",
            reason="canned",
            observations="canned",
            novel_subhypotheses=[],
            confidence=0.9,
        )

    out = run_loop(
        trace=trace,
        scheduler=LinearScheduler(),
        hypothesize_fn=hypothesize_fn,
        experiment_fn=experiment_step,
        evaluate_fn=evaluate_fn,
        max_iters=1,
    )

    hyps = out.all_hypotheses()
    assert len(hyps) == 1
    node = out.get(hyps[0].id)
    assert node.experiment is not None
    assert len(node.experiment.attempts) == 2
    assert node.experiment.attempts[0].exit_code != 0
    assert (
        "nonexistent_column" in node.experiment.attempts[0].stderr
        or "KeyError" in node.experiment.attempts[0].stderr
    )
    assert node.experiment.attempts[1].exit_code == 0
    assert node.experiment.succeeded is True
    # The retry attempt's metrics survived end-to-end into the trace.
    assert node.experiment.attempts[1].metrics.get("mean_age") is not None
