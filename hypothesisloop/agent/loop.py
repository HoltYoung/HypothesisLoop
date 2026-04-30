"""Loop orchestration — pure plumbing, no LLM/sandbox/Langfuse coupling.

Phase 7 refactor: the per-iteration body lives in ``_execute_iteration``, which
is wrapped in ``@observe(name=LOOP_ITERATION)`` so each iteration becomes its
own Langfuse span. The public ``run_loop`` signature stays backward-compatible
(only the new ``safety_fn`` keyword is added, with a default of ``None``).
"""

from __future__ import annotations

from typing import Callable, Optional

from hypothesisloop.agent.scheduler import Scheduler
from hypothesisloop.agent.state import (
    DAGTrace,
    Experiment,
    Hypothesis,
    HypothesisFeedback,
    TraceNode,
)
from hypothesisloop.trace.langfuse_client import LOOP_ITERATION, observe


HypothesizeFn = Callable[[DAGTrace, Optional[TraceNode]], Hypothesis]
ExperimentFn = Callable[[Hypothesis], Experiment]
EvaluateFn = Callable[[Hypothesis, Experiment], HypothesisFeedback]
LearnFn = Callable[[TraceNode], None]
NoveltyFn = Callable[[Hypothesis, DAGTrace], bool]
HITLFn = Callable[[TraceNode], dict]
# Loosely typed to avoid a circular import on hypothesisloop.safety.bias_scanner.
SafetyFn = Callable[[TraceNode], list]


@observe(name=LOOP_ITERATION)
def _execute_iteration(
    iter_idx: int,
    *,
    trace: DAGTrace,
    scheduler: Scheduler,
    hypothesize_fn: HypothesizeFn,
    experiment_fn: ExperimentFn,
    evaluate_fn: EvaluateFn,
    learn_fn: Optional[LearnFn],
    novelty_fn: Optional[NoveltyFn],
    hitl_fn: Optional[HITLFn],
    safety_fn: Optional[SafetyFn],
) -> Optional[dict]:
    """One iteration. Returns ``{"action": "stop"}`` to break the outer loop, else ``None``.

    Novelty rejection still consumes an iteration slot (returns ``None``); the
    rejected hypothesis is appended to ``trace.novelty_rejected``.
    """
    parent = scheduler.next_parent(trace)
    hyp = hypothesize_fn(trace, parent)

    if novelty_fn is not None and not novelty_fn(hyp, trace):
        trace.add_novelty_rejection(hyp)
        return None

    node = trace.add_node(hyp)
    exp = experiment_fn(hyp)
    trace.update_experiment(node.id, exp)
    fb = evaluate_fn(hyp, exp)
    trace.update_feedback(node.id, fb)

    if safety_fn is not None:
        safety_fn(node)  # mutates node.feedback.bias_flags

    if learn_fn is not None:
        learn_fn(node)

    if hitl_fn is not None:
        decision = hitl_fn(node) or {}
        action = decision.get("action")
        if action == "stop":
            return decision
        if action == "redirect" and decision.get("hypothesis"):
            scheduler.inject(decision["hypothesis"])

    return None


def run_loop(
    *,
    trace: DAGTrace,
    scheduler: Scheduler,
    hypothesize_fn: HypothesizeFn,
    experiment_fn: ExperimentFn,
    evaluate_fn: EvaluateFn,
    learn_fn: Optional[LearnFn] = None,
    novelty_fn: Optional[NoveltyFn] = None,
    hitl_fn: Optional[HITLFn] = None,
    safety_fn: Optional[SafetyFn] = None,
    max_iters: int = 5,
) -> DAGTrace:
    """Run Hypothesize -> Experiment -> Evaluate -> Learn for up to ``max_iters`` rounds.

    Phase 7 adds ``safety_fn`` — a callback (typically
    :func:`hypothesisloop.safety.bias_scanner.scan_node`) that runs after
    evaluate and may mutate ``node.feedback.bias_flags`` in place.
    """
    for i in range(max_iters):
        result = _execute_iteration(
            i,
            trace=trace,
            scheduler=scheduler,
            hypothesize_fn=hypothesize_fn,
            experiment_fn=experiment_fn,
            evaluate_fn=evaluate_fn,
            learn_fn=learn_fn,
            novelty_fn=novelty_fn,
            hitl_fn=hitl_fn,
            safety_fn=safety_fn,
        )
        if result and result.get("action") == "stop":
            break

    return trace


__all__ = [
    "run_loop",
    "HypothesizeFn",
    "ExperimentFn",
    "EvaluateFn",
    "LearnFn",
    "NoveltyFn",
    "HITLFn",
    "SafetyFn",
]
