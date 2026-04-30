"""Phase 5 acceptance — pruner keeps a 5-iter run under the 50K-token budget."""

from __future__ import annotations

from hypothesisloop.agent.pruner import Pruner
from hypothesisloop.agent.state import (
    DAGTrace,
    Experiment,
    ExperimentAttempt,
    Hypothesis,
    HypothesisFeedback,
    new_hypothesis_id,
)


def _hyp(iteration: int, statement: str, parent_id: str | None) -> Hypothesis:
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
    )


def _experiment(hyp_id: str) -> Experiment:
    # Realistic-ish artifact sizes — pruner drops stdout/stderr from the view,
    # so this is mostly a smoke test that big payloads don't sneak in.
    big_stdout = "result line\n" * 80   # ~1KB
    big_stderr = "warn line\n" * 80     # ~800 chars
    realistic_code = (
        "from scipy import stats\n"
        "import numpy as np\n"
        "x = df['age']\n"
        "y = df['hours-per-week']\n"
        "r, p = stats.pearsonr(x, y)\n"
        "n = int(len(df))\n"
        "print(f'r={r:.4f}, p={p:.6f}, n={n}')\n"
        "hl_emit('r', float(r))\n"
        "hl_emit('p_value', float(p))\n"
        "hl_emit('n', n)\n"
    ) * 4  # ~1.5KB of code
    attempt = ExperimentAttempt(
        attempt_idx=0,
        code=realistic_code,
        exit_code=0,
        stdout=big_stdout,
        stderr=big_stderr,
        figures=[],
        metrics={"p_value": 0.001, "effect_size": 0.31, "n": 1000},
        blocked_reason=None,
        duration_s=1.4,
        timed_out=False,
        oom_killed=False,
    )
    return Experiment(hypothesis_id=hyp_id, attempts=[attempt], succeeded=True)


def _feedback(hyp_id: str) -> HypothesisFeedback:
    return HypothesisFeedback(
        hypothesis_id=hyp_id,
        decision="confirmed",
        reason=(
            "Pearson r=0.31, p=0.001, n=1000 — a modest but reliable positive "
            "association; effect size warrants follow-up controls."
        ),
        observations="r=0.31, p=0.001",
        novel_subhypotheses=["does the relationship persist within education levels?"],
        confidence=0.78,
    )


def test_5_iter_trace_under_50k_tokens():
    trace = DAGTrace(
        session_id="prune-5iter",
        dataset_path="data/adult.csv",
        question="What demographic factors predict income > $50K?",
    )

    parent = None
    for i in range(1, 6):
        h = _hyp(i, f"hypothesis #{i}: explore feature interaction {i}", parent)
        trace.add_node(h)
        trace.update_experiment(h.id, _experiment(h.id))
        trace.update_feedback(h.id, _feedback(h.id))
        parent = h.id

    # Plus 2 novelty-rejected hypotheses on the side.
    trace.add_novelty_rejection(_hyp(99, "near-duplicate of iter 2", None))
    trace.add_novelty_rejection(_hyp(99, "near-duplicate of iter 4", None))

    pruner = Pruner()
    n_tokens = pruner.estimate_tokens(trace)

    assert n_tokens > 0, "estimate_tokens should produce a real count"
    assert n_tokens < 50_000, f"5-iter trace blew the 50K budget: {n_tokens} tokens"
