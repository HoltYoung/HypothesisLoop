"""Phase 7 report-generator tests."""

from __future__ import annotations

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
from hypothesisloop.safety.bias_scanner import DISCLAIMER
from hypothesisloop.steps.report import render_report


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
# Minimal valid 1x1 PNG, ~67 bytes.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cb\x00"
    b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _hyp(
    iteration: int = 1,
    *,
    statement: str = "education-num positively associates with income",
    re_explore: bool = False,
) -> Hypothesis:
    return Hypothesis(
        id=new_hypothesis_id(),
        parent_id=None,
        iteration=iteration,
        statement=statement,
        null="not " + statement,
        test_type="correlation",
        target_columns=["education-num", "income"],
        expected_outcome="r > 0.2, p < 0.05",
        concise_reason="education is the most plausible single predictor",
        concise_observation="education-num spans 1-16",
        concise_justification="cheap and targets the headline question",
        concise_knowledge="tells us if education is even a candidate factor",
        re_explore=re_explore,
    )


def _experiment(
    hyp_id: str,
    *,
    code: str = "from scipy import stats\nprint(stats.pearsonr(df['education-num'], df['income']))\n",
    succeeded: bool = True,
    figures: list[str] | None = None,
    metrics: dict | None = None,
) -> Experiment:
    attempt = ExperimentAttempt(
        attempt_idx=0,
        code=code,
        exit_code=0 if succeeded else 1,
        stdout="r=0.42, p=0.001\n",
        stderr="" if succeeded else "Traceback...\nKeyError\n",
        figures=list(figures or []),
        metrics=dict(metrics or {"p_value": 0.001, "effect_size": 0.42, "n": 1000}),
        blocked_reason=None,
        duration_s=0.5,
        timed_out=False,
        oom_killed=False,
    )
    return Experiment(hypothesis_id=hyp_id, attempts=[attempt], succeeded=succeeded)


def _feedback(
    hyp_id: str,
    *,
    decision: str = "confirmed",
    reason: str = "p=0.001, effect size moderate.",
    confidence: float = 0.78,
    bias_flags: list[dict] | None = None,
) -> HypothesisFeedback:
    return HypothesisFeedback(
        hypothesis_id=hyp_id,
        decision=decision,
        reason=reason,
        observations="r=0.42, p=0.001",
        novel_subhypotheses=["does the effect persist within education levels?"],
        confidence=confidence,
        bias_flags=list(bias_flags or []),
    )


def _build_minimal_trace(question: str = "Does education predict income?") -> DAGTrace:
    trace = DAGTrace(
        session_id="test-session",
        dataset_path="data/adult.csv",
        question=question,
    )
    h = _hyp(1)
    trace.add_node(h)
    trace.update_experiment(h.id, _experiment(h.id))
    trace.update_feedback(h.id, _feedback(h.id))
    return trace


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_render_report_basic(tmp_path: Path):
    trace = _build_minimal_trace()
    result = render_report(trace, output_dir=tmp_path, cli_command="python -m hypothesisloop.cli --auto --question test --seed 42")

    md_path = result["md"]
    txt_path = result["txt"]
    assert md_path is not None and md_path.exists()
    assert txt_path is not None and txt_path.exists()

    md = md_path.read_text(encoding="utf-8")
    for header in (
        "## 1. Run metadata",
        "## 2. Question & approach",
        "## 3. Hypothesis chain",
        "## 4. Key findings",
        "## 5. Rejections & dead-ends",
        "## 6. Bias flags raised",
        "## 7. Reasoning chain",
        "## 8. Limitations & caveats",
        "## 9. Reproduction",
    ):
        assert header in md, f"missing section: {header!r}"

    txt = txt_path.read_text(encoding="utf-8")
    assert "**" not in txt
    assert "<details>" not in txt


def test_render_report_embeds_figures(tmp_path: Path):
    fig_path = tmp_path / "fig.png"
    fig_path.write_bytes(_TINY_PNG)

    trace = DAGTrace(session_id="figs", dataset_path="d.csv", question="?")
    h = _hyp(1)
    trace.add_node(h)
    trace.update_experiment(h.id, _experiment(h.id, figures=[str(fig_path)]))
    trace.update_feedback(h.id, _feedback(h.id))

    out = tmp_path / "out"
    result = render_report(trace, output_dir=out)
    md = result["md"].read_text(encoding="utf-8")
    txt = result["txt"].read_text(encoding="utf-8")
    assert "data:image/png;base64," in md
    assert "data:image/png;base64," not in txt


def test_render_report_bias_flags_section(tmp_path: Path):
    trace = DAGTrace(session_id="bias", dataset_path="d.csv", question="?")
    h = _hyp(1)
    trace.add_node(h)
    trace.update_experiment(h.id, _experiment(h.id))
    fb = _feedback(
        h.id,
        bias_flags=[
            {
                "sensitive_var": "race",
                "causal_verb": "causes",
                "snippet": "Race causes income disparities.",
                "source": "hypothesis",
            }
        ],
    )
    trace.update_feedback(h.id, fb)

    result = render_report(trace, output_dir=tmp_path)
    md = result["md"].read_text(encoding="utf-8")
    assert "## 6. Bias flags raised" in md
    assert "race" in md and "causes" in md
    assert "Race causes income disparities" in md


def test_render_report_disclaimer_in_text(tmp_path: Path):
    trace = DAGTrace(session_id="disc", dataset_path="d.csv", question="?")
    h = _hyp(1, statement="Race causes income disparities in the dataset.")
    trace.add_node(h)
    trace.update_experiment(h.id, _experiment(h.id))
    trace.update_feedback(h.id, _feedback(h.id))

    result = render_report(trace, output_dir=tmp_path)
    md = result["md"].read_text(encoding="utf-8")
    first_line = DISCLAIMER.split("\n", 1)[0]
    assert first_line in md


def test_render_report_empty_trace(tmp_path: Path):
    trace = DAGTrace(session_id="empty", dataset_path="d.csv", question="What about it?")
    result = render_report(trace, output_dir=tmp_path)
    md = result["md"].read_text(encoding="utf-8")
    assert "## 3. Hypothesis chain" in md
    assert "No iterations completed" in md


def test_render_report_partial_trace_handles_missing_feedback(tmp_path: Path):
    trace = DAGTrace(session_id="partial", dataset_path="d.csv", question="?")
    h = _hyp(1)
    trace.add_node(h)
    # No experiment, no feedback (mid-run snapshot).
    result = render_report(trace, output_dir=tmp_path)
    md = result["md"].read_text(encoding="utf-8")
    assert "Iteration 1" in md
    assert "(no feedback yet)" in md


def test_render_report_reproduction_section(tmp_path: Path):
    trace = _build_minimal_trace()
    cmd = "python -m hypothesisloop.cli --auto --question 'Q?' --seed 42"
    result = render_report(trace, output_dir=tmp_path, cli_command=cmd, seed=42)
    md = result["md"].read_text(encoding="utf-8")
    assert "## 9. Reproduction" in md
    assert "--question" in md
    assert "Seed: `42`" in md
    # SHA must be present (real or "unknown").
    assert "Git SHA:" in md


def test_render_report_returns_aggregate_flags(tmp_path: Path):
    trace = DAGTrace(session_id="agg", dataset_path="d.csv", question="?")
    h = _hyp(1, statement="Sex causes pay gaps directly.")  # report-level flag
    trace.add_node(h)
    trace.update_experiment(h.id, _experiment(h.id))
    fb = _feedback(
        h.id,
        bias_flags=[
            {
                "sensitive_var": "race",
                "causal_verb": "causes",
                "snippet": "race-x",
                "source": "hypothesis",
            },
            {
                "sensitive_var": "race",
                "causal_verb": "leads to",
                "snippet": "race-y",
                "source": "feedback_reason",
            },
        ],
    )
    trace.update_feedback(h.id, fb)

    result = render_report(trace, output_dir=tmp_path)
    flags = result["bias_flags"]
    # 2 pre-populated node flags + at least 1 report-level flag from the hypothesis statement.
    assert len(flags) >= 3
    sources = [f["source"] for f in flags]
    assert "hypothesis" in sources
    assert "feedback_reason" in sources
    assert "report" in sources
