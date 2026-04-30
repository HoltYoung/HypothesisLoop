"""Phase 3 evaluate step tests — mocked LLM unless HL_RUN_INTEGRATION=1."""

from __future__ import annotations

import json
import os
from typing import Optional

import pytest
from pydantic import ValidationError

from hypothesisloop.agent.state import (
    Experiment,
    ExperimentAttempt,
    Hypothesis,
    HypothesisFeedback,
    new_hypothesis_id,
)
from hypothesisloop.steps.evaluate import Evaluate


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------
class StubMessage:
    def __init__(self, content: str):
        self.content = content


class StubLLM:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt: Optional[str] = None
        self.bound_kwargs: dict = {}

    def bind(self, **kwargs) -> "StubLLM":
        new = StubLLM(self.response_text)
        new.bound_kwargs = {**self.bound_kwargs, **kwargs}
        return new

    def invoke(self, prompt: str, **_kwargs) -> StubMessage:
        self.last_prompt = prompt
        return StubMessage(self.response_text)


def _make_hypothesis() -> Hypothesis:
    return Hypothesis(
        id=new_hypothesis_id(),
        parent_id=None,
        iteration=1,
        statement="education-num positively associates with income",
        null="education-num does not associate with income",
        test_type="correlation",
        target_columns=["education-num", "income"],
        expected_outcome="r > 0.2, p < 0.05",
        concise_reason="r",
        concise_observation="o",
        concise_justification="j",
        concise_knowledge="k",
    )


def _make_experiment(
    hyp_id: str,
    *,
    exit_code: int = 0,
    stdout: str = "p_value: 0.001\neffect_size: 0.42\n",
    stderr: str = "",
    metrics: Optional[dict] = None,
    succeeded: bool = True,
) -> Experiment:
    return Experiment(
        hypothesis_id=hyp_id,
        attempts=[
            ExperimentAttempt(
                attempt_idx=0,
                code="from scipy import stats\n...",
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                figures=[],
                metrics=metrics or {"p_value": 0.001, "effect_size": 0.42},
                blocked_reason=None,
                duration_s=0.5,
                timed_out=False,
                oom_killed=False,
            )
        ],
        succeeded=succeeded,
    )


VALID_FB_JSON = json.dumps(
    {
        "decision": "confirmed",
        "reason": "Strong positive Spearman correlation (rho=0.42) at p=0.001; meaningful effect size.",
        "observations": "rho=0.42, p=0.001, n=1000",
        "novel_subhypotheses": ["Does the effect persist after controlling for hours-per-week?"],
        "confidence": 0.82,
    }
)


# ---------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------
def test_evaluate_renders_prompt_with_experiment_summary():
    h = _make_hypothesis()
    exp = _make_experiment(h.id)
    llm = StubLLM(VALID_FB_JSON)
    step = Evaluate(llm=llm)

    fb = step(h, exp)

    assert isinstance(fb, HypothesisFeedback)
    assert fb.hypothesis_id == h.id
    assert fb.decision == "confirmed"
    assert fb.confidence == pytest.approx(0.82)
    assert "Does the effect persist" in fb.novel_subhypotheses[0]

    prompt = step.last_prompt
    # Experiment summary surfaces stdout / metrics / exit_code.
    assert "p_value" in prompt
    assert "exit_code: 0" in prompt
    # Hypothesis fields present.
    assert "education-num positively associates with income" in prompt


def test_evaluate_handles_failed_experiment():
    h = _make_hypothesis()
    exp = _make_experiment(
        h.id,
        exit_code=1,
        stdout="",
        stderr="Traceback (most recent call last):\n  ...\nKeyError: 'income-num'\n",
        metrics={},
        succeeded=False,
    )
    invalid_fb = json.dumps(
        {
            "decision": "invalid",
            "reason": "Script raised KeyError; the named column does not exist.",
            "observations": "exit_code=1; KeyError on missing column.",
            "novel_subhypotheses": [],
            "confidence": 0.95,
        }
    )
    llm = StubLLM(invalid_fb)
    step = Evaluate(llm=llm)
    fb = step(h, exp)
    assert fb.decision == "invalid"
    assert "KeyError" in step.last_prompt


def test_evaluate_invalid_confidence_raises():
    h = _make_hypothesis()
    exp = _make_experiment(h.id)
    bad = json.dumps({**json.loads(VALID_FB_JSON), "confidence": 1.5})
    llm = StubLLM(bad)
    step = Evaluate(llm=llm)
    with pytest.raises(ValidationError):
        step(h, exp)


# ---------------------------------------------------------------------------
# integration (gated)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.getenv("HL_RUN_INTEGRATION"),
    reason="set HL_RUN_INTEGRATION=1 to run live Kimi",
)
def test_evaluate_live_kimi():
    from hypothesisloop.llm.dispatch import get_llm

    h = _make_hypothesis()
    exp = _make_experiment(h.id)
    llm = get_llm(model="moonshot-v1-128k", temperature=0.3)
    step = Evaluate(llm=llm)

    fb = step(h, exp)
    assert fb.decision in {"confirmed", "rejected", "inconclusive", "invalid"}
    assert 0.0 <= fb.confidence <= 1.0
    assert fb.reason.strip()
