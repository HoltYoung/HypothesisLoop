"""Phase 3 hypothesize step tests — mocked LLM unless HL_RUN_INTEGRATION=1."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import pytest
from pydantic import ValidationError

from hypothesisloop.agent.scheduler import LinearScheduler
from hypothesisloop.agent.state import (
    DAGTrace,
    Experiment,
    ExperimentAttempt,
    Hypothesis,
    HypothesisFeedback,
    new_hypothesis_id,
)
from hypothesisloop.steps.hypothesize import Hypothesize


# ---------------------------------------------------------------------------
# stubs
# ---------------------------------------------------------------------------
class StubMessage:
    def __init__(self, content: str):
        self.content = content


class StubLLM:
    """Minimal LangChain-Runnable shape for tests: bind() + invoke() + .content."""

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


def _make_retriever(chunks: list[dict]):
    def _r(_query: str) -> list[dict]:
        return chunks

    return _r


VALID_DRAFT = {
    "statement": "education-num positively associates with income",
    "null": "education-num does not associate with income",
    "test_type": "correlation",
    "target_columns": ["education-num", "income"],
    "expected_outcome": "Spearman rho > 0.2, p < 0.05",
    "concise_reason": "education is the most plausible single predictor",
    "concise_observation": "education-num spans 1-16 with reasonable spread",
    "concise_justification": "cheap and targets the headline question directly",
    "concise_knowledge": "tells us if education is even a candidate factor",
}


# ---------------------------------------------------------------------------
# unit tests (no LLM)
# ---------------------------------------------------------------------------
def test_hypothesize_renders_first_iter_prompt():
    trace = DAGTrace(session_id="s1", dataset_path="data/adult.csv", question="Q?")
    chunks = [
        {"text": "codebook chunk", "source": "adult_codebook.md", "heading": "income", "score": 0.81},
        {"text": "test guide chunk", "source": "test_selection.md", "heading": "Correlations", "score": 0.74},
    ]
    llm = StubLLM(json.dumps(VALID_DRAFT))
    step = Hypothesize(llm=llm, retriever=_make_retriever(chunks))

    hyp = step(trace, None)

    assert isinstance(hyp, Hypothesis)
    assert hyp.iteration == 1
    assert hyp.parent_id is None
    assert len(hyp.id) == 32  # uuid4 hex
    assert hyp.statement == VALID_DRAFT["statement"]
    assert hyp.test_type == "correlation"
    assert hyp.target_columns == ["education-num", "income"]

    # Bound LLM has json_object response_format set.
    bound = step.llm
    assert bound.bound_kwargs.get("response_format") == {"type": "json_object"}
    # First-iter language present in prompt.
    assert "iteration 1" in step.last_prompt
    assert "adult_codebook.md" in step.last_prompt


def test_hypothesize_renders_later_iter_with_priors():
    trace = DAGTrace(session_id="s2", dataset_path="data/adult.csv", question="Q?")

    # Seed two prior nodes with feedback so they show up in the priors list.
    for i, statement in enumerate(["age relates to income", "hours relates to income"], start=1):
        h = Hypothesis(
            id=new_hypothesis_id(),
            parent_id=None,
            iteration=i,
            statement=statement,
            null="not " + statement,
            test_type="correlation",
            target_columns=["age", "income"],
            expected_outcome="?",
            concise_reason="?",
            concise_observation="?",
            concise_justification="?",
            concise_knowledge="?",
        )
        trace.add_node(h)
        trace.update_feedback(
            h.id,
            HypothesisFeedback(
                hypothesis_id=h.id,
                decision=("confirmed" if i == 1 else "rejected"),
                reason=f"reason text for {statement}, padded out " * 5,
                observations="obs",
                novel_subhypotheses=[],
                confidence=0.7,
            ),
        )

    llm = StubLLM(json.dumps(VALID_DRAFT))
    step = Hypothesize(llm=llm, retriever=_make_retriever([]))
    hyp = step(trace, None)

    assert hyp.iteration == 3, "third iteration after two priors"
    prompt = step.last_prompt
    assert "iteration 3" in prompt
    assert "age relates to income" in prompt
    assert "hours relates to income" in prompt
    # Reason should be truncated to 120 chars + ellipsis in the prompt body.
    assert "reason text for age" in prompt


def test_hypothesize_invalid_json_raises():
    trace = DAGTrace(session_id="s3", dataset_path="d.csv", question="Q?")
    llm = StubLLM("not json at all")
    step = Hypothesize(llm=llm, retriever=_make_retriever([]))
    with pytest.raises(RuntimeError) as excinfo:
        step(trace, None)
    assert "not json at all" in str(excinfo.value)


def test_hypothesize_invalid_test_type_raises():
    trace = DAGTrace(session_id="s4", dataset_path="d.csv", question="Q?")
    bogus = dict(VALID_DRAFT)
    bogus["test_type"] = "bogus"
    llm = StubLLM(json.dumps(bogus))
    step = Hypothesize(llm=llm, retriever=_make_retriever([]))
    with pytest.raises(ValidationError):
        step(trace, None)


def test_hypothesize_consumes_scheduler_injection():
    trace = DAGTrace(session_id="s5", dataset_path="d.csv", question="Q?")
    sched = LinearScheduler()
    sched.inject("focus on the gender-income gap")

    llm = StubLLM(json.dumps(VALID_DRAFT))
    step = Hypothesize(llm=llm, retriever=_make_retriever([]), scheduler=sched)

    step(trace, None)
    assert "focus on the gender-income gap" in step.last_prompt
    assert sched.consume_injection() is None, "injection should have been consumed exactly once"


# ---------------------------------------------------------------------------
# integration (gated)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.getenv("HL_RUN_INTEGRATION"),
    reason="set HL_RUN_INTEGRATION=1 to run live Kimi",
)
def test_hypothesize_live_kimi():
    from hypothesisloop.llm.dispatch import get_llm

    trace = DAGTrace(
        session_id="phase3-live-hyp",
        dataset_path="data/adult.csv",
        question="What demographic factors most predict income > $50K?",
        schema_summary="14 columns, 32k rows; numeric: age, education-num, hours-per-week; categorical: education, marital-status, race, sex, income.",
    )
    llm = get_llm(model="moonshot-v1-128k", temperature=0.7)
    chunks = [
        {"text": "age, education-num are continuous; income is binary.",
         "source": "adult_codebook.md", "heading": "Schema", "score": 0.9},
    ]
    step = Hypothesize(llm=llm, retriever=_make_retriever(chunks))

    hyp = step(trace, None)
    assert hyp.statement and len(hyp.statement) < 200
    for f in (hyp.concise_reason, hyp.concise_observation, hyp.concise_justification, hyp.concise_knowledge):
        assert f.strip(), "concise_* fields must be non-empty"
    assert hyp.test_type in {"correlation", "group_diff", "regression", "distribution", "custom"}
    assert hyp.target_columns and all(isinstance(c, str) for c in hyp.target_columns)
