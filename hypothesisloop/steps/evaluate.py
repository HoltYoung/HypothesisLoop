"""Evaluate step — real LLM-backed feedback writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field, ValidationError

from hypothesisloop.agent.state import (
    Experiment,
    Hypothesis,
    HypothesisFeedback,
)
from hypothesisloop.trace.langfuse_client import FEEDBACK_HYPOTHESIS, observe

load_dotenv(override=True)


_DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "evaluate.j2"
)
_DecisionLiteral = Literal["confirmed", "rejected", "inconclusive", "invalid"]
_TAIL_CAP = 1500


class FeedbackDraft(BaseModel):
    decision: _DecisionLiteral
    reason: str
    observations: str
    novel_subhypotheses: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


def _bind_json(llm: Any) -> Any:
    bind = getattr(llm, "bind", None)
    if callable(bind):
        try:
            return bind(response_format={"type": "json_object"})
        except Exception:
            return llm
    return llm


def _content_of(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _tail(text: str, n: int = _TAIL_CAP) -> str:
    if not text:
        return ""
    if len(text) <= n:
        return text
    return text[-n:]


def _format_experiment_summary(experiment: Experiment) -> str:
    """Pre-format the experiment for the prompt body.

    Pulls from ``experiment.final_attempt`` because retries land in the same
    attempts list and the tail attempt is the result the LLM should judge.
    """
    attempt = experiment.final_attempt
    if attempt is None:
        return "exit_code: <no attempts recorded>"

    parts = [f"exit_code: {attempt.exit_code}"]
    if attempt.timed_out:
        parts.append("timed_out: true")
    if attempt.oom_killed:
        parts.append("oom_killed: true")
    if attempt.blocked_reason:
        parts.append(f"blocked_reason: {attempt.blocked_reason}")
    parts.append(f"succeeded: {experiment.succeeded}")
    parts.append(f"attempts: {len(experiment.attempts)}")

    if attempt.metrics:
        parts.append("metrics: " + json.dumps(attempt.metrics, default=str))

    parts.append(f"\nstdout (last {_TAIL_CAP} chars):\n{_tail(attempt.stdout)}")
    parts.append(f"\nstderr (last {_TAIL_CAP} chars):\n{_tail(attempt.stderr)}")

    if attempt.figures:
        parts.append("\nfigures saved: " + ", ".join(attempt.figures))

    return "\n".join(parts)


class Evaluate:
    """Real LLM-backed evaluator. Wrapped in Langfuse ``feedback.hypothesis_feedback``."""

    def __init__(
        self,
        *,
        llm: Any,
        prompt_path: Path | str = _DEFAULT_PROMPT_PATH,
    ):
        self.llm = _bind_json(llm)
        prompt_path = Path(prompt_path)
        env = Environment(
            loader=FileSystemLoader(str(prompt_path.parent)),
            undefined=StrictUndefined,
            trim_blocks=False,
            lstrip_blocks=False,
        )
        self._template = env.get_template(prompt_path.name)
        self.last_prompt: Optional[str] = None

    def _render(self, hypothesis: Hypothesis, experiment: Experiment) -> str:
        return self._template.render(
            hypothesis={
                "statement": hypothesis.statement,
                "null": hypothesis.null,
                "test_type": hypothesis.test_type,
                "target_columns": list(hypothesis.target_columns),
                "expected_outcome": hypothesis.expected_outcome,
            },
            experiment_summary=_format_experiment_summary(experiment),
        )

    @observe(name=FEEDBACK_HYPOTHESIS)
    def __call__(
        self, hypothesis: Hypothesis, experiment: Experiment
    ) -> HypothesisFeedback:
        prompt = self._render(hypothesis, experiment)
        self.last_prompt = prompt

        response = self.llm.invoke(prompt)
        raw = _content_of(response).strip()

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"evaluate: LLM output is not valid JSON ({e}). Raw output:\n{raw!r}"
            ) from e
        draft = FeedbackDraft.model_validate(payload)

        return HypothesisFeedback(
            hypothesis_id=hypothesis.id,
            decision=draft.decision,
            reason=draft.reason,
            observations=draft.observations,
            novel_subhypotheses=list(draft.novel_subhypotheses),
            confidence=float(draft.confidence),
        )


__all__ = ["Evaluate", "FeedbackDraft"]
