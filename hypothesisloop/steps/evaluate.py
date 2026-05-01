"""Evaluate step — real LLM-backed feedback writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field, ValidationError

from hypothesisloop.agent.predict_score import cv_score, is_improvement
from hypothesisloop.agent.state import (
    EngineeredFeature,
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
        predict_state: Optional[dict] = None,
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
        # Predict mode hook. When non-None, ``__call__`` overrides the LLM's
        # decision based on a CV proxy score and mutates the shared state.
        # Expected keys: trace, train_df, target_column, task_type,
        # metric_name, prev_score, seed.
        self.predict_state = predict_state

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

        feedback = HypothesisFeedback(
            hypothesis_id=hypothesis.id,
            decision=draft.decision,
            reason=draft.reason,
            observations=draft.observations,
            novel_subhypotheses=list(draft.novel_subhypotheses),
            confidence=float(draft.confidence),
        )

        # Predict mode: override the LLM's decision with a deterministic
        # threshold check on the proxy CV delta.
        if self.predict_state is not None:
            try:
                _apply_predict_decision(feedback, hypothesis, experiment, self.predict_state)
            except Exception as exc:  # pragma: no cover — surface but don't crash the loop
                feedback.decision = "invalid"
                feedback.reason = (
                    f"{feedback.reason}\n\n[predict-eval failed: "
                    f"{type(exc).__name__}: {exc}]"
                )
                feedback.confidence = 0.0

        return feedback


def _apply_predict_decision(
    feedback: HypothesisFeedback,
    hypothesis: Hypothesis,
    experiment: Experiment,
    state: dict,
) -> None:
    """Run CV on engineered df, override decision, mutate state if accepted."""
    import pandas as pd

    # If the experiment failed, mark invalid — no CV to run.
    if not experiment.succeeded or not experiment.attempts:
        feedback.decision = "invalid"
        feedback.observations = (
            (feedback.observations or "")
            + f"\n[predict] experiment did not succeed; no CV measurement."
        )
        return

    code = experiment.attempts[-1].code
    train_df: pd.DataFrame = state["train_df"]
    target_column: str = state["target_column"]
    task_type: str = state["task_type"]
    metric_name: str = state["metric_name"]
    prev_score = state["prev_score"]
    seed: int = state.get("seed", 42)

    # Apply the LLM's code to a fresh copy of the current best train_df.
    # The sandbox already verified the code runs without crashing; we re-execute
    # in-process here so the engineered df survives for CV scoring.
    new_df = train_df.copy()
    local_ns = {
        "df": new_df,
        "pd": __import__("pandas"),
        "np": __import__("numpy"),
        "hl_emit": lambda k, v: None,
    }
    try:
        exec(compile(code, "<feature-engineering>", "exec"), local_ns, local_ns)
    except Exception as exc:
        feedback.decision = "invalid"
        feedback.observations = (
            (feedback.observations or "")
            + f"\n[predict] in-process re-exec failed: {type(exc).__name__}: {exc}"
        )
        return
    new_df = local_ns.get("df", new_df)

    # CV-score the engineered df.
    new_score = cv_score(
        new_df,
        target_column=target_column,
        task_type=task_type,  # type: ignore[arg-type]
        metric_name=metric_name,  # type: ignore[arg-type]
        seed=seed,
    )
    accepted, delta = is_improvement(prev_score, new_score)

    feedback.decision = "confirmed" if accepted else "rejected"
    feedback.observations = (
        f"[predict] {metric_name} prev={prev_score.value:.4f}, "
        f"new={new_score.value:.4f}, delta={delta:+.4f}, "
        f"predicted={hypothesis.predicted_metric_delta:+.4f}, "
        f"accepted={accepted}\n"
        + (feedback.observations or "")
    )

    # Trace mutation. The shared trace is in state["trace"] so that we can
    # update engineered_features and current_best_score atomically with the
    # decision override.
    trace = state.get("trace")
    feature_name = (hypothesis.target_columns[0] if hypothesis.target_columns else "<unknown>")
    if trace is not None:
        trace.engineered_features.append(
            EngineeredFeature(
                name=feature_name,
                code=code,
                iteration_added=hypothesis.iteration,
                hypothesis_id=hypothesis.id,
                predicted_delta=float(hypothesis.predicted_metric_delta or 0.0),
                actual_delta=float(delta),
                accepted=bool(accepted),
                rejection_reason=(None if accepted else f"delta {delta:+.4f} below threshold"),
            )
        )

    if accepted:
        state["train_df"] = new_df
        state["prev_score"] = new_score
        if trace is not None:
            trace.current_best_score = float(new_score.value)


__all__ = ["Evaluate", "FeedbackDraft"]
