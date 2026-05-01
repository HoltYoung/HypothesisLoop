"""Hypothesize step — real LLM-backed hypothesis proposer.

Stateless after construction. Implements the ``HypothesizeFn`` callable
shape from ``hypothesisloop.agent.loop`` so it can be plugged into
``run_loop`` directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, ValidationError

from hypothesisloop.agent.state import (
    DAGTrace,
    Hypothesis,
    TraceNode,
    new_hypothesis_id,
)
from hypothesisloop.trace.langfuse_client import RESEARCH_HYPOTHESIS, observe

# CRITICAL: override=True — feedback_env_override.
load_dotenv(override=True)


_DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "hypothesize.j2"
)
_PREDICT_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "hypothesize_predict.j2"
)
_TestTypeLiteral = Literal[
    "correlation", "group_diff", "regression", "distribution", "custom", "classification"
]


class HypothesisDraft(BaseModel):
    """Explore-mode LLM-boundary validator."""

    statement: str
    null: str
    test_type: _TestTypeLiteral
    target_columns: list[str]
    expected_outcome: str
    concise_reason: str
    concise_observation: str
    concise_justification: str
    concise_knowledge: str


class PredictHypothesisDraft(HypothesisDraft):
    """Predict-mode draft — adds metric-delta commitment + feature operation."""

    predicted_metric_delta: float
    feature_op: str


def _bind_json(llm: Any) -> Any:
    """Bind ``response_format={"type":"json_object"}`` if the LLM supports it.

    Real LangChain runnables have a ``.bind`` method; test stubs may not.
    """
    bind = getattr(llm, "bind", None)
    if callable(bind):
        try:
            return bind(response_format={"type": "json_object"})
        except Exception:
            return llm
    return llm


def _content_of(response: Any) -> str:
    """Extract the text content of an LLM response, tolerantly."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        # langchain may return list-of-parts on multi-modal responses.
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _reason_short(reason: str, n: int = 120) -> str:
    if len(reason) <= n:
        return reason
    return reason[: n - 1].rstrip() + "…"


class Hypothesize:
    """Real LLM-backed hypothesis proposer.

    Construct once, then call as ``hypothesize_fn(trace, parent)``. Wrapped in
    ``@observe`` so every call lands in Langfuse under ``research.hypothesis``.
    """

    def __init__(
        self,
        *,
        llm: Any,
        retriever: Callable[[str], list[dict]],
        prompt_path: Path | str = _DEFAULT_PROMPT_PATH,
        predict_prompt_path: Path | str = _PREDICT_PROMPT_PATH,
        rag_k: int = 4,
        scheduler: Any = None,
        pruner: Any = None,
    ):
        self.llm = _bind_json(llm)
        self.retriever = retriever
        self.rag_k = rag_k
        self.scheduler = scheduler
        self.pruner = pruner

        explore_path = Path(prompt_path)
        predict_path = Path(predict_prompt_path)
        # Both templates live under hypothesisloop/prompts/, so a single Jinja2
        # environment with that directory as its loader root suffices.
        env = Environment(
            loader=FileSystemLoader(str(explore_path.parent)),
            undefined=StrictUndefined,
            trim_blocks=False,
            lstrip_blocks=False,
        )
        self._template = env.get_template(explore_path.name)
        try:
            self._predict_template = env.get_template(predict_path.name)
        except Exception:
            # Predict template optional during tests that mock the explore path.
            self._predict_template = None
        # Stash the most recently rendered prompt so tests can inspect it.
        self.last_prompt: Optional[str] = None

    def _render(
        self,
        *,
        trace: DAGTrace,
        iteration_idx: int,
        rag_chunks: list[dict],
        injected_redirect: Optional[str],
    ) -> str:
        if self.pruner is not None:
            prior_hypotheses = self.pruner.prior_hypotheses_view(trace)
            rejected_hypotheses = self.pruner.rejected_view(trace)
        else:
            prior_hypotheses = self._build_priors_no_pruner(trace)
            rejected_hypotheses = self._build_rejected_no_pruner(trace)

        # Cross-iteration error context: pull the last 3 iterations' code
        # failures so the LLM proactively avoids repeating dtype/import bugs
        # across the run, not just within a single iteration's retries.
        prior_failures = self._build_prior_failures(trace, k=3)

        if getattr(trace, "mode", "explore") == "predict" and self._predict_template is not None:
            return self._predict_template.render(
                dataset_path=trace.dataset_path,
                target_column=trace.target_column or "",
                task_type=trace.task_type or "classification",
                metric_name=trace.metric_name or "roc_auc",
                baseline_score=trace.baseline_score if trace.baseline_score is not None else 0.0,
                current_best_score=trace.current_best_score if trace.current_best_score is not None else 0.0,
                schema_summary=trace.schema_summary or "_(schema not yet profiled)_",
                rag_chunks=rag_chunks,
                engineered_features=list(trace.engineered_features),
                injected_redirect=injected_redirect,
                iteration_idx=iteration_idx,
                prior_failures=prior_failures,
            )
        return self._template.render(
            dataset_path=trace.dataset_path,
            question=trace.question,
            schema_summary=trace.schema_summary or "_(schema not yet profiled)_",
            rag_chunks=rag_chunks,
            prior_hypotheses=prior_hypotheses,
            rejected_hypotheses=rejected_hypotheses,
            injected_redirect=injected_redirect,
            iteration_idx=iteration_idx,
            prior_failures=prior_failures,
        )

    @staticmethod
    def _build_prior_failures(trace: DAGTrace, k: int = 3) -> list[dict]:
        """Pull the most informative stderr line from each failed attempt in
        the last `k` iterations. Returns a flat list of dicts.

        Shape per entry: {"iteration": int, "error": str, "blocked_reason": str|None}
        """
        out: list[dict] = []
        order = list(trace._order)[-k:]
        for node_id in order:
            try:
                node = trace.get(node_id)
            except Exception:
                continue
            if node.experiment is None:
                continue
            for a in node.experiment.attempts:
                if a.exit_code == 0 and not a.blocked_reason:
                    continue
                # Last meaningful stderr line — exception type + message.
                err_line = ""
                if a.stderr:
                    for line in reversed(a.stderr.splitlines()):
                        line = line.strip()
                        if not line:
                            continue
                        # Exception lines look like "FooError: bar"
                        if any(s in line for s in ("Error:", "Exception:", "Warning:")) or line[:1].isupper():
                            err_line = line[:160]
                            break
                if not err_line and a.blocked_reason:
                    err_line = f"blocked: {a.blocked_reason}"
                if not err_line:
                    err_line = f"exit_code={a.exit_code}"
                out.append({
                    "iteration": node.iteration,
                    "error": err_line,
                    "blocked_reason": a.blocked_reason,
                })
        return out

    @staticmethod
    def _build_priors_no_pruner(trace: DAGTrace) -> list[dict]:
        priors: list[dict] = []
        for node_id in trace._order:  # private but stable; preserves insertion order
            node = trace.get(node_id)
            if node.feedback is None:
                continue
            metrics: Optional[dict] = None
            if node.experiment is not None and node.experiment.attempts:
                metrics = dict(node.experiment.attempts[-1].metrics or {})
            priors.append(
                {
                    "statement": node.hypothesis.statement,
                    "decision": node.feedback.decision,
                    "reason_short": _reason_short(node.feedback.reason),
                    "metrics": metrics,
                    "code_snippet": None,
                    "re_explore": bool(node.hypothesis.re_explore),
                }
            )
        return priors

    @staticmethod
    def _build_rejected_no_pruner(trace: DAGTrace) -> list[dict]:
        return [
            {
                "statement": h.statement,
                "rejection_reason": "embedding similarity above novelty gate",
            }
            for h in trace.novelty_rejected
        ]

    @observe(name=RESEARCH_HYPOTHESIS)
    def __call__(self, trace: DAGTrace, parent: Optional[TraceNode]) -> Hypothesis:
        iteration_idx = trace.iteration_count() + 1

        # Truncate the schema in the retrieval query so we don't blow embedding tokens.
        retrieval_query = (trace.question + " " + (trace.schema_summary or ""))[:1000]
        rag_chunks = self.retriever(retrieval_query)[: self.rag_k]

        injected_redirect: Optional[str] = None
        if self.scheduler is not None:
            consume = getattr(self.scheduler, "consume_injection", None)
            if callable(consume):
                injected_redirect = consume()

        prompt = self._render(
            trace=trace,
            iteration_idx=iteration_idx,
            rag_chunks=rag_chunks,
            injected_redirect=injected_redirect,
        )
        self.last_prompt = prompt

        response = self.llm.invoke(prompt)
        raw = _content_of(response).strip()

        # Parse JSON first so a malformed body becomes a clean RuntimeError
        # carrying the raw output (callers can choose to retry). Schema-level
        # mismatches still surface as pydantic ValidationError.
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"hypothesize: LLM output is not valid JSON ({e}). Raw output:\n{raw!r}"
            ) from e

        is_predict = getattr(trace, "mode", "explore") == "predict"
        draft = (
            PredictHypothesisDraft.model_validate(payload)
            if is_predict
            else HypothesisDraft.model_validate(payload)
        )

        hyp = Hypothesis(
            id=new_hypothesis_id(),
            parent_id=(parent.id if parent is not None else None),
            iteration=iteration_idx,
            statement=draft.statement,
            null=draft.null,
            test_type=draft.test_type,
            target_columns=list(draft.target_columns),
            expected_outcome=draft.expected_outcome,
            concise_reason=draft.concise_reason,
            concise_observation=draft.concise_observation,
            concise_justification=draft.concise_justification,
            concise_knowledge=draft.concise_knowledge,
        )
        if is_predict:
            hyp.predicted_metric_delta = float(draft.predicted_metric_delta)  # type: ignore[attr-defined]
            hyp.feature_op = str(draft.feature_op)  # type: ignore[attr-defined]
        return hyp


__all__ = ["Hypothesize", "HypothesisDraft", "PredictHypothesisDraft"]
