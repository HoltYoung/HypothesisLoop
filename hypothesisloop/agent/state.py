"""Agent state: hypothesis/experiment/feedback dataclasses + the DAG trace.

Pure data + lookup. No I/O outside ``DAGTrace.save`` / ``DAGTrace.load``, no
logging, no Langfuse — Phase 4 wraps callers in spans, this module stays
silent.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional


TestType = Literal["correlation", "group_diff", "regression", "distribution", "custom", "classification"]
Decision = Literal["confirmed", "rejected", "inconclusive", "invalid"]
LoopMode = Literal["explore", "predict"]
TaskType = Literal["classification", "regression"]
MetricName = Literal["roc_auc", "log_loss", "r2"]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def new_hypothesis_id() -> str:
    """Fresh uuid4 hex — one per Hypothesis."""
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Hypothesis:
    id: str
    parent_id: Optional[str]
    iteration: int
    statement: str
    null: str
    test_type: TestType
    target_columns: list[str]
    expected_outcome: str
    concise_reason: str
    concise_observation: str
    concise_justification: str
    concise_knowledge: str
    embedding: list[float] = field(default_factory=list)
    re_explore: bool = False
    created_at: str = field(default_factory=_utc_iso)
    # Predict-mode optional fields. Explore-mode hypotheses leave these None.
    predicted_metric_delta: Optional[float] = None
    feature_op: Optional[str] = None  # "create:<name>" | "drop:<name>" | "transform:<name>" | "derive:<name>"


@dataclass
class EngineeredFeature:
    """One accepted-or-rejected feature engineering operation (Predict mode)."""

    name: str
    code: str
    iteration_added: int
    hypothesis_id: str
    predicted_delta: float
    actual_delta: float
    accepted: bool
    rejection_reason: Optional[str] = None


@dataclass
class ExperimentAttempt:
    """One run of generated code. Multiple attempts on retry."""

    attempt_idx: int
    code: str
    exit_code: int
    stdout: str
    stderr: str
    figures: list[str]
    metrics: dict
    blocked_reason: Optional[str]
    duration_s: float
    timed_out: bool
    oom_killed: bool


@dataclass
class Experiment:
    hypothesis_id: str
    attempts: list[ExperimentAttempt]
    succeeded: bool

    @property
    def final_attempt(self) -> Optional[ExperimentAttempt]:
        return self.attempts[-1] if self.attempts else None


@dataclass
class HypothesisFeedback:
    hypothesis_id: str
    decision: Decision
    reason: str
    observations: str
    novel_subhypotheses: list[str]
    confidence: float
    bias_flags: list[dict] = field(default_factory=list)


@dataclass
class TraceNode:
    id: str
    parent_id: Optional[str]
    iteration: int
    hypothesis: Hypothesis
    experiment: Optional[Experiment] = None
    feedback: Optional[HypothesisFeedback] = None
    stale: bool = False


# ---------------------------------------------------------------------------
# from_dict helpers (explicit so Optional / nested lists round-trip cleanly)
# ---------------------------------------------------------------------------
def _hypothesis_from_dict(d: dict) -> Hypothesis:
    return Hypothesis(**d)


def _attempt_from_dict(d: dict) -> ExperimentAttempt:
    return ExperimentAttempt(**d)


def _experiment_from_dict(d: Optional[dict]) -> Optional[Experiment]:
    if d is None:
        return None
    attempts = [_attempt_from_dict(a) for a in d.get("attempts", [])]
    return Experiment(
        hypothesis_id=d["hypothesis_id"],
        attempts=attempts,
        succeeded=d["succeeded"],
    )


def _feedback_from_dict(d: Optional[dict]) -> Optional[HypothesisFeedback]:
    if d is None:
        return None
    return HypothesisFeedback(**d)


def _node_from_dict(d: dict) -> TraceNode:
    return TraceNode(
        id=d["id"],
        parent_id=d.get("parent_id"),
        iteration=d["iteration"],
        hypothesis=_hypothesis_from_dict(d["hypothesis"]),
        experiment=_experiment_from_dict(d.get("experiment")),
        feedback=_feedback_from_dict(d.get("feedback")),
        stale=d.get("stale", False),
    )


# ---------------------------------------------------------------------------
# DAGTrace
# ---------------------------------------------------------------------------
class DAGTrace:
    """In-memory DAG of hypothesis nodes for one HypothesisLoop session.

    The DAG-readiness lives here (parent_id pointers, leaves(), ancestors()).
    Schedulers are dumb consumers; novelty/pruner read but don't mutate.
    """

    def __init__(
        self,
        session_id: str,
        dataset_path: Path | str,
        question: str,
        schema_summary: str = "",
        *,
        mode: LoopMode = "explore",
        target_column: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        metric_name: Optional[MetricName] = None,
    ):
        self.session_id: str = session_id
        self.dataset_path: str = str(dataset_path)
        self.question: str = question
        self.schema_summary: str = schema_summary
        self.created_at: str = _utc_iso()
        self._nodes: dict[str, TraceNode] = {}
        self._children: dict[str, list[str]] = {}
        self._order: list[str] = []
        # Hypotheses the novelty gate rejected — kept out of the DAG but
        # surfaced in subsequent prompts so the LLM doesn't re-propose them.
        self.novelty_rejected: list[Hypothesis] = []
        # Predict-mode state. ``mode == "explore"`` leaves these untouched.
        self.mode: LoopMode = mode
        self.target_column: Optional[str] = target_column
        self.task_type: Optional[TaskType] = task_type
        self.metric_name: Optional[MetricName] = metric_name
        self.baseline_score: Optional[float] = None
        self.current_best_score: Optional[float] = None
        self.engineered_features: list[EngineeredFeature] = []

    # ---- mutation -------------------------------------------------------
    def add_node(self, hypothesis: Hypothesis) -> TraceNode:
        if hypothesis.id in self._nodes:
            raise ValueError(f"duplicate node id: {hypothesis.id!r}")
        node = TraceNode(
            id=hypothesis.id,
            parent_id=hypothesis.parent_id,
            iteration=hypothesis.iteration,
            hypothesis=hypothesis,
        )
        self._nodes[node.id] = node
        self._order.append(node.id)
        self._children.setdefault(node.id, [])
        if hypothesis.parent_id is not None:
            self._children.setdefault(hypothesis.parent_id, []).append(node.id)
        return node

    def update_experiment(self, node_id: str, experiment: Experiment) -> None:
        if node_id not in self._nodes:
            raise KeyError(node_id)
        self._nodes[node_id].experiment = experiment

    def update_feedback(self, node_id: str, feedback: HypothesisFeedback) -> None:
        if node_id not in self._nodes:
            raise KeyError(node_id)
        self._nodes[node_id].feedback = feedback

    def mark_stale(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise KeyError(node_id)
        self._nodes[node_id].stale = True

    def add_novelty_rejection(self, hypothesis: Hypothesis) -> None:
        """Append to ``novelty_rejected``. Hypothesis is NOT added to the DAG."""
        self.novelty_rejected.append(hypothesis)

    # ---- reads ----------------------------------------------------------
    def get(self, node_id: str) -> TraceNode:
        if node_id not in self._nodes:
            raise KeyError(node_id)
        return self._nodes[node_id]

    def leaves(self) -> list[TraceNode]:
        """Non-stale nodes with no (active) children."""
        out: list[TraceNode] = []
        for node_id in self._order:
            node = self._nodes[node_id]
            if node.stale:
                continue
            if not self._children.get(node_id):
                out.append(node)
        return out

    def ancestors(self, node_id: str) -> list[TraceNode]:
        """Root → ... → parent (excluding the node itself)."""
        if node_id not in self._nodes:
            raise KeyError(node_id)
        chain: list[TraceNode] = []
        cur = self._nodes[node_id].parent_id
        while cur is not None:
            chain.append(self._nodes[cur])
            cur = self._nodes[cur].parent_id
        chain.reverse()
        return chain

    def all_hypotheses(self) -> list[Hypothesis]:
        return [self._nodes[nid].hypothesis for nid in self._order]

    def iter_nodes(self) -> list[TraceNode]:
        """Return nodes in insertion order (public alternative to ``_order``)."""
        return [self._nodes[nid] for nid in self._order]

    def latest(self) -> Optional[TraceNode]:
        if not self._order:
            return None
        return self._nodes[self._order[-1]]

    def iteration_count(self) -> int:
        if not self._nodes:
            return 0
        return max(node.iteration for node in self._nodes.values())

    # ---- serialization --------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "dataset_path": self.dataset_path,
            "question": self.question,
            "schema_summary": self.schema_summary,
            "created_at": self.created_at,
            "nodes": [asdict(self._nodes[nid]) for nid in self._order],
            "children": {nid: list(kids) for nid, kids in self._children.items()},
            "order": list(self._order),
            "novelty_rejected": [asdict(h) for h in self.novelty_rejected],
            # Phase 9 — Predict-mode state. Loaders default these for back-compat.
            "mode": self.mode,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "metric_name": self.metric_name,
            "baseline_score": self.baseline_score,
            "current_best_score": self.current_best_score,
            "engineered_features": [asdict(f) for f in self.engineered_features],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DAGTrace":
        trace = cls(
            session_id=d["session_id"],
            dataset_path=d["dataset_path"],
            question=d["question"],
            schema_summary=d.get("schema_summary", ""),
            mode=d.get("mode", "explore"),
            target_column=d.get("target_column"),
            task_type=d.get("task_type"),
            metric_name=d.get("metric_name"),
        )
        trace.baseline_score = d.get("baseline_score")
        trace.current_best_score = d.get("current_best_score")
        trace.engineered_features = [
            EngineeredFeature(**f) for f in d.get("engineered_features", [])
        ]
        # Restore the original timestamp (don't re-stamp).
        trace.created_at = d.get("created_at", trace.created_at)
        for node_dict in d.get("nodes", []):
            node = _node_from_dict(node_dict)
            trace._nodes[node.id] = node
        trace._order = list(d.get("order", []))
        trace._children = {nid: list(kids) for nid, kids in d.get("children", {}).items()}
        # Defensive: every node should have a children entry.
        for nid in trace._nodes:
            trace._children.setdefault(nid, [])
        # Back-compat: older trace files predate the novelty_rejected slot.
        nr_list = d.get("novelty_rejected", [])
        trace.novelty_rejected = [_hypothesis_from_dict(h) for h in nr_list]
        return trace

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "DAGTrace":
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


__all__ = [
    "TestType",
    "Decision",
    "LoopMode",
    "TaskType",
    "MetricName",
    "Hypothesis",
    "EngineeredFeature",
    "ExperimentAttempt",
    "Experiment",
    "HypothesisFeedback",
    "TraceNode",
    "DAGTrace",
    "new_hypothesis_id",
]
