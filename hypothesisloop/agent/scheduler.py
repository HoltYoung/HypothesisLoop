"""Scheduling: which DAG node should the next hypothesis branch from?

Phase 2 ships a dumb-linear scheduler. Branching/probabilistic variants land
when (and if) we need them — the DAG-readiness already lives in
``DAGTrace.leaves`` / ``DAGTrace.ancestors``, so this module stays small.
"""

from __future__ import annotations

from typing import Optional, Protocol

from hypothesisloop.agent.state import DAGTrace, TraceNode


class Scheduler(Protocol):
    def next_parent(self, trace: DAGTrace) -> Optional[TraceNode]: ...
    def mark_stale(self, node_id: str, trace: DAGTrace) -> None: ...
    def inject(self, hypothesis_text: str) -> None: ...


class LinearScheduler:
    """Always returns the most recently added non-stale leaf as the parent.

    Returning ``None`` means "no parent — this is a root hypothesis." That is
    the correct state for round 0 (empty trace) and also when every leaf has
    been marked stale; the loop's termination is governed by ``max_iters``,
    not by this method.
    """

    def __init__(self) -> None:
        self._injected: list[str] = []

    def next_parent(self, trace: DAGTrace) -> Optional[TraceNode]:
        leaves = trace.leaves()
        if not leaves:
            return None
        # leaves() iterates in insertion order; the last one is most recent.
        return leaves[-1]

    def mark_stale(self, node_id: str, trace: DAGTrace) -> None:
        trace.mark_stale(node_id)

    def inject(self, hypothesis_text: str) -> None:
        self._injected.append(hypothesis_text)

    def consume_injection(self) -> Optional[str]:
        return self._injected.pop(0) if self._injected else None


__all__ = ["Scheduler", "LinearScheduler"]
