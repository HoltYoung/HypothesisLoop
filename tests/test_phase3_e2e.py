"""Phase 3 end-to-end acceptance test — one full iteration with real Kimi.

Gated behind ``HL_RUN_INTEGRATION=1`` so it does not run on every pytest
invocation. Costs ~one Kimi call to hypothesize + one Kimi call to evaluate
+ one OpenAI embedding call to retrieve.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.skipif(
    not os.getenv("HL_RUN_INTEGRATION"),
    reason="set HL_RUN_INTEGRATION=1 to run live Kimi + OpenAI embeddings",
)


def test_one_iteration_end_to_end_real_llm(tmp_path: Path, capsys):
    from hypothesisloop.agent.state import (
        DAGTrace,
        Experiment,
        ExperimentAttempt,
    )
    from hypothesisloop.llm.dispatch import get_llm
    from hypothesisloop.primitives.rag import load_index, retrieve
    from hypothesisloop.steps.evaluate import Evaluate
    from hypothesisloop.steps.hypothesize import Hypothesize
    from hypothesisloop.steps.profile import profile_dataset
    from hypothesisloop.trace.langfuse_client import observe, start_session

    start_session(session_id="phase3-e2e")

    project_root = Path(__file__).resolve().parents[1]

    # 1. Load 1000-row sample of UCI Adult.
    csv_path = project_root / "data" / "adult.csv"
    df_full = pd.read_csv(csv_path)
    df = df_full.sample(n=min(1000, len(df_full)), random_state=42).reset_index(drop=True)

    # 2. Build trace seeded with the dataset profile.
    schema_summary = profile_dataset(df, dataset_path=str(csv_path), max_categories=5)
    trace = DAGTrace(
        session_id="phase3-e2e",
        dataset_path=str(csv_path),
        question="What demographic factors most predict income > $50K?",
        schema_summary=schema_summary,
    )

    # 3. Build LLMs at the locked temperatures.
    llm_hyp = get_llm(model="moonshot-v1-128k", temperature=0.7)
    llm_eval = get_llm(model="moonshot-v1-128k", temperature=0.3)

    # 4. Build retriever from the on-disk RAG index.
    index_path = project_root / "knowledge" / "rag.index"
    chunks_path = project_root / "knowledge" / "rag_chunks.pkl"
    assert index_path.exists() and chunks_path.exists(), (
        "Run `python scripts/build_rag_index.py` before this test."
    )
    index, chunks = load_index(index_path, chunks_path)

    def retriever(q: str) -> list[dict]:
        return retrieve(q, index, chunks, k=3)

    # 5. Step instances.
    hyp_step = Hypothesize(llm=llm_hyp, retriever=retriever, rag_k=3)
    eval_step = Evaluate(llm=llm_eval)

    # 6. Hypothesize + Evaluate inside a parent @observe so we get one trace
    #    URL covering the whole iteration.
    captured: dict = {}

    @observe(name="phase3.e2e")
    def _run_iteration():
        h = hyp_step(trace, None)
        canned_local = Experiment(
            hypothesis_id=h.id,
            attempts=[
                ExperimentAttempt(
                    attempt_idx=0,
                    code="from scipy import stats\n...",
                    exit_code=0,
                    stdout="p_value=0.001\neffect=0.42\n",
                    stderr="",
                    figures=[],
                    metrics={"p_value": 0.001, "effect_size": 0.42, "n": 1000},
                    blocked_reason=None,
                    duration_s=1.2,
                    timed_out=False,
                    oom_killed=False,
                )
            ],
            succeeded=True,
        )
        f = eval_step(h, canned_local)
        # Capture trace URL while we're still inside the active span.
        try:
            from langfuse import get_client  # type: ignore

            client = get_client()
            try:
                client.update_current_trace(session_id="phase3-e2e")
            except Exception:
                pass
            try:
                captured["trace_url"] = client.get_trace_url()
            except Exception:
                captured["trace_url"] = None
        except ImportError:
            captured["trace_url"] = None
        return h, canned_local, f

    hyp, canned_inside, fb = _run_iteration()
    canned = canned_inside  # use the canned built inside the observe span

    # 9. Persist into the trace.
    trace.add_node(hyp)
    trace.update_experiment(hyp.id, canned)
    trace.update_feedback(hyp.id, fb)

    # 10. Save + reload round-trip.
    out = tmp_path / "trace.json"
    trace.save(out)
    loaded = DAGTrace.load(out)
    assert loaded.get(hyp.id) == trace.get(hyp.id)

    # 11. Field-level assertions.
    assert hyp.statement.strip()
    assert hyp.test_type in {"correlation", "group_diff", "regression", "distribution", "custom"}
    assert hyp.target_columns
    df_cols = set(df.columns)
    assert all(c in df_cols for c in hyp.target_columns), (
        f"target_columns referenced unknown columns: "
        f"{[c for c in hyp.target_columns if c not in df_cols]} "
        f"(df has {sorted(df_cols)})"
    )
    assert fb.decision in {"confirmed", "rejected", "inconclusive", "invalid"}
    assert 0.0 <= fb.confidence <= 1.0

    # Flush + surface the trace URL captured inside the @observe span.
    try:
        from langfuse import get_client  # type: ignore

        get_client().flush()
    except ImportError:
        pass
    print(f"\n[phase3-e2e] Langfuse trace URL: {captured.get('trace_url')}")
