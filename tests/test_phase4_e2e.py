"""Phase 4 end-to-end acceptance — one full live iteration.

Profile → hypothesize (live Kimi) → experiment (live Kimi codegen + real
sandbox) → evaluate (live Kimi). Gated behind ``HL_RUN_INTEGRATION=1``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.skipif(
    not os.getenv("HL_RUN_INTEGRATION"),
    reason="set HL_RUN_INTEGRATION=1 to run live Kimi + sandbox",
)


def test_phase4_full_iteration_e2e(tmp_path: Path):
    from hypothesisloop.agent.state import DAGTrace
    from hypothesisloop.llm.dispatch import get_llm
    from hypothesisloop.primitives.rag import load_index, retrieve
    from hypothesisloop.steps.evaluate import Evaluate
    from hypothesisloop.steps.experiment import ExperimentStep
    from hypothesisloop.steps.hypothesize import Hypothesize
    from hypothesisloop.steps.profile import profile_dataset
    from hypothesisloop.trace.langfuse_client import observe, start_session

    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "adult.csv"

    # 1. Load 1000-row sample.
    df_full = pd.read_csv(csv_path)
    df = df_full.sample(n=min(1000, len(df_full)), random_state=42).reset_index(drop=True)
    sample_csv = tmp_path / "adult_sample.csv"
    df.to_csv(sample_csv, index=False)

    # 2. Build trace.
    schema_summary = profile_dataset(df, dataset_path=str(sample_csv), max_categories=5)
    trace = DAGTrace(
        session_id="phase4-e2e",
        dataset_path=str(sample_csv),
        question="What demographic factors most predict income > $50K?",
        schema_summary=schema_summary,
    )
    start_session(session_id="phase4-e2e")

    # 3. Steps.
    llm_hyp = get_llm(model="moonshot-v1-128k", temperature=0.7)
    llm_code = get_llm(model="moonshot-v1-128k", temperature=0.7)
    llm_eval = get_llm(model="moonshot-v1-128k", temperature=0.3)

    index_path = project_root / "knowledge" / "rag.index"
    chunks_path = project_root / "knowledge" / "rag_chunks.pkl"
    assert index_path.exists() and chunks_path.exists(), (
        "Run `python scripts/build_rag_index.py` before this test."
    )
    index, chunks = load_index(index_path, chunks_path)

    def retriever(q: str) -> list[dict]:
        return retrieve(q, index, chunks, k=3)

    hyp_step = Hypothesize(llm=llm_hyp, retriever=retriever, rag_k=3)
    exp_step = ExperimentStep(
        llm=llm_code,
        session_root=tmp_path / "session",
        dataset_path=sample_csv,
        schema_summary=schema_summary,
        max_retries=3,
        timeout_s=30,
    )
    eval_step = Evaluate(llm=llm_eval)

    captured: dict = {}

    @observe(name="phase4.e2e")
    def _run_iteration():
        h = hyp_step(trace, None)
        e = exp_step(h)
        f = eval_step(h, e)
        try:
            from langfuse import get_client  # type: ignore

            client = get_client()
            try:
                client.update_current_trace(session_id="phase4-e2e")
            except Exception:
                pass
            try:
                captured["trace_url"] = client.get_trace_url()
            except Exception:
                captured["trace_url"] = None
        except ImportError:
            captured["trace_url"] = None
        return h, e, f

    hyp, exp, fb = _run_iteration()

    # If experiment failed, dump every attempt's stderr so failures are
    # debuggable from CI output before the assertion fires.
    if not exp.succeeded:
        print("\n[phase4-e2e] EXPERIMENT FAILED — attempt summaries:")
        for a in exp.attempts:
            print(
                f"  attempt {a.attempt_idx}: exit={a.exit_code} blocked={a.blocked_reason} "
                f"timed_out={a.timed_out} oom={a.oom_killed}"
            )
            print(f"    stderr tail: {a.stderr[-500:]!r}")

    trace.add_node(hyp)
    trace.update_experiment(hyp.id, exp)
    trace.update_feedback(hyp.id, fb)

    out = tmp_path / "trace.json"
    trace.save(out)
    loaded = DAGTrace.load(out)
    assert loaded.get(hyp.id) == trace.get(hyp.id), "save/load round-trip must preserve everything"

    assert hyp.id == exp.hypothesis_id == fb.hypothesis_id
    assert exp.succeeded is True, (
        f"experiment didn't succeed within {exp_step.max_retries + 1} attempts"
    )
    assert any(bool(a.metrics) for a in exp.attempts), "at least one attempt must have non-empty metrics"
    assert fb.decision in {"confirmed", "rejected", "inconclusive", "invalid"}
    assert 0.0 <= fb.confidence <= 1.0

    try:
        from langfuse import get_client  # type: ignore

        get_client().flush()
    except ImportError:
        pass
    print(f"\n[phase4-e2e] Langfuse trace URL: {captured.get('trace_url')}")
    print(f"[phase4-e2e] attempts={len(exp.attempts)}  succeeded={exp.succeeded}")
    print(f"[phase4-e2e] decision={fb.decision}  confidence={fb.confidence}")
    print(f"[phase4-e2e] hypothesis statement: {hyp.statement}")
