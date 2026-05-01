"""HypothesisLoop CLI — main entry point.

Wires every component (profile -> hypothesize -> experiment -> evaluate, plus
novelty, pruner, bias scanner) and runs ``agent.loop.run_loop`` to completion.

Run modes:
    python -m hypothesisloop.cli --smoke-test                     # one Kimi call, sanity
    python -m hypothesisloop.cli --auto --question "..."          # full unattended run
    python -m hypothesisloop.cli --question "..."                 # interactive HITL
    python -m hypothesisloop.cli --resume <session-id>            # continue saved trace
    python -m hypothesisloop.cli --report-only --resume <id>      # re-render report.md

Phase 8 lazy-imports the heavy modules (langchain, faiss, langfuse, pandas,
all the steps) inside the functions that need them. ``--help`` and
``--report-only`` no longer pay the full import cost.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# CRITICAL: must run before any module that reads env vars (dispatch, embed,
# trace). The user's shell may carry a stale OPENAI_API_KEY.
load_dotenv(override=True)


# Hardcoded mirror of ``hypothesisloop.llm.dispatch.HL_MODEL_DEFAULT`` — kept
# in sync manually so ``--help`` doesn't have to import dispatch (which pulls
# in langchain). If you change the default in dispatch.py, change it here too.
_DEFAULT_MODEL = "moonshot-v1-128k"
DEFAULT_RAG_INDEX = "knowledge/rag.index"
DEFAULT_RAG_CHUNKS = "knowledge/rag_chunks.pkl"


def _gen_session_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"hl-{ts}-{uuid.uuid4().hex[:4]}"


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hypothesisloop.cli",
        description=(
            "HypothesisLoop -- autonomous Hypothesize/Experiment/Evaluate/Learn agent."
        ),
    )
    p.add_argument(
        "--data",
        default="data/adult.csv",
        help="Path to a CSV dataset (default: data/adult.csv).",
    )
    p.add_argument(
        "--question",
        help="Research question for the agent. Required for real runs; optional with --smoke-test or --resume.",
    )
    p.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Chat model name (default: {_DEFAULT_MODEL} = Kimi K2.6).",
    )
    p.add_argument(
        "--max-iters",
        type=int,
        default=5,
        help="Maximum loop iterations (default: 5).",
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help="Skip per-iteration HITL pauses (run unattended).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--session-id",
        help="Langfuse session id (default: auto-generated hl-<timestamp>-<random4>).",
    )
    p.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="Resume an existing run by session id (loads <output_dir>/<id>/trace.json).",
    )
    p.add_argument(
        "--output-dir",
        default="reports",
        help="Where session artifacts go (default: reports/).",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Make one trivial @observed Kimi call, print the response and the Langfuse "
            "trace URL, then exit. Phase 0 acceptance test."
        ),
    )
    p.add_argument(
        "--rag-index",
        default=DEFAULT_RAG_INDEX,
        help="Path to FAISS index (default: knowledge/rag.index).",
    )
    p.add_argument(
        "--rag-chunks",
        default=DEFAULT_RAG_CHUNKS,
        help="Path to RAG chunks pickle (default: knowledge/rag_chunks.pkl).",
    )
    p.add_argument(
        "--rag-k",
        type=int,
        default=4,
        help="Top-k chunks retrieved per hypothesize call (default: 4).",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Re-render report.md and report.txt from a saved trace; requires --resume.",
    )
    # Phase 9 — Predict-mode flags.
    p.add_argument(
        "--mode",
        choices=["explore", "predict"],
        default="explore",
        help="Loop mode (default: explore).",
    )
    p.add_argument(
        "--target",
        default=None,
        help="Target column name (Predict mode only; required when --mode=predict).",
    )
    p.add_argument(
        "--task-type",
        choices=["classification", "regression", "auto"],
        default="auto",
        help="Predict-mode task type (default: auto-detected from target dtype).",
    )
    p.add_argument(
        "--provider",
        choices=["openai", "moonshot"],
        default=None,
        help="Override provider routing (default: inferred from --model prefix).",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Runtime API key override (rare — usually set via .env).",
    )
    p.add_argument(
        "--automl-time-budget",
        type=int,
        default=120,
        help="AutoGluon time budget in seconds (Predict mode; default: 120).",
    )
    return p


# ---------------------------------------------------------------------------
# Phase 0 smoke test
# ---------------------------------------------------------------------------
def _run_smoke_test(args) -> int:
    # Lazy-load: only --smoke-test needs Langfuse + dispatch.
    from hypothesisloop.llm.dispatch import get_llm
    from hypothesisloop.trace.langfuse_client import observe, start_session

    @observe(name="phase0_smoke_test")
    def _smoke_test_call(model: str, session_id: str) -> tuple[str, Optional[str]]:
        try:
            from langfuse import get_client  # type: ignore

            client = get_client()
            try:
                client.update_current_trace(session_id=session_id)
            except Exception:
                pass
        except Exception:
            client = None

        llm = get_llm(model=model, temperature=0.7).bind(max_tokens=10)
        response = llm.invoke("Reply with the word 'ok'.")
        text = getattr(response, "content", str(response))

        url: Optional[str] = None
        if client is not None:
            try:
                url = client.get_trace_url()
            except Exception:
                url = None
        return text, url

    session_id = args.session_id or _gen_session_id()
    start_session(session_id=session_id)
    print(f"[smoke-test] session_id = {session_id}")
    print(f"[smoke-test] model      = {args.model}")
    try:
        response, trace_url = _smoke_test_call(args.model, session_id)
    except Exception as exc:
        print(f"[smoke-test] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"[smoke-test] response   = {response!r}")
    try:
        from langfuse import get_client  # type: ignore

        client = get_client()
        if trace_url is None:
            try:
                trace_url = client.get_trace_url()
            except Exception:
                pass
        client.flush()
    except Exception:
        pass

    if trace_url:
        print(f"[smoke-test] trace_url  = {trace_url}")
    else:
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        print(
            f"[smoke-test] trace_url  = (not captured; check {host} for session {session_id})"
        )
    print("[smoke-test] OK")
    return 0


# ---------------------------------------------------------------------------
# --report-only
# ---------------------------------------------------------------------------
def _run_report_only(args) -> int:
    if not args.resume:
        raise SystemExit("--report-only requires --resume <session-id>.")
    session_root = Path(args.output_dir) / args.resume
    trace_path = session_root / "trace.json"
    if not trace_path.exists():
        raise SystemExit(f"--report-only: trace not found at {trace_path}")

    # Lazy-load: report-only doesn't need langchain / faiss / pandas.
    from hypothesisloop.agent.state import DAGTrace
    from hypothesisloop.steps.report import render_report

    trace = DAGTrace.load(trace_path)
    # --report-only skips the Langfuse usage rollup (~7s API call) — the
    # user is iterating on the report; the original run already paid for that.
    empty_usage = {
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost_usd": 0.0,
        "wall_time_s": 0.0,
        "trace_count": 0,
        "_error": "skipped on --report-only (run without --report-only to refresh)",
    }
    result = render_report(
        trace,
        output_dir=session_root,
        format="both",
        cli_command=None,
        seed=args.seed,
        usage_override=empty_usage,
    )
    print(f"[hypothesisloop] report -> {result['md']}", file=sys.stderr)
    print(f"[hypothesisloop] report -> {result['txt']}", file=sys.stderr)
    if result["bias_flags"]:
        print(
            f"[hypothesisloop] bias flags raised: {len(result['bias_flags'])}",
            file=sys.stderr,
        )
    return 0


# ---------------------------------------------------------------------------
# Real-run helpers
# ---------------------------------------------------------------------------
def _load_or_create_trace(args):
    """Return ``(trace, session_root, df)``.

    On ``--resume``: loads ``<output_dir>/<id>/trace.json`` and reads the
    dataset from ``trace.dataset_path``. Otherwise: profiles the CSV, stamps
    the schema summary onto a new trace.
    """
    # Lazy-load — pandas + the state module + the profiler.
    import pandas as pd

    from hypothesisloop.agent.state import DAGTrace
    from hypothesisloop.steps.profile import profile_dataset

    if args.resume:
        session_id = args.resume
        session_root = Path(args.output_dir) / session_id
        trace_path = session_root / "trace.json"
        if not trace_path.exists():
            raise SystemExit(f"--resume: trace file not found at {trace_path}")
        trace = DAGTrace.load(trace_path)
        df = pd.read_csv(trace.dataset_path)
        return trace, session_root, df

    if args.mode == "predict":
        if not args.target:
            raise SystemExit("--mode=predict requires --target <column>.")
        # Predict-mode: question text is auxiliary; default if absent.
        question = args.question or f"Predict {args.target}"
    else:
        if not args.question:
            raise SystemExit(
                "--question is required for non-resume, non-smoke-test runs."
            )
        question = args.question

    session_id = args.session_id or _gen_session_id()
    session_root = Path(args.output_dir) / session_id
    session_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    schema = profile_dataset(df, dataset_path=args.data)

    task_type = None
    metric_name = None
    if args.mode == "predict":
        if args.target not in df.columns:
            raise SystemExit(
                f"--target {args.target!r} not in dataset columns: {sorted(df.columns)}"
            )
        if args.task_type == "auto":
            from hypothesisloop.steps.baseline import auto_metric_for, auto_task_type

            task_type = auto_task_type(df[args.target])
            metric_name = auto_metric_for(task_type)
        else:
            task_type = args.task_type
            metric_name = "roc_auc" if task_type == "classification" else "r2"

    trace = DAGTrace(
        session_id=session_id,
        dataset_path=str(Path(args.data).resolve()),
        question=question,
        schema_summary=schema,
        mode=args.mode,
        target_column=(args.target if args.mode == "predict" else None),
        task_type=task_type,
        metric_name=metric_name,
    )
    return trace, session_root, df


def _build_steps(args, trace, session_root, *, tracker=None, predict_state=None):
    """Thin wrapper over :func:`hypothesisloop.agent.factory.build_steps`."""
    from hypothesisloop.agent.factory import build_steps

    components = build_steps(
        trace=trace,
        session_root=session_root,
        model=args.model,
        seed=args.seed,
        rag_index_path=args.rag_index,
        rag_chunks_path=args.rag_chunks,
        rag_k=args.rag_k,
        api_key=args.api_key,
        tracker=tracker,
        predict_state=predict_state,
    )
    return (
        components["hypothesize_fn"],
        components["experiment_fn"],
        components["evaluate_fn"],
        components["novelty_fn"],
        components["scheduler"],
        components["safety_fn"],
    )


def _run(args) -> int:
    """Real-run path. Wraps the loop in @observe so step spans nest under one trace."""
    # Lazy-load — the loop, hitl, report, langfuse all live behind this function.
    from hypothesisloop.agent.loop import run_loop
    from hypothesisloop.llm.cost_tracker import CostTracker
    from hypothesisloop.steps.report import render_report
    from hypothesisloop.trace.langfuse_client import observe, start_session
    from hypothesisloop.ui.hitl import hitl_prompt, print_run_summary

    random.seed(args.seed)

    trace, session_root, df = _load_or_create_trace(args)
    print(f"[hypothesisloop] session_id = {trace.session_id}", file=sys.stderr)
    print(f"[hypothesisloop] artifacts  -> {session_root}", file=sys.stderr)
    if trace.mode == "predict":
        print(
            f"[hypothesisloop] mode       = predict (target={trace.target_column}, "
            f"task={trace.task_type}, metric={trace.metric_name})",
            file=sys.stderr,
        )

    start_session(trace.session_id)

    tracker = CostTracker()

    # Predict mode: run iter-0 baseline before the loop. The shared
    # ``predict_state`` dict gives Evaluate the train_df + prev_score it
    # needs to compute deterministic accept/reject deltas.
    predict_state = None
    automl_input: dict | None = None
    if trace.mode == "predict":
        from hypothesisloop.steps.baseline import run_baseline

        baseline = run_baseline(trace, df, seed=args.seed)
        print(
            f"[hypothesisloop] baseline   = {trace.metric_name}={baseline.baseline_cv.value:.4f} "
            f"({baseline.baseline_cv.n_folds}-fold CV)",
            file=sys.stderr,
        )
        predict_state = {
            "trace": trace,
            "train_df": baseline.train_df,
            "target_column": trace.target_column,
            "task_type": trace.task_type,
            "metric_name": trace.metric_name,
            "prev_score": baseline.baseline_cv,
            "seed": args.seed,
        }
        automl_input = {
            "train_df_baseline": baseline.train_df,
            "test_df": baseline.test_df,
        }

    hyp_fn, exp_fn, eval_fn, nov_fn, scheduler, safety_fn = _build_steps(
        args, trace, session_root, tracker=tracker, predict_state=predict_state
    )
    hitl_fn = None if args.auto else hitl_prompt

    # --max-iters is the *session* budget, not the per-invocation budget.
    already_run = trace.iteration_count()
    remaining = max(0, args.max_iters - already_run)
    if args.resume and remaining == 0:
        print(
            f"[hypothesisloop] trace already has {already_run} iterations; "
            f"--max-iters={args.max_iters} budget exhausted. Nothing to do.",
            file=sys.stderr,
        )

    @observe(name="loop.iteration")
    def _do_run():
        run_loop(
            trace=trace,
            scheduler=scheduler,
            hypothesize_fn=hyp_fn,
            experiment_fn=exp_fn,
            evaluate_fn=eval_fn,
            novelty_fn=nov_fn,
            hitl_fn=hitl_fn,
            safety_fn=safety_fn,
            max_iters=remaining,
        )

    try:
        _do_run()
    finally:
        try:
            trace.save(session_root / "trace.json")
        except Exception as e:
            print(f"[hypothesisloop] trace save failed: {e}", file=sys.stderr)

        # Predict mode: run AutoGluon on the engineered train_df + held-out
        # test split, write leaderboard + feature_importance + summary JSON.
        # Failure here is loud but non-fatal — the report still renders.
        if trace.mode == "predict" and predict_state is not None and automl_input is not None:
            try:
                from hypothesisloop.automl.autogluon_runner import (
                    run_automl,
                    write_automl_summary,
                )

                print(
                    f"[hypothesisloop] AutoGluon training "
                    f"(time_budget={args.automl_time_budget}s)...",
                    file=sys.stderr,
                )
                # The shared predict_state["train_df"] now contains every
                # accepted engineered feature; the test split was reserved at
                # baseline and didn't see any of it.
                summary = run_automl(
                    train_df=predict_state["train_df"],
                    test_df=automl_input["test_df"],
                    target_column=trace.target_column,
                    task_type=trace.task_type,
                    output_dir=session_root,
                    engineered_features=trace.engineered_features,
                    time_budget_s=args.automl_time_budget,
                    seed=args.seed,
                )
                write_automl_summary(summary, session_root)
                print(
                    f"[hypothesisloop] AutoGluon test {summary['test_metric']}="
                    f"{summary['test_score']:.4f} ({summary['best_model']})",
                    file=sys.stderr,
                )
            except ImportError as e:
                print(
                    f"[hypothesisloop] AutoGluon unavailable: {e}. "
                    "The proxy CV ledger in the report still reflects "
                    "feature engineering decisions.",
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f"[hypothesisloop] AutoGluon training failed: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )

        try:
            result = render_report(
                trace,
                output_dir=session_root,
                format="both",
                cli_command=" ".join(sys.argv),
                seed=args.seed,
            )
            print(f"[hypothesisloop] report -> {result['md']}", file=sys.stderr)
            print(f"[hypothesisloop] report -> {result['txt']}", file=sys.stderr)
            if result["bias_flags"]:
                print(
                    f"[hypothesisloop] bias flags raised: {len(result['bias_flags'])}",
                    file=sys.stderr,
                )
        except Exception as e:
            print(
                f"[hypothesisloop] report rendering failed: {e}",
                file=sys.stderr,
            )

        try:
            from langfuse import get_client  # type: ignore

            get_client().flush()
        except Exception:
            pass

    print_run_summary(trace)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.smoke_test:
        return _run_smoke_test(args)
    if args.report_only:
        return _run_report_only(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
