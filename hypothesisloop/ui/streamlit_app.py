"""HypothesisLoop Streamlit UI — mission control.

Run:
    streamlit run hypothesisloop/ui/streamlit_app.py

Phase 9 mode-aware rewrite:
- Mode toggle (Explore / Predict) drives the rest of the form. Predict mode
  shows target dropdown + task-type radio + AutoML time budget; Explore
  shows the question textarea.
- File-upload widget accepts a CSV; target dropdown auto-populates with the
  uploaded file's columns.
- Provider radio + API-key input + model dropdown — runtime overrides flow
  through ``factory.build_steps`` -> ``dispatch.get_llm`` without touching env.
- Cost tracker is the source of truth for live token / cost metrics.
- Per-iteration metrics table sits below the timeline cards in Predict mode.
- ↻ NEW RUN sticky button + ↻ CONTINUE +5 ITERS after completion.
"""

from __future__ import annotations

import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)

import pandas as pd
import streamlit as st

from hypothesisloop.agent.factory import build_steps
from hypothesisloop.agent.loop import _execute_iteration
from hypothesisloop.agent.state import DAGTrace
from hypothesisloop.llm.cost_tracker import CostTracker
from hypothesisloop.steps.profile import profile_dataset
from hypothesisloop.steps.report import render_report
from hypothesisloop.trace.langfuse_client import start_session
from hypothesisloop.ui.theme import (
    DECISION_BADGE_CLASS,
    DECISION_CARD_CLASS,
    inject_css,
)


st.set_page_config(
    page_title="HypothesisLoop",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()


# Provider → list of model names supported by the model dropdown.
PROVIDER_MODELS = {
    "moonshot": ["moonshot-v1-128k", "moonshot-v1-32k", "moonshot-v1-8k"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
}


# ---------------------------------------------------------------------------
# session-state init
# ---------------------------------------------------------------------------
def _zero_metrics() -> dict:
    return {
        "tokens_total": 0,
        "tokens_input": 0,
        "tokens_output": 0,
        "cost_usd": 0.0,
        "wall_time_s": 0.0,
        "calls": 0,
    }


def _init_state() -> None:
    defaults = {
        "phase": "config",         # "config" | "running" | "paused" | "complete"
        "trace": None,
        "session_root": None,
        "components": None,
        "iter_idx": 0,
        "max_iters": 5,
        "redirect_open": False,
        "last_error": None,
        "metrics_cache": _zero_metrics(),
        "cost_tracker": None,
        "predict_state": None,
        "automl_input": None,
        "automl_summary": None,
        "wall_start": None,
        # Config form state
        "uploaded_df": None,
        "uploaded_data_path": None,
        "uploaded_columns": [],
        "config_mode": "explore",
        "config_provider": "moonshot",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _refresh_metrics_cache() -> None:
    s = st.session_state
    tracker: Optional[CostTracker] = s.get("cost_tracker")
    wall_start = s.get("wall_start")
    wall = (
        (datetime.now(timezone.utc) - wall_start).total_seconds()
        if wall_start is not None
        else 0.0
    )
    if tracker is None:
        s["metrics_cache"] = {**_zero_metrics(), "wall_time_s": wall}
        return
    s["metrics_cache"] = {
        "tokens_total": tracker.total_tokens,
        "tokens_input": tracker.total_input_tokens,
        "tokens_output": tracker.total_output_tokens,
        "cost_usd": tracker.total_cost_usd,
        "wall_time_s": wall,
        "calls": tracker.total_calls,
    }


# ---------------------------------------------------------------------------
# top bar
# ---------------------------------------------------------------------------
def _render_topbar() -> None:
    s = st.session_state
    state_attr = {
        "config": "idle",
        "running": "running",
        "paused": "running",
        "complete": "complete",
    }.get(s.phase, "idle")

    cols = st.columns([3, 2, 2, 2, 2])
    with cols[0]:
        st.markdown('<div class="hl-brand">◐ HYPOTHESISLOOP</div>', unsafe_allow_html=True)

    if s.trace is None:
        with cols[4]:
            st.markdown(
                f'<div class="hl-stat hl-status" data-state="{state_attr}">'
                f'<span class="hl-stat-label">STATUS</span><code>READY</code></div>',
                unsafe_allow_html=True,
            )
        return

    mode_label = (s.trace.mode or "explore").upper()
    with cols[1]:
        st.markdown(
            f'<div class="hl-stat"><span class="hl-stat-label">SESSION</span>'
            f'<code>{s.trace.session_id[-12:]}</code></div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f'<div class="hl-stat"><span class="hl-stat-label">MODE</span>'
            f'<code>{mode_label}</code></div>',
            unsafe_allow_html=True,
        )
    with cols[3]:
        completed = s.trace.iteration_count()
        st.markdown(
            f'<div class="hl-stat"><span class="hl-stat-label">ITERS</span>'
            f'<code>{completed} / {s.max_iters}</code></div>',
            unsafe_allow_html=True,
        )
    with cols[4]:
        status_label = {
            "config": "READY",
            "running": "RUNNING",
            "paused": "PAUSED",
            "complete": "COMPLETE",
        }[s.phase]
        st.markdown(
            f'<div class="hl-stat hl-status" data-state="{state_attr}">'
            f'<span class="hl-stat-label">STATUS</span><code>{status_label}</code></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    s = st.session_state
    with st.sidebar:
        # NEW RUN — present after a run starts.
        if s.phase != "config":
            st.markdown('<div class="hl-newrun">', unsafe_allow_html=True)
            if st.button(
                "↻ NEW RUN",
                use_container_width=True,
                type="primary",
                key="hl_new_run_btn",
            ):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="hl-section">RUN CONFIG</div>', unsafe_allow_html=True)
        if s.phase == "config":
            _render_config_form()
        else:
            preview_q = (s.trace.question or "")[:80]
            preview = f"dataset : {s.trace.dataset_path}\nquestion: {preview_q}"
            if s.trace.mode == "predict":
                preview += (
                    f"\ntarget  : {s.trace.target_column}"
                    f"\ntask    : {s.trace.task_type}"
                    f"\nmetric  : {s.trace.metric_name}"
                )
            st.markdown(f"<pre>{preview}</pre>", unsafe_allow_html=True)

        st.markdown('<div class="hl-section">LIVE METRICS</div>', unsafe_allow_html=True)
        m = s.get("metrics_cache", _zero_metrics())
        rejects = len(s.trace.novelty_rejected) if s.trace is not None else 0
        st.markdown(
            (
                f'<pre class="hl-metrics">'
                f"calls  : {int(m['calls']):>8,}\n"
                f"tokens : {int(m['tokens_total']):>8,}\n"
                f"cost   : ${float(m['cost_usd']):>7.4f}\n"
                f"wall   : {float(m['wall_time_s']):>7.1f}s\n"
                f"rejects: {rejects:>8}"
                f"</pre>"
            ),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="hl-section">SESSION</div>', unsafe_allow_html=True)
        if s.phase == "complete" and s.session_root is not None:
            md_path = s.session_root / "report.md"
            txt_path = s.session_root / "report.txt"
            lb_path = s.session_root / "leaderboard.csv"
            fi_path = s.session_root / "feature_importance.csv"
            features_path = s.session_root / "features.parquet"

            if md_path.exists():
                st.download_button(
                    "⬇ report.md",
                    md_path.read_bytes(),
                    file_name="report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            if txt_path.exists():
                st.download_button(
                    "⬇ report.txt",
                    txt_path.read_bytes(),
                    file_name="report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            if lb_path.exists():
                st.download_button(
                    "⬇ leaderboard.csv",
                    lb_path.read_bytes(),
                    file_name="leaderboard.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            if fi_path.exists():
                st.download_button(
                    "⬇ feature_importance.csv",
                    fi_path.read_bytes(),
                    file_name="feature_importance.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            if features_path.exists():
                st.download_button(
                    "⬇ features.parquet",
                    features_path.read_bytes(),
                    file_name="features.parquet",
                    mime="application/octet-stream",
                    use_container_width=True,
                )

            if st.button(
                "↻ CONTINUE +5 ITERS",
                use_container_width=True,
                key="hl_continue_5",
            ):
                _continue_plus_5()
        else:
            st.markdown(
                '<div class="hl-empty">downloads after run completes</div>',
                unsafe_allow_html=True,
            )


def _render_config_form() -> None:
    s = st.session_state
    mode = st.radio(
        "MODE",
        ["explore", "predict"],
        index=0 if s.config_mode == "explore" else 1,
        horizontal=True,
        key="cfg_mode_radio",
    )
    s.config_mode = mode

    # CSV upload — populates the target dropdown.
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="cfg_csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            s.uploaded_df = df
            s.uploaded_columns = list(df.columns)
            s.uploaded_data_path = uploaded.name
        except Exception as e:
            st.error(f"could not read CSV: {e}")

    # Manual fallback: a path field for users who don't want to upload.
    data_path = st.text_input(
        "or path to CSV",
        value=s.uploaded_data_path or "data/adult.csv",
        key="cfg_data_path",
    )

    # Mode-dependent inputs.
    target = None
    task_type = "auto"
    question = "What demographic factors most predict income > $50K?"
    if mode == "predict":
        cols = s.uploaded_columns
        if not cols and Path(data_path).exists():
            try:
                cols = list(pd.read_csv(data_path, nrows=1).columns)
                s.uploaded_columns = cols
            except Exception:
                cols = []
        if cols:
            default_idx = max(0, len(cols) - 1)  # last column heuristic
            target = st.selectbox("Target column", cols, index=default_idx, key="cfg_target")
        else:
            target = st.text_input(
                "Target column (no CSV loaded; enter the name)",
                key="cfg_target_text",
            )
        task_type = st.radio(
            "Task type",
            ["auto", "classification", "regression"],
            index=0,
            horizontal=True,
            key="cfg_task_type",
        )
    else:
        question = st.text_area(
            "Question",
            value="What demographic factors most predict income > $50K?",
            key="cfg_question",
        )

    # Provider + API key + model.
    provider = st.radio(
        "Provider",
        ["moonshot", "openai"],
        index=0 if s.config_provider == "moonshot" else 1,
        horizontal=True,
        key="cfg_provider",
    )
    s.config_provider = provider
    api_key = st.text_input(
        "API key",
        value="",
        type="password",
        help="Not stored anywhere — re-enter on each session.",
        key="cfg_api_key",
    )
    st.caption("Key is held in this Streamlit session only — never written to disk or logged.")
    model = st.selectbox(
        "Model",
        PROVIDER_MODELS[provider],
        key="cfg_model",
    )

    max_iters = st.slider("Max iterations", 1, 10, 5, key="cfg_max_iters")
    auto_run = st.checkbox("Auto-run (no HITL pause)", value=True, key="cfg_auto_run")
    seed = int(st.number_input("Seed", value=42, key="cfg_seed"))

    automl_budget = 120
    if mode == "predict":
        automl_budget = st.select_slider(
            "AutoML time budget (s)",
            options=[60, 120, 300, 600],
            value=120,
            key="cfg_automl_budget",
        )

    if st.button("▶ START RUN", use_container_width=True, type="primary"):
        _start_run(
            mode=mode,
            data_path=data_path,
            target=target,
            task_type=task_type,
            question=question,
            provider=provider,
            api_key=api_key or None,
            model=model,
            max_iters=int(max_iters),
            auto_run=bool(auto_run),
            seed=seed,
            automl_budget=int(automl_budget),
        )


# ---------------------------------------------------------------------------
# main column — timeline + per-iteration metrics + action bar
# ---------------------------------------------------------------------------
def _render_main() -> None:
    s = st.session_state
    if s.trace is None:
        st.markdown('<div class="hl-empty">awaiting run config</div>', unsafe_allow_html=True)
        return
    if s.last_error:
        st.error(s.last_error)

    nodes = s.trace.iter_nodes()
    if not nodes:
        st.markdown('<div class="hl-empty">no iterations yet</div>', unsafe_allow_html=True)
    else:
        for node in nodes:
            _render_iteration_card(node)

        # Predict mode: per-iteration metrics table beneath the cards.
        if s.trace.mode == "predict":
            _render_predict_metrics_table()

    _render_action_bar()


def _is_plot_safe(value) -> bool:
    if not isinstance(value, (int, float)):
        return False
    if isinstance(value, bool):
        return False
    return not math.isinf(value) and not math.isnan(value)


def _render_iteration_card(node) -> None:
    fb = node.feedback
    decision = fb.decision if fb else "pending"
    card_class = DECISION_CARD_CLASS.get(decision, "")
    badge_class = DECISION_BADGE_CLASS.get(decision, "")
    badge_glyph = {
        "confirmed": "✓",
        "rejected": "✗",
        "inconclusive": "⚠",
        "invalid": "✕",
    }.get(decision, "○")
    decision_label = decision.upper() if fb else "PENDING"
    confidence = f"c={fb.confidence:.2f}" if fb else ""
    re_ex = " · re-explored" if node.hypothesis.re_explore else ""
    duration = (
        f"{node.experiment.attempts[-1].duration_s:.1f}s"
        if node.experiment and node.experiment.attempts
        else "—"
    )
    n_attempts = (
        len(node.experiment.attempts) if node.experiment and node.experiment.attempts else 0
    )
    attempts_str = (
        f"{n_attempts} attempt{'s' if n_attempts != 1 else ''}" if n_attempts else "—"
    )

    card_classes = "hl-iter-card" + (f" {card_class}" if card_class else "")
    badge_classes = "hl-iter-badge" + (f" {badge_class}" if badge_class else "")
    st.markdown(
        f"""
        <div class="{card_classes}">
          <div class="hl-iter-header">
            <span class="hl-iter-num">ITER {node.iteration:03d}</span>
            <span class="hl-iter-meta">{duration} · {attempts_str}{re_ex}</span>
            <span class="{badge_classes}">{badge_glyph} {decision_label} {confidence}</span>
          </div>
          <div class="hl-iter-statement">{node.hypothesis.statement}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if node.experiment is None or not node.experiment.attempts:
        return

    last = node.experiment.attempts[-1]
    is_predict = st.session_state.trace.mode == "predict"
    tab_labels = ["CODE", "OUTPUT", "METRICS", "EVALUATOR"]
    if is_predict:
        tab_labels.append("FEATURE")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        st.code(last.code or "(no code)", language="python")
    with tabs[1]:
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**stdout**")
            st.code(last.stdout or "(empty)", language=None)
        with cols[1]:
            st.markdown("**stderr**")
            st.code(last.stderr or "(empty)", language=None)
    with tabs[2]:
        numeric_metrics = {
            k: float(v) for k, v in (last.metrics or {}).items() if _is_plot_safe(v)
        }
        if numeric_metrics:
            st.bar_chart(numeric_metrics)
        else:
            st.markdown(
                '<div class="hl-empty">no numeric metrics emitted</div>',
                unsafe_allow_html=True,
            )
        if last.metrics:
            st.json(last.metrics)
        for fig_path_str in last.figures:
            fig_path = Path(fig_path_str)
            if fig_path.exists():
                st.image(str(fig_path))
    with tabs[3]:
        if fb is not None:
            st.markdown(f"**Decision:** `{fb.decision}` (confidence: {fb.confidence:.2f})")
            st.markdown(f"**Reason:** {fb.reason}")
            st.markdown(f"**Observations:** {fb.observations}")
            if fb.bias_flags:
                st.warning(f"⚠ {len(fb.bias_flags)} bias flag(s) raised")
            if fb.novel_subhypotheses:
                st.markdown("**Suggested follow-ups:**")
                for sh in fb.novel_subhypotheses:
                    st.markdown(f"- {sh}")
        else:
            st.markdown(
                '<div class="hl-empty">no feedback yet</div>', unsafe_allow_html=True
            )

    if is_predict:
        with tabs[4]:
            features = [
                f for f in st.session_state.trace.engineered_features
                if f.hypothesis_id == node.hypothesis.id
            ]
            if not features:
                st.markdown(
                    '<div class="hl-empty">no engineered feature recorded for this iteration</div>',
                    unsafe_allow_html=True,
                )
            else:
                f = features[0]
                st.markdown(
                    f"**Feature:** `{f.name}`  ·  "
                    f"**Predicted Δ:** `{f.predicted_delta:+.4f}`  ·  "
                    f"**Actual Δ:** `{f.actual_delta:+.4f}`  ·  "
                    f"**Kept:** {'✅' if f.accepted else '❌'}"
                )
                if f.rejection_reason:
                    st.caption(f.rejection_reason)


def _render_predict_metrics_table() -> None:
    s = st.session_state
    features = s.trace.engineered_features
    if not features:
        return
    rows = []
    running = s.trace.baseline_score or 0.0
    for f in features:
        if f.accepted:
            running += f.actual_delta
        rows.append(
            {
                "iter": f.iteration_added,
                "feature": f.name,
                "predicted Δ": f"{f.predicted_delta:+.4f}",
                "actual Δ": f"{f.actual_delta:+.4f}",
                "kept": "✅" if f.accepted else "❌",
                "running score": f"{running:.4f}" if f.accepted else "(unchanged)",
            }
        )
    st.markdown(
        '<div class="hl-section" style="margin-top:24px;">FEATURE-ENGINEERING LEDGER</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_action_bar() -> None:
    s = st.session_state
    if s.phase != "paused":
        return

    st.markdown('<div class="hl-actionbar">', unsafe_allow_html=True)
    cols = st.columns([2, 2, 2, 6])
    with cols[0]:
        if st.button("▶ CONTINUE", use_container_width=True, type="primary", key="hl_continue"):
            _continue()
    with cols[1]:
        if st.button("■ STOP", use_container_width=True, key="hl_stop"):
            _stop()
    with cols[2]:
        if st.button("↳ REDIRECT…", use_container_width=True, key="hl_redirect_toggle"):
            s.redirect_open = not s.redirect_open
    with cols[3]:
        if s.redirect_open:
            new_hyp = st.text_input(
                "inject hypothesis",
                key="hl_redirect_input",
                label_visibility="collapsed",
                placeholder="propose a different direction…",
            )
            if st.button("INJECT & CONTINUE", use_container_width=True, key="hl_redirect_go"):
                _redirect(new_hyp)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# run lifecycle
# ---------------------------------------------------------------------------
def _new_session_id() -> str:
    return f"hl-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def _start_run(
    *,
    mode: str,
    data_path: str,
    target: Optional[str],
    task_type: str,
    question: str,
    provider: str,
    api_key: Optional[str],
    model: str,
    max_iters: int,
    auto_run: bool,
    seed: int,
    automl_budget: int,
) -> None:
    s = st.session_state

    # Resolve dataframe: uploaded takes precedence, else read from path.
    df: Optional[pd.DataFrame] = None
    if s.uploaded_df is not None and (s.uploaded_data_path == data_path or data_path == ""):
        df = s.uploaded_df.copy()
    else:
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            st.error(f"could not read dataset at {data_path}: {e}")
            return

    if mode == "predict":
        if not target or target not in df.columns:
            st.error(f"target column {target!r} not in dataset columns: {sorted(df.columns)}")
            return

    schema = profile_dataset(df, dataset_path=data_path)

    session_id = _new_session_id()
    session_root = Path("reports") / session_id
    session_root.mkdir(parents=True, exist_ok=True)

    # Persist uploaded df under the session for reproducibility.
    if data_path == s.uploaded_data_path or s.uploaded_df is not None:
        canonical_path = session_root / "data.csv"
        df.to_csv(canonical_path, index=False)
        dataset_path_str = str(canonical_path.resolve())
    else:
        dataset_path_str = str(Path(data_path).resolve())

    # Derive task type / metric for predict mode.
    resolved_task: Optional[str] = None
    resolved_metric: Optional[str] = None
    if mode == "predict":
        from hypothesisloop.steps.baseline import auto_metric_for, auto_task_type

        if task_type == "auto":
            resolved_task = auto_task_type(df[target])
        else:
            resolved_task = task_type
        resolved_metric = auto_metric_for(resolved_task)

    trace = DAGTrace(
        session_id=session_id,
        dataset_path=dataset_path_str,
        question=question if mode == "explore" else f"Predict {target}",
        schema_summary=schema,
        mode=mode,
        target_column=(target if mode == "predict" else None),
        task_type=resolved_task,
        metric_name=resolved_metric,
    )

    tracker = CostTracker()
    predict_state = None
    automl_input = None
    if mode == "predict":
        try:
            from hypothesisloop.steps.baseline import run_baseline

            baseline = run_baseline(trace, df, seed=seed)
            predict_state = {
                "trace": trace,
                "train_df": baseline.train_df,
                "target_column": trace.target_column,
                "task_type": trace.task_type,
                "metric_name": trace.metric_name,
                "prev_score": baseline.baseline_cv,
                "seed": seed,
            }
            automl_input = {"test_df": baseline.test_df}
        except Exception as e:
            st.error(f"baseline failed: {type(e).__name__}: {e}")
            return

    try:
        components = build_steps(
            trace=trace,
            session_root=session_root,
            model=model,
            seed=seed,
            api_key=api_key,
            tracker=tracker,
            predict_state=predict_state,
        )
    except Exception as e:
        st.error(f"could not build agent components: {e}")
        return

    start_session(session_id)

    s.trace = trace
    s.session_root = session_root
    s.components = components
    s.cost_tracker = tracker
    s.predict_state = predict_state
    s.automl_input = automl_input
    s.max_iters = max_iters
    s.iter_idx = 0
    s.phase = "running"
    s.redirect_open = False
    s.last_error = None
    s.metrics_cache = _zero_metrics()
    s.wall_start = datetime.now(timezone.utc)
    s["_auto_run"] = auto_run
    s["_automl_budget"] = automl_budget
    s["_provider"] = provider
    s["_model"] = model
    s["_api_key"] = api_key

    _run_one_iteration()
    if s.phase == "paused" and s.get("_auto_run"):
        # Auto mode — keep firing iterations until done.
        _drive_auto()
    st.rerun()


def _run_one_iteration() -> None:
    s = st.session_state
    s.phase = "running"
    try:
        with st.status(f"iter {s.iter_idx + 1} in progress", expanded=True) as status:
            status.write("→ hypothesizing → generating code → running sandbox → evaluating")
            _execute_iteration(
                s.iter_idx,
                trace=s.trace,
                scheduler=s.components["scheduler"],
                hypothesize_fn=s.components["hypothesize_fn"],
                experiment_fn=s.components["experiment_fn"],
                evaluate_fn=s.components["evaluate_fn"],
                learn_fn=None,
                novelty_fn=s.components["novelty_fn"],
                hitl_fn=None,
                safety_fn=s.components["safety_fn"],
            )
            status.update(label=f"iter {s.iter_idx + 1} complete", state="complete")
    except Exception as e:
        s.last_error = f"iteration {s.iter_idx + 1} failed: {type(e).__name__}: {e}"

    s.iter_idx += 1
    try:
        s.trace.save(s.session_root / "trace.json")
    except Exception as e:
        s.last_error = f"trace save failed: {e}"

    _refresh_metrics_cache()

    if s.iter_idx >= s.max_iters or s.last_error:
        _complete()
    else:
        s.phase = "paused"


def _drive_auto() -> None:
    """Auto-mode: keep stepping until budget is hit or an error stops us."""
    s = st.session_state
    while s.phase == "paused" and s.iter_idx < s.max_iters:
        _run_one_iteration()


def _continue() -> None:
    _run_one_iteration()
    st.rerun()


def _continue_plus_5() -> None:
    s = st.session_state
    s.max_iters += 5
    s.phase = "paused"
    if s.get("_auto_run"):
        _drive_auto()
    else:
        _run_one_iteration()
    st.rerun()


def _stop() -> None:
    _complete()
    st.rerun()


def _redirect(text: str) -> None:
    s = st.session_state
    if text and s.components is not None:
        s.components["scheduler"].inject(text)
    s.redirect_open = False
    _run_one_iteration()
    if s.phase == "paused" and s.get("_auto_run"):
        _drive_auto()
    st.rerun()


def _complete() -> None:
    s = st.session_state
    s.phase = "complete"
    if s.trace is None or s.session_root is None:
        return
    try:
        s.trace.save(s.session_root / "trace.json")
    except Exception as e:
        s.last_error = f"trace save failed: {e}"

    # Predict mode: train AutoGluon if we have the test split + at least the baseline.
    if (
        s.trace.mode == "predict"
        and s.predict_state is not None
        and s.automl_input is not None
        and s.automl_summary is None  # don't re-train on subsequent _complete calls
    ):
        try:
            from hypothesisloop.automl.autogluon_runner import (
                run_automl,
                write_automl_summary,
            )

            with st.status("training AutoGluon ensemble", expanded=True) as status:
                status.write(f"→ time budget: {s.get('_automl_budget', 120)}s")
                summary = run_automl(
                    train_df=s.predict_state["train_df"],
                    test_df=s.automl_input["test_df"],
                    target_column=s.trace.target_column,
                    task_type=s.trace.task_type,
                    output_dir=s.session_root,
                    time_budget_s=s.get("_automl_budget", 120),
                    seed=42,
                )
                write_automl_summary(summary, s.session_root)
                s.automl_summary = summary
                status.update(
                    label=f"AutoGluon test {summary['test_metric']}={summary['test_score']:.4f}",
                    state="complete",
                )
        except ImportError as e:
            s.last_error = f"AutoGluon unavailable: {e}"
        except Exception as e:
            s.last_error = f"AutoGluon training failed: {type(e).__name__}: {e}"

    try:
        render_report(
            s.trace,
            output_dir=s.session_root,
            format="both",
            cli_command="streamlit run hypothesisloop/ui/streamlit_app.py",
            seed=42,
        )
    except Exception as e:
        s.last_error = f"report rendering failed: {e}"
    _refresh_metrics_cache()


# ---------------------------------------------------------------------------
# page render
# ---------------------------------------------------------------------------
_render_topbar()
_render_sidebar()
_render_main()
