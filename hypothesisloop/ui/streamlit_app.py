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

MODEL_DISPLAY_NAMES = {
    "moonshot-v1-128k": "Kimi K2.6 (128K context)",
    "moonshot-v1-32k": "Kimi K2.6 (32K context)",
    "moonshot-v1-8k": "Kimi K2.6 (8K context)",
    "gpt-4o-mini": "GPT-4o mini",
    "gpt-4o": "GPT-4o",
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
        "pending_iteration": False,
        "progress_substep": "",
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

    # Row/col preview (fix #8) — show counts so the user knows the dataset was parsed.
    _preview_df = s.uploaded_df
    if _preview_df is None and data_path and Path(data_path).exists():
        try:
            _preview_df = pd.read_csv(data_path, nrows=200_000)
        except Exception:
            _preview_df = None
    if _preview_df is not None:
        st.caption(
            f"📊 dataset: {len(_preview_df):,} rows × {len(_preview_df.columns)} columns"
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
    # Fix #7 (round 2): make the selectbox key provider-specific. Streamlit's
    # selectbox with a stable key + format_func leaves the closed-dropdown
    # label stuck on the old value when the underlying options change. Using
    # a per-provider key forces a fresh widget render with the right default.
    model = st.selectbox(
        "Model",
        PROVIDER_MODELS[provider],
        key=f"cfg_model_{provider}",
        format_func=lambda m: MODEL_DISPLAY_NAMES.get(m, m),
    )

    max_iters = st.slider("Max iterations", 1, 10, 5, key="cfg_max_iters")
    auto_run = st.checkbox("Auto-run (no HITL pause)", value=True, key="cfg_auto_run")
    seed = 42  # pinned internally for reproducibility; not user-configurable in UI

    automl_budget = 120
    if mode == "predict":
        automl_budget = st.select_slider(
            "AutoML time budget (s)",
            options=[60, 120, 300, 600],
            value=120,
            key="cfg_automl_budget",
        )

    # Fix #5: gate START — explore mode requires a non-empty question; predict
    # requires a target column.
    blocking_reason = None
    if mode == "explore" and not (question or "").strip():
        blocking_reason = "Enter a research question to start an Explore run."
    elif mode == "predict" and not (target or "").strip():
        blocking_reason = "Pick a target column to start a Predict run."

    if blocking_reason:
        st.caption(f"⚠️ {blocking_reason}")

    if st.button(
        "▶ START RUN",
        use_container_width=True,
        type="primary",
        disabled=blocking_reason is not None,
    ):
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
def _render_progress_bar() -> None:
    """Main-area progress bar (fix #1, #2, #4 + user request).

    Replaces the cramped sidebar `st.status` blocks. Shows iter N/M, a phase
    sub-label, and a real progress meter. Visible only when a run is in flight
    or completed.
    """
    s = st.session_state
    if s.trace is None:
        return
    if s.phase not in ("running", "paused", "complete"):
        return

    completed = s.trace.iteration_count()
    total = max(1, int(s.max_iters or 1))

    if s.phase == "complete":
        label = f"Complete · {completed} / {total} iterations"
        sub = "All iterations done. Reports written to disk."
        pct = 1.0
    elif s.phase == "running":
        # Show the iteration that's currently in flight (1-indexed for humans).
        in_flight = min(completed + 1, total)
        label = f"Running iteration {in_flight} / {total}"
        sub = s.get("progress_substep", "→ hypothesizing → generating code → running sandbox → evaluating")
        pct = (completed) / total
    else:  # paused
        label = f"Paused after iteration {completed} / {total}"
        sub = "Review the latest card and click CONTINUE / STOP / REDIRECT below."
        pct = completed / total

    # Predict mode: after the last iteration, AutoGluon trains. Surface that
    # transition explicitly when we've left the loop and ensemble training is
    # underway.
    if (
        s.phase == "running"
        and s.trace.mode == "predict"
        and completed >= total
    ):
        label = "Training AutoGluon ensemble"
        sub = f"→ time budget: {s.get('_automl_budget', 120)}s"
        pct = 0.99

    st.markdown(f'<div class="hl-progress-block">', unsafe_allow_html=True)
    st.markdown(f'<div class="hl-progress-label">{label}</div>', unsafe_allow_html=True)
    st.progress(min(1.0, max(0.0, pct)))
    st.markdown(f'<div class="hl-progress-sub">{sub}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def _render_main() -> None:
    s = st.session_state
    if s.trace is None:
        st.markdown('<div class="hl-empty">awaiting run config</div>', unsafe_allow_html=True)
        return

    # Progress bar at the top of the timeline (fixes mid-run UX).
    _render_progress_bar()

    # Question-level bias banner (fix #6 round 2): if the user's question
    # itself frames causation about a sensitive variable, surface that as a
    # persistent top-of-run banner. This catches the demo case where Kimi
    # self-disciplines and the per-iteration bias scanner doesn't fire.
    qflags = s.get("question_bias_flags") or []
    if qflags:
        n = len(qflags)
        bullets = "<br>".join(
            f"&nbsp;&nbsp;• <b>{f['sensitive_var']}</b> + causal verb <code>{f['causal_verb']}</code>: \"{f['snippet'][:200]}\""
            for f in qflags
        )
        st.markdown(
            f'<div class="hl-bias-chip">⚠ Your question implied causation about a sensitive '
            f'variable ({n} match{"es" if n != 1 else ""}). The agent will produce '
            f'<b>correlational</b> findings only — never causal. Treat all results accordingly.<br>'
            f'{bullets}</div>',
            unsafe_allow_html=True,
        )

    # Sam's audit fix #10: off-topic question banner — surfaces above the
    # iteration cards so the user knows the question may not match the dataset.
    if s.get("question_offtopic"):
        st.markdown(
            f'<div class="hl-bias-chip" style="border-color:#FBBF24;color:#FDE68A;'
            f'background:rgba(251,191,36,0.08);">'
            f'⚠ <b>Off-topic question.</b> {s.get("question_offtopic_reason", "")}'
            f'</div>',
            unsafe_allow_html=True,
        )

    if s.last_error:
        st.error(s.last_error)

    nodes = s.trace.iter_nodes()
    if not nodes:
        st.markdown('<div class="hl-empty">no iterations yet — first one is in flight</div>', unsafe_allow_html=True)
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
    # Fix #6: bias chip surfaced directly on the card so the demo audience sees
    # the safety system fire without having to download the report.
    bias_html = ""
    if fb and fb.bias_flags:
        n = len(fb.bias_flags)
        flag_word = "flag" if n == 1 else "flags"
        bias_html = (
            f'<div class="hl-bias-chip">⚠ {n} bias {flag_word} raised — '
            f"causal claim about a sensitive variable. Interpret as correlational only; "
            f"see EVALUATOR tab for details.</div>"
        )
    st.markdown(
        f"""
        <div class="{card_classes}">
          <div class="hl-iter-header">
            <span class="hl-iter-num">ITER {node.iteration:03d}</span>
            <span class="hl-iter-meta">{duration} · {attempts_str}{re_ex}</span>
            <span class="{badge_classes}">{badge_glyph} {decision_label} {confidence}</span>
          </div>
          <div class="hl-iter-statement">{node.hypothesis.statement}</div>
          {bias_html}
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
        # Sam's audit fix #9: drop the bar chart that compared heterogeneous
        # quantities (n=32561 dwarfing p_value=0.005). Render as a labeled
        # table instead, with category hints.
        m = last.metrics or {}
        if not m:
            st.markdown(
                '<div class="hl-empty">no metrics emitted</div>',
                unsafe_allow_html=True,
            )
        else:
            _METRIC_CATEGORIES = {
                "p_value":       ("p-value",            "smaller is more significant"),
                "effect_size":   ("effect size",        "magnitude of the relationship"),
                "cohens_d":      ("Cohen's d",          "small=0.2 · medium=0.5 · large=0.8"),
                "eta_squared":   ("η²",                 "small=0.01 · medium=0.06 · large=0.14"),
                "cramers_v":     ("Cramer's V",         "small=0.1 · medium=0.3 · large=0.5"),
                "pearson_r":     ("Pearson r",          "small=0.1 · medium=0.3 · large=0.5"),
                "r2":            ("R²",                 "fraction of variance explained"),
                "cliffs_delta":  ("Cliff's delta",      "abs: small=0.11 · medium=0.28 · large=0.43"),
                "odds_ratio":    ("odds ratio",         "OR=1 means no effect"),
                "n":             ("sample size",        ""),
                "n_male":        ("n (male)",           ""),
                "n_female":      ("n (female)",         ""),
                "feature_name":  ("feature created",    ""),
                "feature_op":    ("feature op",         ""),
            }
            rows = []
            for k, v in m.items():
                label, hint = _METRIC_CATEGORIES.get(k, (k, ""))
                if isinstance(v, float):
                    if abs(v) < 0.001 and v != 0:
                        v_fmt = f"{v:.2e}"
                    else:
                        v_fmt = f"{v:.4f}" if abs(v) < 1000 else f"{v:,.2f}"
                elif isinstance(v, int):
                    v_fmt = f"{v:,}"
                else:
                    v_fmt = str(v)
                rows.append({"metric": label, "value": v_fmt, "interpretation": hint})
            try:
                import pandas as _pd
                st.dataframe(
                    _pd.DataFrame(rows),
                    hide_index=True,
                    use_container_width=True,
                )
            except Exception:
                # Fallback if pandas display fails
                for r in rows:
                    st.markdown(f"- **{r['metric']}**: `{r['value']}`  · {r['interpretation']}")

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
                st.warning(f"⚠ {len(fb.bias_flags)} bias flag(s) raised:")
                for flag in fb.bias_flags:
                    sv = flag.get("sensitive_var", "?")
                    cv = flag.get("causal_verb", "?")
                    src = flag.get("source", "?")
                    snip = (flag.get("snippet", "") or "")[:240]
                    st.markdown(
                        f"- **{sv}** + causal verb \"`{cv}`\" in *{src}*: \"{snip}\""
                    )
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

    # Fix #6 (round 2): scan the *question itself* for causal framing on a
    # sensitive variable. The bias scanner that runs per-iteration only flags
    # the LLM's output, but Kimi self-disciplines on causal language so its
    # generated hypothesis usually reads as correlational even when the user
    # explicitly asked "find the true cause of X". Surfacing the *question's*
    # framing as a top-of-timeline banner keeps the safety story visible in
    # the demo regardless of what the LLM does downstream.
    s.question_bias_flags = []
    if (trace.question or "").strip():
        try:
            from hypothesisloop.safety.bias_scanner import scan_text
            flags = scan_text(trace.question, source="user_question")
            s.question_bias_flags = [
                {
                    "sensitive_var": f.sensitive_var,
                    "causal_verb": f.causal_verb,
                    "snippet": f.snippet,
                    "source": f.source,
                }
                for f in flags
            ]
        except Exception:
            pass

    # Sam's audit fix #10: lightweight off-topic detector. Cheap heuristic —
    # if the question references zero columns AND no statistical keywords,
    # flag it as off-topic. Surfaces a banner; doesn't block the run.
    s.question_offtopic = False
    s.question_offtopic_reason = ""
    if mode == "explore" and (trace.question or "").strip():
        q_lower = (trace.question or "").lower()
        col_terms = {c.lower() for c in df.columns}
        col_terms |= {c.replace("_", " ").lower() for c in df.columns}
        col_terms |= {c.replace("-", " ").lower() for c in df.columns}
        # Strip very short tokens (e.g. "n", "id") that would false-positive.
        col_terms = {c for c in col_terms if len(c) >= 3}
        stat_keywords = {
            "predict", "test", "correlat", "compar", "relationship", "differ",
            "effect", "associat", "regress", "distribut", "outlier", "missing",
            "trend", "cluster", "group", "mean", "median", "average", "ratio",
            "rate", "proportion", "income", "data", "feature", "variable",
            "column", "sample", "analysis", "analyze", "understand", "explore",
            "what", "why", "how", "which", "explain",
        }
        col_hit = any(c in q_lower for c in col_terms)
        kw_hit = any(k in q_lower for k in stat_keywords)
        if not col_hit and not kw_hit:
            s.question_offtopic = True
            s.question_offtopic_reason = (
                f"Your question doesn't reference any column from the dataset "
                f"({len(df.columns)} columns: {', '.join(list(df.columns)[:5])}…) "
                f"and doesn't contain statistical-analysis vocabulary. The agent "
                f"will run anyway, but the results may not address what you asked."
            )

    # Fix #1, #2, #3, #4: the run is driven from the bottom of the script via
    # `s.pending_iteration`, which gives the topbar/progress-bar a chance to
    # paint between iterations and ensures status indicators land in the main
    # area (not the cramped sidebar).
    s.pending_iteration = True
    st.rerun()


def _run_one_iteration() -> None:
    s = st.session_state
    s.phase = "running"
    s.progress_substep = "→ hypothesizing → generating code → running sandbox → evaluating"
    try:
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
        s.pending_iteration = False
    else:
        s.phase = "paused"


def _continue() -> None:
    s = st.session_state
    s.pending_iteration = True
    s.phase = "running"
    st.rerun()


def _continue_plus_5() -> None:
    s = st.session_state
    # Fix #9: sync iter_idx to actual completed count, not the previous budget.
    if s.trace is not None:
        s.iter_idx = s.trace.iteration_count()
    s.max_iters = s.iter_idx + 5
    s.phase = "running"
    s.pending_iteration = True
    s.automl_summary = None  # let AutoGluon retrain on the extended feature set
    st.rerun()


def _stop() -> None:
    s = st.session_state
    s.pending_iteration = False
    _complete()
    st.rerun()


def _redirect(text: str) -> None:
    s = st.session_state
    if text and s.components is not None:
        s.components["scheduler"].inject(text)
    s.redirect_open = False
    s.phase = "running"
    s.pending_iteration = True
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
                    engineered_features=s.trace.engineered_features,
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
        # Sam's audit fix #8: pull live numbers from the in-process tracker
        # so the report header matches what's been ticking in the sidebar
        # all along. Langfuse rollup lags and returns zeros at this moment.
        tracker_usage = None
        tracker = s.get("cost_tracker")
        if tracker is not None:
            try:
                wall_s = 0.0
                if s.get("wall_start") is not None:
                    wall_s = (datetime.now(timezone.utc) - s.wall_start).total_seconds()
                tracker_usage = {
                    "total_tokens":   tracker.total_tokens,
                    "input_tokens":   sum(r.input_tokens for r in tracker.records),
                    "output_tokens":  sum(r.output_tokens for r in tracker.records),
                    "total_cost_usd": tracker.total_cost_usd,
                    "wall_time_s":    wall_s,
                    "trace_count":    tracker.total_calls,
                }
            except Exception:
                tracker_usage = None
        render_report(
            s.trace,
            output_dir=s.session_root,
            format="both",
            cli_command="streamlit run hypothesisloop/ui/streamlit_app.py",
            seed=42,
            usage_override=tracker_usage,
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

# ---------------------------------------------------------------------------
# auto-iteration driver — fixes #1, #2, #4 (status pulse / streaming cards /
# main-area progress instead of cramped sidebar st.status). One iteration per
# render pass; the topbar, progress bar, and any new card paint BEFORE the
# next iteration runs.
# ---------------------------------------------------------------------------
def _drive_pending() -> None:
    s = st.session_state
    if not s.get("pending_iteration"):
        return
    if s.get("phase") != "running":
        return
    if s.get("trace") is None or s.get("components") is None:
        return
    if s.get("last_error"):
        return

    _run_one_iteration()

    if s.get("phase") == "complete":
        s.pending_iteration = False
    elif s.get("_auto_run"):
        s.pending_iteration = True
        s.phase = "running"
    else:
        s.pending_iteration = False  # HITL mode → wait for user click

    st.rerun()


_drive_pending()
