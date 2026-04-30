"""HypothesisLoop Streamlit UI — mission control.

Run:
    streamlit run hypothesisloop/ui/streamlit_app.py

Phase 8.1 audit-driven fixes layered on top of Phase 8:
- Live-metrics cache refreshed after each iteration (not just on completion).
- Single ``<pre class="hl-metrics">`` render so we don't double-stack panels.
- Sticky "↻ NEW RUN" button at the top of the sidebar after a run starts.
- Top-bar status panel uses ``data-state="running|paused|complete|idle"``;
  the CSS pulses on ``running``, holds steady-green on ``complete``.
- Iteration cards switch from inline ``border-left-color`` to decision-class
  CSS (``.hl-iter-card.invalid``, ``.rejected``, ``.pivot``) so hover and
  border-color animations come from the stylesheet.
- METRICS tab gates on numeric + finite (no NaN, no Inf) before plotting,
  fixing the Vega-Lite "infinite extent" warnings.
"""

from __future__ import annotations

import math
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# CRITICAL: load .env before any module reads env vars.
load_dotenv(override=True)

import pandas as pd
import streamlit as st

from hypothesisloop.agent.factory import build_steps
from hypothesisloop.agent.loop import _execute_iteration
from hypothesisloop.agent.state import DAGTrace
from hypothesisloop.steps.profile import profile_dataset
from hypothesisloop.steps.report import render_report
from hypothesisloop.trace.langfuse_client import get_session_usage, start_session
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


# ---------------------------------------------------------------------------
# session-state init
# ---------------------------------------------------------------------------
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
        "_usage_cache": _zero_usage(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _zero_usage() -> dict:
    return {
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_cost_usd": 0.0,
        "wall_time_s": 0.0,
        "trace_count": 0,
    }


def _refresh_usage_cache() -> None:
    """Refresh the Langfuse usage rollup. Called after each iteration."""
    s = st.session_state
    if s.trace is None:
        s["_usage_cache"] = _zero_usage()
        return
    try:
        s["_usage_cache"] = get_session_usage(s.trace.session_id)
    except Exception as e:  # pragma: no cover — defensive
        s["_usage_cache"] = {**_zero_usage(), "_error": f"{type(e).__name__}: {e}"}


_init_state()


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

    with cols[1]:
        st.markdown(
            f'<div class="hl-stat"><span class="hl-stat-label">SESSION</span>'
            f'<code>{s.trace.session_id[-12:]}</code></div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            '<div class="hl-stat"><span class="hl-stat-label">MODEL</span>'
            '<code>kimi-k2.6</code></div>',
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
# sidebar: NEW RUN / config / live metrics / downloads
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    s = st.session_state
    with st.sidebar:
        # NEW RUN — always present after a run starts.
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
            data_path = st.text_input("dataset", value="data/adult.csv")
            question = st.text_area(
                "question",
                value="What demographic factors most predict income > $50K?",
            )
            max_iters = st.slider("max iterations", 1, 10, 5)
            seed = st.number_input("seed", value=42)
            if st.button("▶ START RUN", use_container_width=True, type="primary"):
                _start_run(data_path, question, int(max_iters), int(seed))
        else:
            preview_q = (s.trace.question or "")[:80]
            st.markdown(
                f"<pre>dataset : {s.trace.dataset_path}\n"
                f"question: {preview_q}</pre>",
                unsafe_allow_html=True,
            )

        st.markdown('<div class="hl-section">LIVE METRICS</div>', unsafe_allow_html=True)
        usage = st.session_state.get("_usage_cache", _zero_usage())
        tokens = int(usage.get("total_tokens", 0) or 0)
        cost = float(usage.get("total_cost_usd", 0.0) or 0.0)
        wall = float(usage.get("wall_time_s", 0.0) or 0.0)
        rejects = len(s.trace.novelty_rejected) if s.trace is not None else 0
        # Single render — no nested st.code under st.markdown, so we don't
        # double-stack <pre> elements.
        st.markdown(
            (
                f'<pre class="hl-metrics">'
                f"tokens : {tokens:>8,}\n"
                f"cost   : ${cost:>7.4f}\n"
                f"wall   : {wall:>7.1f}s\n"
                f"rejects: {rejects:>8}"
                f"</pre>"
            ),
            unsafe_allow_html=True,
        )
        if usage.get("_error"):
            st.caption(f"usage: {usage['_error']}")

        st.markdown('<div class="hl-section">SESSION</div>', unsafe_allow_html=True)
        if s.phase == "complete" and s.session_root is not None:
            md_path = s.session_root / "report.md"
            txt_path = s.session_root / "report.txt"
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
        else:
            st.markdown(
                '<div class="hl-empty">downloads after run completes</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# iteration timeline
# ---------------------------------------------------------------------------
def _render_timeline() -> None:
    s = st.session_state
    if s.trace is None:
        st.markdown('<div class="hl-empty">awaiting run config</div>', unsafe_allow_html=True)
        return
    if s.last_error:
        st.error(s.last_error)
    nodes = s.trace.iter_nodes()
    if not nodes:
        st.markdown('<div class="hl-empty">no iterations yet</div>', unsafe_allow_html=True)
        return
    for node in nodes:
        _render_iteration_card(node)


def _is_plot_safe(value) -> bool:
    if not isinstance(value, (int, float)):
        return False
    if isinstance(value, bool):  # bool is a subclass of int — exclude
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
    tabs = st.tabs(["CODE", "OUTPUT", "METRICS", "EVALUATOR"])
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
        # Always show the raw dict beneath the chart (or in lieu of it).
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
                '<div class="hl-empty">no feedback yet</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# action bar
# ---------------------------------------------------------------------------
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


def _start_run(data_path: str, question: str, max_iters: int, seed: int) -> None:
    s = st.session_state
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        st.error(f"could not read dataset at {data_path}: {e}")
        return
    schema = profile_dataset(df, dataset_path=data_path)

    session_id = _new_session_id()
    session_root = Path("reports") / session_id
    session_root.mkdir(parents=True, exist_ok=True)

    trace = DAGTrace(
        session_id=session_id,
        dataset_path=str(Path(data_path).resolve()),
        question=question,
        schema_summary=schema,
    )
    try:
        components = build_steps(trace=trace, session_root=session_root, seed=seed)
    except Exception as e:
        st.error(f"could not build agent components: {e}")
        return

    start_session(session_id)

    s.trace = trace
    s.session_root = session_root
    s.components = components
    s.max_iters = max_iters
    s.iter_idx = 0
    s.phase = "running"
    s.redirect_open = False
    s.last_error = None
    s["_usage_cache"] = _zero_usage()
    _run_one_iteration()
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
                hitl_fn=None,                    # Streamlit IS the HITL gate
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

    # Refresh metrics cache once per iteration so the sidebar updates.
    _refresh_usage_cache()

    if s.iter_idx >= s.max_iters or s.last_error:
        _complete()
    else:
        s.phase = "paused"


def _continue() -> None:
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
    # Final metrics refresh so sidebar reflects the completed run.
    _refresh_usage_cache()


# ---------------------------------------------------------------------------
# page render
# ---------------------------------------------------------------------------
_render_topbar()
_render_sidebar()
_render_timeline()
_render_action_bar()
