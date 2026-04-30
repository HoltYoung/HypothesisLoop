"""Mission-control theme injector for the Streamlit UI.

Phase 8.1 replaces the original CSS wholesale with the audit-driven patch
documented in the Phase 8.1 prompt §2. Numbered sections match the audit
finding numbers so future fixes can be traced back.
"""

from __future__ import annotations

import streamlit as st


PALETTE = {
    "bg": "#0F172A",       # slate-900
    "panel": "#1E293B",    # slate-800
    "border": "#334155",   # slate-700
    "fg": "#E2E8F0",       # slate-200
    "muted": "#94A3B8",    # slate-400
    "accent": "#06B6D4",   # cyan-500
}

DECISION_COLORS = {
    "confirmed": "#34D399",     # emerald-400
    "rejected": "#F87171",      # red-400
    "inconclusive": "#FBBF24",  # amber-400
    "invalid": "#64748B",       # slate-500
    "pending": "#94A3B8",       # slate-400
}

# Card class for the .hl-iter-card decision-color override (left border + hover).
# Confirmed/pending fall through to the default cyan-accent border.
DECISION_CARD_CLASS = {
    "confirmed": "",
    "invalid": "invalid",
    "rejected": "rejected",
    "inconclusive": "pivot",
    "pending": "",
}

# Badge class for the .hl-iter-badge background swatch.
DECISION_BADGE_CLASS = {
    "confirmed": "confirmed",
    "invalid": "invalid",
    "rejected": "rejected",
    "inconclusive": "inconclusive",
    "pending": "",
}


def decision_color(decision: str) -> str:
    """Return the hex color for a decision label, or the muted fallback."""
    return DECISION_COLORS.get(decision, DECISION_COLORS["pending"])


def inject_css() -> None:
    """Write the mission-control stylesheet (audit §2) into the Streamlit page."""
    st.markdown(
        """
        <style>
        /* ---------- 1. Real font load ---------- */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

        /* ---------- 2. Background everywhere ---------- */
        html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stMain"] {
            background:#0F172A !important;
            color:#E2E8F0;
        }

        /* ---------- 3. Scope monospace, leave icons alone ---------- */
        .stApp, .stApp *:not([class*="material-symbols"]):not([class*="material-icons"]):not([data-testid="stIconMaterial"]):not([data-testid="stIconMaterial"] *) {
            font-family: "JetBrains Mono","Fira Code","SF Mono",Consolas,monospace;
        }
        [data-testid="stIconMaterial"],
        [class*="material-symbols"], [class*="material-icons"],
        [data-testid="stSidebarCollapseButton"] *,
        [data-testid="stSidebarCollapsedControl"] * {
            font-family: "Material Symbols Rounded","Material Icons",sans-serif !important;
        }

        /* ---------- 4. Sidebar surface ---------- */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            background:#0B1220 !important;
            border-right:1px solid #1E293B;
        }
        [data-testid="stSidebar"] .hl-section {
            color:#94A3B8; font-size:10px; letter-spacing:0.14em;
            text-transform:uppercase; margin:18px 12px 6px;
        }

        /* ---------- 5. Top bar lockup ---------- */
        .hl-brand { height:56px; display:flex; align-items:center;
            color:#06B6D4; font-size:18px; font-weight:600; letter-spacing:0.08em; }
        .hl-stat  { height:56px; box-sizing:border-box; padding:8px 14px;
            background:#1E293B; border:1px solid #334155; border-radius:4px;
            display:flex; flex-direction:column; justify-content:center; }
        .hl-stat-label { display:block; font-size:10px; color:#94A3B8;
            letter-spacing:0.12em; margin-bottom:4px; text-transform:uppercase; }
        .hl-stat code { font-size:13px; color:#E2E8F0;
            background:transparent !important; padding:0 !important; }

        /* ---------- 6. Status pulse ---------- */
        .hl-status code::before { content:"●"; color:#22D3EE; margin-right:6px;
            display:inline-block; }
        .hl-status[data-state="running"] code::before { animation: hl-pulse 1.2s ease-in-out infinite; }
        .hl-status[data-state="complete"] code::before { color:#34D399; animation:none; }
        .hl-status[data-state="error"]    code::before { color:#F87171; animation:none; }
        @keyframes hl-pulse { 0%,100%{opacity:1;} 50%{opacity:0.25;} }

        /* ---------- 7. Iteration cards ---------- */
        .hl-iter-card { background:#1E293B; border:1px solid #334155;
            border-left:3px solid #06B6D4; border-radius:4px;
            padding:14px 18px; margin-bottom:16px;
            transition: border-color 150ms, background 150ms; }
        .hl-iter-card:hover { background:#243349; }
        .hl-iter-card.invalid   { border-left-color:#475569; }
        .hl-iter-card.rejected  { border-left-color:#F87171; }
        .hl-iter-card.pivot     { border-left-color:#FBBF24; }
        .hl-iter-card.exploring { border-left-color:#A78BFA; }
        .hl-iter-num   { color:#06B6D4; font-weight:600; font-size:11px;
            letter-spacing:0.1em; margin-right:10px; }
        .hl-iter-meta  { color:#94A3B8; font-size:11px; }
        .hl-iter-statement { color:#E2E8F0; font-size:14px;
            line-height:1.55; margin-top:6px; }

        /* ---------- 8. Badges ---------- */
        .hl-iter-badge { display:inline-flex; align-items:center; gap:4px;
            padding:3px 10px; border-radius:3px; font-size:11px;
            font-weight:600; letter-spacing:0.06em; }
        .hl-iter-badge.confirmed    { background:#34D399; color:#0F172A; }
        .hl-iter-badge.invalid      { background:#475569; color:#F1F5F9; }
        .hl-iter-badge.rejected     { background:#F87171; color:#0F172A; }
        .hl-iter-badge.inconclusive { background:#FBBF24; color:#0F172A; }
        .hl-iter-badge.pivot        { background:#FBBF24; color:#0F172A; }

        /* ---------- 9. Tabs ---------- */
        .stTabs [data-baseweb="tab-list"] {
            gap:24px !important; border-bottom:1px solid #334155 !important;
            padding:0 !important; }
        .stTabs [data-baseweb="tab"] {
            padding:8px 4px !important; font-size:11px !important;
            letter-spacing:0.1em; color:#94A3B8 !important;
            background:transparent !important; border-radius:0 !important; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color:#06B6D4 !important; border-bottom:2px solid #06B6D4 !important; }
        .stTabs [role="tabpanel"] { padding-top:14px !important; max-height: 320px; overflow:auto; }

        /* ---------- 10. Code blocks scoped to cards ---------- */
        .hl-iter-card .stCodeBlock pre,
        .hl-iter-card pre { max-height:280px !important; overflow:auto !important;
            background:#0B1220 !important; border:1px solid #1E293B; border-radius:4px;
            font-size:12px !important; line-height:1.5; }

        /* ---------- 11. Buttons ---------- */
        .stButton > button, .stDownloadButton > button {
            background:#0F172A !important; color:#E2E8F0 !important;
            border:1px solid #334155 !important; border-radius:4px !important;
            font-family:"JetBrains Mono",monospace !important;
            font-size:12px !important; letter-spacing:0.04em;
            transition: border-color 120ms, color 120ms; }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color:#06B6D4 !important; color:#06B6D4 !important;
            background:#0F172A !important; }
        .stButton > button[kind="primary"] {
            background:#06B6D4 !important; color:#0F172A !important;
            border-color:#06B6D4 !important; font-weight:600; }
        .stButton > button[kind="primary"]:hover {
            background:#22D3EE !important; border-color:#22D3EE !important; }

        /* ---------- 12. Sidebar pre + metrics ---------- */
        [data-testid="stSidebar"] pre {
            background:transparent !important;
            white-space: pre-wrap !important; word-break: break-all;
            font-size: 11px; line-height: 1.5;
            color:#E2E8F0 !important; padding:8px 12px !important;
            border:1px solid #1E293B; border-radius:4px;
        }
        .hl-metrics {
            font-family: "JetBrains Mono","Fira Code",monospace !important;
            font-size: 12px !important; color:#E2E8F0 !important;
            background:transparent !important; padding:0 !important;
            white-space: pre; line-height: 1.6;
        }

        /* ---------- 13. Kill Streamlit chrome ---------- */
        [data-testid="stDeployButton"], [data-testid="stToolbar"] { display:none !important; }
        #MainMenu, footer, header { visibility:hidden; height:0px; }

        /* ---------- 14. Block container width ---------- */
        .block-container { max-width:1400px !important; padding: 16px 32px !important; }

        /* ---------- 15. New-run button helper ---------- */
        .hl-newrun button { background:#06B6D4 !important; color:#0F172A !important;
            font-weight:600; letter-spacing:0.06em; border:none; }
        .hl-newrun button:hover { background:#22D3EE !important; }

        /* ---------- Empty-state pill ---------- */
        .hl-empty {
            color:#94A3B8; text-align:center; padding:1.5rem 1rem;
            font-size:0.9rem; letter-spacing:0.1em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


__all__ = [
    "PALETTE",
    "DECISION_COLORS",
    "DECISION_CARD_CLASS",
    "DECISION_BADGE_CLASS",
    "decision_color",
    "inject_css",
]
