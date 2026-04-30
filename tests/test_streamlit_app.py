"""Phase 8 Streamlit-app tests.

Streamlit is hard to unit-test the UI of; we cover what we can:
- the app file imports cleanly
- the theme constants are stable
- ``inject_css`` writes the full palette
- ``factory.build_steps`` returns the expected dict
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hypothesisloop.ui.theme as theme_module
from hypothesisloop.agent.factory import build_steps
from hypothesisloop.agent.state import DAGTrace
from hypothesisloop.ui.theme import DECISION_COLORS, PALETTE, decision_color


def test_streamlit_app_imports_clean():
    """Smoke-check: the app module is syntactically valid + imports resolve."""
    import importlib

    # Fresh import to exercise the module-load side-effects (set_page_config,
    # inject_css). Streamlit logs a warning when these run outside a real
    # Streamlit context but doesn't raise.
    if "hypothesisloop.ui.streamlit_app" in __import__("sys").modules:
        del __import__("sys").modules["hypothesisloop.ui.streamlit_app"]
    importlib.import_module("hypothesisloop.ui.streamlit_app")


def test_decision_color_returns_hex():
    assert decision_color("confirmed") == "#34D399"
    assert decision_color("rejected") == "#F87171"
    assert decision_color("inconclusive") == "#FBBF24"
    assert decision_color("invalid") == "#64748B"
    assert decision_color("pending") == "#94A3B8"
    # Unknown decision falls back to the muted (pending) color.
    assert decision_color("totally-made-up") == "#94A3B8"


def test_inject_css_writes_palette_constants(monkeypatch):
    """Capture every ``st.markdown`` payload and assert the palette is in it."""
    captured: list[str] = []

    def fake_markdown(s, *args, **kwargs):  # noqa: ARG001
        captured.append(s)

    # Patch the binding the theme module uses.
    monkeypatch.setattr(theme_module.st, "markdown", fake_markdown)

    theme_module.inject_css()
    blob = "\n".join(captured)

    for hex_code in list(PALETTE.values()) + list(DECISION_COLORS.values()):
        # DECISION_COLORS aren't directly written into the CSS — they're used
        # at render time by decision_color() — so only assert PALETTE.
        if hex_code in PALETTE.values():
            assert hex_code in blob, f"missing palette color in injected CSS: {hex_code}"

    # Sanity: a few class names we depend on at render time. Phase 8.1
    # dropped the standalone ``.hl-actionbar`` rule (action-bar buttons
    # inherit the global .stButton restyle), so it's not in this list.
    for cls in (
        ".hl-brand",
        ".hl-stat",
        ".hl-iter-card",
        ".hl-iter-badge",
        "@keyframes hl-pulse",
    ):
        assert cls in blob, f"injected CSS missing rule: {cls}"


def test_inject_css_includes_google_fonts_import(monkeypatch):
    """Audit Critical #1 — JetBrains Mono must be loaded via @import."""
    captured: list[str] = []
    monkeypatch.setattr(
        theme_module.st, "markdown", lambda s, *a, **kw: captured.append(s)
    )
    theme_module.inject_css()
    blob = "\n".join(captured)
    assert "@import" in blob
    assert "fonts.googleapis.com/css2?family=JetBrains" in blob


def test_inject_css_includes_data_state_animation(monkeypatch):
    """Audit Critical #7 — status pulse keyed off data-state attribute."""
    captured: list[str] = []
    monkeypatch.setattr(
        theme_module.st, "markdown", lambda s, *a, **kw: captured.append(s)
    )
    theme_module.inject_css()
    blob = "\n".join(captured)
    assert '[data-state="running"]' in blob
    assert "@keyframes hl-pulse" in blob
    assert '[data-state="complete"]' in blob


def test_factory_build_steps_returns_dict(tmp_path: Path):
    """build_steps wires every component when the RAG index exists."""
    project_root = Path(__file__).resolve().parents[1]
    rag_index = project_root / "knowledge" / "rag.index"
    rag_chunks = project_root / "knowledge" / "rag_chunks.pkl"
    if not rag_index.exists() or not rag_chunks.exists():
        pytest.skip(
            "RAG index missing; run scripts/build_rag_index.py first"
        )

    trace = DAGTrace(
        session_id="factory-test",
        dataset_path=str(project_root / "data" / "adult.csv"),
        question="?",
        schema_summary="14 cols, 32k rows.",
    )

    components = build_steps(
        trace=trace,
        session_root=tmp_path / "session",
        rag_index_path=str(rag_index),
        rag_chunks_path=str(rag_chunks),
    )

    expected_keys = {
        "hypothesize_fn",
        "experiment_fn",
        "evaluate_fn",
        "novelty_fn",
        "safety_fn",
        "scheduler",
        "pruner",
    }
    assert set(components.keys()) == expected_keys
    for key in expected_keys:
        assert components[key] is not None, f"{key} should not be None"
    # The scheduler is the LinearScheduler instance (callable + .inject + .next_parent).
    assert callable(components["safety_fn"])
    assert hasattr(components["scheduler"], "inject")
    assert hasattr(components["scheduler"], "next_parent")
    assert hasattr(components["pruner"], "estimate_tokens")
