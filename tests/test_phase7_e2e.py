"""Phase 7 end-to-end acceptance — full live run produces a report (gated)."""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.skipif(
    not os.getenv("HL_RUN_INTEGRATION"),
    reason="set HL_RUN_INTEGRATION=1 to run live Kimi + sandbox + report",
)


def test_phase7_full_run_with_report_and_bias_scanner(tmp_path: Path):
    from hypothesisloop import cli
    from hypothesisloop.agent.state import DAGTrace
    from hypothesisloop.trace.langfuse_client import get_session_usage

    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "adult.csv"
    df_full = pd.read_csv(csv_path)
    sample = df_full.sample(n=min(1000, len(df_full)), random_state=42).reset_index(drop=True)
    sample_csv = tmp_path / "adult_sample.csv"
    sample.to_csv(sample_csv, index=False)

    output_dir = tmp_path / "out"

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        rc = cli.main(
            [
                "--auto",
                "--max-iters",
                "2",
                "--question",
                "What demographic factors most predict income > $50K?",
                "--data",
                str(sample_csv),
                "--output-dir",
                str(output_dir),
                "--rag-index",
                str(project_root / "knowledge" / "rag.index"),
                "--rag-chunks",
                str(project_root / "knowledge" / "rag_chunks.pkl"),
            ]
        )

    err = captured_stderr.getvalue()
    print(captured_stdout.getvalue())
    print(err, file=__import__("sys").stderr)

    assert rc == 0, err

    sessions = [p for p in output_dir.iterdir() if p.is_dir()]
    assert len(sessions) == 1, f"expected one session dir, saw {sessions}"
    session_dir = sessions[0]

    md_path = session_dir / "report.md"
    txt_path = session_dir / "report.txt"
    assert md_path.exists(), f"report.md missing at {md_path}"
    assert txt_path.exists(), f"report.txt missing at {txt_path}"

    md = md_path.read_text(encoding="utf-8")
    for header in (
        "## 1. Run metadata",
        "## 2. Question & approach",
        "## 3. Hypothesis chain",
        "## 4. Key findings",
        "## 5. Rejections & dead-ends",
        "## 6. Bias flags raised",
        "## 7. Reasoning chain",
        "## 8. Limitations & caveats",
        "## 9. Reproduction",
    ):
        assert header in md, f"missing section: {header!r}"

    trace = DAGTrace.load(session_dir / "trace.json")
    for node in trace.iter_nodes():
        if node.feedback is not None:
            assert isinstance(node.feedback.bias_flags, list), (
                "bias_flags must be a list (possibly empty) on every evaluated node"
            )

    usage = get_session_usage(trace.session_id)
    assert isinstance(usage, dict)
    for k in ("total_tokens", "input_tokens", "output_tokens", "total_cost_usd", "wall_time_s"):
        assert k in usage

    print(f"\n[phase7-e2e] report.md -> {md_path}")
    print(f"[phase7-e2e] report.txt -> {txt_path}")
