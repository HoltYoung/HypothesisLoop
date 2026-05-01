"""Phase 9 end-to-end Predict-mode acceptance — gated.

Full pipeline: load Adult sample → run baseline → run loop with live Kimi
+ proxy CV → run AutoGluon (60s budget) → render report. Verifies the
trace, ledger, AutoGluon artifacts, and report sections are all populated.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import pandas as pd
import pytest


pytestmark = pytest.mark.skipif(
    not os.getenv("HL_RUN_INTEGRATION"),
    reason="set HL_RUN_INTEGRATION=1 to run live Kimi + AutoGluon",
)


def test_phase9_predict_full_run(tmp_path: Path):
    from hypothesisloop import cli
    from hypothesisloop.agent.state import DAGTrace

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
                "--mode", "predict",
                "--target", "income",
                "--data", str(sample_csv),
                "--auto",
                "--max-iters", "2",
                "--automl-time-budget", "60",
                "--output-dir", str(output_dir),
                "--rag-index", str(project_root / "knowledge" / "rag.index"),
                "--rag-chunks", str(project_root / "knowledge" / "rag_chunks.pkl"),
            ]
        )
    print(captured_stdout.getvalue())
    print(captured_stderr.getvalue(), file=__import__("sys").stderr)
    assert rc == 0, captured_stderr.getvalue()

    sessions = [p for p in output_dir.iterdir() if p.is_dir()]
    assert len(sessions) == 1
    session_dir = sessions[0]

    trace = DAGTrace.load(session_dir / "trace.json")
    assert trace.mode == "predict"
    assert trace.target_column == "income"
    assert trace.baseline_score is not None
    assert trace.current_best_score is not None
    # At least one engineered feature attempted.
    assert len(trace.engineered_features) >= 1

    # AutoGluon artifacts.
    assert (session_dir / "leaderboard.csv").exists()
    assert (session_dir / "automl_summary.json").exists()
    assert (session_dir / "model").is_dir()

    # Report includes the Predict section.
    md = (session_dir / "report.md").read_text(encoding="utf-8")
    assert "## 10. Predictive modeling" in md
    assert "AutoGluon ensemble" in md
    assert "Feature engineering ledger" in md
