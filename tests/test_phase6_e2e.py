"""Phase 6 end-to-end CLI acceptance — gated.

Calls ``cli.main()`` with ``--auto --max-iters 2`` against real Kimi. Uses a
1000-row sample of UCI Adult written into ``tmp_path`` so the live run does
not pollute ``reports/`` and CSV-load latency stays low.
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
    reason="set HL_RUN_INTEGRATION=1 to run live Kimi + sandbox",
)


def test_cli_auto_run_2_iters(tmp_path: Path):
    from hypothesisloop import cli

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

    out = captured_stdout.getvalue()
    err = captured_stderr.getvalue()

    print(out)
    print(err, file=__import__("sys").stderr)

    assert rc == 0, err

    # Locate the session directory the CLI minted.
    sessions = [p for p in output_dir.iterdir() if p.is_dir()]
    assert len(sessions) == 1, f"expected one session dir, saw {sessions}"
    session_dir = sessions[0]

    trace_path = session_dir / "trace.json"
    assert trace_path.exists(), f"trace.json missing at {trace_path}"

    from hypothesisloop.agent.state import DAGTrace

    trace = DAGTrace.load(trace_path)
    accepted = trace.iter_nodes()
    assert 1 <= len(accepted) <= 2, (
        f"expected 1-2 accepted iterations (novelty may reject up to 1), got {len(accepted)}"
    )
    assert "HypothesisLoop run:" in out
    assert "iterations   :" in out
    for node in accepted:
        assert f"iter {node.iteration:>2}" in out, "summary should list each accepted iteration"
