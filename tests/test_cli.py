"""Phase 6 CLI surface tests — argparse, --resume / --question guards.

The live --auto run is gated separately in ``tests/test_phase6_e2e.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

import hypothesisloop.cli as cli


# ---------------------------------------------------------------------------
# --help via subprocess (smoke check that argparse renders without crashing)
# ---------------------------------------------------------------------------
def test_cli_help_runs():
    proc = subprocess.run(
        [sys.executable, "-m", "hypothesisloop.cli", "--help"],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "--smoke-test" in proc.stdout
    assert "--auto" in proc.stdout
    assert "--resume" in proc.stdout
    assert "--question" in proc.stdout


# ---------------------------------------------------------------------------
# guard branches in _load_or_create_trace
# ---------------------------------------------------------------------------
def _ns(**overrides) -> argparse.Namespace:
    """Build a CLI Namespace with sensible defaults for the guards."""
    base = dict(
        data="data/adult.csv",
        question=None,
        model="moonshot-v1-128k",
        max_iters=5,
        auto=False,
        seed=42,
        session_id=None,
        resume=None,
        output_dir="reports",
        smoke_test=False,
        rag_index="knowledge/rag.index",
        rag_chunks="knowledge/rag_chunks.pkl",
        rag_k=4,
        report_only=False,
        # Phase 9 flags
        mode="explore",
        target=None,
        task_type="auto",
        provider=None,
        api_key=None,
        automl_time_budget=120,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cli_resume_missing_trace_errors(tmp_path: Path):
    args = _ns(resume="does-not-exist", output_dir=str(tmp_path))
    with pytest.raises(SystemExit) as excinfo:
        cli._load_or_create_trace(args)
    assert "trace file not found" in str(excinfo.value)


def test_cli_no_question_no_resume_errors(tmp_path: Path):
    args = _ns(output_dir=str(tmp_path))  # question=None, resume=None
    with pytest.raises(SystemExit) as excinfo:
        cli._load_or_create_trace(args)
    assert "--question is required" in str(excinfo.value)


# ---------------------------------------------------------------------------
# offline import sanity for the smoke-test path
# ---------------------------------------------------------------------------
def test_cli_smoke_test_offline_import_path():
    """We can build the smoke-test run plumbing without making any network call.

    The real network call is exercised in Phase 0's manual --smoke-test run; here
    we only verify the function is importable and the argparse plumbing accepts
    --smoke-test.
    """
    parser = cli._build_argparser()
    args = parser.parse_args(["--smoke-test"])
    assert args.smoke_test is True
    assert args.question is None  # not required in smoke mode
    # The function exists and is callable; we don't actually invoke it (would
    # hit the network).
    assert callable(cli._run_smoke_test)
