"""Phase 7 acceptance — ``--report-only --resume`` re-renders without re-running."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from hypothesisloop.agent.state import (
    DAGTrace,
    Experiment,
    ExperimentAttempt,
    Hypothesis,
    HypothesisFeedback,
    new_hypothesis_id,
)


def _seed_session(reports_root: Path, session_id: str = "session-x") -> Path:
    session_root = reports_root / session_id
    session_root.mkdir(parents=True, exist_ok=True)

    trace = DAGTrace(
        session_id=session_id,
        dataset_path="data/adult.csv",
        question="Does education predict income?",
    )
    h = Hypothesis(
        id=new_hypothesis_id(),
        parent_id=None,
        iteration=1,
        statement="education-num positively associates with income",
        null="education-num does not associate with income",
        test_type="correlation",
        target_columns=["education-num", "income"],
        expected_outcome="r > 0.2, p < 0.05",
        concise_reason="education is the strongest single predictor",
        concise_observation="education-num spans 1-16",
        concise_justification="cheap and headline-relevant",
        concise_knowledge="settles whether education is even a candidate",
    )
    trace.add_node(h)
    trace.update_experiment(
        h.id,
        Experiment(
            hypothesis_id=h.id,
            attempts=[
                ExperimentAttempt(
                    attempt_idx=0,
                    code="from scipy import stats\nprint('r=0.42, p=0.001')\n",
                    exit_code=0,
                    stdout="r=0.42, p=0.001\n",
                    stderr="",
                    figures=[],
                    metrics={"p_value": 0.001, "effect_size": 0.42, "n": 1000},
                    blocked_reason=None,
                    duration_s=0.5,
                    timed_out=False,
                    oom_killed=False,
                )
            ],
            succeeded=True,
        ),
    )
    trace.update_feedback(
        h.id,
        HypothesisFeedback(
            hypothesis_id=h.id,
            decision="confirmed",
            reason="Pearson r=0.42, p=0.001 — moderate positive association.",
            observations="r=0.42, p=0.001",
            novel_subhypotheses=[],
            confidence=0.78,
        ),
    )
    trace.save(session_root / "trace.json")
    return session_root


def test_report_only_flag_renders_from_saved_trace(tmp_path: Path):
    reports_root = tmp_path / "reports"
    session_root = _seed_session(reports_root)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "hypothesisloop.cli",
            "--report-only",
            "--resume",
            session_root.name,
            "--output-dir",
            str(reports_root),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"

    md_path = session_root / "report.md"
    txt_path = session_root / "report.txt"
    assert md_path.exists(), "report.md should be created by --report-only"
    assert txt_path.exists(), "report.txt should be created by --report-only"

    md = md_path.read_text(encoding="utf-8")
    for required in (
        "## 1. Run metadata",
        "## 6. Bias flags raised",
        "## 9. Reproduction",
        "education-num positively associates with income",
    ):
        assert required in md, f"missing in re-rendered report: {required!r}"
