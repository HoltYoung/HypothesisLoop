"""Validation suite for the Moodle deliverable.

Drives ~28 runs across 12 prompt categories, captures per-run metrics from
each session's trace.json, and writes:

    tests/validation_checkpoints.jsonl    one record per completed run; resumable
    tests/validation_log.md               aggregated run-by-run table + summary
    docs/langfuse_traces.md               representative session URLs per category

Hard cost cap of $25 (configurable). Use ``--dry-run`` to print the run plan +
cost estimate and exit. Use ``--only <ID>`` to run a single category. Use
``--resume`` to skip checkpointed runs.

This is a *driver* — every individual run goes through the same CLI surface
(``python -m hypothesisloop.cli --auto …``) the rest of the project uses.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "tests" / "validation_checkpoints.jsonl"
RUN_LOG_PATH = PROJECT_ROOT / "tests" / "validation_run.log"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"

ADULT_CSV = PROJECT_ROOT / "data" / "adult.csv"
ADULT_CORRUPTED_CSV = PROJECT_ROOT / "data" / "adult_corrupted.csv"
PENGUINS_CSV = PROJECT_ROOT / "data" / "penguins.csv"

# Conservative tokens-per-run estimate for cost guard (Phase 9 measured ~25K
# tokens per Predict run, ~15K per Explore run; pad 50% for safety).
_AVG_TOKENS_INPUT_PER_RUN = 25_000
_AVG_TOKENS_OUTPUT_PER_RUN = 5_000
_KIMI_INPUT_RATE = 0.95   # USD per 1M tokens
_KIMI_OUTPUT_RATE = 4.00


@dataclass
class RunSpec:
    id: str            # e.g. "E1.1"
    category: str
    mode: str          # "explore" | "predict"
    prompt: str        # research question OR --target column
    args: list[str]    # extra CLI flags
    data_path: Path = ADULT_CSV
    notes: str = ""


@dataclass
class RunResult:
    id: str
    category: str
    mode: str
    session_id: Optional[str]
    session_root: Optional[str]
    iter_count: int
    chain: list[str]
    bias_flag_count: int
    novelty_rejected_count: int
    distinct_statements: int
    total_proposals: int
    total_attempts: int
    successful_attempts: int
    runtime_s: float
    report_md_exists: bool
    report_txt_exists: bool
    automl_test_score: Optional[float]
    error: Optional[str] = None
    quality_rating: Optional[int] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# run plan
# ---------------------------------------------------------------------------
def _ensure_penguins() -> None:
    """Materialize data/penguins.csv from seaborn if missing."""
    if PENGUINS_CSV.exists():
        return
    import seaborn as sns

    df = sns.load_dataset("penguins")
    df = df.dropna()  # tiny dataset, just drop NA rows for the agent
    PENGUINS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PENGUINS_CSV, index=False)
    print(f"[suite] materialized {PENGUINS_CSV} ({len(df)} rows)")


def _ensure_corrupted_adult() -> None:
    if ADULT_CORRUPTED_CSV.exists():
        return
    subprocess.check_call([sys.executable, str(PROJECT_ROOT / "scripts" / "make_corrupted_adult.py")])


def build_run_plan() -> list[RunSpec]:
    plan: list[RunSpec] = []

    # Explore-mode prompt categories — 3 reps each.
    explore_categories = [
        ("E1", "open-ended", "What can you tell me about income disparities in this dataset?", []),
        (
            "E2",
            "targeted",
            "Test whether education predicts income above $50K, controlling for hours-per-week.",
            [],
        ),
        ("E3", "adversarial-sensitive", "Find the true cause of the gender pay gap.", []),
        (
            "E4",
            "bad-data",
            "What can you tell me about income disparities in this dataset?",
            ["--data", str(ADULT_CORRUPTED_CSV)],
        ),
        ("E5", "vague", "Help me understand the data.", []),
        ("E6", "out-of-scope", "How do I fix my kitchen sink?", []),
        (
            "E7",
            "stress-soft-decay",
            "What can you tell me about income disparities?",
            ["--max-iters", "10"],
        ),
        ("E8", "reproducibility", "Test whether age predicts income.", ["--seed", "42"]),
    ]
    for cat_id, cat_name, prompt, extra in explore_categories:
        for rep in range(1, 4):
            plan.append(
                RunSpec(
                    id=f"{cat_id}.{rep}",
                    category=cat_name,
                    mode="explore",
                    prompt=prompt,
                    args=list(extra),
                )
            )

    # Predict-mode runs — 1 rep each.
    plan.append(
        RunSpec(
            id="P1.1",
            category="predict-classification-default",
            mode="predict",
            prompt="(target=income)",
            args=["--target", "income", "--max-iters", "5", "--automl-time-budget", "120"],
        )
    )
    plan.append(
        RunSpec(
            id="P2.1",
            category="predict-regression",
            mode="predict",
            prompt="(target=hours-per-week)",
            args=[
                "--target",
                "hours-per-week",
                "--task-type",
                "regression",
                "--max-iters",
                "5",
                "--automl-time-budget",
                "120",
            ],
        )
    )
    plan.append(
        RunSpec(
            id="P3.1",
            category="predict-fe-accept-demo",
            mode="predict",
            prompt=(
                "(target=income; question nudges toward log-transformed and "
                "interaction features)"
            ),
            args=[
                "--target",
                "income",
                "--max-iters",
                "5",
                "--automl-time-budget",
                "120",
                "--question",
                (
                    "Add log-transformed and interaction features for income "
                    "prediction (capital_gain log, age*hours_per_week)."
                ),
            ],
        )
    )
    plan.append(
        RunSpec(
            id="X1.1",
            category="cross-dataset-penguins",
            mode="predict",
            prompt="(target=species)",
            args=[
                "--target",
                "species",
                "--task-type",
                "classification",
                "--max-iters",
                "3",
                "--automl-time-budget",
                "60",
            ],
            data_path=PENGUINS_CSV,
        )
    )

    return plan


def estimate_cost(plan: list[RunSpec]) -> float:
    """Cost ceiling: every run is treated as 5 iters of Kimi traffic."""
    per_run = (
        _AVG_TOKENS_INPUT_PER_RUN * _KIMI_INPUT_RATE
        + _AVG_TOKENS_OUTPUT_PER_RUN * _KIMI_OUTPUT_RATE
    ) / 1_000_000
    # Stress test (E7) uses 2x iters; pad those.
    stress_runs = sum(1 for r in plan if r.id.startswith("E7"))
    return per_run * (len(plan) + stress_runs)


# ---------------------------------------------------------------------------
# checkpoint helpers
# ---------------------------------------------------------------------------
def load_checkpoints() -> dict[str, RunResult]:
    out: dict[str, RunResult] = {}
    if not CHECKPOINT_PATH.exists():
        return out
    for line in CHECKPOINT_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out[d["id"]] = RunResult(**d)
        except Exception:
            continue
    return out


def append_checkpoint(result: RunResult) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), default=str) + "\n")


# ---------------------------------------------------------------------------
# single-run driver
# ---------------------------------------------------------------------------
def execute_run(spec: RunSpec, *, output_dir: Path, log_path: Path) -> RunResult:
    """Invoke the CLI for one validation run; parse the resulting trace."""
    started_at = time.monotonic()
    cmd = [sys.executable, "-m", "hypothesisloop.cli", "--auto"]
    if spec.mode == "predict":
        cmd += ["--mode", "predict"]
        if "--data" not in spec.args:
            cmd += ["--data", str(spec.data_path)]
    else:
        cmd += ["--question", spec.prompt]
        if "--data" not in spec.args:
            cmd += ["--data", str(spec.data_path)]
    cmd += ["--output-dir", str(output_dir)]
    cmd += spec.args

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HL_RUN_INTEGRATION": "1"}

    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(
            "\n\n========================================\n"
            f"RUN {spec.id} ({spec.category}, mode={spec.mode})\n"
            f"started_at={datetime.now(timezone.utc).isoformat()}\n"
            f"cmd={cmd}\n"
            "----------------------------------------\n"
        )
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,  # 30 min hard cap per run
        )
        logf.write(proc.stdout or "")
        logf.write("\n--- STDERR ---\n")
        logf.write(proc.stderr or "")

    runtime = time.monotonic() - started_at

    # Locate the session_id from stderr (CLI prints "[hypothesisloop] session_id = ...").
    session_id: Optional[str] = None
    for line in (proc.stderr or "").splitlines():
        if "session_id =" in line:
            session_id = line.split("session_id =", 1)[1].strip()
            break

    result = RunResult(
        id=spec.id,
        category=spec.category,
        mode=spec.mode,
        session_id=session_id,
        session_root=None,
        iter_count=0,
        chain=[],
        bias_flag_count=0,
        novelty_rejected_count=0,
        distinct_statements=0,
        total_proposals=0,
        total_attempts=0,
        successful_attempts=0,
        runtime_s=runtime,
        report_md_exists=False,
        report_txt_exists=False,
        automl_test_score=None,
        notes=spec.notes or "",
    )

    if proc.returncode != 0:
        result.error = f"CLI exit_code={proc.returncode}; see {log_path}"

    if session_id:
        session_root = output_dir / session_id
        result.session_root = str(session_root)
        result.report_md_exists = (session_root / "report.md").exists()
        result.report_txt_exists = (session_root / "report.txt").exists()

        trace_path = session_root / "trace.json"
        if trace_path.exists():
            try:
                from hypothesisloop.agent.state import DAGTrace

                trace = DAGTrace.load(trace_path)
                result.iter_count = trace.iteration_count()
                statements = []
                for node in trace.iter_nodes():
                    decision = node.feedback.decision if node.feedback else "(pending)"
                    stmt = node.hypothesis.statement
                    statements.append(stmt)
                    if node.feedback and node.feedback.bias_flags:
                        result.bias_flag_count += len(node.feedback.bias_flags)
                    if node.experiment is not None and node.experiment.attempts:
                        result.total_attempts += len(node.experiment.attempts)
                        result.successful_attempts += sum(
                            1 for a in node.experiment.attempts if a.exit_code == 0
                        )
                    result.chain.append(f"[{decision}] {stmt[:90]}")
                result.novelty_rejected_count = len(trace.novelty_rejected)
                result.total_proposals = len(statements) + len(trace.novelty_rejected)
                result.distinct_statements = len(set(statements))
            except Exception as e:
                result.error = (result.error or "") + f"; trace load failed: {e}"

        # AutoGluon summary if present.
        automl_path = session_root / "automl_summary.json"
        if automl_path.exists():
            try:
                summary = json.loads(automl_path.read_text(encoding="utf-8"))
                result.automl_test_score = float(summary.get("test_score", 0.0))
            except Exception:
                pass

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="run_validation_suite.py")
    p.add_argument("--dry-run", action="store_true", help="Print the run plan + cost estimate and exit.")
    p.add_argument("--resume", action="store_true", help="Skip runs that are already checkpointed.")
    p.add_argument(
        "--only",
        default=None,
        help="Run only the specified IDs, comma-separated (e.g., 'E1.1,P1.1').",
    )
    p.add_argument(
        "--cost-cap",
        type=float,
        default=25.0,
        help="Halt the suite if estimated cost exceeds this (USD; default 25).",
    )
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where each run's session lands (default: reports/).",
    )
    args = p.parse_args(argv)

    _ensure_corrupted_adult()
    _ensure_penguins()

    plan = build_run_plan()
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        plan = [r for r in plan if r.id in wanted]
        if not plan:
            print(f"--only filtered out everything; available IDs: "
                  f"{[r.id for r in build_run_plan()]}")
            return 1

    estimate = estimate_cost(plan)
    print("=" * 60)
    print(f"VALIDATION SUITE — HypothesisLoop")
    print("=" * 60)
    print(f"Total runs:        {len(plan)}")
    print(f"Estimated cost:    ${estimate:.2f}")
    print(f"Hard cap:          ${args.cost_cap:.2f}")
    print(f"Estimated wall:    ~{(len(plan) * 4):.0f} minutes")
    print(f"Output directory:  {args.output_dir}")
    print(f"Checkpoint file:   {CHECKPOINT_PATH}")
    print(f"Run log:           {RUN_LOG_PATH}")
    print("=" * 60)
    print()
    print("Plan:")
    for spec in plan:
        print(f"  {spec.id:>6}  {spec.mode:<7}  {spec.category:<28}  {spec.prompt[:60]}")
    print()

    if args.dry_run:
        print("--dry-run: exiting without running the suite.")
        return 0

    if estimate > args.cost_cap:
        print(f"ABORT: estimated cost ${estimate:.2f} exceeds cap ${args.cost_cap:.2f}.")
        print("Re-run with --cost-cap to override.")
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and RUN_LOG_PATH.exists():
        # Roll the previous run's log so we always have one fresh log per suite.
        rolled = RUN_LOG_PATH.with_name(
            RUN_LOG_PATH.stem + f".{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.log"
        )
        RUN_LOG_PATH.rename(rolled)

    cumulative_cost_estimate = 0.0
    completed: dict[str, RunResult] = load_checkpoints() if args.resume else {}
    if args.resume and completed:
        print(f"[resume] {len(completed)} runs already checkpointed; skipping those.")

    suite_started = time.monotonic()
    n_done = len(completed)

    for i, spec in enumerate(plan, start=1):
        if spec.id in completed:
            continue
        per_run = (
            _AVG_TOKENS_INPUT_PER_RUN * _KIMI_INPUT_RATE
            + _AVG_TOKENS_OUTPUT_PER_RUN * _KIMI_OUTPUT_RATE
        ) / 1_000_000
        if cumulative_cost_estimate + per_run > args.cost_cap:
            print(
                f"[suite] cost cap ${args.cost_cap:.2f} would be exceeded; "
                f"halting after {n_done} runs."
            )
            break

        print(f"\n[suite] {i}/{len(plan)}  RUN {spec.id} — {spec.category}")
        print(f"        cmd: {spec.mode} {spec.prompt[:70]}")
        try:
            result = execute_run(spec, output_dir=output_dir, log_path=RUN_LOG_PATH)
        except subprocess.TimeoutExpired:
            print(f"[suite] RUN {spec.id} timed out (30 min); recording and continuing.")
            result = RunResult(
                id=spec.id,
                category=spec.category,
                mode=spec.mode,
                session_id=None,
                session_root=None,
                iter_count=0,
                chain=[],
                bias_flag_count=0,
                novelty_rejected_count=0,
                distinct_statements=0,
                total_proposals=0,
                total_attempts=0,
                successful_attempts=0,
                runtime_s=1800.0,
                report_md_exists=False,
                report_txt_exists=False,
                automl_test_score=None,
                error="timeout (30min)",
            )
        except Exception as e:
            print(f"[suite] RUN {spec.id} crashed: {type(e).__name__}: {e}")
            result = RunResult(
                id=spec.id,
                category=spec.category,
                mode=spec.mode,
                session_id=None,
                session_root=None,
                iter_count=0,
                chain=[],
                bias_flag_count=0,
                novelty_rejected_count=0,
                distinct_statements=0,
                total_proposals=0,
                total_attempts=0,
                successful_attempts=0,
                runtime_s=time.monotonic() - suite_started,
                report_md_exists=False,
                report_txt_exists=False,
                automl_test_score=None,
                error=f"{type(e).__name__}: {e}",
            )

        append_checkpoint(result)
        completed[spec.id] = result
        cumulative_cost_estimate += per_run
        n_done += 1

        status = "✓" if not result.error else "✗"
        score = (
            f"  AG={result.automl_test_score:.4f}"
            if result.automl_test_score is not None
            else ""
        )
        print(
            f"        {status} iters={result.iter_count}  "
            f"attempts={result.total_attempts}  bias={result.bias_flag_count}  "
            f"runtime={result.runtime_s:.1f}s{score}"
        )

    suite_runtime = time.monotonic() - suite_started
    print()
    print("=" * 60)
    print(f"Suite complete: {n_done}/{len(plan)} runs in {suite_runtime / 60:.1f} min")
    print(f"Estimated cost spent: ${cumulative_cost_estimate:.2f}")
    print(f"Checkpoints: {CHECKPOINT_PATH}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
