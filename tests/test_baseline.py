"""Phase 9 baseline-iteration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hypothesisloop.agent.state import DAGTrace
from hypothesisloop.steps.baseline import auto_metric_for, auto_task_type, run_baseline


def _binary_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(size=n)
    cat = rng.choice(["A", "B", "C"], size=n)
    y = (x1 + (cat == "A") * 0.8 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "cat": cat, "y": y})


def _regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(size=n)
    y = 1.7 * x1 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"x1": x1, "y": y})


def test_auto_task_type_binary_int():
    df = _binary_df()
    assert auto_task_type(df["y"]) == "classification"


def test_auto_task_type_continuous_regression():
    df = _regression_df()
    assert auto_task_type(df["y"]) == "regression"


def test_auto_task_type_object_dtype_is_classification():
    s = pd.Series(["yes", "no", "yes", "yes", "no"])
    assert auto_task_type(s) == "classification"


def test_auto_metric_for_classification_is_auc():
    assert auto_metric_for("classification") == "roc_auc"


def test_auto_metric_for_regression_is_r2():
    assert auto_metric_for("regression") == "r2"


def test_run_baseline_mutates_trace_classification():
    df = _binary_df()
    trace = DAGTrace(
        session_id="b1",
        dataset_path="(memory)",
        question="?",
        mode="predict",
        target_column="y",
        task_type="classification",
        metric_name="roc_auc",
    )
    result = run_baseline(trace, df, seed=42)

    assert trace.baseline_score is not None
    assert trace.current_best_score == trace.baseline_score
    assert 0.5 <= trace.baseline_score <= 1.0
    assert result.train_df is not None and result.test_df is not None
    # 80/20 split.
    assert 0.78 < len(result.train_df) / len(df) < 0.82


def test_run_baseline_regression_sets_r2():
    df = _regression_df()
    trace = DAGTrace(
        session_id="b2",
        dataset_path="(memory)",
        question="?",
        mode="predict",
        target_column="y",
        task_type="regression",
    )
    run_baseline(trace, df, seed=42)
    assert trace.metric_name == "r2"
    assert trace.baseline_score is not None
    assert trace.current_best_score == trace.baseline_score


def test_run_baseline_requires_target_column():
    trace = DAGTrace(
        session_id="b3",
        dataset_path="(memory)",
        question="?",
        mode="predict",
    )
    try:
        run_baseline(trace, _binary_df())
    except ValueError as e:
        assert "target_column" in str(e)
    else:
        raise AssertionError("expected ValueError")
