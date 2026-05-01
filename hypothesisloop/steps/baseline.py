"""Iter-0 baseline (Predict mode only).

Runs the proxy model on the original feature set (no engineering) so the
loop has something to compare engineered features against. Mutates
``trace.baseline_score`` and ``trace.current_best_score`` in place.

The 80/20 train/test split is taken here once; the test set is reserved
for AutoGluon's final evaluation. The baseline CV runs on the train half.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from hypothesisloop.agent.predict_score import CVScore, cv_score, split_train_test
from hypothesisloop.agent.state import DAGTrace, MetricName, TaskType


@dataclass
class BaselineResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    baseline_cv: CVScore


def auto_metric_for(task_type: TaskType) -> MetricName:
    if task_type == "classification":
        return "roc_auc"
    return "r2"


def auto_task_type(series: pd.Series) -> TaskType:
    """Cheap heuristic: ≤20 unique non-null values OR object dtype → classification."""
    if series.dtype == object or str(series.dtype).startswith("category"):
        return "classification"
    n_unique = series.nunique(dropna=True)
    return "classification" if n_unique <= 20 else "regression"


def run_baseline(
    trace: DAGTrace,
    df: pd.DataFrame,
    *,
    seed: int = 42,
) -> BaselineResult:
    """Mutate trace with baseline + best score; return the train/test split + score."""
    if trace.target_column is None:
        raise ValueError("run_baseline requires trace.target_column to be set")
    if trace.task_type is None:
        raise ValueError("run_baseline requires trace.task_type to be set")
    if trace.metric_name is None:
        trace.metric_name = auto_metric_for(trace.task_type)

    train_df, test_df = split_train_test(
        df,
        trace.target_column,
        task_type=trace.task_type,
        seed=seed,
    )
    score = cv_score(
        train_df,
        target_column=trace.target_column,
        task_type=trace.task_type,
        metric_name=trace.metric_name,
        seed=seed,
    )
    trace.baseline_score = float(score.value)
    trace.current_best_score = float(score.value)
    return BaselineResult(train_df=train_df, test_df=test_df, baseline_cv=score)


__all__ = ["BaselineResult", "auto_metric_for", "auto_task_type", "run_baseline"]
