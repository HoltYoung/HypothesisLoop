"""Phase 9 proxy-CV scoring tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypothesisloop.agent.predict_score import (
    ACCEPTANCE_THRESHOLDS,
    CVScore,
    cv_score,
    is_improvement,
    split_train_test,
)


def _toy_classification_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["A", "B", "C"], size=n)
    # Target depends on x1 + cat=="A" — gives the classifier real signal.
    logits = 1.5 * x1 + (cat == "A") * 0.8 + rng.normal(scale=0.5, size=n)
    y = (logits > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})


def _toy_regression_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.7 * x1 - 0.4 * x2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_split_train_test_classification_is_stratified():
    df = _toy_classification_df()
    train, test = split_train_test(df, "y", task_type="classification", seed=42)
    # Stratified split: class proportions in train and test stay close.
    assert len(train) + len(test) == len(df)
    train_pos = (train["y"] == 1).mean()
    test_pos = (test["y"] == 1).mean()
    full_pos = (df["y"] == 1).mean()
    assert abs(train_pos - full_pos) < 0.05
    assert abs(test_pos - full_pos) < 0.05


def test_split_train_test_regression_random():
    df = _toy_regression_df()
    train, test = split_train_test(df, "y", task_type="regression", seed=42)
    assert len(train) + len(test) == len(df)
    # 80/20 split.
    assert 0.78 < len(train) / len(df) < 0.82


def test_cv_score_classification_returns_valid_auc():
    df = _toy_classification_df(n=500)
    train, _ = split_train_test(df, "y", task_type="classification", seed=42)
    score = cv_score(train, "y", task_type="classification", metric_name="roc_auc", seed=42)
    assert isinstance(score, CVScore)
    assert score.metric_name == "roc_auc"
    assert 0.5 < score.value <= 1.0  # signal is real, AUC should beat chance
    assert len(score.fold_values) == score.n_folds


def test_cv_score_regression_returns_valid_r2():
    df = _toy_regression_df(n=500)
    train, _ = split_train_test(df, "y", task_type="regression", seed=42)
    score = cv_score(train, "y", task_type="regression", metric_name="r2", seed=42)
    assert score.metric_name == "r2"
    assert 0.0 < score.value < 1.0


def test_is_improvement_auc_threshold():
    prev = CVScore(metric_name="roc_auc", value=0.85, fold_values=[], n_folds=5)
    new_just_under = CVScore(metric_name="roc_auc", value=0.8505, fold_values=[], n_folds=5)
    new_passes = CVScore(metric_name="roc_auc", value=0.852, fold_values=[], n_folds=5)
    accepted_under, _ = is_improvement(prev, new_just_under)
    accepted_pass, delta = is_improvement(prev, new_passes)
    assert accepted_under is False
    assert accepted_pass is True
    assert delta == pytest.approx(0.002, abs=1e-9)


def test_is_improvement_log_loss_sign_flip():
    """log_loss is a loss — lower is better, so prev - new is the gain."""
    prev = CVScore(metric_name="log_loss", value=0.50, fold_values=[], n_folds=5)
    new_lower = CVScore(metric_name="log_loss", value=0.49, fold_values=[], n_folds=5)
    new_higher = CVScore(metric_name="log_loss", value=0.51, fold_values=[], n_folds=5)
    a_lower, d_lower = is_improvement(prev, new_lower)
    a_higher, d_higher = is_improvement(prev, new_higher)
    assert a_lower is True
    assert d_lower > 0
    assert a_higher is False
    assert d_higher < 0


def test_is_improvement_r2_threshold():
    prev = CVScore(metric_name="r2", value=0.40, fold_values=[], n_folds=5)
    new = CVScore(metric_name="r2", value=0.41, fold_values=[], n_folds=5)
    accepted, delta = is_improvement(prev, new)
    assert accepted is True
    assert delta == pytest.approx(0.01, abs=1e-9)


def test_acceptance_thresholds_table_includes_three_metrics():
    assert set(ACCEPTANCE_THRESHOLDS.keys()) == {"roc_auc", "log_loss", "r2"}
