"""Cheap proxy model + CV scoring for Predict mode.

Per-iteration scoring uses LogisticRegression (classification) or Ridge
(regression) under StratifiedKFold (cv=5; cv=10 for n<2000). The proxy is
deliberately simple and fast — sub-second per fit — so feature engineering
acceptance / rejection happens on the order of the LLM call itself rather
than on the order of an AutoGluon training. AutoGluon is reserved for the
final ensemble (see ``hypothesisloop.automl.autogluon_runner``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Acceptance thresholds — a feature must improve the metric by at least
# this much on cross-validation to be kept. Higher = stricter; tuned to
# UCI Adult's noise floor.
ACCEPTANCE_THRESHOLDS: dict[str, float] = {
    "roc_auc": 0.001,
    "log_loss": 0.001,  # log_loss is a *loss* — we flip the sign internally.
    "r2": 0.005,
}


@dataclass
class CVScore:
    metric_name: str
    value: float                  # mean across folds
    fold_values: list[float]
    n_folds: int


def _ohe_compat(**kwargs):
    """OneHotEncoder ``sparse_output`` is the post-1.2 name; older sklearn
    used ``sparse``. We lean on the new name; if the installed sklearn is
    very old, fall back."""
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(sparse=False, **kwargs)


def _make_preprocessor(features_df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = features_df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in features_df.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", _ohe_compat(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        # Edge case: no columns at all — fall back to a passthrough so we don't crash.
        return ColumnTransformer(transformers=[], remainder="passthrough")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_train_test(
    df: pd.DataFrame,
    target_column: str,
    *,
    task_type: Literal["classification", "regression"],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 split. Stratified for classification, random for regression."""
    if target_column not in df.columns:
        raise KeyError(f"target_column {target_column!r} not in dataframe columns")
    stratify = df[target_column] if task_type == "classification" else None
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def cv_score(
    train_df: pd.DataFrame,
    target_column: str,
    task_type: Literal["classification", "regression"],
    metric_name: Literal["roc_auc", "log_loss", "r2"],
    *,
    seed: int = 42,
) -> CVScore:
    """k-fold CV on the proxy model. k=10 for n<2000, else k=5."""
    if target_column not in train_df.columns:
        raise KeyError(f"target_column {target_column!r} not in train dataframe")

    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    n = len(train_df)
    n_folds = 10 if n < 2000 else 5

    if task_type == "classification":
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        # Encode the target for ROC AUC compatibility: prefer its original
        # (often binary) form, but binarize via sklearn's pos-label conventions.
        y_arr = y.to_numpy()
        classes = sorted(pd.Series(y_arr).dropna().unique().tolist())
        if len(classes) > 2 and metric_name == "roc_auc":
            # Multiclass + AUC is OvR; sklearn handles via decision_function shape.
            pass
        model_factory = lambda: LogisticRegression(max_iter=200, n_jobs=None, random_state=seed)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        y_arr = y.to_numpy()
        classes = []
        model_factory = lambda: Ridge(random_state=seed)

    fold_scores: list[float] = []
    for train_idx, val_idx in splitter.split(X, y_arr):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

        pre = _make_preprocessor(X_tr)
        pipe = Pipeline([("pre", pre), ("model", model_factory())])
        pipe.fit(X_tr, y_tr)

        if task_type == "classification":
            if metric_name == "roc_auc":
                proba = pipe.predict_proba(X_val)
                if proba.shape[1] == 2:
                    fold_scores.append(float(roc_auc_score(y_val, proba[:, 1])))
                else:
                    fold_scores.append(
                        float(roc_auc_score(y_val, proba, multi_class="ovr"))
                    )
            elif metric_name == "log_loss":
                proba = pipe.predict_proba(X_val)
                # log_loss expects label list when classes can vary fold-to-fold.
                fold_scores.append(
                    float(log_loss(y_val, proba, labels=pipe.classes_))
                )
            else:
                raise ValueError(f"unsupported classification metric: {metric_name}")
        else:
            preds = pipe.predict(X_val)
            if metric_name == "r2":
                fold_scores.append(float(r2_score(y_val, preds)))
            else:
                raise ValueError(f"unsupported regression metric: {metric_name}")

    return CVScore(
        metric_name=metric_name,
        value=float(np.mean(fold_scores)),
        fold_values=fold_scores,
        n_folds=n_folds,
    )


def is_improvement(prev: CVScore, new: CVScore) -> tuple[bool, float]:
    """Return ``(accepted, delta)``.

    For ``log_loss`` lower is better, so ``delta = prev - new``. For
    ``roc_auc`` / ``r2`` higher is better, so ``delta = new - prev``.
    Acceptance: ``delta >= ACCEPTANCE_THRESHOLDS[metric_name]``.
    """
    if prev.metric_name != new.metric_name:
        raise ValueError(
            f"metric mismatch: prev={prev.metric_name!r}, new={new.metric_name!r}"
        )
    metric = new.metric_name
    if metric == "log_loss":
        delta = prev.value - new.value
    else:
        delta = new.value - prev.value
    threshold = ACCEPTANCE_THRESHOLDS.get(metric, 0.0)
    return delta >= threshold, float(delta)


__all__ = [
    "ACCEPTANCE_THRESHOLDS",
    "CVScore",
    "split_train_test",
    "cv_score",
    "is_improvement",
]
