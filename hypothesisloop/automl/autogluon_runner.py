"""AutoGluon ensemble training, run once at the end of a Predict-mode session.

Uses the engineered-feature DataFrame (original columns + accepted features)
that the loop has been accumulating. Trains on the 80% train split that was
reserved at baseline; evaluates on the held-out 20% test split.

The AutoGluon import is **deferred** to inside ``run_automl`` — keeping
``--help`` and ``--report-only`` cold-starts fast for users who never hit a
Predict-mode run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import pandas as pd


def run_automl(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    task_type: Literal["classification", "regression"],
    output_dir: Path | str,
    *,
    time_budget_s: int = 120,
    presets: str = "medium_quality",
    seed: int = 42,
) -> dict:
    """Train AutoGluon and evaluate on the held-out test split.

    Returns::

        {
          "leaderboard":         list[dict],   # one row per model
          "feature_importance":  list[dict],   # importance (SHAP-style) per feature
          "test_score":          float,        # primary metric on test
          "test_metric":         str,          # the AG metric name
          "best_model":          str,
          "model_dir":           str,
          "time_budget_s":       int,
        }

    Raises ``ImportError`` if AutoGluon can't be loaded — callers surface a
    clear message and continue rendering the report without the trained model.
    """
    from autogluon.tabular import TabularPredictor  # deferred import

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"

    # Pick the AutoGluon eval metric that matches the loop's metric where
    # possible. AG accepts these strings out of the box.
    if task_type == "classification":
        eval_metric = "roc_auc"
        ag_problem_type = "binary" if train_df[target_column].nunique() == 2 else "multiclass"
        if ag_problem_type == "multiclass":
            eval_metric = "log_loss"
    else:
        eval_metric = "r2"
        ag_problem_type = "regression"

    predictor = TabularPredictor(
        label=target_column,
        eval_metric=eval_metric,
        problem_type=ag_problem_type,
        path=str(model_dir),
        verbosity=1,
    )
    predictor.fit(
        train_data=train_df,
        time_limit=time_budget_s,
        presets=presets,
    )

    # Test-set evaluation
    perf = predictor.evaluate(test_df, silent=True)
    if isinstance(perf, dict):
        test_score = float(perf.get(eval_metric, list(perf.values())[0]))
    else:
        test_score = float(perf)

    leaderboard_df = predictor.leaderboard(test_df, silent=True)
    leaderboard = leaderboard_df.to_dict(orient="records")
    leaderboard_path = output_dir / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)

    # Feature importance — best-effort; some AG versions can fail on
    # certain data shapes (e.g., very small holdout) so we trap and skip.
    importance: list[dict] = []
    try:
        imp_df = predictor.feature_importance(test_df, silent=True)
        # Reset the index so 'feature' column is explicit on disk.
        if hasattr(imp_df, "reset_index"):
            imp_df = imp_df.reset_index().rename(columns={"index": "feature"})
        importance = imp_df.to_dict(orient="records")
        (output_dir / "feature_importance.csv").write_text(
            imp_df.to_csv(index=False), encoding="utf-8"
        )
    except Exception as exc:
        importance = [{"_error": f"feature_importance failed: {type(exc).__name__}: {exc}"}]

    best_model = ""
    if leaderboard:
        # AG ranks descending by score_test; first row is best.
        best_model = str(leaderboard[0].get("model", ""))

    return {
        "leaderboard": leaderboard,
        "feature_importance": importance,
        "test_score": float(test_score),
        "test_metric": str(eval_metric),
        "best_model": best_model,
        "model_dir": str(model_dir),
        "time_budget_s": int(time_budget_s),
    }


def write_automl_summary(result: dict, output_dir: Path | str) -> Path:
    """Write a small JSON summary of the AutoML run for the report."""
    import json

    output_dir = Path(output_dir)
    summary_path = output_dir / "automl_summary.json"
    summary_path.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
    return summary_path


__all__ = ["run_automl", "write_automl_summary"]
