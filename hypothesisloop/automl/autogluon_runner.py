"""AutoGluon ensemble training, run once at the end of a Predict-mode session.

Uses the engineered-feature DataFrame (original columns + accepted features)
that the loop has been accumulating. Trains on the 80% train split that was
reserved at baseline; evaluates on the held-out 20% test split.

The AutoGluon import is **deferred** to inside ``run_automl`` — keeping
``--help`` and ``--report-only`` cold-starts fast for users who never hit a
Predict-mode run.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from hypothesisloop.agent.state import EngineeredFeature


def apply_engineered_features(
    df: pd.DataFrame, features: list[EngineeredFeature]
) -> pd.DataFrame:
    """Re-execute each accepted feature's code against ``df``.

    Returns a new DataFrame; the input is never mutated. Skips rejected
    features. Failures emit a ``RuntimeWarning`` and the failing feature
    is dropped from THIS split (the trace's record stays intact — a
    feature can succeed on train_df during the loop and fail on test_df
    here, e.g. when the test slice has different value ranges).

    The exec namespace exposes ``df``, ``pd``, ``np``, and a no-op
    ``hl_emit``. Code may rebind ``df`` (e.g. ``df = df.assign(...)``);
    we pull the rebound reference back from the namespace post-exec.

    Most LLM-generated FE code is stateless and idempotent — re-running
    the same transform on an already-engineered DataFrame produces the
    same column. Stateful transforms (target encoding, z-scoring against
    full-df stats) silently degrade quality on the test split because
    the helper has no way to share fitted state across calls. The
    Predict-mode prompt instructs the LLM to avoid those.
    """
    out = df.copy()
    failed: list[str] = []
    for feat in features:
        if not feat.accepted:
            continue
        ns: dict = {"df": out, "pd": pd, "np": np, "hl_emit": lambda *a, **k: None}
        try:
            exec(compile(feat.code, f"<feature:{feat.name}>", "exec"), ns)
            rebound = ns.get("df", out)
            if isinstance(rebound, pd.DataFrame):
                out = rebound
            else:
                raise TypeError(
                    f"feature code rebound `df` to {type(rebound).__name__}, expected DataFrame"
                )
        except Exception as e:
            warnings.warn(
                f"Feature {feat.name!r} failed to apply on this split: "
                f"{type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            failed.append(feat.name)
    if failed:
        warnings.warn(
            f"{len(failed)} engineered features dropped from this split: {failed}",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


def run_automl(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    task_type: Literal["classification", "regression"],
    output_dir: Path | str,
    *,
    engineered_features: Optional[list[EngineeredFeature]] = None,
    time_budget_s: int = 120,
    presets: str = "medium_quality",
    seed: int = 42,
) -> dict:
    """Train AutoGluon and evaluate on the held-out test split.

    ``engineered_features`` is the list from the trace
    (``trace.engineered_features``). When provided, every accepted feature's
    code is re-executed against BOTH ``train_df`` and ``test_df`` before
    fitting — idempotent on stateless transforms (the common case), so
    re-running on an already-engineered ``train_df`` produces the same
    columns. Without this re-execution the test split would only contain
    raw columns and the AutoGluon leaderboard would silently report a
    misleading score. See ``apply_engineered_features`` for failure semantics.

    Returns::

        {
          "leaderboard":         list[dict],
          "feature_importance":  list[dict],
          "test_score":          float,
          "test_metric":         str,
          "best_model":          str,
          "model_dir":           str,
          "time_budget_s":       int,
          "engineered_applied":  list[str],   # feature names that landed on test_df
        }

    Raises ``ImportError`` if AutoGluon can't be loaded — callers surface a
    clear message and continue rendering the report without the trained model.
    """
    from autogluon.tabular import TabularPredictor  # deferred import

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"

    # Re-execute accepted features on both splits. Idempotent on stateless
    # FE; the loop's mutated train_df ends up with the same columns it
    # already has. Test split picks up the engineered columns for the
    # first time, fixing the Phase 9 silent-degradation bug.
    accepted_features: list[EngineeredFeature] = []
    if engineered_features:
        accepted_features = [f for f in engineered_features if f.accepted]
    if accepted_features:
        train_df = apply_engineered_features(train_df, accepted_features)
        test_df = apply_engineered_features(test_df, accepted_features)
    engineered_applied = [f.name for f in accepted_features]

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
        "engineered_applied": engineered_applied,
    }


def write_automl_summary(result: dict, output_dir: Path | str) -> Path:
    """Write a small JSON summary of the AutoML run for the report."""
    import json

    output_dir = Path(output_dir)
    summary_path = output_dir / "automl_summary.json"
    summary_path.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
    return summary_path


__all__ = ["apply_engineered_features", "run_automl", "write_automl_summary"]
