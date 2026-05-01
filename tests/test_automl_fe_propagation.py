"""Phase 10A — engineered features must propagate to test_df at AutoGluon time."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from hypothesisloop.agent.state import EngineeredFeature
from hypothesisloop.automl.autogluon_runner import apply_engineered_features


def _fake_feature(
    name: str,
    code: str,
    *,
    accepted: bool = True,
    iter_idx: int = 1,
) -> EngineeredFeature:
    return EngineeredFeature(
        name=name,
        code=code,
        iteration_added=iter_idx,
        hypothesis_id="h1",
        predicted_delta=0.005,
        actual_delta=0.003,
        accepted=accepted,
        rejection_reason=None,
    )


def test_apply_features_to_raw_test_df():
    """Stateless transform applied to raw test split produces the same column train would have."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    feat = _fake_feature("log_x", "df['log_x'] = np.log(df['x'])")
    out = apply_engineered_features(df, [feat])
    np.testing.assert_array_almost_equal(out["log_x"].values, np.log([1, 2, 3, 4, 5]))


def test_apply_skips_rejected_features():
    df = pd.DataFrame({"x": [1, 2, 3]})
    feat = _fake_feature("log_x", "df['log_x'] = np.log(df['x'])", accepted=False)
    out = apply_engineered_features(df, [feat])
    assert "log_x" not in out.columns


def test_apply_failed_feature_emits_warning_and_continues():
    """A feature that crashes shouldn't kill the run; subsequent features still apply."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    bad = _fake_feature("bad", "df['bad'] = df['nonexistent_column']")
    good = _fake_feature("sq", "df['x_sq'] = df['x'] ** 2", iter_idx=2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = apply_engineered_features(df, [bad, good])
    messages = [str(warning.message) for warning in w]
    # First warning: per-feature failure. Second warning: aggregate dropped list.
    assert any("'bad'" in m and "failed" in m for m in messages)
    assert any("dropped" in m and "bad" in m for m in messages)
    assert "x_sq" in out.columns
    assert "bad" not in out.columns


def test_apply_idempotent_on_already_engineered_df():
    """Re-applying a stateless feature should produce the same output."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    feat = _fake_feature("log_x", "df['log_x'] = np.log(df['x'])")
    once = apply_engineered_features(df, [feat])
    twice = apply_engineered_features(once, [feat])
    pd.testing.assert_frame_equal(once, twice)


def test_apply_handles_df_rebinding():
    """LLM may write `df = df.assign(...)`; the helper pulls the rebound df back."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    feat = _fake_feature("log_x", "df = df.assign(log_x=np.log(df['x']))")
    out = apply_engineered_features(df, [feat])
    assert "log_x" in out.columns
    np.testing.assert_array_almost_equal(out["log_x"].values, np.log([1, 2, 3]))


def test_apply_does_not_mutate_input():
    df = pd.DataFrame({"x": [1, 2, 3]})
    feat = _fake_feature("log_x", "df['log_x'] = np.log(df['x'])")
    _ = apply_engineered_features(df, [feat])
    assert "log_x" not in df.columns
    assert list(df.columns) == ["x"]
