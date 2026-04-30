"""Phase 3 profile tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hypothesisloop.steps.profile import profile_dataset


def test_profile_dataset_basic():
    df = pd.DataFrame(
        {
            "age": np.linspace(20, 70, 50),
            "occupation": (["a", "b", "c"] * 17)[:50],
            "income": [100.0, 200.0, None] * 16 + [None, None],
        }
    )
    out = profile_dataset(df, dataset_path="data/test.csv")

    assert "Numeric columns" in out
    assert "Categorical columns" in out
    assert "Missingness" in out
    assert "age" in out and "occupation" in out and "income" in out
    assert len(out) < 4096


def test_profile_truncates_long_categorical():
    values = [f"cat{i}" for i in range(100)]
    df = pd.DataFrame({"thing": values * 2})  # 200 rows, 100 unique
    out = profile_dataset(df, max_categories=5)

    # Header records the full count, but the per-value table caps at 5.
    assert "100 unique" in out
    assert "more values not shown" in out
    # Only top-5 rows in the value table — count "| cat" lines.
    body = out.split("### `thing`", 1)[1]
    cat_rows = [ln for ln in body.splitlines() if ln.startswith("| cat")]
    assert len(cat_rows) == 5


def test_profile_handles_empty_df():
    df = pd.DataFrame()
    out = profile_dataset(df)
    assert "0 rows" in out
    assert "empty dataset" in out
