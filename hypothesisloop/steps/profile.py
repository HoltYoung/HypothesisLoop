"""Round-zero dataset profiler.

Returns a markdown-shaped schema summary suitable for prompt injection. The
target is "<2K tokens" — readable to the LLM, dense enough to pick a sensible
first hypothesis from. Numeric stats are rounded; categorical sections are
top-K only.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def _fmt_num(x) -> str:
    """Compact numeric formatter that survives ints, floats, and NaN."""
    if pd.isna(x):
        return "nan"
    if isinstance(x, (int,)) or (hasattr(x, "is_integer") and x.is_integer()):
        return f"{int(x)}"
    return f"{float(x):.4g}"


def _numeric_section(df: pd.DataFrame) -> str:
    cols = df.select_dtypes(include="number").columns.tolist()
    if not cols:
        return "## Numeric columns\n_(none)_"

    lines = ["## Numeric columns", "", "| column | min | mean | max | missing% |", "|---|---|---|---|---|"]
    for c in cols:
        s = df[c]
        miss_pct = 100.0 * s.isna().mean()
        lines.append(
            f"| `{c}` | {_fmt_num(s.min(skipna=True))} "
            f"| {_fmt_num(s.mean(skipna=True))} "
            f"| {_fmt_num(s.max(skipna=True))} "
            f"| {miss_pct:.1f}% |"
        )
    return "\n".join(lines)


def _categorical_section(df: pd.DataFrame, max_categories: int) -> str:
    cols = df.select_dtypes(exclude="number").columns.tolist()
    if not cols:
        return "## Categorical columns\n_(none)_"

    parts = ["## Categorical columns"]
    for c in cols:
        s = df[c]
        n_unique = s.nunique(dropna=True)
        miss_pct = 100.0 * s.isna().mean()
        counts = s.value_counts(dropna=True).head(max_categories)
        total = len(s)
        parts.append(f"\n### `{c}`  ({n_unique} unique, {miss_pct:.1f}% missing)")
        parts.append("")
        parts.append("| value | count | % |")
        parts.append("|---|---|---|")
        for value, count in counts.items():
            pct = 100.0 * count / total if total else 0.0
            value_str = str(value).replace("|", "\\|")
            parts.append(f"| {value_str} | {int(count)} | {pct:.1f}% |")
        if n_unique > max_categories:
            parts.append(f"_…+{n_unique - max_categories} more values not shown_")
    return "\n".join(parts)


def _missingness_section(df: pd.DataFrame) -> str:
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        return "## Missingness\nNo missing values."
    lines = ["## Missingness", "", "| column | missing% |", "|---|---|"]
    for col, pct in miss.items():
        lines.append(f"| `{col}` | {100.0 * pct:.1f}% |")
    return "\n".join(lines)


def profile_dataset(
    df: pd.DataFrame,
    *,
    max_categories: int = 5,
    dataset_path: Optional[str] = None,
) -> str:
    """Return a markdown schema summary for prompt injection."""
    label = dataset_path or "<unspecified>"
    rows, cols = df.shape
    if rows == 0 or cols == 0:
        return (
            f"# Dataset: {label} ({rows} rows × {cols} columns)\n\n"
            "_(empty dataset — no schema to profile)_"
        )

    header = f"# Dataset: {label} ({rows} rows × {cols} columns)"
    sections = [
        header,
        "",
        _numeric_section(df),
        "",
        _categorical_section(df, max_categories=max_categories),
        "",
        _missingness_section(df),
    ]
    return "\n".join(sections)


__all__ = ["profile_dataset"]
