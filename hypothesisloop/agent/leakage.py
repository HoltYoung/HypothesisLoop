"""Target-leakage AST guard for Predict mode.

Predict mode generates feature-engineering code. If the LLM writes a feature
that references the target column directly, the engineered feature
trivially leaks the answer and the proxy model's CV score becomes
meaningless.

This module AST-scans generated code BEFORE sandbox execution and rejects:
    - ``df['<target>']`` / ``df["<target>"]``  (Subscript with string key)
    - ``df.<target>``                         (Attribute access)

Comparison is case-insensitive and whitespace-stripped. Scanned code may
include the data-loader preamble; the loader's ``pd.read_csv`` references
neither, so it passes through unchanged.
"""

from __future__ import annotations

import ast
from typing import Optional


class TargetLeakageError(Exception):
    """Raised when feature-engineering code references the target column."""


def check_no_target_leakage(code: str, target_column: str) -> Optional[str]:
    """Return ``None`` if no leakage; else a retry-friendly error string."""
    if not target_column:
        return None
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Let the runner surface the SyntaxError normally — don't mask it
        # with a "leakage" complaint when the actual problem is a parse fail.
        return None

    target_norm = target_column.lower().strip()

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            slc = node.slice
            # Python 3.9+ unwraps Index → Constant directly.
            if isinstance(slc, ast.Constant) and isinstance(slc.value, str):
                if slc.value.lower().strip() == target_norm:
                    return _msg(target_column)
        elif isinstance(node, ast.Attribute):
            if node.attr.lower().strip() == target_norm:
                return _msg(target_column)

    return None


def _msg(target_column: str) -> str:
    return (
        f"Target column {target_column!r} must not appear in feature-engineering "
        "code (target leakage). Refactor without referencing the target."
    )


__all__ = ["TargetLeakageError", "check_no_target_leakage"]
