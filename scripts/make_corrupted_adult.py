"""Create ``data/adult_corrupted.csv``: original UCI Adult with mangled column names.

Tests the agent's resilience to bad data. The actual data is unchanged; only
the header line is malformed (mixed case, non-ASCII punctuation, abbreviations,
hyphens vs underscores).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src = project_root / "data" / "adult.csv"
    dst = project_root / "data" / "adult_corrupted.csv"

    df = pd.read_csv(src)
    expected_cols = 15
    if len(df.columns) != expected_cols:
        raise RuntimeError(
            f"expected {expected_cols} columns in {src}, got {len(df.columns)}: "
            f"{list(df.columns)}"
        )

    # Mangled but recognizable. The agent should retry until it figures out
    # which mess of casing/abbreviations corresponds to which target.
    mangled = [
        "AGe",
        "WorkClass!!",
        "fnlwgt_unknown",
        "EDU",
        "education-num",
        "marital",
        "occu",
        "relat",
        "RACE_!",
        "SEx",
        "cap_gain",
        "cap_loss",
        "hpw",
        "country",
        "income",
    ]
    df.columns = mangled
    df.to_csv(dst, index=False)
    print(f"Wrote {dst} with {len(df)} rows.")


if __name__ == "__main__":
    main()
