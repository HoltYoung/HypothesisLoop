# basic_profile

## Purpose
Report a high-level overview of a DataFrame: number of rows, number of columns, and the dtype of each column.

## When to use
Call this first whenever a new dataset is loaded, or when the user asks to "describe" or "summarize" the dataset at a structural level.

## Inputs
- `df`

## Outputs
A dict-like summary containing:
- `n_rows`
- `n_cols`
- `columns` and their dtypes

## Why it helps an agent
Grounds subsequent tool choices by exposing the number and type of columns available.

## Cautions
- Does not detect missing values. Pair with `missingness_table`.
- Does not distinguish numeric vs categorical for plotting purposes. Use `split_columns` for that.
