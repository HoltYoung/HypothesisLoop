# missingness_table

## Purpose
Count and rank missing values per column.

## When to use
Whenever the user asks about missing data, data quality, or completeness.

## Inputs
- `df`

## Outputs
A table with:
- column name
- count of missing values
- percentage missing

## Why it helps an agent
Lets the agent decide whether to drop columns, impute, or warn the user before modeling.

## Cautions
- Only detects true NaN. If missingness is encoded as `?`, `-999`, or a blank string, convert to NaN first.
- On the Adult dataset, `?` is the missing marker in `workclass`, `occupation`, and `native_country`.
