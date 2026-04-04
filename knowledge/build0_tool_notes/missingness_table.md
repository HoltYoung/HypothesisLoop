# missingness_table

## Purpose
Calculate missing counts and missing rates for each variable.

## When to use
Use before modeling, before imputation, and early in exploratory analysis.

## Inputs
- `df`

## Outputs
A table with:
- variable name
- missing count
- missing percentage or rate

## Why it helps an agent
Missing data can influence model choice, usable sample size, and interpretation. This tool makes that visible immediately.

## Example interpretation
If 40 percent of one predictor is missing, the agent should be cautious about using it in a regression without discussing missing-data handling.
