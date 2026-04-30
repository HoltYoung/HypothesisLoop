# summarize_numeric

## Purpose
Return descriptive statistics for numeric columns: count, mean, std, min, quartiles, max.

## When to use
When the user asks for a statistical summary of numeric variables, or wants to understand scale, center, and spread before plotting or modeling.

## Inputs
- `df`
- optional list of numeric columns

## Outputs
A table with one row per numeric column and descriptive-statistics columns.

## Why it helps an agent
Surfaces skew, outliers (via min/max vs quartiles), and scale differences that affect modeling choices.

## Cautions
- Ignores categorical columns.
- Does not handle missing values beyond standard pandas behavior.
