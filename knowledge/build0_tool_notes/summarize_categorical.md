# summarize_categorical

## Purpose
Summarize categorical variables using counts and common values.

## When to use
Use when exploring non-numeric variables such as groups, labels, yes/no indicators, or survey responses.

## Inputs
- `df`
- `cat_cols`
- optional `top_k` for most frequent categories

## Outputs
A table showing:
- number of non-missing values
- missing values
- number of unique levels
- most common categories

## Why it helps an agent
It helps the agent identify dominant groups, rare levels, and variables that may have too many categories for a simple plot.

## Example interpretation
If one category contains 92 percent of observations, the variable is highly imbalanced.
