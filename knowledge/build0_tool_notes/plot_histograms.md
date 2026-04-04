# plot_histograms

## Purpose
Create histograms for one or more numeric variables.

## When to use
Use when checking distribution shape, skewness, outliers, or concentration near zero.

## Inputs
- `df`
- `numeric_cols`
- figure directory
- optional limit on number of variables

## Outputs
Saved histogram figures

## Why it helps an agent
Histograms help the agent understand whether a variable looks symmetric, skewed, zero-heavy, or unusual.

## Example interpretation
If a variable has a long right tail, the agent may suggest a transformation or at least caution in interpretation.
