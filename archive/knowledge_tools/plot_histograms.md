# plot_histograms

## Purpose
Render a histogram for each specified numeric column.

## When to use
When the user asks for the distribution or shape of a numeric variable, or wants to eyeball skew and outliers.

## Inputs
- `df`
- `numeric_cols`: a list of numeric column names

## Outputs
One PNG per column, saved to the report directory.

## Why it helps an agent
Visual complement to `summarize_numeric` — reveals skew and multimodality that a summary table can hide.

## Cautions
- Only works on numeric columns. Use `plot_bar_charts` for categorical distributions.
- On highly skewed columns like `capital_gain` or `capital_loss` in the Adult dataset, consider a log transform before plotting.
