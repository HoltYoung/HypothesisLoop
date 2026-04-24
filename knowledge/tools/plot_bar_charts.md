# plot_bar_charts

## Purpose
Render a bar chart of category counts for each specified categorical column.

## When to use
When the user asks for the distribution, frequency, or breakdown of a categorical variable.

## Inputs
- `df`
- list of categorical column names

## Outputs
One PNG per column, saved to the report directory.

## Why it helps an agent
Makes dominant categories and class imbalance immediately visible.

## Cautions
- Do not use for numeric columns — use `plot_histograms`.
- Do not use this to show associations between a categorical and a numeric variable — use `plot_cat_num_boxplot`.
- High-cardinality columns (like `native_country` on Adult, with 40+ values) may render poorly; consider top-N filtering.
