# pearson_correlation

## Purpose
Compute Pearson correlation coefficients for all pairs of numeric columns.

## When to use
When the user asks about linear relationships, correlations, or associations among numeric variables.

## Inputs
- `df`
- optional list of numeric columns

## Outputs
A symmetric numeric-by-numeric correlation matrix.

## Why it helps an agent
Identifies collinearity before modeling and highlights candidate predictors.

## Cautions
- Pearson assumes approximately linear relationships. Non-linear or ordinal relationships are better measured with Spearman.
- Only computes on numeric columns — encode categoricals separately.
- The output of this tool is the input for `plot_corr_heatmap`. Never call `plot_corr_heatmap` directly on raw data.
