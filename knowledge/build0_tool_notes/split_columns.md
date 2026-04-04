# split_columns

## Purpose
Separate variables into numeric and categorical groups.

## When to use
Use early in the workflow before deciding which summaries and plots are appropriate.

## Inputs
- `df`: a pandas DataFrame

## Outputs
Two lists:
- numeric column names
- categorical column names

## Why it helps an agent
Many analysis choices depend on variable type. Histograms and correlations are usually for numeric variables, while bar charts are usually better for categorical variables.

## Typical use
This function is often called after basic profiling so the agent can decide what kinds of tools to run next.
