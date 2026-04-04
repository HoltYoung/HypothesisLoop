# summarize_numeric

## Purpose
Compute descriptive statistics for numeric variables.

## When to use
Use when the goal is to understand the center, spread, and range of continuous or count variables.

## Inputs
- `df`
- `numeric_cols`: list of numeric columns

## Outputs
A table of descriptive statistics, often including:
- count
- mean
- standard deviation
- minimum
- 25th percentile
- median
- 75th percentile
- maximum

## Why it helps an agent
It helps the agent identify skew, unusual values, limited range, or variables that may need transformation.

## Example interpretation
If income has a mean far above the median, the distribution may be right-skewed.
