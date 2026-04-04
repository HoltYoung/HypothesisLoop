# plot_missingness

## Purpose
Visualize missing data rates across variables.

## When to use
Use when missingness is important to communicate visually instead of only as a table.

## Inputs
- missingness summary table
- output path
- optional limit such as `top_n`

## Outputs
A saved figure, often a bar chart of missingness rates

## Why it helps an agent
Plots make it easier to identify the worst variables quickly and to communicate data quality problems to a user.

## Example interpretation
A missingness plot may reveal that only a small number of variables drive most of the missing-data problem.
