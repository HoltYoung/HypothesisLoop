# correlations

## Purpose
Compute pairwise correlations among numeric variables.

## When to use
Use when exploring linear relationships among multiple numeric variables.

## Inputs
- `df`
- `numeric_cols`

## Outputs
A correlation matrix

## Why it helps an agent
It helps the agent find variables that move together, identify redundancy among predictors, and spot possible inputs for later modeling.

## Caution
Correlation does not imply causation, and correlation only captures specific kinds of association.

## Example interpretation
A correlation of 0.88 between two predictors may suggest multicollinearity concerns in a linear model.
