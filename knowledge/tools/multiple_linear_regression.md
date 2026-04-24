# multiple_linear_regression

## Purpose
Fit an ordinary least squares multiple linear regression model.

## When to use
When the outcome is numeric and the goal is to estimate linear relationships between predictors and the outcome.

## Inputs
- `df`
- `outcome`: name of the outcome column (numeric)
- `predictors`: optional list of predictor column names

Note: the argument is named `outcome`, not `target`.

## Outputs
A structured summary containing:
- formula
- predictors used
- number of observations
- coefficients
- standard errors
- p-values
- confidence intervals
- R-squared and adjusted R-squared
- AIC and BIC

## Why it helps an agent
Provides an interpretable baseline for continuous outcomes and reports effect sizes the agent can cite in its summary.

## Cautions
- Outcome must be numeric. On the Adult dataset, `income` is binary — do NOT use this tool on it. Use `hours_per_week` or another numeric column, or fall back to code generation for logistic regression.
- Missing data trigger listwise deletion.
- Linearity and independence assumptions still matter; discuss them in the interpretation.
- Categorical predictors need encoding before fitting.
