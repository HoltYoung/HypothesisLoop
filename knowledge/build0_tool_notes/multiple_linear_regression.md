# multiple_linear_regression

## Purpose
Fit an ordinary least squares multiple linear regression model.

## When to use
Use when the outcome is numeric and the goal is to estimate linear relationships between predictors and the outcome.

## Inputs
- `df`
- outcome variable
- optional list of predictors

## Outputs
A structured summary containing information such as:
- formula
- predictors used
- number of observations
- coefficients
- standard errors
- p-values
- confidence intervals
- R-squared
- adjusted R-squared
- AIC and BIC

## Why it helps an agent
It provides a simple interpretable baseline model for continuous outcomes.

## Cautions
- outcome should be numeric
- missing data often lead to listwise deletion
- assumptions of linearity and independence still matter
- categorical predictors may need encoding or formula handling
