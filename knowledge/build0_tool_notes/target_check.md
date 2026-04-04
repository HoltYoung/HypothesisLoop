# target_check

## Purpose
Create a focused summary of one target or outcome variable.

## When to use
Use when the user has a specific outcome in mind and the agent needs to determine what kind of variable it is.

## Inputs
- `df`
- target column name

## Outputs
A quick summary including:
- variable type
- missing rate
- number of unique values
- descriptive statistics if numeric
- top levels if categorical

## Why it helps an agent
It helps the agent decide whether the target is continuous, categorical, binary, sparse, or highly missing.

## Example interpretation
If the target is binary, a linear regression tool may not be appropriate.
