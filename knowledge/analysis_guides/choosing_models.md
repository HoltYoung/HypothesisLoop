# Choosing Simple Models

## Descriptive analysis
Choose summaries and plots when the goal is exploration rather than formal modeling.

## Multiple linear regression
Choose this when:
- the outcome is numeric
- the relationship is approximately linear
- predictors may be numeric or categorical

## When not to use linear regression
Avoid it when:
- the outcome is binary
- the outcome is categorical
- the outcome is a count with strong skew or zero inflation
- observations are clearly dependent in ways the model ignores

## Practical rule
A simple model is often best for a first pass. The agent should not jump to regression before understanding the data.
