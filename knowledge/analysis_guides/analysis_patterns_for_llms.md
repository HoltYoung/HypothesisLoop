# Analysis Patterns for LLM-Based Data Analysis Agents

## Variable type patterns
- If the variable is numeric, think about summaries, histograms, correlations, and regression.
- If the variable is categorical, think about counts, proportions, and bar charts.
- If the outcome is binary, do not default to linear regression.

## Question-to-method patterns
- "Show the distribution" often implies a histogram for numeric data.
- "Show counts by category" often implies a bar chart.
- "How are these variables related?" may imply correlations if the variables are numeric.
- "Predict a numeric outcome" may imply multiple linear regression.

## Guardrails
The agent should always check variable type and missingness before choosing a method.
