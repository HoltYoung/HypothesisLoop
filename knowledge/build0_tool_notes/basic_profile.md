# basic_profile

## Purpose
Create a high-level overview of a dataset before deeper analysis.

## When to use
Use at the start of almost any project to understand the size, structure, and missingness of the data.

## Inputs
- `df`: a pandas DataFrame

## Outputs
A summary object or dictionary with information such as:
- number of rows
- number of columns
- column names
- data types
- total missing values
- missing values by column
- memory usage

## Why it helps an agent
This tool gives the agent a quick understanding of the data environment before it chooses plots, summaries, or models.

## Example interpretation
If a dataset has 2,000 rows, 14 columns, and 3 variables with substantial missingness, the agent should treat missing data as an important part of the workflow.
