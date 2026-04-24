# Standard Data-Analysis Workflow

A dependable order of operations for a new dataset.

## 1. Profile
Start with `basic_profile` to confirm row/column counts and dtypes. Run `split_columns` to separate numeric and categorical variables.

## 2. Missingness
Run `missingness_table` and `plot_missingness`. On Adult-style census data, first convert `?` to NaN so missingness is detected correctly.

## 3. Univariate summaries
Use `summarize_numeric` for numeric columns and `summarize_categorical` for categorical columns. Plot with `plot_histograms` and `plot_bar_charts`.

## 4. Bivariate exploration
Use `pearson_correlation` (then `plot_corr_heatmap`) for numeric-numeric relationships. Use `plot_cat_num_boxplot` for categorical-numeric comparisons.

## 5. Target check
Before modeling, run `target_check` on the outcome column to confirm its type matches the intended model.

## 6. Modeling
For numeric outcomes, `multiple_linear_regression` is a baseline. For binary outcomes like `income`, use code generation to fit a logistic regression or similar classifier.

## 7. Interpretation
Always follow tool output with a short natural-language summary. Cite effect sizes, p-values, or model fit statistics by name — do not just echo numbers.
