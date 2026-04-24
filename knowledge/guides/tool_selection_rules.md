# Tool Selection Rules

## Dataset overview requests
If the user asks to describe or summarize the dataset, start with `basic_profile` and `split_columns` to separate numeric and categorical variables.

## Missing-data requests
If the user asks about missing values, use `missingness_table`. For a visual pattern, also call `plot_missingness`. On the Adult dataset, remember that `?` is the missing marker in `workclass`, `occupation`, and `native_country`.

## Distribution requests
If the user asks for the distribution of a numeric variable, use `plot_histograms`.
If the user asks for the distribution of a categorical variable, use `plot_bar_charts` or `summarize_categorical`.

## Relationship requests
If the user asks about the linear relationship between numeric variables, use `pearson_correlation`.
If they ask for a picture of those correlations, use `plot_corr_heatmap` on the correlation matrix returned by `pearson_correlation`. Do NOT call `plot_corr_heatmap` on raw data.
If the user asks how a numeric variable differs across a categorical one, use `plot_cat_num_boxplot`.

## Modeling requests
If the user asks to predict or explain a numeric outcome (e.g., `hours_per_week`), `multiple_linear_regression` is a reasonable baseline.
If the outcome is binary (e.g., `income`), linear regression is the wrong tool — fall back to code generation for logistic regression.

## Safety rules
- Do not choose a model before validating the outcome column with `target_check`.
- Do not use a regression tool on a categorical outcome.
- Prefer tools from the allow-list when one clearly matches the request; only fall back to code generation for analyses no tool covers.
