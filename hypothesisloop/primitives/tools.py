# tools registry for ai data analysis agent

from hypothesisloop.primitives import checks
from hypothesisloop.primitives import io_utils
from hypothesisloop.primitives import modeling
from hypothesisloop.primitives import plotting
from hypothesisloop.primitives import profiling
from hypothesisloop.primitives import summaries

TOOLS = {
    # summaries
    "summarize_numeric": summaries.summarize_numeric,
    "summarize_categorical": summaries.summarize_categorical,
    "missingness_table": summaries.missingness_table,
    "pearson_correlation": summaries.pearson_correlation,
    # profiling
    "basic_profile": profiling.basic_profile,
    "split_columns": profiling.split_columns,
    # modeling
    "multiple_linear_regression": modeling.multiple_linear_regression,
    # plotting
    "plot_missingness": plotting.plot_missingness,
    "plot_corr_heatmap": plotting.plot_corr_heatmap,
    "plot_histograms": plotting.plot_histograms,
    "plot_bar_charts": plotting.plot_bar_charts,
    "plot_cat_num_boxplot": plotting.plot_cat_num_boxplot,
    # checks
    "assert_json_safe": checks.assert_json_safe,
    "target_check": checks.target_check,
    # io
    "ensure_dirs": io_utils.ensure_dirs,
    "read_data": io_utils.read_data,
}

# Optional (recommended): descriptions kept separate from callables
TOOL_DESCRIPTIONS = {
    "summarize_numeric": "Descriptive statistics for numeric columns.",
    "summarize_categorical": "Frequency table for categorical columns. Use args: {column: '<col>'} for one column, or {cat_cols: ['<col1>', '<col2>']} for multiple.",
    "missingness_table": "Summary of missing values per column. No args needed.",
    "pearson_correlation": "Compute Pearson correlations for numeric columns. Use this for correlation analysis (NOT plot_corr_heatmap).",
    "plot_corr_heatmap": "Plot a heatmap from a PRE-COMPUTED correlation matrix. DO NOT use for raw data — use pearson_correlation instead.",
    "plot_histograms": "Histograms for numeric columns. Args: {numeric_cols: ['<col1>', '<col2>']}.",
    "plot_bar_charts": "Bar chart of category counts for categorical columns (NOT associations with numeric variables).",
    "plot_cat_num_boxplot": "Boxplot showing the distribution of a numeric variable grouped by a categorical variable (categorical-numeric association).",
    "plot_missingness": "Visual heatmap of missing data patterns.",
    "multiple_linear_regression": "OLS regression. Args: {outcome: '<target_col>', predictors: ['<col1>', '<col2>']}. Use 'outcome' NOT 'target'.",
    "basic_profile": "Overview of dataset: row count, column count, dtypes. No args needed.",
    "target_check": "Validate an outcome/target column for modeling.",
}
