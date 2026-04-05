# Langfuse Tracing & Error Log — Build 3: HITL + Tool Router Agent

**QAC387 — Spring 2026**
**Holt Young & Sam Penn**
**Date:** April 5, 2026

## Overview

This document summarizes the Langfuse tracing output, errors encountered, and resolutions for our Build 3 HITL + Tool Router Agent running on the UCI Adult Income dataset (32,561 rows, 15 columns). All traces are available in our local Langfuse instance at `http://127.0.0.1:3000`.

**Model:** gpt-4o-mini
**Tracing:** Langfuse v4.0.6 with LangChain CallbackHandler + @observe decorator
**Dataset:** UCI Adult Income — 32,561 rows, 15 columns
**Langfuse Host:** http://127.0.0.1:3000

---

## Test Log

We tested all CLI commands (`suggest`, `ask`, `tool`, `code`) with multiple prompts across several sessions.

### suggest command

| # | Prompt | Outcome | Notes |
|---|--------|---------|-------|
| 1 | `suggest what research questions can I explore about income inequality` | Pass | Returned structured research questions with outcomes, predictors, and suggested analysis types. Also provided clarifying follow-up questions. |

### ask command (router decides tool vs codegen)

| # | Prompt | Router Decision | Outcome | Notes |
|---|--------|-----------------|---------|-------|
| 3 | `ask show a frequency table for education` | tool (summarize_categorical) | Pass (after fix) | Initially failed — see Error 2 below. After fixing tool descriptions, ran successfully. |
| 4 | `ask show histograms for age and hours_per_week` | tool (plot_histograms) | Pass | Router correctly selected plot_histograms with args `{"numeric_cols": ["age", "hours_per_week"]}`. Generated histogram PNGs. |
| 5 | `ask show a correlation heatmap for all numeric columns` | codegen | Pass (after fix) | Initially the router selected plot_corr_heatmap which expects a pre-computed matrix — see Error 3 below. After prompt fix, router correctly falls back to codegen. |
| 6 | `ask fit a regression predicting hours_per_week from age and education_num` | tool (multiple_linear_regression) | Pass (after fix) | Router initially used wrong arg names (target/features instead of outcome/predictors) — see Error 4. Fixed by adding explicit arg rules to router prompt. |
| 7 | `ask run a chi-square test of independence between income and sex` | codegen | Pass | No chi-square tool exists, so router correctly fell back to codegen. Generated a valid script using scipy.stats.chi2_contingency. |

### tool command (forced tool mode)

| # | Prompt | Tool Selected | Outcome | Notes |
|---|--------|---------------|---------|-------|
| 8 | `tool show a frequency table for sex` | summarize_categorical | Pass | Correctly selected tool with args `{"column": "sex"}`. Returned frequency counts: Male 21,790, Female 10,771. Summarizer provided interpretation. |
| 9 | `tool run a regression of hours_per_week on age and education_num` | multiple_linear_regression | Pass | Correctly used `{"outcome": "hours_per_week", "predictors": ["age", "education_num"]}`. OLS regression ran successfully and summarizer explained results. |

### code command (forced codegen mode)

| # | Prompt | Outcome | Notes |
|---|--------|---------|-------|
| 10 | `code create a violin plot of hours_per_week grouped by income and save it` | Pass | Generated a standalone Python script using matplotlib violinplot. Script used argparse with --data and --report_dir, handled missing values with listwise deletion, validated column existence. Executed successfully (return code 0), saved plot to reports/. |
| 11 | `code calculate the chi-square test of independence between sex and income` | Pass | Generated script using scipy.stats.chi2_contingency. Correctly built contingency table with pd.crosstab, saved chi2 statistic, p-value, and degrees of freedom to a report file. Executed successfully (return code 0). |

### schema command

| # | Prompt | Outcome | Notes |
|---|--------|---------|-------|
| 12 | `schema` | Pass | Displayed all 15 columns with dtypes. Local command, no LLM call. |

---

## Errors Encountered and Resolutions

### Error 1: OpenAI API Key Override (Pre-test)

**Prompt:** N/A — error occurred on startup before any prompt was entered.

**Problem:** The agent returned `openai.RateLimitError: insufficient_quota` despite having a valid API key in the `.env` file.

**Root Cause:** An older, expired OpenAI API key was set as a system-level environment variable (`OPENAI_API_KEY`). The `load_dotenv()` call does not override existing environment variables by default, so the expired system key took precedence over the valid `.env` key.

**Where in the workflow:** Before any LLM call — the agent could not start.

**Resolution:** Changed `load_dotenv(PROJECT_ROOT / ".env")` to `load_dotenv(PROJECT_ROOT / ".env", override=True)` so the `.env` file always takes priority over system environment variables.

---

### Error 2: Summarize Chain JSON Parse Failure (Test 3)

**Prompt:** `ask show a frequency table for education`

**Problem:** After the `summarize_categorical` tool successfully returned a frequency table for the `education` column, the summarize chain failed with:
```
Error code: 400 - "We could not parse the JSON body of your request."
```

**Root Cause:** The tool output contained a pandas DataFrame string representation with Unicode ellipsis characters and special formatting that caused the LangChain/OpenAI JSON serialization to produce an invalid request body.

**Where in the workflow:** Tool execution succeeded → summarize chain (results_summarizer) failed when sending the tool output to the LLM for interpretation.

**Resolution:** Updated `TOOL_DESCRIPTIONS` in `src/tools.py` to provide clearer output formatting guidance. On re-run, the summarize chain successfully processed the same tool output without errors.

---

### Error 3: Router Argument Mismatch — plot_corr_heatmap (Test 5)

**Prompt:** `ask show a correlation heatmap for all numeric columns`

**Problem:** The router selected `plot_corr_heatmap` with args `{"numeric_cols": ["age", "fnlwgt", ...]}`, but the function signature is `plot_corr_heatmap(corr, out_path, missing)` — it expects a pre-computed correlation DataFrame, not raw column names.

**What happened:** The router chose a tool that can't accept raw data. The user would need to first run `pearson_correlation` to get a correlation matrix, then pass that to `plot_corr_heatmap`. The router didn't understand this two-step dependency.

**Where in the workflow:** Router decision step — the router selected the wrong tool because the system prompt didn't clarify the tool's input requirements.

**Resolution:**
1. Updated the router's system prompt to explicitly state: `Correlations -> pearson_correlation (NOT plot_corr_heatmap, which needs a pre-computed matrix)`
2. Added a tool description: `"plot_corr_heatmap": "Plot a heatmap from a PRE-COMPUTED correlation matrix. DO NOT use for raw data — use pearson_correlation instead."`
3. After the fix, the router correctly falls back to codegen mode for correlation heatmap requests.

---

### Error 4: Router Argument Name Mismatch — multiple_linear_regression (Test 6)

**Prompt:** `ask fit a regression predicting hours_per_week from age and education_num`

**Problem:** The router selected `multiple_linear_regression` with args `{"target": "hours_per_week", "features": ["age", "education_num"]}`, but the actual function signature uses `outcome` and `predictors` as parameter names.

**What happened:** The LLM used common ML terminology (target/features) instead of the function's actual parameter names (outcome/predictors). The tool execution failed with a TypeError because it received unexpected keyword arguments.

**Where in the workflow:** Router decision step — correct tool was selected but with wrong argument names, causing the tool execution step to fail.

**Resolution:**
1. Added explicit argument rules to the router's system prompt:
   ```
   - multiple_linear_regression: args={"outcome":"<target_col>","predictors":["<col1>","<col2>"]}
     IMPORTANT: use "outcome" NOT "target", use "predictors" NOT "features"
   ```
2. Added a tool description reinforcing the correct parameter names.
3. After the fix, the router correctly uses `outcome` and `predictors`.

---

### Error 5: Toolplan Chain Missing Template Variables (Test 9, pre-fix)

**Prompt:** `tool run a regression of hours_per_week on age and education_num`

**Problem:** The `tool` command crashed with: `KeyError: "Input to ChatPromptTemplate is missing variables {'tool_arg_hints', 'allow_str'}"`

**What happened:** The toolplan chain's system prompt contained `{allow_str}` and `{tool_arg_hints}` as LangChain template variables, but they were not being passed at invoke time. The tool list and argument hints were never visible to the LLM, so it couldn't select the right tool.

**Where in the workflow:** Toolplan generation step — the LLM call failed before it could generate a tool plan because the prompt template was missing required input variables.

**Resolution:** Changed the toolplan system prompt to format the tool list and argument hints at chain build time (using an f-string) rather than treating them as runtime template variables. After the fix, the `tool` command correctly shows the LLM the full list of available tools and their argument signatures.

---

## Agent Performance Summary

### Routing Accuracy

After prompt fixes, the router achieved correct routing decisions across all test scenarios:
- **Tool mode** was correctly selected for: frequency tables, histograms, and linear regression
- **Codegen mode** was correctly selected for: chi-square tests and correlation heatmaps (no appropriate tool)
- **Suggest mode** correctly answered open-ended research questions

### Tool Execution

All pre-built tools executed successfully when given correct arguments:
- `summarize_categorical` produced accurate frequency tables
- `plot_histograms` generated histogram PNGs for age and hours_per_week distributions
- `multiple_linear_regression` ran OLS regression and returned model results

### Code Generation

The codegen chain produced valid, executable Python scripts that:
- Used argparse with `--data` and `--report_dir` as required
- Handled missing values with listwise deletion
- Validated column existence before analysis
- Saved artifacts to the report directory

### LLM Summarization

The results summarizer chain provided clear natural-language interpretations including:
- What analysis was performed
- Key findings in bullet points
- Plain-language interpretation
- Caveats and assumptions
- Suggested next steps

### Key Takeaways

1. **HITL is essential.** The argument-name mismatches (Errors 3 and 4) would have been caught by human review of the tool plan before execution. This validates the HITL design.
2. **Tool descriptions matter.** Generic routing guidance led to incorrect tool selection. Explicit argument-name rules in the system prompt resolved the issues.
3. **Codegen fallback works well.** When no tool fits (chi-square test), the router correctly falls back to generating a standalone Python script.
4. **Langfuse tracing provides full auditability.** Every LLM call, tool execution, and error is captured with timestamps and metadata, making debugging straightforward.
