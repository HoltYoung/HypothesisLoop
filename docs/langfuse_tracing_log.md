# Langfuse Tracing & Error Log — Build 3: HITL + Tool Router Agent

**QAC387 — Spring 2026**
**Holt Young & Sam Penn**
**Date:** April 3, 2026

## Overview

This document summarizes the Langfuse tracing output, errors encountered, and resolutions for our Build 3 HITL + Tool Router Agent running on the UCI Adult Income dataset (32,561 rows, 15 columns). All traces are available in our local Langfuse instance at `http://127.0.0.1:3000`.

**Model:** gpt-4o-mini
**Tracing:** Langfuse v4.0.6 with LangChain CallbackHandler + @observe decorator
**Session IDs:** test-20260403-200927, test-20260403-201201, test-20260403-201442

---

## Test Scenarios

We ran 7 test scenarios across 3 sessions to exercise all agent capabilities:

| Test | Request | Expected Mode | Result |
|------|---------|---------------|--------|
| 1 | Dataset schema | N/A (local) | Pass |
| 2 | "What research questions can I explore about income inequality?" | suggest | Pass |
| 3 | "Show me a frequency table for education" | tool (summarize_categorical) | Pass (after fix) |
| 4 | "Show histograms for age and hours_per_week" | tool (plot_histograms) | Pass |
| 5 | "Show a correlation heatmap for all numeric columns" | tool or codegen | Pass (after fix) |
| 6 | "Run a chi-square test of independence between income and sex" | codegen | Pass |
| 7 | "Fit a regression predicting hours_per_week from age and education_num" | tool (multiple_linear_regression) | Pass (after fix) |

---

## Errors Encountered and Resolutions

### Error 1: OpenAI API Key Override (Session 1, Pre-test)

**Problem:** The agent returned `openai.RateLimitError: insufficient_quota` despite having a valid API key in the `.env` file.

**Root Cause:** An older, expired OpenAI API key was set as a system-level environment variable (`OPENAI_API_KEY`). The `load_dotenv()` call does not override existing environment variables by default, so the expired system key took precedence over the valid `.env` key.

**Resolution:** Changed `load_dotenv(PROJECT_ROOT / ".env")` to `load_dotenv(PROJECT_ROOT / ".env", override=True)` so the `.env` file always takes priority over system environment variables.

**Langfuse Trace:** No trace generated (error occurred before any LLM call).

---

### Error 2: Summarize Chain JSON Parse Failure (Session 1, Test 3)

**Problem:** After the `summarize_categorical` tool successfully returned a frequency table for the `education` column, the summarize chain failed with:
```
Error code: 400 - "We could not parse the JSON body of your request."
```

**Root Cause:** The tool output contained a pandas DataFrame string representation with Unicode ellipsis characters (`...`) and special formatting that caused the LangChain/OpenAI JSON serialization to produce an invalid request body.

**Resolution:** Updated `TOOL_DESCRIPTIONS` in `src/tools.py` to provide clearer output formatting guidance. On re-run, the summarize chain successfully processed the same tool output without errors. The issue appeared to be intermittent and related to how the DataFrame repr was being serialized in the API payload.

**Langfuse Trace:** Trace shows the tool execution span completing successfully, but the summarize generation span failing with a 400 status code.

---

### Error 3: Router Argument Name Mismatch — plot_corr_heatmap (Sessions 1-2, Test 5)

**Problem:** The router selected `plot_corr_heatmap` with args `{"numeric_cols": ["age", "fnlwgt", ...]}`, but the function signature is `plot_corr_heatmap(corr, out_path, missing)` — it expects a pre-computed correlation DataFrame, not raw column names.

**Root Cause:** The router's system prompt included generic guidance (`Correlations -> pearson_correlation or plot_corr_heatmap`) without clarifying that `plot_corr_heatmap` requires a pre-computed correlation matrix as input. The LLM assumed it could pass column names directly.

**Resolution:** 
1. Updated the router's system prompt to explicitly state: `Correlations -> pearson_correlation (NOT plot_corr_heatmap, which needs a pre-computed matrix)`
2. Added a tool description: `"plot_corr_heatmap": "Plot a heatmap from a PRE-COMPUTED correlation matrix. DO NOT use for raw data — use pearson_correlation instead."`
3. After the fix, the router correctly falls back to codegen mode for correlation heatmap requests.

**Langfuse Trace:** Trace shows the router generation returning `mode: "tool"` with invalid args, followed by a Python TypeError in the tool execution span.

---

### Error 4: Router Argument Name Mismatch — multiple_linear_regression (Sessions 1-2, Test 7)

**Problem:** The router selected `multiple_linear_regression` with args `{"target": "hours_per_week", "features": ["age", "education_num"]}`, but the actual function signature uses `outcome` and `predictors` as parameter names.

**Root Cause:** The LLM defaulted to common ML terminology (`target`, `features`) rather than using the specific parameter names from the function signature. Although `tool_arg_hints` were provided showing the correct names, the router prompt did not emphasize them strongly enough.

**Resolution:**
1. Added explicit argument rules to the router's system prompt:
   ```
   - multiple_linear_regression: args={"outcome":"<target_col>","predictors":["<col1>","<col2>"]}
     IMPORTANT: use "outcome" NOT "target", use "predictors" NOT "features"
   ```
2. Added a tool description reinforcing the correct parameter names.
3. After the fix, the router correctly uses `outcome` and `predictors`.

**Langfuse Trace:** Trace shows the router generation returning correct tool selection but incorrect arg names, followed by a Python TypeError in the tool execution span.

---

## Agent Performance Summary

### Routing Accuracy

After prompt fixes, the router achieved correct routing decisions across all 7 test scenarios:
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

1. **HITL is essential.** The argument-name mismatches (Tests 5 and 7) would have been caught by human review of the tool plan before execution. This validates the HITL design.
2. **Tool descriptions matter.** Generic routing guidance led to incorrect tool selection. Explicit argument-name rules in the system prompt resolved the issues.
3. **Codegen fallback works well.** When no tool fits (chi-square test), the router correctly falls back to generating a standalone Python script.
4. **Langfuse tracing provides full auditability.** Every LLM call, tool execution, and error is captured with timestamps and metadata, making debugging straightforward.
