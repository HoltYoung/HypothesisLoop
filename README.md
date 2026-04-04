# Build 3: HITL + Tool Router Agent — UCI Adult Income Analysis

**QAC387 — Spring 2026**
**Holt Young & Sam Penn**

## Purpose

This application is an LLM-powered data analysis agent that uses a **human-in-the-loop (HITL)** workflow with **intelligent tool routing** to analyze the UCI Adult Income dataset. It is part of our larger **HypothesisLoop** project — an iterative hypothesis-testing system for automated data analysis.

The agent accepts natural-language analysis requests and uses an LLM (OpenAI via LangChain) to decide whether to:

1. **Route to an existing analysis tool** from a pre-built registry (e.g., correlation heatmap, regression, frequency table), or
2. **Generate custom Python code** when no tool fits the request.

All actions require **human approval** before execution, and every LLM call is traced with **Langfuse** for full auditability.

## Dataset

**UCI Adult Income Dataset** — 32,561 records with 15 attributes from the 1994 Census database. The prediction task is to determine whether a person earns more than $50K/year based on demographic features (age, education, occupation, race, sex, etc.).

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-api-key-here

# Langfuse (for tracing)
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=http://localhost:3000
```

### 3. Run the agent

```bash
python builds/build3_hitl_tool_router_agent.py --data data/adult.csv --report_dir reports --tags build3 --memory
```

Add `--stream` to stream LLM output in real time.

## Usage

Once running, the agent provides an interactive CLI with these commands:

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `schema` | Print the dataset schema (columns + dtypes) |
| `suggest <question>` | Ask questions about the dataset (LLM answers) |
| `ask <request>` | **Router decides**: run an existing tool OR generate code (HITL) |
| `tool <request>` | Force tool mode: pick a tool from the registry (HITL) |
| `code <request>` | Force code generation mode (HITL) |
| `run` | Execute the last approved generated script |
| `exit` | Quit |

### Example session

```
> ask show a frequency table for education
  [Router selects: summarize_categorical with column="education"]
  Run tool 'summarize_categorical' now? (y/n) y
  [Runs tool, displays results, then LLM summarizes findings]

> ask fit a regression predicting hours_per_week from age and education_num
  [Router selects: multiple_linear_regression]
  Run tool 'multiple_linear_regression' now? (y/n) y

> ask create a violin plot of income by occupation
  [Router falls back to codegen — no violin plot tool exists]
  Approve and save this code? (y/n) y
  > run
  Execute build3_generated_analysis.py now? (y/n) y
```

## Available Tools

The agent has access to these pre-built analysis tools:

- `basic_profile` — Dataset overview (rows, columns, dtypes)
- `summarize_numeric` — Descriptive statistics for numeric columns
- `summarize_categorical` — Frequency tables for categorical columns
- `missingness_table` / `plot_missingness` — Missing data analysis
- `pearson_correlation` / `plot_corr_heatmap` — Correlation analysis
- `plot_histograms` — Histograms for numeric distributions
- `plot_bar_charts` — Bar charts for categorical counts
- `plot_cat_num_boxplot` — Boxplots for categorical-numeric associations
- `multiple_linear_regression` — OLS regression
- `target_check` — Validate outcome/target column

When a request cannot be handled by any tool, the agent generates a standalone Python script instead.

## Cautions

- **LLM outputs are non-deterministic.** Running the same query twice may produce different code or tool selections. Use `--temperature 0` for more consistent results.
- **Generated code requires human review.** Always inspect code before approving execution. The agent uses subprocess sandboxing but does not restrict all imports.
- **The dataset contains sensitive demographic variables** (race, sex, native_country). The LLM may make observations about correlations involving these variables — correlation does not imply causation.
- **API costs.** Each query makes one or more OpenAI API calls. The default model (`gpt-4o-mini`) is cost-effective, but usage adds up with many queries.
- **Langfuse tracing sends data to your Langfuse instance.** If running a cloud-hosted Langfuse, be aware that prompts and dataset schemas are logged.

## Project Structure

```
├── builds/
│   └── build3_hitl_tool_router_agent.py   # Main application (this assignment)
├── data/
│   └── adult.csv                          # UCI Adult Income dataset
├── src/
│   ├── tools.py                           # Tool registry
│   ├── summaries.py                       # Summary/stats tools
│   ├── plotting.py                        # Plotting tools
│   ├── modeling.py                        # Regression tools
│   ├── profiling.py                       # Data profiling tools
│   ├── checks.py                          # Validation tools
│   └── io_utils.py                        # I/O utilities
├── reports/                               # Generated outputs (gitignored)
├── requirements.txt
├── .env                                   # API keys (gitignored)
└── README.md
```
