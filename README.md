# Build 4: RAG + HITL + Tool Router Agent — UCI Adult Income Analysis

**QAC387 — Spring 2026**
**Holt Young & Sam Penn**

## Purpose

An LLM-powered data-analysis agent that combines **retrieval-augmented generation (RAG)**, a **human-in-the-loop (HITL)** approval flow, and an **LLM router** that picks a tool from an allow-list or falls back to custom code generation. The agent runs on the UCI Adult Income dataset and is fully traced in Langfuse.

On every analysis request, the agent:

1. Sends the request to a router LLM that decides between a **tool-run** or **code generation**.
2. Before code generation, retrieves the top-k most relevant chunks from a **FAISS index** of our knowledge corpus (dataset codebook, tool notes, analysis guides) and injects them into the codegen prompt.
3. Requires **human approval** before any tool or generated script runs.
4. Writes a text summary/interpretation of the output.
5. Logs every LLM call to **Langfuse**.

## Dataset

**UCI Adult Income Dataset** — 32,561 records, 15 columns, from the 1994 US Census. Task: predict whether an individual earns more than $50K/year. File: `data/adult.csv`. See `knowledge/dataset/adult_codebook.md` for the full column dictionary.

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

# Langfuse (local instance by default)
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=http://localhost:3000
```

### 3. Build the RAG index (one-time, ≈ $0.01 in OpenAI credits)

```bash
PYTHONPATH=. python scripts/build_rag_index.py
```

Writes `knowledge/rag_faiss.index` and `knowledge/rag_chunks.pkl`.

### 4. Run the agent

```bash
PYTHONPATH=. python builds/build4_rag_router_agent.py --data data/adult.csv --knowledge_dir knowledge --report_dir reports --session_id my-session
```

Useful flags: `--memory` (conversation memory), `--stream` (stream LLM output), `--model <id>`, `--rag_k 4` (number of retrieved chunks).

> **Why `PYTHONPATH=.`?** The teacher's reference scripts are invoked from the project root but do not mutate `sys.path`. Prefixing with `PYTHONPATH=.` lets them find the `src/` package.

## Usage

Once the REPL is running, these commands are available at the `> ` prompt:

| Command | Description |
|---|---|
| `help` | Show available commands |
| `schema` | Print the dataset schema |
| `suggest <question>` | LLM-only suggestion pass (no tool run, no codegen) |
| `ask <request>` | **Router decides**: tool-run or codegen (HITL) |
| `tool <request>` | Force a tool-run |
| `code <request>` | Force code generation + approval |
| `run` | Execute the last approved generated script |
| `exit` | Quit |

### Example session

```
> ask summarize missing values in the dataset
  [Router → missingness_table]
  Run tool 'missingness_table' now? (y/n) y
  [Tool output + LLM interpretation]

> ask fit a logistic regression predicting income from age and hours_per_week
  [Router → codegen]
  [RAG retrieves 4 chunks; top: adult_codebook.md > Cautions (score=0.601)]
  [Codegen produces script]
  Approve and save this code? (y/n) y

> run
  Execute agent_generated_analysis.py now? (y/n) y
```

## Available tools

The router can pick any of these from `src/tools.py`:

- `basic_profile` — row/column count, dtypes
- `summarize_numeric` / `summarize_categorical` — descriptive statistics
- `missingness_table` / `plot_missingness`
- `pearson_correlation` / `plot_corr_heatmap`
- `plot_histograms` / `plot_bar_charts` / `plot_cat_num_boxplot`
- `multiple_linear_regression`
- `target_check` — outcome-column validation
- `assert_json_safe` — serialization check

When no tool fits, the router falls back to code generation and the agent writes a standalone Python script.

## RAG knowledge corpus

Markdown files in `knowledge/` are chunked by heading and indexed by `text-embedding-3-small`:

```
knowledge/
├── dataset/adult_codebook.md       Adult dataset codebook (columns, types, cautions)
├── guides/
│   ├── tool_selection_rules.md     How the router should pick tools
│   └── analysis_workflow.md        Standard EDA + modeling steps
└── tools/                          One short note per Build-0 tool
    ├── basic_profile.md
    ├── missingness_table.md
    ├── summarize_numeric.md
    ├── pearson_correlation.md
    ├── plot_histograms.md
    ├── plot_bar_charts.md
    └── multiple_linear_regression.md
```

74 chunks total at the current corpus size. See `docs/Langfuse_Tracing_Log_Build4.md` for an analysis of RAG's measured impact on agent behavior.

## Cautions

- **LLM outputs are non-deterministic.** Running the same query twice may produce different routing decisions, tool arguments, or generated code. Use `--temperature 0` for more reproducible behavior.
- **Generated code requires human review.** Always read the code before approving `run`. The agent executes the saved script in a subprocess but does not restrict imports or filesystem access.
- **Sensitive demographic variables.** The dataset contains `race`, `sex`, `native_country`. Correlations involving these variables do not imply causation.
- **API cost.** Each run calls OpenAI for embeddings (index build only), routing, codegen, and summarization. `gpt-4o-mini` is cheap but costs add up across many queries.
- **Langfuse logs prompts and responses.** If you point `LANGFUSE_HOST` at a hosted instance, be aware that dataset schemas and query text are persisted there.

## Project structure

```
├── builds/
│   └── build4_rag_router_agent.py         Main agent (Build 4 deliverable)
├── scripts/
│   └── build_rag_index.py                 RAG index builder
├── src/
│   ├── rag_faiss_utils_pdf.py             FAISS helpers (embed / index / retrieve)
│   ├── tools.py                           Tool registry
│   ├── summaries.py / plotting.py / ...   Build-0 analysis modules
│   └── ...
├── knowledge/
│   ├── dataset/, guides/, tools/          Markdown corpus (10 files)
│   ├── rag_faiss.index                    FAISS index (generated)
│   └── rag_chunks.pkl                     Chunk metadata (generated)
├── data/
│   └── adult.csv                          UCI Adult Income dataset
├── docs/
│   └── Langfuse_Tracing_Log_Build4.md     Tracing log + RAG impact analysis
├── reports/                               Generated outputs (gitignored)
├── requirements.txt
├── .env                                   (gitignored)
└── README.md
```
