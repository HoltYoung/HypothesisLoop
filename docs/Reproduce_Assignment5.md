# Reproducing the Assignment 5 Validation Run

This file lets Holt or Sam re-run the exact same seven prompts that produced
`Assignment5_Validation_Log.md` and check the output against what the log
reports. Nothing in the run depends on a VM; it all runs locally with the
project's `.env`.

## Prerequisites

1. Repo cloned and `cd` into the repo root.
2. `.venv` activated (or whichever Python environment has `requirements.txt`
   installed).
3. `.env` present in the repo root with at least:
   - `OPENAI_API_KEY=sk-...`
   - `LANGFUSE_PUBLIC_KEY=...`, `LANGFUSE_SECRET_KEY=...`, `LANGFUSE_HOST=https://cloud.langfuse.com`
   - `DEFAULT_CHAT_PROVIDER=openai`
   - `DEFAULT_CHAT_MODEL_OPENAI=gpt-4o-mini`
4. `data/adult.csv` present (already in the repo).
5. `knowledge/rag_faiss.index` present (built once with
   `python scripts/build_rag_index.py` if missing).

## Running one prompt at a time

Each command below is a single-prompt session. It pipes the prompt and the
human-in-the-loop responses (`y`, `run`, `y`, `exit`) into the agent's stdin.

The four lines mean:
1. `ask <prompt>` -> route the prompt
2. `y` -> approve the tool or save the generated code
3. `run` -> execute the saved code (no-op for tool prompts; the agent prints
   "Unrecognized command" and moves on)
4. `y` -> confirm execution
5. `exit` -> close the session

For tool-only prompts the `run` and second `y` are harmless extras.

```bash
# P1. Simple tool
printf "ask compute pearson correlations between numeric columns\ny\ny\nexit\n" \
  | PYTHONPATH=. python builds/build4_rag_router_agent.py \
      --data data/adult.csv --knowledge_dir knowledge --report_dir reports \
      --provider openai --session_id a5-p1 --tags build4,assignment5,fresh

# P2. Simple codegen
printf "ask write code that bins age into 5 quantiles and shows income rate per quantile\ny\nrun\ny\nexit\n" \
  | PYTHONPATH=. python builds/build4_rag_router_agent.py \
      --data data/adult.csv --knowledge_dir knowledge --report_dir reports \
      --provider openai --session_id a5-p2 --tags build4,assignment5,fresh

# P3. RAG-conceptual
printf "ask according to the knowledge base when should I use multiple regression?\ny\nrun\ny\nexit\n" \
  | PYTHONPATH=. python builds/build4_rag_router_agent.py \
      --data data/adult.csv --knowledge_dir knowledge --report_dir reports \
      --provider openai --session_id a5-p3 --tags build4,assignment5,fresh

# P4. Mixed
printf "ask use the knowledge base to recommend an analysis and then run it\ny\nrun\ny\nexit\n" \
  | PYTHONPATH=. python builds/build4_rag_router_agent.py \
      --data data/adult.csv --knowledge_dir knowledge --report_dir reports \
      --provider openai --session_id a5-p4 --tags build4,assignment5,fresh

# P5. Ambiguous
printf "ask help me analyze this dataset\ny\nrun\ny\nexit\n" \
  | PYTHONPATH=. python builds/build4_rag_router_agent.py \
      --data data/adult.csv --knowledge_dir knowledge --report_dir reports \
      --provider openai --session_id a5-p5 --tags build4,assignment5,fresh

# P6. Bad input
printf "ask analyze the column nonexistent_variable_xyz\ny\nrun\ny\nexit\n" \
  | PYTHONPATH=. python builds/build4_rag_router_agent.py \
      --data data/adult.csv --knowledge_dir knowledge --report_dir reports \
      --provider openai --session_id a5-p6 --tags build4,assignment5,fresh

# P7. Unrelated
printf "ask how do I fix my kitchen sink?\ny\nrun\ny\nexit\n" \
  | PYTHONPATH=. python builds/build4_rag_router_agent.py \
      --data data/adult.csv --knowledge_dir knowledge --report_dir reports \
      --provider openai --session_id a5-p7 --tags build4,assignment5,fresh
```

## What to expect per prompt (so the log can be checked)

| ID | Should print router mode | Should produce artifact | Known bug |
|---|---|---|---|
| P1 | `mode:"tool"`, `tool:"pearson_correlation"` | `reports/tool_outputs/session_a5-p1/pearson_correlation_output.txt` | Args are hallucinated; summary names only `age` and `fnlwgt`. |
| P2 | `mode:"codegen"` | `reports/session_a5-p2/agent_generated_analysis.py` and a PNG | Code is the wrong analysis (a boxplot, not age quantile income rate). |
| P3 | `mode:"answer"` | None | Validator prints `ERROR: Router 'mode' must be 'tool' or 'codegen'`. |
| P4 | `mode:"tool"`, `tool:"basic_profile"` | `reports/tool_outputs/session_a5-p4/basic_profile_output.txt` | Final summary contains placeholder values like `X` and `Y`. |
| P5 | `mode:"answer"` | None | Same validator error as P3. |
| P6 | `mode:"answer"` (router detects missing column) | None | Same validator error as P3. |
| P7 | `mode:"answer"` (router classifies off-topic) | None | Same validator error as P3. |

## Saving stdout for evidence

To save each transcript the way the log was built, redirect stdout:

```bash
mkdir -p docs/a5_runs
# Add the redirection at the end of each command, e.g.:
... --session_id a5-p1 --tags build4,assignment5,fresh > docs/a5_runs/p1.out 2>&1
```

## Rebuilding the PDF

After re-running, regenerate the PDF from the markdown:

```bash
PYTHONPATH=. python scripts/build_assignment5_pdf.py
```

The output is `docs/Assignment5_Validation_Log.pdf`.

## Verifying traces in Langfuse

Each run is tagged `build4,assignment5,fresh` and has a session id of the form
`a5-p<N>`. Filter by tag or session to see the router decision span, retrieval
spans, and tool or codegen spans for each prompt.
