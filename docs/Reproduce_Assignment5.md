# Reproducing the Assignment 5 Validation Run

This file lets Holt or Sam re-run the same seven prompts that produced
`Assignment5_Validation_Log.md` and check the output against what the log
reports. The flow is: launch the agent once, then type each prompt at the
`>` prompt the same way a normal user would.

## Prerequisites

1. Repo cloned, terminal open at the repo root.
2. Python environment activated (the one with `requirements.txt` installed).
3. `.env` present in the repo root with at least:
   - `OPENAI_API_KEY=sk-...`
   - `LANGFUSE_PUBLIC_KEY=...`, `LANGFUSE_SECRET_KEY=...`, `LANGFUSE_HOST=https://cloud.langfuse.com`
4. `data/adult.csv` and `knowledge/rag_faiss.index` are already in the repo.

## Step 1. Launch the agent

Run this once:

```
PYTHONPATH=. python builds/build4_rag_router_agent.py \
    --data data/adult.csv \
    --knowledge_dir knowledge \
    --report_dir reports \
    --provider openai \
    --session_id a5-fresh \
    --tags build4,assignment5,fresh
```

When startup finishes you will see a `>` prompt. From here, every line you
type is one of three things:

- `ask <prompt>` to send a new request to the router
- `y` or `n` to answer a HITL question (approve a tool, save code, run code)
- `run` to execute the most recently approved code
- `exit` to quit

## Step 2. Type the seven prompts

For each prompt below, type the `ask` line at the `>` prompt, then answer
the HITL questions the agent asks you. The expected HITL flow per prompt
type is:

| Prompt type | What the agent will ask | What to type |
|---|---|---|
| Tool prompt (P1, P4) | `Run tool '<name>' now? (y/n)` | `y` |
| Codegen prompt (P2) | `Approve and save this code? (y/n)` then `Execute ... now? (y/n)` | `y`, then `run`, then `y` |
| Answer-mode prompt (P3, P5, P6, P7) | nothing (validator rejects and prints ERROR) | nothing |

The seven prompts (paste each line one at a time):

```
ask compute pearson correlations between numeric columns
ask write code that bins age into 5 quantiles and shows income rate per quantile
ask according to the knowledge base when should I use multiple regression?
ask use the knowledge base to recommend an analysis and then run it
ask help me analyze this dataset
ask analyze the column nonexistent_variable_xyz
ask how do I fix my kitchen sink?
```

When you are done, type `exit`.

## Step 3. What to look for per prompt

Compare what you see to the log. The router decision and the failure mode
should match what's in `Assignment5_Validation_Log.md` section 11.

| ID | Should print | Should produce artifact | Known bug to confirm |
|---|---|---|---|
| P1 | `mode:"tool"`, `tool:"pearson_correlation"` | `pearson_correlation_output.txt` under `reports/tool_outputs/session_a5-fresh/` | Args are hallucinated; the summary names only `age` and `fnlwgt`. |
| P2 | `mode:"codegen"` | `agent_generated_analysis.py` and a PNG under `reports/session_a5-fresh/` | Code is the wrong analysis (a boxplot, not age-quantile income rate). |
| P3 | `mode:"answer"` | None | Validator prints `ERROR: Router 'mode' must be 'tool' or 'codegen'`. |
| P4 | `mode:"tool"`, `tool:"basic_profile"` | `basic_profile_output.txt` under `reports/tool_outputs/session_a5-fresh/` | Final summary contains placeholder values like `X` and `Y`. |
| P5 | `mode:"answer"` | None | Same validator error as P3. |
| P6 | `mode:"answer"` (router detects missing column) | None | Same validator error as P3. |
| P7 | `mode:"answer"` (router classifies off-topic) | None | Same validator error as P3. |

## Step 4. Verify in Langfuse

Open Langfuse and filter by tag `assignment5` (or session id `a5-fresh`).
You should see one trace per `ask` you typed, each with a router-decision
span, retrieval spans (when RAG fired), and a tool or codegen span (when
the validator did not block it).

## Optional: rebuild the PDF

After the run, regenerate the PDF from the markdown:

```
PYTHONPATH=. python scripts/build_assignment5_pdf.py
```

Output: `docs/Assignment5_Validation_Log.pdf`.

## Optional: rerun all seven prompts non-interactively

The transcripts in `docs/a5_runs/p<N>.out` were captured by piping the
prompt and HITL answers into the agent's stdin so the run could be saved
to a file. This is not how you would normally use the CLI, but if you
want to reproduce the saved transcripts byte for byte, the commands are
in the git history (commit that added `docs/a5_runs/`). Use Step 1 + Step
2 above as the primary path.
