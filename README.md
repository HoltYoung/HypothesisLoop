---
title: HypothesisLoop
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Autonomous Hypothesize-Experiment-Evaluate-Learn agent
---

# HypothesisLoop

**QAC387-01 · Wesleyan · Spring 2026**
**Holt Young & Sam Penn**

> **Status: shipped (2026-04-30).** All 10 build phases complete.
> See `docs/FINAL_REPORT.md` for the writeup, `tests/validation_log.md`
> for the 28-run validation suite, and `docs/langfuse_traces.md` for
> representative trace URLs.

HypothesisLoop is an LLM-powered agent that wraps the **scientific method** —
Hypothesize → Experiment → Evaluate → Learn → Repeat — around a tabular dataset
and runs autonomously for N iterations (default 5). The user supplies a CSV and
a research question; the agent proposes a testable hypothesis, writes and runs
Python code in a sandbox, interprets the statistics, and feeds the learning
into the next round. Every step is traced in Langfuse. At the end, the agent
emits a Markdown report with embedded charts and a reasoning chain. HITL is
*optional* per iteration — the user can pause, inspect, redirect, or let the
agent run to completion.

This is a deliberate **pivot** from the cookie-cutter Build 0–4 router/REPL
architecture back to the team's original proposal: autonomous iteration over a
persistent multi-round trace, where the agent itself decides what to test next
based on what previous rounds learned. ~40% of Build 4's code (analysis
primitives, multi-provider dispatch, Langfuse plumbing) survives as
subroutines; ~60% is archived. See `docs/SPEC.md` for the full design and
`docs/DESIGN_DECISIONS.md` for the locked decisions.

## Status

The agent is feature-complete through Phase 6. Bias scanner + Markdown report
land in Phase 7; Streamlit UI is Phase 8.

- Phase 0 complete: package scaffolding, LLM dispatch, Langfuse plumbing, CLI smoke test.
- Phase 1 complete: sandboxed Python execution (AST denylist, timeout, RAM cap on POSIX, output capture).
- Phase 2 complete: state dataclasses + DAGTrace + linear scheduler + loop skeleton.
- Phase 3 complete: hypothesize + evaluate steps with Jinja2 prompts and FAISS RAG.
- Phase 4 complete: experiment step with sandboxed code execution and LLM-driven retry-on-error (max 3 retries).
- Phase 5 complete: novelty detection (embedding gate + soft-decay) and context pruning (tiktoken-based, <50K tokens for 5-iter runs).
- Phase 6 complete: HITL CLI with `--auto`, `--resume`, per-iteration redirect.
- Phase 7 complete: bias scanner + Markdown/plain-text report generator (auto-rendered at end of every run).
- Phase 8 complete: Streamlit UI with mission-control theme.
- Phase 9 complete: Predict mode (feature engineering loop + AutoGluon ensemble) + per-call cost tracker + UI overhaul.

## Quick start

**One command, anywhere:**

```powershell
# Windows PowerShell
git clone https://github.com/HoltYoung/HypothesisLoop.git
cd HypothesisLoop
.\run.ps1
```

```bash
# Mac / Linux / WSL
git clone https://github.com/HoltYoung/HypothesisLoop.git
cd HypothesisLoop
./run.sh
```

`run.ps1` / `run.sh` is idempotent and self-contained:

1. Checks Python 3.11+ is installed
2. Creates `.venv` if missing
3. Activates it
4. Installs dependencies (`pip install -e .`) — only on first run or when `pyproject.toml` changes
5. Verifies `.env` exists; prints a template if not (see below)
6. Launches the Streamlit UI at `http://localhost:8501`

Re-run anytime — subsequent launches skip the install step automatically.

**If you hit a PowerShell execution-policy block:**

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

### One-time `.env` setup

Create a file named exactly `.env` at the repo root with:

```ini
OPENAI_API_KEY=sk-...
KIMI_API_KEY=...                 # or MOONSHOT_API_KEY
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Usage

### Smoke test

One `@observe`'d call to the default model (Kimi `moonshot-v1-128k`); prints
a Langfuse trace URL.

```bash
python -m hypothesisloop.cli --smoke-test
```

### Interactive run (HITL mode)

```bash
python -m hypothesisloop.cli --question "What demographic factors most predict income > $50K?"
```

After each iteration the agent prints its hypothesis and result, then waits
for `[c]ontinue / [s]top / [r]edirect <text>`.

### Unattended run

```bash
python -m hypothesisloop.cli --auto --question "..." --max-iters 5
```

### Predict mode

Drives the loop to engineer features for a target column and trains an
AutoGluon ensemble at the end:

```bash
python -m hypothesisloop.cli --mode predict --data data/adult.csv \
    --target income --auto --max-iters 5 --automl-time-budget 120
```

After each iteration the loop:

1. proposes a feature operation (`create:<name>` / `transform:<name>` / etc.),
2. checks for target leakage + ASCII identifiers,
3. runs the code in the sandbox,
4. CV-scores the engineered DataFrame with a proxy model (LogReg / Ridge),
5. accepts the feature deterministically iff CV improvement ≥ threshold.

When the loop finishes, AutoGluon trains an ensemble on the engineered
training split and evaluates on the held-out test split. Outputs land in
`reports/<session-id>/`:

- `leaderboard.csv` — model leaderboard
- `feature_importance.csv` — SHAP-style importances from AG
- `automl_summary.json` — best model, test score, time budget
- `model/` — AutoGluon's `TabularPredictor` on disk

### Configuring providers

Default provider/model: Moonshot Kimi K2.6 (`moonshot-v1-128k`). Switch via
`--model gpt-4o-mini`, `--provider openai`, or `--api-key` (overrides env).
Set keys in `.env`:

```ini
KIMI_API_KEY=...               # or MOONSHOT_API_KEY
OPENAI_API_KEY=sk-...
```

The Streamlit UI exposes a Provider radio + API-key field; entered keys are
held only in the in-memory Streamlit session.

### Resume a saved run

```bash
python -m hypothesisloop.cli --resume hl-20260430-143000-abcd
```

Outputs land in `reports/<session-id>/`:

- `trace.json` — full DAG-tracked trace (saved on completion *and* on early stop / crash)
- `iter_<NNN>/attempt_<KK>/exp.py` — generated code per attempt
- `iter_<NNN>/attempt_<KK>/metrics.json` — emitted metrics
- `iter_<NNN>/attempt_<KK>/*.png` — generated figures

For all flags, run `python -m hypothesisloop.cli --help`.

### Reproducing the validation suite

```bash
python scripts/make_corrupted_adult.py        # builds data/adult_corrupted.csv
python scripts/run_validation_suite.py --dry-run    # prints plan + cost
python scripts/run_validation_suite.py --resume     # ~110 min wall, ~$2 in API
python scripts/aggregate_validation.py              # rebuilds the markdown deliverables
```

The 28-run suite covers 8 Explore-mode prompt categories (3 reps each) +
3 Predict-mode categories + 1 cross-dataset Predict run. See
`tests/validation_log.md` for the full log + success metrics, and
`docs/langfuse_traces.md` for representative trace URLs.

### Building / rebuilding the RAG index

```bash
python scripts/build_rag_index.py
```

Indexes `knowledge/adult_codebook.md` + `knowledge/test_selection.md` into
`knowledge/rag.index` + `knowledge/rag_chunks.pkl`.

## Streamlit UI

```bash
streamlit run hypothesisloop/ui/streamlit_app.py
```

Mission-control theme: live iteration cards (decision-colored borders),
in-app HITL (`▶ CONTINUE` / `■ STOP` / `↳ REDIRECT…`), report download. The
CLI and Streamlit share the same `agent.factory.build_steps`, so behavior
is identical across front-ends.

## Reports

Every run produces `reports/<session-id>/report.md` and `report.txt`
automatically. Re-render an existing run's report without re-running the loop:

```bash
python -m hypothesisloop.cli --report-only --resume <session-id>
```

The report has 9 sections: run metadata, question, hypothesis chain (with
embedded charts), key findings, rejections, bias flags, reasoning chain,
limitations, reproduction command. The plain-text version is for pasting into
the Final Report appendix.

## Layout

```
hypothesisloop/        # the package — agent/, steps/, sandbox/, llm/, trace/, primitives/, ...
data/adult.csv         # primary dataset
knowledge/             # RAG corpus (codebook, test-selection guide)
docs/SPEC.md           # full build spec (read first)
docs/DESIGN_DECISIONS.md
archive/               # Build 0–4 + assignment artifacts (frozen reference)
```

See `docs/SPEC.md` §5 for the full target layout and `docs/SPEC.md` §11 for
the phased build order.
