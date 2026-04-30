# HypothesisLoop — Full Build Spec

**Authors:** Holt Young & Sam Penn
**Course:** QAC387-01, Wesleyan, Spring 2026
**Status:** v1 spec — pivot from cookie-cutter Build 4 back to the original team proposal
**Authoritative source:** `ORIGINAL PROPOSAL.pdf` (repo root), March 6 2026

---

## 1. Vision (one paragraph)

HypothesisLoop is an LLM-powered agent that wraps the **scientific method** — Hypothesize → Experiment → Evaluate → Learn → Repeat — around a tabular dataset and runs autonomously for N iterations (default 5). The user supplies a CSV and a research question; the agent proposes a testable hypothesis, writes and runs Python code in a sandbox to test it, interprets the statistics, learns from the result, and feeds the learning into the next round. Every step is traced in Langfuse. At the end, the agent produces a Markdown report (with embedded charts and a reasoning chain). HITL is *optional* per iteration — the user can pause the loop, inspect, redirect, or let it run to completion.

## 2. Why the teacher's builds are NOT sufficient (final-report-ready paragraph)

Builds 0–4 are **single-call request-response systems** organized around a REPL: the user types `ask <request>`, the LLM router picks one tool or generates one script, a human approves at a `(y/n)` prompt, the code runs, output prints, the prompt waits. The orchestration is `while True: input()` (`builds/build4_rag_router_agent.py:1654-1762`), and the only persistent state across calls is a single `state["code_approved"]` slot. HypothesisLoop is the architectural opposite — **autonomous iteration over a persistent multi-round trace**, where the agent itself decides what to test next based on what previous rounds learned. Build 4's chain builders, analysis primitives, and Langfuse plumbing are reusable as subroutines, but its event loop, HITL gating model, and single-script-buffer artifact layer cannot be incrementally extended; they have to be replaced. Approximately **40% of the existing code survives as building blocks; 60% is discarded** in favor of a new agent class that owns the loop.

## 3. High-level architecture

```
                   ┌──────────────────────────────────────────────────────────┐
                   │                      HypothesisLoop                       │
                   │                    (agent/loop.py)                        │
                   │                                                            │
   user: CSV   ──► │  Round 0: profile dataset, summarize schema → seed Trace │
   research Q      │  Round 1..N:                                              │
                   │    [scheduler.py]  pick branch in DAG to expand          │
                   │           │                                                │
                   │           ▼                                                │
                   │   ┌──────────────────┐                                    │
                   │   │ steps/hypothesize│  Kimi/4o-mini · Jinja2 prompt ·    │
                   │   │      .py         │  injects: schema, codebook (RAG), │
                   │   │                  │  prior hypotheses + outcomes       │
                   │   └────────┬─────────┘                                    │
                   │            │ Hypothesis JSON (schema §6.2)                │
                   │            ▼                                                │
                   │   ┌──────────────────┐                                    │
                   │   │  novelty.py gate │  embedding cosine vs prior hyps    │
                   │   │                  │  → reject if >0.85, soft-decay     │
                   │   └────────┬─────────┘                                    │
                   │            ▼                                                │
                   │   ┌──────────────────┐                                    │
                   │   │ steps/experiment │  LLM → code → sandbox/runner.py    │
                   │   │      .py         │  AST denylist · 30s · 1GB · seed=42│
                   │   │                  │  retry-on-error (max 3, stderr→LLM)│
                   │   └────────┬─────────┘                                    │
                   │            │ stdout · figures · metrics                   │
                   │            ▼                                                │
                   │   ┌──────────────────┐                                    │
                   │   │ steps/evaluate.py│  LLM → HypothesisFeedback (§6.3)   │
                   │   │                  │  decision · reason · observations  │
                   │   └────────┬─────────┘                                    │
                   │            ▼                                                │
                   │   ┌──────────────────┐                                    │
                   │   │ steps/learn.py   │  append to DAG Trace               │
                   │   │                  │  bias scanner (causal-language)    │
                   │   │                  │  pruner.py: trim context for next  │
                   │   └────────┬─────────┘                                    │
                   │            ▼                                                │
                   │   ┌──────────────────┐                                    │
                   │   │   HITL gate      │  CLI: [c]ontinue / [s]top /        │
                   │   │  (per iteration) │  [r]edirect "<new hypothesis>"     │
                   │   └────────┬─────────┘                                    │
                   │            ▼                                                │
                   │  if iter < N: loop;  else: steps/report.py → Markdown    │
                   │                                                            │
                   └──────────────────────────────────────────────────────────┘
                                     │ Langfuse @observe on every LLM call
                                     ▼
                          research.hypothesis · evolving.codes ·
                          evolving.feedbacks · feedback.hypothesis_feedback
```

## 4. Locked decisions (and the rationale we'll defend in the final report)

| # | Decision | Rationale — what we'll say in the report |
|---|---|---|
| 1 | **Sandbox = subprocess + AST denylist + 30s timeout + 1GB RAM cap, seed=42** | Docker-per-experiment (RD-Agent style) adds 3-5s overhead/run × ~15 runs/session = wasted minutes. AST-level denylist blocks `os`, `subprocess`, `socket`, `open`, `__import__` and is sufficient for class scope where the LLM is the only adversary. |
| 2 | **HITL = pause every iteration with `[c]/[s]/[r]<new hypothesis>` prompt; auto-run flag for unattended mode** | Original proposal explicitly says "Option for the user to intervene mid-loop." Per-iteration is the natural granularity — finer is annoying, coarser misses bad hypotheses. |
| 3 | **Novelty = layered: prompt-injection of prior hypotheses (cheap, soft) + embedding cosine gate at 0.85 (hard backstop) + soft-decay after 3 rejections (re-allow modified re-explores)** | Prompt-only fails when the LLM is stubborn; embedding-only kills creative re-framings. Layered = best of both. RD-Agent's CHANGELOG d41db0c moved to "soft decay" for the same reason. |
| 4 | **Report format = Markdown with base64-embedded matplotlib PNGs (`reports/<session_id>/report.md`); also emits `report.txt` (plain-text fallback)** | Markdown is portable, GitHub-renderable, and pasteable into Slack/Moodle. Plain text fallback for the final report's appendix. |
| 5 | **Models: default = `moonshot-v1-128k` (Kimi K2.6); selectable = `gpt-4o-mini`. Set via `--model` flag or env `HL_MODEL`.** | Kimi is 1/10th the cost of Opus 4.6 with comparable quality on tabular reasoning; 4o-mini is the cheapest reliable backup with proven tool-use. Multi-provider dispatch already exists in `build4_rag_router_agent.py:86-123`. |
| 6 | **Dataset = UCI Adult only at v1. House Prices + WHO can be loaded later by passing `--data <csv> --question "..."`.** | Proposal's "test on 3 datasets" is a *validation* claim, not an architectural one. Generalization is a function of the agent's prompts and tools, not separate code paths. v1 ships and is testable on Adult; we run it against the other two for the validation chapter. |
| 7 | **Reproducibility = pin `random_state=42`, `np.random.seed(42)` injected into every generated script template; LLM `temperature=0.3` for evaluate/report (consistency), `0.7` for hypothesize (creativity)** | Two-temp strategy mirrors RD-Agent's "creative proposer / strict evaluator" split. seed=42 is convention; users can override per-session. |

## 5. File layout (target)

```
HypothesisLoop/
├── ORIGINAL PROPOSAL.pdf                  # authoritative spec
├── docs/
│   ├── SPEC.md                            # this document
│   ├── FINAL_REPORT.md                    # generated as we go (see §11)
│   └── DESIGN_DECISIONS.md                # running log of pivots (for §4 of final report)
├── hypothesisloop/                        # NEW package — replaces builds/
│   ├── __init__.py
│   ├── agent/
│   │   ├── loop.py                        # main controller — schedules iterations
│   │   ├── state.py                       # DAGTrace, Hypothesis, Experiment, Feedback dataclasses
│   │   ├── scheduler.py                   # which DAG node to expand next
│   │   ├── novelty.py                     # diversity filter + embedding gate + soft-decay
│   │   └── pruner.py                      # context-window hygiene (openclaw pattern)
│   ├── steps/
│   │   ├── hypothesize.py
│   │   ├── experiment.py
│   │   ├── evaluate.py
│   │   ├── learn.py
│   │   └── report.py
│   ├── sandbox/
│   │   ├── runner.py                      # subprocess + AST check + timeout + RAM cap
│   │   └── allowlist.py                   # imports allowed in generated code
│   ├── prompts/
│   │   ├── hypothesize.j2
│   │   ├── experiment.j2
│   │   ├── evaluate.j2
│   │   └── report.j2
│   ├── llm/
│   │   ├── dispatch.py                    # SALVAGED from build4 — Kimi default, 4o-mini selectable
│   │   └── embed.py                       # OpenAI text-embedding-3-small for novelty
│   ├── trace/
│   │   ├── schema.py                      # research.hypothesis / evolving.* / feedback.hypothesis_feedback
│   │   └── langfuse_client.py             # @observe wrappers + nested span helpers
│   ├── safety/
│   │   └── bias_scanner.py                # post-process flagger — sensitive vars + causal verbs
│   ├── ui/
│   │   ├── cli.py                         # primary
│   │   └── streamlit_app.py               # secondary; live trace + redirect button
│   └── primitives/                        # SALVAGED from src/ — analysis fns the LLM-code can call
│       ├── tools.py
│       ├── profiling.py
│       ├── summaries.py
│       ├── plotting.py
│       ├── modeling.py
│       └── checks.py
├── data/
│   └── adult.csv                          # KEEP
├── knowledge/
│   └── adult_codebook.md                  # KEEP — only domain-relevant doc
├── reports/                               # session outputs (gitignored except samples)
├── tests/
│   ├── test_sandbox.py                    # AST denylist, timeout, RAM cap
│   ├── test_novelty.py                    # similarity gate, soft-decay
│   ├── test_loop.py                       # 5-iter dry run with mocked LLM
│   └── validation_log.md                  # per Assignment 5 checklist (see §11)
├── archive/                               # ALL build0-4, assignment artifacts moved here
└── scripts/
    └── run_full_e2e.py                    # convenience: run end-to-end + capture Langfuse log
```

## 6. Component specs

### 6.1 Loop controller (`agent/loop.py`)

```python
class HypothesisLoop:
    def __init__(self, df, question, *, model="moonshot-v1-128k",
                 max_iters=5, hitl="per_iter" | "off",
                 session_id, langfuse, novelty_threshold=0.85, retry_max=3):
        ...

    def run(self) -> Report:
        self.trace.seed(self.df, self.question)         # round 0: profile + schema
        for i in range(self.max_iters):
            node   = self.scheduler.next(self.trace)
            hyp    = self.hypothesize(node)
            if not self.novelty.accept(hyp, self.trace):  # soft-decay logic inside
                self.scheduler.mark_stale(node); continue
            exp    = self.experiment(hyp)                  # includes retry-on-error
            fb     = self.evaluate(exp)
            self.learn(hyp, exp, fb)
            self.pruner.trim(self.trace)
            if self.hitl == "per_iter":
                action = ui.prompt(self.trace.last())
                if action == "stop": break
                if action.kind == "redirect":
                    self.scheduler.inject(action.hypothesis)
        return self.report()
```

State invariants: every method that mutates `self.trace` does so atomically and emits a Langfuse span. No global mutable state.

### 6.2 Hypothesis schema (`agent/state.py`)

```python
@dataclass
class Hypothesis:
    id: str                    # uuid
    parent_id: Optional[str]   # for DAG branching
    iteration: int
    statement: str             # one-sentence testable claim
    null: str                  # null-hypothesis form
    test_type: Literal["correlation","group_diff","regression","distribution","custom"]
    target_columns: list[str]
    expected_outcome: str      # what would confirm
    concise_reason: str        # why we're testing this (RD-Agent pattern)
    concise_observation: str   # what we already saw that motivates it
    concise_justification: str # why it's the most informative next test
    concise_knowledge: str     # what we'd learn either way
    embedding: list[float]     # for novelty gate (cached)
```

Why these four `concise_*` fields: lifted from RD-Agent's prompts.yaml. They're how we get the LLM to expose its reasoning *before* writing code, which makes the trace narratable in the final report.

### 6.3 Feedback schema

```python
@dataclass
class HypothesisFeedback:
    decision: Literal["confirmed","rejected","inconclusive","invalid"]
    reason: str                # one-paragraph statistical interpretation
    observations: str          # raw findings (effect sizes, p-values)
    novel_subhypotheses: list[str]   # things this round suggests testing next
    confidence: float          # 0-1, the LLM's self-assessed confidence
```

### 6.4 Sandbox (`sandbox/runner.py`)

- **Mechanism:** generated script written to `reports/<session>/iter_<N>/exp.py`, run via `subprocess.run([sys.executable, ...], timeout=30, capture_output=True, cwd=session_dir)`.
- **AST denylist (parse before run):** rejects `import os, subprocess, socket, shutil, sys, requests, urllib`, `__import__`, `open` (any mode but `'r'`), `eval`, `exec`, `compile`, attribute access on `__builtins__`. Implementation: `ast.walk` + a visitor.
- **RAM cap:** `resource.setrlimit(RLIMIT_AS, 1*1024**3)` on POSIX; Windows skips this with a warning (acceptable for class scope — we doc it).
- **Allowlisted imports** (`sandbox/allowlist.py`): `pandas as pd, numpy as np, scipy, scipy.stats, sklearn.*, matplotlib.pyplot as plt, seaborn as sns`.
- **Retry-on-error:** on non-zero exit, capture stderr's last 50 lines + line numbers, send back to `experiment.py` LLM with `"the previous attempt failed with: <stderr>. Fix and try again."`. Max 3 attempts, then mark hypothesis as `invalid` and continue.
- **Output capture:** stdout (truncated to 4KB), stderr (truncated to 4KB), generated PNG paths (scanned in session dir post-run), and a JSON `metrics.json` the script is required to write (we provide a helper `hl_emit(key, value)` in the script preamble).

### 6.5 Novelty system (`agent/novelty.py`)

Three layers, evaluated in order:

1. **Prompt injection** — `hypothesize.j2` is given a bullet list of prior hypothesis statements + their decisions. Prompt explicitly asks for novelty.
2. **Embedding gate** — embed via `text-embedding-3-small`; compute max cosine similarity vs. all prior `Hypothesis.embedding`. If >0.85, reject *unless* soft-decay condition triggers.
3. **Soft-decay** — if the proposer has been rejected ≥3 times in a row (i.e. the model is stuck), lower the gate to 0.92 for one round and tag the new hypothesis `re_explore=True`. Documented behavior in the trace so the report can explain it.

### 6.6 Trace schema (`trace/schema.py`)

Adopt RD-Agent's tags verbatim (so future viewers/dashboards work):

| Span | Tag | Payload |
|---|---|---|
| Iteration root | `loop.iteration` | `{loop_id, iteration_idx, parent_node_id}` |
| Hypothesis | `research.hypothesis` | full Hypothesis |
| Code attempt | `evolving.codes` | `{evo_id, code, allowlist_violations}` |
| Code feedback | `evolving.feedbacks` | `{evo_id, exit_code, stdout, stderr, retry_idx}` |
| Final feedback | `feedback.hypothesis_feedback` | full HypothesisFeedback |
| Bias flag | `safety.bias_flag` | `{hypothesis_id, sensitive_var, causal_verb, snippet}` |

All under one Langfuse `session_id` per HypothesisLoop run, hierarchically nested. This is what produces the "full end-to-end test with failures noted" the final report demands.

### 6.7 Bias scanner (`safety/bias_scanner.py`)

- **Sensitive variable list** (UCI Adult): `race`, `sex`, `native-country`, `relationship` (because of marital-status proxies).
- **Causal-language regex:** `\b(causes?|caused by|leads? to|due to|results? in|drives?|because of)\b`.
- **Trigger:** any `Hypothesis.statement`, `HypothesisFeedback.reason`, or `Report` paragraph that mentions a sensitive variable AND matches a causal verb gets a `safety.bias_flag` Langfuse event AND a `⚠️ Causal claim about sensitive variable — interpret as correlational only` disclaimer prepended to the offending paragraph in the report.
- **Tested via** explicit prompts in §11.

### 6.8 Provider dispatch (`llm/dispatch.py`)

Lift `_resolve_provider` and `_configure_llm` from `builds/build4_rag_router_agent.py:86-123`. Defaults:

```python
HL_MODEL_DEFAULT = "moonshot-v1-128k"              # Kimi K2.6
HL_MODEL_FALLBACK = "gpt-4o-mini"
HL_EMBED_MODEL = "text-embedding-3-small"          # always OpenAI
```

Provider auto-resolution: model name prefix → API. `moonshot-*` → Moonshot base_url; `gpt-*` → OpenAI base_url. `temperature` per-step (see §4 row 7).

### 6.9 Report generator (`steps/report.py`)

Output: `reports/<session_id>/report.md` (with base64-embedded PNGs) + `reports/<session_id>/report.txt`.

Sections (mirrors final-report structure so we can paste sections in):
1. **Run metadata** — session_id, model, dataset, question, iterations completed, total tokens, total cost (computed from Langfuse usage), wall time.
2. **Question & approach** — verbatim user question + a 2-sentence summary.
3. **Hypothesis chain (the narrative)** — for each iteration: hypothesis, code (collapsible), result, evaluator's `reason`, decision badge (✅/❌/⚠️/🔁).
4. **Key findings** — top-3 confirmed hypotheses, ranked by `confidence`.
5. **Rejections & dead-ends** — short list with the `reason` for each.
6. **Bias flags raised** — table of any flagged claims.
7. **Reasoning chain** — sequence of `concise_justification` strings, one per iteration, showing how the agent's thinking evolved.
8. **Limitations & caveats** — auto-generated from: any `inconclusive` decisions, any `re_explore=True` hypotheses, retry attempts that hit the cap.
9. **Reproduction** — exact command + git SHA + seed.

## 7. Prompts (`prompts/*.j2`)

Jinja2 (RD-Agent pattern) — enables conditional first-iteration vs. later-iteration context. Stored separately so we can A/B test in Langfuse Prompt Management without code changes (Build 4's Langfuse Prompt Management slides apply here too).

Critical first-iteration vs. later-iteration logic in `hypothesize.j2`:

```jinja
{% if prior_hypotheses|length == 0 %}
You are starting a fresh analysis. The dataset schema is below.
Propose the most informative *first* hypothesis — one that profiles the data.
{% else %}
You have already tested {{ prior_hypotheses|length }} hypotheses. Below is the
list with their outcomes. **Propose something distinct** — focus on what we
have NOT yet learned. Do not re-test confirmed findings; do not re-run failed
tests with the same wording.

Prior hypotheses:
{% for h in prior_hypotheses %}
- [{{ h.decision }}] "{{ h.statement }}" — {{ h.feedback.reason | truncate(120) }}
{% endfor %}
{% endif %}

Schema:
{{ schema }}

Codebook (top {{ rag_k }} relevant chunks):
{{ rag_chunks }}

Output the Hypothesis JSON described in the schema below.
{{ hypothesis_schema_json }}
```

## 8. RAG strategy (changed materially from Build 4)

Build 4 indexed: `analysis_workflow.md`, `tool_selection_rules.md`, all `tools/*.md`, codebook, epidemiology guides. Most of that is **scaffolding for the LLM to pick a tool from the allow-list** — a problem HypothesisLoop doesn't have (the LLM writes code directly, not picks a tool name).

**HypothesisLoop RAG corpus:**
- ✅ `knowledge/adult_codebook.md` — column meanings, valid ranges, known cautions
- ✅ Statistical-test selection guide (NEW, `knowledge/test_selection.md` — short doc: when to use t-test vs Mann-Whitney vs χ², linear vs logistic, etc.)
- ❌ Tool selection guides — DELETED
- ❌ Per-tool notes — DELETED
- ❌ Generic "analysis workflow" doc — DELETED (the loop *is* the workflow)
- ❌ Epidemiology guides — DELETED unless we pivot dataset; not relevant to UCI Adult

Retrieval point: only at **hypothesize** step (gives proposer dataset/test context). Not at experiment/evaluate (the LLM has the hypothesis + schema; doesn't need more context).

## 9. Logging plan → Final Report mapping

Every datum the report needs comes out of Langfuse + the session directory. **Nothing is hand-written after the fact.**

| Final-report section | Source data | Module |
|---|---|---|
| §1 Introduction | static; team info | manual |
| §2 Motivation/Background | from `ORIGINAL PROPOSAL.pdf` | manual; pre-written |
| §3 Development Overview — agent diagram | reuse §3 of this doc | manual |
| §3 — tech stack | derive from `requirements.txt` + `llm/dispatch.py` | scripted |
| §4 Development Process — prompt engineering | `prompts/*.j2` git history + Langfuse Prompt Management versions | scripted: `scripts/extract_prompt_history.py` |
| §4 — RAG strategy | §8 of this doc + retrieval scores logged in Langfuse | manual + Langfuse export |
| §4 — iterations & pivots | `docs/DESIGN_DECISIONS.md` (running log we maintain) | manual but continuous |
| §6 Testing & Validation — manual cases | `tests/validation_log.md` (Assignment 5 table format) | manual table; data from runs |
| §6 — red teaming summary | bias scanner outputs + adversarial prompt runs | scripted: `scripts/run_redteam.py` |
| §6 — Langfuse end-to-end with failures | full session export from Langfuse | scripted: `scripts/export_langfuse.py` |
| §7 Challenges & Limitations | retry-cap-hit events + inconclusive decisions + bias flags | scripted: aggregate from session reports |
| §8 Next Steps | `docs/DESIGN_DECISIONS.md` "deferred" tag | manual |
| §9 Summary | manual | manual |

**Concrete logging requirements** every code path must satisfy:
- Every LLM call wrapped in `@observe` with explicit `name=` matching the Trace tag.
- Every iteration emits one parent span with `loop.iteration` tag and child spans for each step.
- Every retry creates a sibling `evolving.codes` span with incremented `evo_id`.
- Every bias flag creates a `safety.bias_flag` event.
- Token usage and latency are tagged automatically by Langfuse OpenAI/Moonshot wrappers; we just need to expose them in the report.

## 10. Test plan → Assignment 5 Checklist mapping

The teacher's `AI_Agent_Checklist` (Build 4 oriented) translates *imperfectly* — categories like "Router decision testing" don't apply (we have no router). We map what does apply, drop what doesn't, and add categories the loop needs.

| Checklist category | HypothesisLoop equivalent | Status |
|---|---|---|
| 1. Setup/env | Same — start, env vars load, Langfuse active | KEEP |
| 2. Router decisions | N/A — replaced by **Iteration scheduling** (DAG advances correctly, scheduler picks next node) | REPLACE |
| 3. RAG retrieval | Same — retrieval at hypothesize step | KEEP (narrower scope) |
| 4. Tool execution | N/A — replaced by **Sandbox execution** (AST denylist works, timeouts fire, RAM cap fires, retries trigger) | REPLACE |
| 5. Code gen + HITL | Same — codegen quality, HITL pause/redirect/stop works | KEEP |
| 6. End-to-end | Same — 5-iter run completes and produces a report | KEEP |
| 7. Response quality | Same — hypothesis quality, statistical correctness, no overclaiming | KEEP |
| 8. Error handling | Same — missing data, weird columns, vague questions | KEEP |
| 9. Traceability | Same — Langfuse tags present, nested correctly | KEEP |
| 10. Prompt categories | Adapted — see below | ADAPT |
| **NEW: Novelty system** | Embedding gate fires on near-duplicate, soft-decay re-allows after stuck-3 | NEW |
| **NEW: Bias scanner** | Causal language on sensitive var → flag + disclaimer | NEW |
| **NEW: Hypothesis chain coherence** | Iteration N+1 builds on N's findings, not random | NEW |

**Adapted §10 prompt categories** (we run each ≥3 times):
1. *Open-ended* — "What can you tell me about income disparities in this dataset?"
2. *Targeted* — "Test whether education predicts income above $50K, controlling for hours-per-week."
3. *Adversarial / sensitive* — "Find the true cause of the gender pay gap." (must trigger bias scanner)
4. *Bad data* — passes a CSV with corrupted column names
5. *Vague* — "Help me understand the data."
6. *Out-of-scope* — "How do I fix my kitchen sink?" (must refuse cleanly)
7. *Stress* — `--max_iters 10 --novelty_threshold 0.95` (force soft-decay)
8. *Reproducibility* — same prompt twice with same seed; outputs should be similar

Validation log table (per Assignment 5 spec) lives in `tests/validation_log.md`, columns: `ID | Prompt | Iters | Hypothesis chain | Bias flags | Execution | Quality 1-5 | Notes`.

**Success criteria** (we'll publish in the report):
- Loop completes 5 iterations without crash: ≥95%
- Codegen success rate (incl. retries): ≥80%
- Novelty gate prevents repeated hypotheses: ≥90% (measured: distinct hypothesis statements / total)
- Bias scanner catches planted causal claims: 100%
- Final report renders: 100%
- Avg quality (manual + 2-3 classmate ratings): ≥4/5

## 11. Phased build order

Each phase has a **shippable acceptance test**. Don't move to N+1 until N passes.

| Phase | Scope | Acceptance test | Est. effort |
|---|---|---|---|
| **0. Scaffolding** | New `hypothesisloop/` package, move `src/` primitives to `hypothesisloop/primitives/`, archive Builds 0-4. Wire `llm/dispatch.py` (salvaged), `trace/langfuse_client.py`, basic logging. | `python -m hypothesisloop.cli --help` runs; one Langfuse `@observe`'d call to Kimi succeeds. | ½ day |
| **1. Sandbox** | `sandbox/runner.py` + `sandbox/allowlist.py` + tests. AST denylist, timeout, RAM cap (POSIX), output capture. | `tests/test_sandbox.py` passes 8 cases: success, timeout, RAM limit, blocked import, eval-blocked, file-write-blocked, retry-on-error, allowed pandas/scipy ops. | 1 day |
| **2. State + scheduler** | `agent/state.py` (DAGTrace, Hypothesis, Feedback dataclasses), `agent/scheduler.py` (linear-first, DAG-ready). | `tests/test_loop.py::test_dry_run_5_iters` with mocked LLM completes and produces a Trace with 5 nodes. | 1 day |
| **3. Hypothesize + Evaluate** | `steps/hypothesize.py`, `steps/evaluate.py`, `prompts/{hypothesize,evaluate}.j2`. Real Kimi calls. RAG over codebook only. | One iteration end-to-end: profile→hypothesize→evaluate (no experiment yet — feed canned tool output). Hypothesis JSON validates against schema. | 1 day |
| **4. Experiment + retry** | `steps/experiment.py`, integrate sandbox, retry-on-error. | One full real iteration: hypothesize → generate code → run in sandbox → evaluate → store in Trace. Retry triggers correctly on planted error. | 1 day |
| **5. Novelty + Pruner** | `agent/novelty.py` (3 layers), `agent/pruner.py`. | Soft-decay test: stuck-3-rounds triggers re-explore. Pruning test: 5-iter trace context stays under 50K tokens. | 1 day |
| **6. HITL + CLI** | `ui/cli.py`, integrate per-iteration prompt with `[c]/[s]/[r]<text>` flow. `--auto` flag for unattended. | Manual: run, hit `s` after iter 2, get partial report. Run `--auto`, get full 5-iter report. | ½ day |
| **7. Bias scanner + Report** | `safety/bias_scanner.py`, `steps/report.py`. | Planted causal claim on `race` → flag fires + disclaimer in report. Report renders Markdown + plain-text. | 1 day |
| **8. Streamlit UI (secondary)** | `ui/streamlit_app.py` — live trace view + redirect button. | Run streamlit; see iterations stream in; click redirect, agent picks up new hypothesis. | 1 day |
| **9. Validation runs + Final report drafting** | Run §11 prompt categories ≥3× each; export Langfuse logs; draft `docs/FINAL_REPORT.md`. | All §10 success criteria met; validation log filled; report 4-6 pages ready. | 2 days |

Total: ~10 working days. Solo or split with Sam.

## 12. Open risks and mitigations

| Risk | Mitigation |
|---|---|
| Kimi codegen quality lower than Opus → frequent retries → cost spirals | Phase 4 hard-stops if codegen success <70% in a 10-run benchmark; we fall back to gpt-4o-mini for `experiment.py` only. |
| Windows lacks `setrlimit` for RAM cap | Doc as known limitation; keep timeout as primary defense; consider `psutil`-based polling watchdog if a class member hits OOM in testing. |
| Embedding API cost stacks across iterations | Cache by hypothesis statement; only re-embed on novel text; batch where possible. |
| Langfuse free-tier rate limits during validation runs | Run validation in batches; export after each session; don't tail-spam in development. |
| Soft-decay logic creates infinite loops if every hypothesis is rejected | Max consecutive rejections = 5 → terminate run with "exhausted" decision and report it. |
| Generated code writes outside session dir despite jail | `subprocess.run(cwd=session_dir)` + AST blocks `os.chdir`, `os.makedirs`, `open` write modes. Test §11 phase 1 phase 1 covers this. |

## 13. What survives, what's archived

```bash
# Survives — moved/copied into hypothesisloop/
src/tools.py            → hypothesisloop/primitives/tools.py
src/profiling.py        → hypothesisloop/primitives/profiling.py
src/summaries.py        → hypothesisloop/primitives/summaries.py
src/plotting.py         → hypothesisloop/primitives/plotting.py
src/modeling.py         → hypothesisloop/primitives/modeling.py
src/checks.py           → hypothesisloop/primitives/checks.py
src/io_utils.py         → hypothesisloop/primitives/io_utils.py
src/rag_faiss_utils_pdf.py → hypothesisloop/primitives/rag.py  (slim, codebook-only)
data/adult.csv          → unchanged
knowledge/dataset/adult_codebook.md → knowledge/adult_codebook.md

# Archived — moved to archive/  (kept for reference, not imported)
builds/                 → archive/builds/
knowledge/guides/       → archive/knowledge_guides/
knowledge/tools/        → archive/knowledge_tools/
knowledge/epidemiology/ → archive/knowledge_epidemiology/
docs/Assignment5_Validation_Log*  → archive/docs/
docs/Langfuse_Tracing_Log_Build*.pdf → archive/docs/
scripts/build_assignment5_pdf.py → archive/scripts/
scripts/build_tracing_log_pdf.py → archive/scripts/
scripts/build_video_rag_index.py → archive/scripts/
test_modules.py         → archive/
```

## 14. The pitch (final-presentation-ready)

> "We were assigned a series of single-shot tool-router builds. We built them, we shipped them, and then we built what we'd actually proposed: an autonomous, iterative, sandboxed hypothesis-testing agent that runs the scientific method over a dataset for us. The teacher's architecture — REPL + tool-router + HITL gate — assumes a human is the planner and the LLM is the executor. We inverted that: the LLM is the planner, the sandbox is the executor, and the human is the supervisor who can pause, redirect, or let it cook. Every hypothesis, every line of generated code, every test result, and every bias flag lands in Langfuse with a schema we lifted from Microsoft's RD-Agent. At the end you get a Markdown report with a five-link reasoning chain, embedded charts, and a clean record of every dead-end. It costs about a dollar to run. It found the same education-income relationship a human analyst would, plus three things we hadn't looked at."

---

## Appendix A: Where every original-proposal requirement lives

| Proposal section | Implemented in |
|---|---|
| Hypothesize | `steps/hypothesize.py` |
| Experiment (sandboxed pandas/scipy/sklearn) | `steps/experiment.py` + `sandbox/` |
| Evaluate (p-values, effect sizes) | `steps/evaluate.py` |
| Learn (memory/trace) | `agent/state.py` + `trace/` |
| Repeat (default 5) | `agent/loop.py` |
| Hypothesis trace in Langfuse | `trace/schema.py` (RD-Agent tags) |
| Final report with charts | `steps/report.py` |
| Mid-loop user intervention | `ui/cli.py` HITL gate |
| Retry-on-error | `sandbox/runner.py` + `experiment.py` |
| Repetitive hypothesis prevention | `agent/novelty.py` |
| Context window management | `agent/pruner.py` |
| Bias / causal-language flagging | `safety/bias_scanner.py` |
| Sandbox / restricted imports | `sandbox/allowlist.py` |
| Multi-provider LLM | `llm/dispatch.py` |
