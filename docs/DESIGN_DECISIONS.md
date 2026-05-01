# Design Decisions Log — HypothesisLoop

This file is the **running record of architectural decisions and pivots** during the build. It feeds directly into Final Report §4 ("Iterations & Pivots").

Append, don't edit. Each entry: date, decision, alternatives considered, why, trade-off.

---

## 2026-04-30 — Pivot back to original proposal

**Decision:** Abandon Build 4's RAG-router-HITL REPL architecture. Rebuild around the original proposal's iterative Hypothesize→Experiment→Evaluate→Learn→Repeat loop.

**Alternatives considered:**
- (a) Extend Build 4 incrementally by adding a `for` loop around its REPL.
- (b) Keep Build 4 as the deliverable; submit it as HypothesisLoop with a relabeling pass.
- (c) **CHOSEN:** Salvage ~40% of Build 4 (analysis primitives, FAISS utils, chain builders, Langfuse plumbing, multi-provider dispatch) into a new `hypothesisloop/` package; archive the rest.

**Why:** Build 4 is a single-call request-response REPL with one persistent state slot (`code_approved`). The loop the proposal describes is autonomous, multi-round, with hypothesis novelty tracking and DAG-able branching. Wrapping a `for` loop around the REPL doesn't give us that; it gives us the same REPL run 5 times, with no learning between rounds.

**Trade-off:** ~10 working days of fresh build vs. 1-2 days to relabel Build 4. Worth it because the proposal *is* the project, and the final-report narrative ("we built what we proposed, not the cookie-cutter assignment") is much stronger with real autonomous iteration than with a relabeled router agent.

**Owner:** Holt + Sam.

---

## 2026-04-30 — Default LLM = Moonshot Kimi K2.6, fallback = OpenAI 4o-mini

**Decision:** `moonshot-v1-128k` is the default model for all steps; `gpt-4o-mini` is the documented fallback, selectable via `--model gpt-4o-mini` or `HL_MODEL` env var.

**Alternatives considered:**
- The frontier reasoning model originally specified in the proposal — abandoned for cost.
- All-OpenAI (4o-mini for all steps) — workable but loses Kimi's 128K context advantage for long traces.
- Split: Kimi for hypothesize+evaluate, 4o-mini for experiment codegen — deferred until we benchmark codegen quality.

**Why:** Kimi is roughly 1/10th the cost of Opus 4.6 and benchmarks comparably on tabular reasoning. 128K context handles full iteration traces without aggressive pruning. 4o-mini is the cheapest reliable backup.

**Trade-off:** Kimi's tool-use API is slightly less battle-tested than OpenAI's; we work around this by using JSON-mode + Pydantic validation rather than function-calling.

---

## 2026-04-30 — Sandbox = subprocess + AST denylist (not Docker)

**Decision:** Generated Python code runs via `subprocess.run([sys.executable, "exp.py"], cwd=session_dir, timeout=30)`; pre-execution AST scan blocks dangerous imports; POSIX `RLIMIT_AS` caps RAM at 1GB.

**Alternatives considered:**
- Docker-per-experiment (RD-Agent's approach) — too heavy for class scope (3-5s startup × ~15 runs/session).
- `RestrictedPython` — restricts the language too much; our generated code needs full pandas/scipy idioms.
- `eval`/`exec` in-process — unsafe and crashes the agent on any error.

**Why:** AST denylist + subprocess + timeout + RAM cap is sufficient for class scope where the only potential adversary is the LLM itself. The threat model is "LLM writes a buggy or wasteful script," not "attacker compromises the box."

**Trade-off:** Windows can't enforce RAM cap (no `setrlimit`); we document this as a known limitation. Could add a `psutil` poll-based watchdog later if needed.

---

## 2026-04-30 — Novelty = layered (prompt + embedding gate + soft-decay)

**Decision:** Three-layer novelty system: (1) prompt-injection of prior hypotheses, (2) embedding cosine gate at 0.85, (3) soft-decay after 3 consecutive rejections (loosens gate to 0.92 + tags `re_explore=True`).

**Alternatives considered:**
- Prompt-only (cheap, but LLM is sometimes stubborn).
- Embedding-only (kills creative re-framings).
- Hard ban on duplicates (loses re-exploration after a plateau).

**Why:** Layered gives us the best of all three. RD-Agent's CHANGELOG d41db0c moved to soft-decay for the same reason — pure ban left the agent stuck.

**Trade-off:** Embedding cost stacks linearly with iterations. Mitigated by caching by statement hash.

---

## 2026-04-30 — RAG corpus shrunk to codebook + test-selection guide

**Decision:** Drop Build 4's `analysis_workflow.md`, `tool_selection_rules.md`, `tools/*.md` from the FAISS index. Index only `adult_codebook.md` and a new `test_selection.md` (when to use t-test vs Mann-Whitney etc.).

**Alternatives considered:**
- Keep full Build 4 corpus — wasteful: most of it was scaffolding for the LLM to pick a tool name from a fixed allow-list, a problem the loop doesn't have.
- Drop RAG entirely — loses the column-cautions in the codebook, which prevent obvious mistakes (e.g., LLM hypothesizing on `fnlwgt` without knowing it's a survey weight).

**Why:** RAG should inject *domain* knowledge the LLM doesn't have, not workflow scaffolding. The loop *is* the workflow.

**Trade-off:** Less retrieved context per call → some risk of the LLM missing edge-case cautions. Mitigated by including the codebook as a static system-prompt block (not just retrieval) for the hypothesize step.

---

## 2026-04-30 — Trace schema = RD-Agent's tags verbatim

**Decision:** Adopt RD-Agent's Langfuse tag set: `research.hypothesis`, `evolving.codes`, `evolving.feedbacks`, `feedback.hypothesis_feedback`, plus our own `loop.iteration`, `safety.bias_flag`.

**Alternatives considered:**
- Custom HypothesisLoop schema — saves nothing, costs ~½ day designing it.
- Plain Langfuse with no schema discipline — hard to do per-iteration analysis later.

**Why:** RD-Agent's schema is battle-tested across multiple scenarios and gives us a hierarchy that nests naturally under one Langfuse session. Future RD-Agent trace viewers would work on our logs.

**Trade-off:** None significant.

---

## 2026-04-30 — HITL = per-iteration pause with `[c]/[s]/[r]<text>` (not per-step)

**Decision:** After each Hypothesize→Experiment→Evaluate→Learn cycle, pause with a CLI prompt: `[c]ontinue / [s]top / [r]edirect "<new hypothesis>"`. `--auto` flag skips the pause.

**Alternatives considered:**
- Per-step approval (Build 4 style) — annoying; defeats the autonomy.
- No HITL at all — loses the proposal's "user can intervene mid-loop" requirement and the safety value of catching a runaway agent.
- Approval only on bias-flag fires — too narrow; user might want to redirect for non-safety reasons.

**Why:** Per-iteration is the natural granularity. Each iteration is an atomic unit of work. Finer is annoying, coarser misses the chance to redirect early.

**Trade-off:** A 5-iter run with HITL takes maybe 2-3 minutes of human attention; `--auto` for unattended cases.

---

## 2026-04-30 — CLI lives at `hypothesisloop/cli.py`, not `hypothesisloop/ui/cli.py`

**Decision:** Phase 0 puts the argparse entry point at `hypothesisloop/cli.py`
(top-level module) so it can be invoked as `python -m hypothesisloop.cli`.
The `hypothesisloop/ui/` subpackage stays empty in Phase 0 and will house the
Streamlit app (Phase 8) and any richer UI shells later.

**Alternatives considered:**
- `hypothesisloop/ui/cli.py` per SPEC §5 — would need either a top-level
  re-export shim or `python -m hypothesisloop.ui.cli`, both worse for UX.

**Why:** The acceptance criterion is `python -m hypothesisloop.cli --help`.
A top-level `cli.py` is the simplest path to that, and it keeps the Streamlit
UI (a separate concern) in a dedicated subpackage.

**Trade-off:** SPEC §5 file map is no longer literal; will be updated when
SPEC.md is next touched.

---

## 2026-04-30 — Sandbox AST denylist also blocks bare `Name` references to interpreter dunders

**Decision:** Add a `DENIED_NAMES = frozenset({"__builtins__", "__import__"})`
set to `hypothesisloop/sandbox/allowlist.py`. The AST walker rejects any
`ast.Name` node whose `id` is in that set, in addition to the spec'd
`Call`/`Attribute`/`Import` checks.

**Alternatives considered:**
- (a) Rely on the spec'd `Call` + `Attribute` checks alone. Insufficient:
  `__builtins__["__import__"]("os")` reaches `__builtins__` via a bare
  `Name` plus a `Subscript`, and reaches `__import__` via a string key
  inside the subscript — neither path is `Call(func=Name(...))` nor
  `Attribute(...)`.
- (b) Block `Subscript` access generally. Way too aggressive — every
  `df['col']` is a Subscript. Would force us to whitelist subscript bases.
- (c) String-scan source for the dunder substrings. Brittle (false
  positives on string literals) and AST-mode is already in place.

**Why:** The sandbox is the only thing standing between LLM-generated code
and the host. The spec called out this exact bypass class ("If a bypass
is obvious (e.g. `__builtins__["__import__"]("os")`), add it to the
denylist and test for it"). A bare-`Name` check is the minimum surface
that defeats it.

**Trade-off:** Two identifiers (`__builtins__`, `__import__`) are now
unusable in user code. They're already reserved by CPython, so legitimate
generated code has no reason to bind or reference either. No measurable
cost.

**Owner:** Builder, Phase 1.

---

## 2026-04-30 — `--max-iters` is the session budget, not the per-invocation budget

**Decision:** When the CLI is invoked with `--resume <id> --max-iters N`, the
loop runs at most `max(0, N - trace.iteration_count())` more iterations. A
fresh run runs at most `N`. So `--max-iters 5` means "this session ends at 5
total iterations," regardless of how many `--resume` invocations got us there.

**Alternatives considered:**
- (a) Per-invocation budget — `--max-iters 4` after a 2-iter resume runs 4 more
  for a total of 6. Mechanically simpler (just pass through to `run_loop`), but
  a user has to do mental arithmetic to know "I want a 5-iter session" after
  resuming.
- (b) Add a separate `--add-iters` flag for resume — more flags, more confusion;
  no real win over a single coherent meaning of `--max-iters`.

**Why:** A user who runs `--max-iters 5`, hits Ctrl+C at iter 3, then resumes
with `--max-iters 5` expects 2 more iterations to finish the run. With the
session-budget interpretation, that just works. The per-invocation
interpretation would make resume + `--max-iters` together actively misleading.
SPEC §11 row 6's manual sanity test (`--resume … --max-iters 4` should give
4 total) confirms this is the intended reading.

**Trade-off:** Resuming with a budget already met is a no-op (the CLI prints
"budget exhausted, nothing to do" and exits 0 after re-emitting the summary).
A user who *wants* to add iterations beyond the original budget has to pass a
larger `--max-iters`. That feels right — the original budget was their
declared intent.

**Owner:** Builder, Phase 6.

---

## 2026-04-30 — `cli.py` lazy-imports + hardcoded `HL_MODEL_DEFAULT` mirror

**Decision:** All heavy imports in `hypothesisloop/cli.py` (langchain via `dispatch`, faiss via `primitives.rag`, langfuse via `trace.langfuse_client`, `pandas`, every step) are moved inside the functions that use them, not at module top level. Module-level imports are restricted to stdlib + `dotenv`. The `HL_MODEL_DEFAULT` constant is mirrored as a hardcoded string in `cli.py` (with a `# keep in sync with hypothesisloop/llm/dispatch.py` comment) so that `argparse` defaults can resolve without importing `dispatch.py`.

**Alternatives considered:**
- (a) Import `dispatch.py` for the constant — would re-introduce the langchain import at module load. Defeats the optimization.
- (b) Add a `hypothesisloop/constants.py` module with stdlib-only imports — cleaner but adds a file just for one constant. Rejected as over-engineering.
- (c) Accept slower `--help` and `--report-only` paths — fails Phase 8 acceptance criterion #2 (<1s and <5s respectively).

**Why:** Phase 7's report builder flagged that `--report-only` took ~20s mostly from module-level imports the path doesn't need. Phase 8 acceptance required <1s `--help` and <5s `--report-only`. The lazy-import refactor took both to 0.44s and 0.56s respectively. The hardcoded constant mirror is the smallest concession that preserves stdlib-only top-level imports.

**Trade-off:** `HL_MODEL_DEFAULT` now lives in two places (`dispatch.py` and `cli.py`). If a future contributor changes the default in `dispatch.py`, they must also update `cli.py`. A sync-comment flags this. Risk is low (the default rarely changes) and the alternative was worse.

**Owner:** Builder, Phase 8.

---

## 2026-04-30 — Predict-mode CV measurement lives in Evaluate, not ExperimentStep

**Decision:** In Predict mode, the proxy-model cross-validation measurement
that drives the deterministic accept/reject decision happens inside
``Evaluate.__call__`` (when ``predict_state`` is non-None), not inside
``ExperimentStep``. ExperimentStep still owns codegen, sandbox execution,
and the new target-leakage check.

**Alternatives considered:**
- (a) Spec sketch — put CV measurement in ExperimentStep. Cleaner if you read
  "experiment = the whole evaluation pipeline." But it forces ExperimentStep
  to reach across into the trace's running ``train_df`` AND know about
  metric thresholds, which couples it to Predict-mode internals.
- (b) Add a fourth step (``measure_predict``) to the loop. Clean separation
  but invasive: requires a new callable in ``run_loop`` and mode-conditional
  loop logic.

**Why:** Evaluate already has the LLM-call-and-override pattern (it
synthesizes the LLM-prose-reason while the deterministic threshold check
overrides the LLM's decision). The CV measurement is part of "deciding
whether this hypothesis was confirmed" — it belongs in the same step as the
decision. ExperimentStep stays narrowly focused on safety
(AST denylist + ASCII lint + leakage check + sandbox), which made the
retry-on-leakage path a one-liner via the existing ``blocked_reason``
machinery.

**Trade-off:** Evaluate now does double duty in Predict mode (LLM call +
in-process CV via ``exec``). The exec path is safe because the same code
already passed the AST denylist + leakage check inside ExperimentStep — but
it does mean Evaluate carries a slightly larger blast radius than its
single-responsibility name suggests. Documented in the module docstring.

**Owner:** Builder, Phase 9.

---

## Template for future entries

```markdown
## YYYY-MM-DD — <decision title>

**Decision:** <what we did>

**Alternatives considered:** <list>

**Why:** <reasoning>

**Trade-off:** <costs>

**Owner:** <who decided>
```
