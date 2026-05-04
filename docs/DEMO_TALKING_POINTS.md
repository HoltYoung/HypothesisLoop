# HypothesisLoop — Demo Talking Points (for Sam)

**Audience:** the teacher / TA. They want two things: (1) does our project actually
extend Build 4, the way the assignments scaffolded toward, and (2) does it work?

**Total time budget:** ~7 minutes pitch + ~3 minutes live demo + Q&A.

---

## The 30-second elevator pitch (open with this)

> "HypothesisLoop is an LLM agent that wraps the **scientific method** —
> Hypothesize → Experiment → Evaluate → Learn → Repeat — around a tabular
> dataset and runs autonomously for five iterations. The user gives it a CSV
> and either a research question or a target column to predict; the agent
> does the rest. Every step is traced in Langfuse; the final output is a
> Markdown report with embedded charts plus, in Predict mode, a trained
> AutoGluon ensemble ready to deploy."

---

## How it connects to the assignments she assigned

The assignments scaffolded us toward exactly this kind of system, one capability per build:

| Build | What it gave us | Where it lives in HypothesisLoop |
|---|---|---|
| **Build 0** | Stateless analysis primitives (profiling, summaries, regression, plotting) | `hypothesisloop/primitives/` — same code, the agent calls these from inside generated scripts |
| **Build 1** | LLM-only assistant (LangChain + OpenAI dispatch) | `hypothesisloop/llm/dispatch.py` — extended to multi-provider (Kimi K2.6 + GPT-4o-mini) |
| **Build 2** | Human-in-the-loop code-approval gate | `hypothesisloop/agent/loop.py` — the per-iteration `[c]ontinue / [s]top / [r]edirect` pause |
| **Build 3** | Tool routing — LLM picks an analysis tool from an allow-list | The hypothesis step decides what kind of test to run; routing logic moved to a JSON-schema constraint on the LLM output |
| **Build 4** | RAG over a knowledge corpus + Langfuse tracing | `hypothesisloop/primitives/rag.py` (FAISS), `hypothesisloop/trace/` (Langfuse) — both directly salvaged |

**The pitch line:** *"Build 4 was a single-call request-response REPL — you ask one question, it answers, you ask the next. HypothesisLoop took everything Build 4 gave us and wrapped it in the autonomous iteration loop our original proposal called for."*

About **40% of Build 4's code** lives inside HypothesisLoop unchanged
(analysis primitives, FAISS utilities, multi-provider LLM dispatch,
Langfuse `@observe` decorators). The other 60% — the REPL event loop,
the per-call HITL gates, the single-script-buffer state model — was
replaced because it couldn't be incrementally extended into a
multi-iteration agent. This pivot is documented in `docs/DESIGN_DECISIONS.md`.

---

## How it works — high-level flow (one minute)

The agent runs five iterations by default. Each iteration:

1. **Hypothesize.** LLM proposes a falsifiable, testable claim (in JSON, with predicted effect size). Sees prior iterations' hypotheses + decisions + code failures so it doesn't repeat itself.
2. **Experiment.** LLM writes Python; we run it in a **sandboxed subprocess** (AST denylist blocks `os`, `subprocess`, `eval`, file writes outside session dir; 30 s timeout; RAM cap on Linux). Up to 4 attempts on failure, with stderr fed back to the LLM for self-correction.
3. **Evaluate.** LLM interprets the output (p-values, effect sizes, etc.). In Predict mode, a deterministic threshold check on cross-validated AUC overrides the LLM's verdict so accept/reject is reproducible, not vibes.
4. **Learn.** Result lands in a DAG-tracked trace. Novelty gate (embedding cosine ≥ 0.85) rejects near-duplicates of prior hypotheses. Bias scanner flags causal language about sensitive variables (race, sex, native-country, marital status).
5. **Repeat** until the budget runs out or the user hits **[s]top**.

Final output: `reports/<session>/report.md` with the full reasoning chain, embedded charts, and an auto-generated "Limitations & Caveats" section.

---

## A little in the weeds (have these ready if asked)

**Why Kimi K2.6 instead of GPT-4o or Claude?**
- 1/10th the cost of comparable frontier models on tabular reasoning
- 128K context handles full iteration traces without aggressive pruning
- Multi-provider dispatch means we can swap to GPT-4o-mini per call — same architecture

**Why a sandbox instead of just `eval`?**
- The LLM is the only adversary, but it's an unpredictable one — infinite loops, OOM, accidental file writes
- Subprocess + AST denylist is lightweight (no Docker overhead) and sufficient at class scope

**Why the novelty gate?**
- Without it, an LLM that found a strong correlation in iter 1 will propose 4 variations of the same finding in iters 2-5
- Layered defense: prompt-level "avoid duplicates" + embedding-cosine gate at 0.85 + soft-decay (loosens to 0.92 after 3 consecutive rejections so the loop doesn't starve)

**What's the "Predict mode" version?**
- User picks a target column from the CSV instead of writing a research question
- The loop iteratively engineers features (`log_capital_gain`, `age × hours_per_week`, etc.) and CV-tests whether each one moves the held-out ROC AUC by > +0.001
- After the iterations, AutoGluon trains an ensemble on the curated feature set
- On UCI Adult: baseline `roc_auc = 0.9051` → final ensemble `0.9315` ≈ **+2.7 percentage points**

**What gets traced in Langfuse?**
- Every LLM call, every iteration, every retry, every novelty rejection, every bias flag
- Schema borrowed verbatim from Microsoft's RD-Agent (`research.hypothesis`, `evolving.codes`, `evolving.feedbacks`, `feedback.hypothesis_feedback`) so future tooling on top of those tags works

**What's the bias scanner doing?**
- Regex match: any text mentioning a sensitive variable (`race`, `sex`, `native_country`, `marital_status`) AND a causal verb (`causes`, `leads to`, `due to`, `drives`, `because of`, etc.) → flagged
- Two layers: (1) scans the LLM's hypothesis statements + feedback reasons; (2) scans the user's question itself, since the LLM tends to self-discipline language even when prompted causally
- Flagged paragraphs in the report get a `⚠ correlational only` disclaimer prepended

---

## Validation numbers (have these in your back pocket)

We ran a 28-run validation suite covering 8 prompt categories (open-ended, targeted, adversarial, vague, off-topic, stress, reproducibility, plus 4 Predict-mode runs). Real numbers:

- **Loop completion (no crash):** 89% (25/28)
- **Codegen success rate (with retries):** 87% — Kimi sometimes writes valid analysis followed by a crashing plot block; metrics still emit
- **Novelty gate firing:** 28 rejections across 149 total iterations
- **Bias scanner — node level:** 0 fires (LLM self-disciplines)
- **Bias scanner — question level:** catches every causal-framed prompt (added as backstop)
- **AutoGluon Predict lift on UCI Adult:** +2.6 ROC-AUC points over baseline
- **Cost:** ~$1.30 for the entire 28-run suite

Full log at `tests/validation_log.md`; trace URLs at `docs/langfuse_traces.md`.

---

## The 3-minute live demo script

Open `https://huggingface.co/spaces/holtyoung/HypothesisLoop` in the browser (or run `.\run.ps1` locally if HF is flaky).

1. **Set up (15 sec).** "Default dataset is UCI Adult Income — 32K records, 15 columns, predicting whether income exceeds $50K." Switch to **Predict** mode. Target = `income`. Auto-detect picks classification + ROC AUC.
2. **Start the run (15 sec).** Hit **▶ START RUN**. "Auto mode is on, so it'll run 5 iterations end-to-end without asking."
3. **Narrate iter 1 (1 min).** "The LLM just proposed adding `log_capital_gain` as a feature, predicted it would improve AUC by +0.001 — let's see what actually happens." Click into the iteration card → CODE tab ("here's the Python it wrote") → METRICS tab ("p_value, effect size, all labeled") → FEATURE tab ("predicted Δ vs actual Δ — kept or rejected").
4. **Skip ahead to AutoGluon (1 min).** Wait for the AutoGluon training card. "After the loop curates features, we hand the engineered set to AutoGluon for the final ensemble. On Adult, baseline 0.905 → final 0.93. The leaderboard.csv is downloadable from the sidebar."
5. **Show the report (30 sec).** Click **⬇ report.md**. Open it. "Auto-generated, 9 sections, embedded charts, full reasoning chain. This is what we'd hand to a stakeholder."

---

## Anticipated questions + tight answers

**Q: How is this different from ChatGPT's Code Interpreter?**
A: One-shot vs iterative. Code Interpreter answers your question; HypothesisLoop runs the whole research loop and produces a multi-step report you can audit.

**Q: How do you stop the LLM from making causal claims about sensitive variables?**
A: Three layers: (1) prompt-level instruction; (2) post-hoc regex scanner that flags causal verbs near sensitive vars and prepends a disclaimer; (3) question-level scan for causal framing in the user's prompt itself.

**Q: What if the LLM writes broken code?**
A: It does, ~50% of the time on first attempt. Sandbox catches the crash; the stderr is fed back to the LLM as part of the next attempt's prompt. Up to 4 attempts per iteration. Effective codegen success rate (after retries) is ~87%.

**Q: Why didn't you just extend Build 4?**
A: Build 4 is a single-call REPL — `while True: input()`. The original proposal needed an autonomous multi-round loop with persistent state across iterations. Wrapping a `for` loop around the REPL gave us the same REPL run 5 times, with no learning between rounds. We salvaged ~40% of Build 4's code (the primitives, the RAG, the Langfuse plumbing) and rewrote the orchestration layer.

**Q: What's the cost per run?**
A: ~$0.05–$0.15 per session in API calls. The 28-run validation suite was $1.30 total.

**Q: What couldn't you get working?**
A: Mostly LLM nondeterminism — Kimi sometimes writes a valid analysis block followed by a crashing plot block. The retry catches it but the raw exit-code success rate is only 45% (87% with retries). Listed honestly in `docs/FINAL_REPORT.md` §7.

---

## Closing line

> "The Build 0 through 4 sequence built up the components — primitives, LLM
> dispatch, HITL, tool routing, RAG, tracing. Our project takes those
> components and assembles them into the autonomous loop the original
> proposal asked for. Build 4 stops at one query and one answer; we run the
> agent until it has a story to tell."
