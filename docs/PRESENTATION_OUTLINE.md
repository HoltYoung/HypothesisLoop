# HypothesisLoop — Presentation Outline

**Holt Young & Sam Penn · QAC387 Spring 2026**

> Rough outline only. Sam writes the actual words on the slides. Mirrors the Final Report's 9 sections + a live demo segment. Final Report Instructions explicitly says the report "should be a summary of your presentation following the outline below, excluding the Streamlit demo."

Target: ~10–12 slides, ~10 minutes + Q&A.

---

## Slide 1 — Title

- HypothesisLoop: An LLM-Powered Iterative Hypothesis-Testing Agent
- Holt Young & Sam Penn · QAC387 Spring 2026

## Slide 2 — The Problem

- ChatGPT Code Interpreter: one-shot. You ask, it answers, you figure out the next question.
- Real data analysis: hypothesize → test → learn → repeat.
- Most people lack stats/coding background to do that loop.
- *(Visual: split screen — ChatGPT screenshot vs. our Loop diagram)*

## Slide 3 — What We Built

- An LLM agent that wraps the scientific method around a CSV.
- Default 5 iterations. Hypothesizes → writes Python → runs in sandbox → evaluates → learns.
- Two modes: **Explore** (free-form research question → narrative report) and **Predict** (pick a target → engineered features + AutoGluon ensemble).
- *(Visual: the agent diagram from `docs/SPEC.md` §3)*

## Slide 4 — Tech Stack

- LLMs: Moonshot Kimi K2.6 (default, ~10× cheaper than Opus), OpenAI 4o-mini fallback
- Embeddings: OpenAI `text-embedding-3-small` for novelty detection
- Vector store: FAISS over codebook + statistical-test guide
- Tracing: Langfuse (every LLM call, every iteration)
- ML: pandas, scipy, sklearn, AutoGluon (final ensemble in Predict mode)
- UI: Streamlit + custom CSS (mission-control aesthetic)
- Sandbox: subprocess + AST denylist (no Docker; lightweight)

## Slide 5 — How the Loop Actually Works

One iteration broken down:

1. **Hypothesize** — LLM proposes a falsifiable claim (commits to a metric delta in Predict mode)
2. **Experiment** — LLM writes Python; runs in sandbox; retries on error up to 3×
3. **Evaluate** — LLM interprets output (in Predict mode, deterministic accept/reject overrides LLM)
4. **Learn** — DAG-tracked trace, novelty gate, soft-decay, bias scanner

*(Visual: per-iteration flowchart with each step labeled)*

## Slide 6 — Key Pivots (Iterations & Pivots story)

- We **abandoned the cookie-cutter Build 4 architecture** mid-project and rebuilt around the original proposal.
- Why: Build 4 was a single-call REPL; the proposal needed an autonomous multi-round loop. ~40% of code survived as primitives; the orchestration layer was rewritten.
- Other pivots: Kimi over Opus (cost), sklearn → AutoGluon (performance), layered novelty (prompt + embedding + soft-decay), session-budget `--max-iters` semantics, mode-aware loop for Predict.
- *(Visual: timeline of phases 0→10B with key decisions marked)*
- **Source:** every pivot is documented in `docs/DESIGN_DECISIONS.md`.

## Slide 7 — Bias Scanner & Safety

- Pattern: any text mentioning a sensitive variable (race, sex, native-country, marital status) AND a causal verb ("causes," "leads to," "due to") triggers a flag.
- Flagged paragraphs get a `⚠️ correlational only` disclaimer prepended in the report.
- Sandbox AST denylist prevents `os`, `subprocess`, `eval`, file writes outside session dir.
- *(Visual: a flagged-paragraph screenshot from a red-team run)*

## Slide 8 — Validation

- 28-run validation suite across 8 prompt categories + Predict mode + cross-dataset.
- Success metrics (numbers fill in from `tests/validation_log.md` after suite completes):
  - Loop completion: X% (target ≥95%)
  - Codegen success: X% (target ≥80% — Kimi nondeterminism flagged as known limitation)
  - Novelty gate: X% distinct hypotheses (target ≥90%)
  - Bias scanner caught planted causal claims: X%
- *(Visual: validation log table excerpt)*

## Slide 9 — Limitations

- Windows can't enforce RAM cap (POSIX-only `setrlimit`)
- Kimi sometimes writes valid analysis followed by a crashing plot block (~50% clean exit rate; metrics still emitted)
- Predict mode's conservative threshold rejects most LLM-proposed features (proves rigor; demo-able with permissive threshold)
- Stateful FE transforms degrade quality on test split (re-execution uses test stats)

## Slide 10 — Next Steps

- sklearn fit/transform pipeline for stateful FE (the right long-term fix)
- Multi-target prediction
- Persistent embedding cache
- Cross-dataset benchmarks (House Prices, WHO Life Expectancy)
- Schedule-driven scheduler (replace LinearScheduler with RD-Agent-style probabilistic)

## Slide 11 — Live Demo

- Open Streamlit, upload `data/adult.csv`, switch to Predict mode, target `income`, hit START.
- Show iteration cards streaming in.
- Show CONTINUE +5 ITERS extending the run.
- Open the generated `report.md` and the AutoGluon leaderboard.

## Slide 12 — Q&A / Takeaways

- We pivoted away from the assignment scaffolding to build what we actually proposed.
- The agent is auditable: every hypothesis, every line of code, every decision lives in Langfuse + a Markdown report.
- ~$0.05/run. 100% local sandbox. Two modes from one loop.

---

## Source map (where each slide's content comes from)

| Slide | Primary source | Notes |
|---|---|---|
| 1 | `ORIGINAL PROPOSAL.pdf` (title page) | verbatim |
| 2 | `ORIGINAL PROPOSAL.pdf` §3 + team's own framing | the "ChatGPT is one-shot" line is your articulation |
| 3 | `docs/SPEC.md` §1 + §3 | summarized |
| 4 | `requirements.txt` + `docs/SPEC.md` §3 + `docs/DESIGN_DECISIONS.md` (model choice) | factual |
| 5 | `docs/SPEC.md` §6.1 (loop sketch) + Phases 3-5 implementations | architecture |
| 6 | `docs/DESIGN_DECISIONS.md` (every entry is a pivot) | direct quotes available |
| 7 | `hypothesisloop/safety/bias_scanner.py` + `docs/SPEC.md` §6.7 | factual |
| 8 | `tests/validation_log.md` (auto-generated by suite) | LIVE NUMBERS — fill in after suite finishes |
| 9 | Open questions across phase reports + `docs/SPEC.md` §12 | known issues |
| 10 | `docs/SPEC.md` §1 future work + `ORIGINAL PROPOSAL.pdf` §11 | wishlist |
| 11 | live | actual app |
| 12 | `docs/SPEC.md` §14 (the pitch) | rephrased |
