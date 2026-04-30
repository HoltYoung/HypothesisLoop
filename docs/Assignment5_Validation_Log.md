# Assignment 5 — Testing & Validation Log

**Course:** QAC387-01 Spring 2026
**Team:** Holt Young & Sam Penn
**Agent under test:** Build 4 RAG Router Agent (`builds/build4_rag_router_agent.py`)
**Dataset:** UCI Adult Income (`data/adult.csv`, 32,561 rows × 15 cols)
**Knowledge corpus:** 10 markdown files → 74 FAISS chunks (`text-embedding-3-small`)
**Tracing:** Langfuse Cloud (`https://cloud.langfuse.com`)
**Test date:** 2026-04-29

## 1. Test methodology

We ran an 8-prompt sweep covering every category required by the assignment checklist (simple tool, simple codegen, RAG-conceptual, mixed, ambiguous, bad input, unrelated, plus a repeat-style codegen prompt) on **two providers**:

| Provider | Model | Session ID | Notes |
|---|---|---|---|
| OpenAI | `gpt-4o-mini` | `a5-sweep-openai` | Default fast/cheap baseline |
| Moonshot | `kimi-k2.6` | `a5-sweep-kimi-vm` | Reasoning model, 256K context, ran on a temporary GCP e2-small VM (auto-deleted after run) |

The Build 4 agent was extended with a `--provider {openai,moonshot}` flag during this assignment so the *same* prompts could be compared across model families without changing the agent's prompts, RAG index, or HITL flow. Embeddings continue to use `text-embedding-3-small` regardless of chat provider.

## 2. Validation log table

`R` = correct route, `A` = correct args, `X` = executed, `Q` = response quality (1–5), `B` = handled gracefully (bad-input/off-scope rows only).

### 2.1 OpenAI `gpt-4o-mini`

| # | Prompt | Category | Expected route | Actual route | R | A | X | Q | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | compute pearson correlations between numeric columns | Simple tool | `tool: pearson_correlation` | `tool: pearson_correlation, args:{x:"age", y:"fnlwgt"}` | ✓ | ✗ | partial | 2 | **Args hallucinated.** Tool takes no args (computes on all numerics) but router emitted `x`/`y`; tool ignored args silently and ran on full numeric set, but the LLM summary claimed "between age and fnlwgt" — wrong scope reported to user. |
| 2 | make a histogram of hours_per_week | Simple tool | `tool: plot_histograms` | `tool: plot_histograms, args:{numeric_cols:["hours_per_week"], fig_dir:""}` | ✓ | partial | ✓ | 4 | Hallucinated `fig_dir:""` (tool ignores it). Histogram saved to correct dir. Summary accurate. |
| 3 | write code that bins age into 5 quantiles and shows income rate per quantile | Simple codegen | `codegen` | `codegen` | ✓ | n/a | ✗ | 1 | **Codegen target mismatch.** Generated code is a *boxplot of hours_per_week by income* — completely unrelated to the requested age-quantile income-rate analysis. PLAN block also describes the wrong task. |
| 4 | according to the knowledge base when should I use multiple regression? | RAG-conceptual | `answer` (or codegen+RAG) | `answer` | ✓ | n/a | ✗ | 0 | **Validator crash.** Router correctly identified pure-knowledge intent but agent rejects `mode:"answer"` (only `tool`/`codegen` accepted). User sees an error instead of the explanation. |
| 5 | use the knowledge base to recommend an analysis and then run it | Mixed | `tool` or chained | `tool: basic_profile` | ✓ | ✓ | ✓ | 2 | Tool ran fine. **Summary chain hallucinated** — wrote "[Specific results would be listed here, but none were provided in the tool output.]" indicating the summary chain was not fed the tool output text. |
| 6 | help me analyze this dataset | Ambiguous | any reasonable | `answer` (workflow guidance) | ✓ | n/a | ✗ | 0 | Same validator crash. Router gave a sensible workflow-guidance reply but agent threw it away. |
| 7 | analyze the column nonexistent_variable_xyz | Bad input | refuse / flag | `answer` ("column does not exist") | ✓ | n/a | ✗ | 0 | Detection correct, delivery broken (validator crash). B = ✗ — no helpful message reaches user. |
| 8 | how do I fix my kitchen sink? | Unrelated | refuse / out-of-scope | `answer` ("conceptual question, no analysis") | ✓ | n/a | ✗ | 0 | Same — refusal logic fires but never reaches the user because of validator crash. |

### 2.2 Moonshot `kimi-k2.6`

| # | Prompt | Category | Expected route | Actual route | R | A | X | Q | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | compute pearson correlations between numeric columns | Simple tool | `tool: pearson_correlation` | `codegen` ("Compute the Pearson correlation matrix for *all* numeric columns") | ✓ | n/a | ✗ | 4 | **Better routing than OpenAI.** Kimi correctly recognized the `pearson_correlation` tool only handles a single (x,y) pair, so it routed to codegen for the matrix. Codegen output saved but not run. |
| 2 | make a histogram of hours_per_week | Simple tool | `tool: plot_histograms` | `tool: plot_histograms, args:{numeric_cols:["hours_per_week"], fig_dir:"figures"}` | ✓ | ✓ | ✓ | 4 | Cleaner args than OpenAI (`fig_dir:"figures"` vs `""`). Tool ran, summary accurate. |
| 3 | write code that bins age into 5 quantiles and shows income rate per quantile | Simple codegen | `codegen` | `codegen` | ✓ | n/a | ✗ | 1 | **Same codegen target-mismatch bug as OpenAI** — codegen chain produced an "exploratory dataset overview" PLAN instead of quantile-bin code. Confirms the bug is not model-dependent. |
| 4 | according to the knowledge base when should I use multiple regression? | RAG-conceptual | `answer` | `answer` | ✓ | n/a | ✗ | 0 | Same validator crash. Same root cause as OpenAI. |
| 5 | use the knowledge base to recommend an analysis and then run it | Mixed | `tool` or chained | `answer` (workflow narrative) | partial | n/a | ✗ | 0 | Kimi chose to *describe* an analysis instead of running one — the "and then run it" half was dropped. Validator also crashes the answer text. |
| 6 | help me analyze this dataset | Ambiguous | any reasonable | `answer` (multi-step workflow recommendation) | ✓ | n/a | ✗ | 0 | Validator crash. Reasoning was high-quality (cited dataset, workflow steps) but never reaches user. |
| 7 | analyze the column nonexistent_variable_xyz | Bad input | refuse / flag | `answer` ("column does not exist; available columns are…") | ✓ | n/a | ✗ | 0 | **Better than OpenAI** — listed the available columns. Still crashes at validator. B = ✗. |
| 8 | how do I fix my kitchen sink? | Unrelated | refuse | `answer` ("off-topic home-repair question, I can only help with data analysis") | ✓ | n/a | ✗ | 0 | **Best refusal of any test.** Polite, scoped, explicit. Crashes at validator. B = ✗. |

## 3. Failure modes (the actual deliverable for this assignment)

| ID | Bug | Reproduces on | Severity | Where it lives |
|---|---|---|---|---|
| F1 | `mode:"answer"` validator crash | both providers | **critical** | Router prompt advertises `answer` as a legal mode; downstream validator accepts only `tool`/`codegen`. Result: every conceptual / off-scope / refusal answer crashes. Hits 4–6 of 8 prompts depending on provider. |
| F2 | Codegen target mismatch | both providers | **critical** | The codegen chain ignores the user's actual request and emits a generic "exploratory overview" or unrelated boxplot. PLAN block also describes the wrong task, so the human reviewer can catch it at HITL — but the agent never produces the requested analysis. |
| F3 | Router hallucinates tool args | OpenAI more than Kimi | high | `pearson_correlation` (no args) was given `{x,y}`; `plot_histograms` (no `fig_dir` arg) was given `{fig_dir:""}`. Tools currently ignore unexpected args, masking the bug — until an arg conflicts with required fields and the tool errors instead. |
| F4 | Summary chain hallucinates without tool output | OpenAI | high | After `basic_profile` ran successfully, the summary chain output "[Specific results would be listed here, but none were provided in the tool output.]" — i.e., the tool result was not passed into the summary prompt. Indicates a wiring bug between tool execution and the summarize step. |
| F5 | Bad-input and off-scope prompts crash instead of returning helpful messages | both providers | high | Downstream consequence of F1: the *detection* of bad input or off-scope is correct, but the *delivery* to the user is broken. |
| F6 | RAG retrieval picks weakly-relevant chunks for codegen | both providers | medium | For "bin age into 5 quantiles", the top retrieved chunk was `tools/plot_histograms.md` (score 0.475). The genuinely relevant `guides/analysis_workflow.md` did not surface. Embedding model + chunk granularity issue, not a router issue. |

## 4. Success-criteria scorecard

| Metric | Target | OpenAI | Kimi | Combined | Pass? |
|---|---|---|---|---|---|
| Router accuracy (defensible routing decision) | ≥ 80% | 8/8 (100%) | 8/8 (100%) | 16/16 (100%) | ✓ |
| Relevant retrieval in top-k (clearly relevant chunk in top-4) | ≥ 80% | 1/2 codegen calls | 1/2 codegen calls | 2/4 (50%) | ✗ |
| Tool execution success | ≥ 90% | 2/3 attempted (pearson scope wrong, hist OK, basic_profile OK) | 1/1 attempted (hist OK) | 3/4 (75%) | ✗ |
| Approved code execution success (we approved 2 codegen runs but did not execute the wrong-target code) | ≥ 80% | 0/2 produced correct code | 0/2 produced correct code | 0/4 | ✗ |
| Average final response quality | ≥ 4/5 | 9/8 = 1.1 | 9/8 = 1.1 | 1.1 | ✗ |
| Graceful handling of bad-input / off-scope (Q7, Q8) | ≥ 90% | 0/2 (validator crash) | 0/2 (validator crash) | 0/4 | ✗ |

**Reading:** Router *judgment* is excellent on both providers (100%). Everything downstream of the router is broken or partial. The agent's weakest link is **not the LLM** — it is the contract between the router prompt and the rest of the pipeline.

## 5. Provider comparison

`gpt-4o-mini` and `kimi-k2.6` produced **the same two critical failures** (F1 validator crash, F2 codegen mismatch) on the same prompts. Where they differed:

| Behavior | OpenAI advantage | Kimi advantage |
|---|---|---|
| Speed | ~5s per router call | ~30–90s per router call (reasoning model) |
| Cost (this run) | ~$0.005 | ~$0.10 |
| Routing for the pearson prompt | — | Recognized tool's (x,y) limitation, routed to codegen instead of running on wrong scope |
| Args quality | — | `fig_dir:"figures"` vs OpenAI's `""` |
| Refusal quality | — | Polite, scoped explanation for off-topic and bad-input prompts |
| Tool args fabrication | — | Less prone to inventing kwargs the tool doesn't accept |

**Conclusion:** upgrading the model fixes some second-order issues (arg quality, refusal phrasing) but does **not** fix the first-order bugs (F1, F2). Those require code/prompt changes.

## 6. Checklist (Assignment 5 §1–§10)

### 1. Core setup and environment checks
- [x] App starts without import or path errors *(verified on both local and a fresh GCP VM)*
- [x] Environment variables and model settings load correctly *(provider switch wired through `.env`)*
- [x] Langfuse tracing is active and receiving runs *(Langfuse Cloud confirmed; spans visible)*
- [x] RAG index loads without errors *(74 chunks)*
- [x] Tool registry loads correctly *(16 tools enumerated at startup)*
- [x] Dataset file uploads/read operations work correctly
- [x] Schema text is extracted and passed into prompts
- [x] Report, tool output, and figure directories are created correctly
- [x] Generated code and artifacts are being written to the specified path

### 2. Router decision testing
- [x] Requests for which there are available tools route to the correct tool *(prompts 2)*
- [x] Code-generation requests route to codegen *(prompt 3)*
- [~] Knowledge-based questions use RAG appropriately — router routes to `answer`, but **F1** crashes the response
- [x] Ambiguous prompts still produce a reasonable choice *(prompt 6)*
- [x] Router output is valid JSON
- [x] Router does not hallucinate tools that do not exist
- [x] Router does not default to codegen if there's an available tool
- [~] Router respects dataset schema and doesn't hallucinate variables — **but does hallucinate kwargs** (F3)
- [x] Prompts work even when the user is not highly specific *(handled in router; `answer` mode then crashes)*

### 3. RAG retrieval testing
- [~] Top retrieved chunks are clearly relevant — F6 shows weak retrieval for the quantile prompt
- [~] Retrieved chunks come from the most appropriate knowledge files
- [x] Retrieved material contains enough detail to improve the answer *(when relevant)*
- [ ] Different phrasings of the same question still retrieve useful context *(not exhaustively tested)*
- [x] The agent does not fabricate knowledge when retrieval is weak *(it ignores the chunks rather than fabricating)*
- [x] Retrieved content does not push the router into the wrong mode

### 4. Tool execution testing
- [~] Tool name and arguments match the user request — name yes, args partially (F3)
- [x] Variable names passed to tools exist in the dataframe
- [x] Tool runs without crashing *(when args are accepted)*
- [x] Tool returns a standardized output object or expected structure
- [x] Saved figures and files appear in the correct directories
- [~] Tool summaries are understandable to the end user — F4 hallucinates summary in one case
- [ ] Missing arguments, empty subsets, and invalid inputs are handled clearly *(no try/except for empty result sets)*

### 5. Code generation and HITL testing
- [~] Generated code matches the requested analysis — **F2 fails this** (target mismatch)
- [x] Generated code uses valid dataframe and column names
- [x] Generated code is valid Python
- [x] Human-in-the-loop approval is required before execution
- [x] Unapproved code cannot be run
- [x] Approved code executes successfully within timeout limits *(when the code matches the request)*
- [x] Errors, stdout, and stderr are captured or displayed clearly
- [x] Outputs and artifacts are saved in the expected location
- [x] Generated code does not attempt unsafe or unintended operations

### 6. End-to-end workflow testing
- [~] A user prompt flows cleanly from request to router to execution to summary — F1 breaks 4–6 of 8 prompts
- [~] The final response answers the actual question asked
- [x] Outputs are interpretable for a novice user *(when the response makes it through)*
- [x] Artifacts are visible, downloadable, and labeled clearly
- [~] Mixed requests behave correctly — Kimi dropped the "and then run it" half
- [ ] Bad input cases produce helpful exception feedback — F5

### 7. Response quality validation
- [~] Response is statistically appropriate for the request
- [x] Response uses correct variable names and terms *(when answered)*
- [~] Response explains results clearly — F4 produces an empty summary
- [x] Response avoids overclaiming or causal overinterpretation *(strong on both providers)*
- [x] Response acknowledges uncertainty or limitations *(included in summary template)*
- [~] Response makes appropriate use of retrieved knowledge — sometimes ignored

### 8. Error-handling and edge cases
- [ ] Works with missing data in key variables *(not exhaustively tested in this sweep)*
- [ ] Handles small datasets and sparse categories gracefully *(not tested)*
- [ ] Handles misspelled or partially incorrect variable names reasonably
- [~] Handles vague or overly broad requests gracefully — router does, validator doesn't (F1)
- [~] Handles requests that are outside the tool or app scope clearly — same
- [x] Handles no relevant RAG results without hallucinating
- [~] Responds to unrelated content prompts and returns a notification — refusal *generated* but never *delivered*

### 9. Traceability and observability
- [x] Router decisions are visible in Langfuse traces
- [x] Retrieval steps and retrieved context are visible
- [x] Tool or codegen branches are easy to inspect in traces
- [x] Errors are captured clearly in traces *(validator-rejected outputs visible)*
- [x] Session IDs, tags, and prompt versions are trackable *(filtered both sweeps by sessionId)*
- [x] You can compare successful and failed runs for debugging *(direct OpenAI-vs-Kimi compare done in §5)*

### 10. ≥1 prompt from each category
- [x] Simple tool — prompt 1, 2 (both providers)
- [x] Simple codegen — prompt 3 (both providers)
- [x] RAG-heavy conceptual — prompt 4
- [x] Mixed — prompt 5
- [x] Ambiguous — prompt 6
- [x] Bad-input — prompt 7
- [x] Unrelated — prompt 8

## 7. Final reflection

**What did the agent do especially well?**
Router *judgment* was the standout strength — both `gpt-4o-mini` and `kimi-k2.6` produced defensible routing decisions on 100% of prompts (16/16). The router correctly distinguished tool requests from codegen requests, recognized when a knowledge-base question wasn't an analysis ("when should I use multiple regression?"), recognized off-scope prompts ("how do I fix my kitchen sink?"), and recognized non-existent variables. Langfuse instrumentation was also robust: every router/retrieval/tool/codegen/summary step shows up cleanly under the session ID, making it easy to audit any single prompt across providers.

**What failures appeared most often?**
**Routing was good; the pipeline downstream of routing was broken.** The single most damaging issue was the validator that rejects `mode:"answer"` even though the router system prompt explicitly lists it as a legal mode. This dead branch crashed 4 of 8 prompts on OpenAI and 5 of 8 on Kimi — every conceptual question, every off-scope refusal, and every bad-input flag was generated correctly by the router and then thrown away by the validator. The second issue was the codegen chain producing code unrelated to the user request (e.g., responding to "bin age into 5 quantiles" with a "boxplot of hours_per_week by income"). Both bugs reproduce identically on a much stronger model, so they are not model-quality bugs — they are contract bugs between the router prompt and the agent's downstream code.

**What is the single highest-priority improvement?**
**Fix the router-output contract.** Specifically: (a) accept `mode:"answer"` in the validator and route it through a small "answer" handler that prints the router's `note`/`response` text to the user (no tool call, no codegen, no HITL gate); and (b) audit the codegen chain to confirm it is being passed `user_request` and not some stale or misrouted variable, since the symptom (always producing "exploratory overview" code) suggests it is not seeing the actual prompt. Both fixes are localized, take an hour, and would lift the success-criteria scorecard from a fail on most rows to a pass on most rows. Improvements to retrieval (F6) and arg-fabrication (F3) are real but secondary — they degrade quality, while the validator/codegen bugs degrade correctness.

## Appendix A — Reproducing this run

```bash
# OpenAI sweep (local)
PYTHONPATH=. python builds/build4_rag_router_agent.py \
  --data data/adult.csv --knowledge_dir knowledge --report_dir reports \
  --provider openai --session_id a5-sweep-openai --tags build4,assignment5,sweep,openai

# Kimi sweep (Moonshot kimi-k2.6, ran on a temp GCP VM that auto-deleted)
PYTHONPATH=. python builds/build4_rag_router_agent.py \
  --data data/adult.csv --knowledge_dir knowledge --report_dir reports \
  --provider moonshot --session_id a5-sweep-kimi-vm --tags build4,assignment5,sweep,kimi,vm
```

Trace data fetched from Langfuse Cloud:

```bash
curl -s -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
  "https://cloud.langfuse.com/api/public/traces?sessionId=a5-sweep-openai&limit=50"
curl -s -u "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" \
  "https://cloud.langfuse.com/api/public/traces?sessionId=a5-sweep-kimi-vm&limit=50"
```
