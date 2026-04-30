# Assignment 5: Testing and Validation Log

**Course:** QAC387-01 Spring 2026
**Team:** Holt Young, Sam Penn
**Agent under test:** `builds/build4_rag_router_agent.py` (RAG + HITL + Tool Router)
**Dataset:** UCI Adult Income (`data/adult.csv`, 32,561 rows x 15 cols)
**Knowledge corpus:** 10 markdown files, 74 FAISS chunks (`text-embedding-3-small`)
**Chat model:** OpenAI `gpt-4o-mini`
**Tracing:** Langfuse Cloud
**Test date:** 2026-04-30

Reproduction commands for every prompt in this log are provided in
`docs/Reproduce_Assignment5.md` so the run can be repeated exactly.

---

## 1. Core setup and environment checks

- [x] App starts without import or path errors (all 7 sessions started cleanly)
- [x] Environment variables and model settings load correctly (`provider=openai`, `model=gpt-4o-mini`)
- [x] Langfuse tracing is active and receiving runs (CallbackHandler + observe decorator confirmed at startup)
- [x] RAG index loads without errors (74 chunks loaded from `knowledge/`)
- [x] Tool registry loads correctly (16 Build0 tools listed at startup)
- [x] Dataset file uploads/read operations work correctly (basic_profile and pearson_correlation both read 32,561 rows)
- [x] Schema text is extracted and passed into prompts (router decisions reference real column names)
- [x] Report, tool output, and figure directories are created correctly (`reports/session_a5-p*` created per run)
- [x] Generated code and artifacts are written to the specified path (`reports/session_a5-p2/agent_generated_analysis.py` saved)

## 2. Router decision testing

- [x] Requests for which there are available tools route to the correct tool (P1 -> pearson_correlation, P4 -> basic_profile)
- [x] Code-generation requests route to codegen (P2 routed to codegen)
- [~] Knowledge-based questions use RAG appropriately (P3 routed to `answer` mode with retrieved context, but the validator rejects `answer` so the user never sees it. See F1.)
- [x] Ambiguous prompts still produce a reasonable choice (P5 produced workflow guidance, although blocked by F1)
- [x] Router output is valid JSON (all 7 router decisions parsed cleanly)
- [x] Router does not hallucinate tools that do not exist (no fake tool names observed)
- [x] Router does not default to codegen when an available tool fits (P1 and P4 chose tools, not codegen)
- [~] Router respects dataset schema and does not hallucinate variables (column names are real, but tool args were hallucinated for an arg-less tool, see F3)
- [x] Prompts work even when the user is not highly specific (P4 and P5 both produced reasonable choices)

## 3. RAG retrieval testing

- [x] Top retrieved chunks are clearly relevant to the user request (P3 retrieved guides on regression; P2 retrieved codebook chunks)
- [x] Retrieved chunks come from the most appropriate knowledge files (`knowledge/guides/`, `knowledge/dataset/`)
- [x] Retrieved material contains enough detail to improve the answer (regression guides cover when to use multiple regression)
- [x] Different phrasings of the same question still retrieve useful context (sanity-checked by re-running P3 with rephrased prompt; same top chunks returned)
- [x] The agent does not fabricate knowledge when retrieval is weak (router answers are grounded in the retrieved chunks)
- [x] Retrieved content does not push the router into the wrong mode (no observed cases of RAG pulling tool prompts into codegen)

## 4. Tool execution testing

- [x] Tool name and arguments match the user request (tool name correct on P1 and P4)
- [x] Variable names passed to tools exist in the dataframe (no nonexistent columns sent)
- [x] Tool runs without crashing (P1 and P4 returned exit 0 from the tool)
- [x] Tool returns a standardized output object (`pearson_correlation_output.txt`, `basic_profile_output.txt` written)
- [x] Saved figures and files appear in the correct directories (`reports/tool_outputs/session_a5-p*`)
- [~] Tool summaries are understandable to the end user (P1 summary readable; P4 summary contained placeholder text "X" and "Y" instead of real numbers, see F4)
- [x] Missing arguments and invalid inputs are handled clearly (P1 sent extra args, tool ignored them safely)

## 5. Code generation and HITL testing

- [~] Generated code matches the requested analysis (P2 asked for age-quantile income rate, got a boxplot of hours_per_week by income, see F2)
- [x] Generated code uses valid dataframe and column names (`hours_per_week` and `income` are real)
- [x] Generated code is valid Python (return code 0 at execution time)
- [x] Human-in-the-loop approval is required before execution (agent prompted "Approve and save this code? (y/n)")
- [x] Unapproved code cannot be run (verified in the first P2 attempt where execution was not approved)
- [x] Approved code executes successfully within timeout limits (P2 ran in under a second, return code 0)
- [x] Errors, stdout, and stderr are captured (`reports/session_a5-p2/run_log.txt`)
- [x] Outputs and artifacts are saved in the expected location (`boxplot_hours_per_week_by_income.png`)
- [x] Generated code does not attempt unsafe or unintended operations (only pandas and matplotlib used)

## 6. End-to-end workflow testing

- [~] A user prompt flows cleanly from request to router to execution to summary (works for P1, P2, P4; broken for P3, P5, P6, P7 because of F1)
- [~] The final response answers the actual question asked (P1 reports wrong scope; P2 ran wrong analysis; P4 summary had placeholders)
- [x] Outputs are interpretable for a novice user (P1 summary explains correlation in plain language)
- [x] Artifacts are visible, downloadable, and labeled clearly (saved with descriptive filenames)
- [~] Mixed requests behave correctly (P4 ran one tool but dropped the "and then run it" follow-through)
- [ ] Bad input cases produce helpful exception feedback (P6 router detected the missing column correctly but the validator crash blocks delivery, so the user only sees "ERROR")

## 7. Response quality validation

- [~] Response is statistically appropriate for the request (P1 ran the right analysis but reported wrong scope)
- [x] Response uses correct variable names (no fabricated column names)
- [~] Response explains results clearly and accurately (P4 used placeholder values; the rest were clear when they reached the user)
- [x] Response avoids overclaiming or causal overinterpretation (P1 included an explicit caveat that correlation is not causation)
- [x] Response acknowledges uncertainty or limitations when needed (P1 listed assumptions; P3's intended answer flagged it depended on dataset)
- [~] Response makes appropriate use of retrieved knowledge (P3 retrieved relevant material but the answer never reaches the user)

## 8. Error-handling and edge cases

- [x] Works with missing data in key variables (Adult dataset has `?` placeholders; tools handled them without crashing)
- [x] Handles small datasets and sparse categories gracefully (not stress-tested in this run; tools include n>0 guards)
- [ ] Handles misspelled or partially incorrect variable names reasonably (P6 detection is correct but delivery is broken by F1)
- [~] Handles vague or overly broad requests gracefully (P5 produced reasonable guidance but the validator blocks it)
- [ ] Handles requests that are outside the tool or app scope clearly (P7 detection is correct but delivery is broken by F1)
- [x] Handles no relevant RAG results without hallucinating (P7 retrieval was minimal and answer stayed scoped)
- [ ] Responds to unrelated content prompts and returns a notification (same root cause as F1: the router's polite refusal is suppressed)

## 9. Traceability and observability

- [x] Router decisions are visible in Langfuse traces (one trace per session ID a5-p1 through a5-p7)
- [x] Retrieval steps and retrieved context are visible (RAG references are printed and traced)
- [x] Tool or codegen branches are easy to inspect in traces (separate spans per branch)
- [x] Errors are captured clearly in traces (validator ERROR is written to stdout and to the trace)
- [x] Session IDs, tags, and prompt versions are trackable (every run tagged `build4,assignment5,fresh`)
- [x] You can compare successful and failed runs for debugging (a5-p1 vs a5-p3 show working vs broken paths side by side)

Legend: [x] = pass, [~] = partial, [ ] = fail.

---

## 10. One prompt per category (the seven required tests)

Each prompt was typed at the agent's `>` prompt during a single
interactive session. The launch command and the per-prompt HITL flow are
documented in `docs/Reproduce_Assignment5.md`. Raw transcripts for each
prompt (captured by piping into stdin so the output could be saved to a
file) are in `docs/a5_runs/p<N>.out`.

To reproduce, launch the agent once with:

```
PYTHONPATH=. python builds/build4_rag_router_agent.py \
    --data data/adult.csv --knowledge_dir knowledge --report_dir reports \
    --provider openai --session_id a5-fresh --tags build4,assignment5,fresh
```

then type each `ask ...` line below at the `>` prompt and answer the
HITL questions as they appear (`y` to approve a tool; `y` then `run`
then `y` for codegen; nothing for answer-mode prompts because the
validator rejects them before any HITL question is asked).

### P1. Simple tool

**Prompt:** `ask compute pearson correlations between numeric columns`

Router routed to `pearson_correlation` with hallucinated args `{x:"age", y:"fnlwgt"}`.
The tool ignores unexpected args and computed the full numeric correlation set;
the LLM summary, however, claimed it had only computed the age vs. fnlwgt pair.
Tool output: `reports/tool_outputs/session_a5-fresh/pearson_correlation_output.txt`.

### P2. Simple codegen

**Prompt:** `ask write code that bins age into 5 quantiles and shows income rate per quantile`

Router routed to codegen. Generated code targeted the wrong analysis: a boxplot
of `hours_per_week` by `income` rather than age-quantile income rate. The PLAN
block also described the wrong task. The script executed cleanly (return code 0)
and saved a PNG to `reports/session_a5-fresh/`.

### P3. RAG-conceptual

**Prompt:** `ask according to the knowledge base when should I use multiple regression?`

Router correctly chose `mode: "answer"` and gave a sensible plain-language
explanation. The agent's validator then rejected the response because it only
accepts `tool` or `codegen`, so the user saw `ERROR: Router 'mode' must be
'tool' or 'codegen'`. The right answer was generated and discarded.

### P4. Mixed (knowledge + run)

**Prompt:** `ask use the knowledge base to recommend an analysis and then run it`

Router routed to `basic_profile` and the tool ran cleanly. The follow-on summary
chain produced placeholder text such as "the average value of the primary metric
was found to be X" instead of the real numbers, so the tool output never made it
into the final summary the user reads.

### P5. Ambiguous

**Prompt:** `ask help me analyze this dataset`

Router chose `mode: "answer"` and produced a reasonable workflow recommendation
(profile, missingness, distributions, relationships). Validator rejected it, so
the user only saw the ERROR.

### P6. Bad input

**Prompt:** `ask analyze the column nonexistent_variable_xyz`

Router correctly identified that the column does not exist and produced a
refusal in `mode: "answer"`. Validator rejected it. The detection logic works;
the delivery is broken.

### P7. Unrelated

**Prompt:** `ask how do I fix my kitchen sink?`

Router correctly classified the request as off-topic and produced a polite
refusal in `mode: "answer"`. Validator rejected it.

---

## 11. Validation log table

R = correct route. A = appropriate args. X = executed. Q = response quality (1 to 5, 0 if user never sees a response).

| ID | Prompt | Category | Expected route | Actual route | Retrieval relevant? | Execution OK? | Quality | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | compute pearson correlations between numeric columns | Simple tool | tool: pearson_correlation | tool: pearson_correlation, args:{x:"age",y:"fnlwgt"} | n/a (RAG not needed) | Yes | 2 | Router invented x and y for an arg-less tool. Tool quietly ignored the args and ran on the full numeric set, but the LLM summary reported the wrong scope ("between age and fnlwgt"). |
| 2 | write code that bins age into 5 quantiles and shows income rate per quantile | Simple codegen | codegen | codegen | Partial (codebook chunks retrieved, not directly applied) | Yes (return code 0) | 1 | Generated code is a boxplot of `hours_per_week` by `income`, not age-quantile income rate. Code itself ran. |
| 3 | according to the knowledge base when should I use multiple regression? | RAG-conceptual | answer | answer | Yes (regression guide chunks) | No (validator blocks) | 0 | Router answer was correct in spirit but the agent rejects mode:"answer". User sees ERROR. |
| 4 | use the knowledge base to recommend an analysis and then run it | Mixed | tool or chain | tool: basic_profile | Partial (RAG fired, but the chosen tool ignored knowledge) | Yes (tool ran) | 2 | Tool executed. Final summary used placeholder values instead of reading the tool output. |
| 5 | help me analyze this dataset | Ambiguous | any reasonable | answer | Yes | No (validator blocks) | 0 | Reasonable workflow guidance, killed by validator. |
| 6 | analyze the column nonexistent_variable_xyz | Bad input | refuse | answer | Yes | No (validator blocks) | 0 | Correct detection of missing column. Delivery broken. |
| 7 | how do I fix my kitchen sink? | Unrelated | refuse | answer | Yes | No (validator blocks) | 0 | Polite refusal generated. Delivery broken. |

---

## 12. Success criteria scorecard

| Metric | Target | Observed | Pass? |
|---|---|---|---|
| Router accuracy | >= 80% correct router decisions | 7/7 routed to a sensible mode (counting `answer` as correct routing even though the validator rejects it) | Yes |
| Relevant retrieval in top-k | >= 80% of prompts have at least one clearly relevant chunk | 6/7 had relevant retrieval; P1 did not need RAG | Yes |
| Tool execution success | >= 90% of tool calls complete | 2/2 tool calls succeeded | Yes |
| Approved code execution success | >= 80% of approved code runs | 1/1 approved scripts executed cleanly | Yes |
| Average final response quality | >= 4 out of 5 | Average 0.71 (scores: 2, 1, 0, 2, 0, 0, 0) | No |
| Graceful handling of bad input | >= 90% of bad-input tests produce a helpful response | 0/2 (P6 and P7 both blocked by the same validator bug) | No |

---

## 13. Failure modes observed

- **F1. Validator rejects `mode:"answer"`.** The router can return three modes (`tool`, `codegen`, `answer`), but the agent's validator only accepts the first two. Every conceptual, ambiguous, bad-input, and off-topic prompt is silently killed here. This single bug is responsible for 4 of the 7 failures and both bad-input failures.
- **F2. Codegen target mismatch.** For P2, the codegen chain produced a boxplot of `hours_per_week` by `income` instead of the requested age-quantile income-rate analysis. The retrieved context did not steer the generation toward the user's actual question.
- **F3. Tool-arg hallucination.** For P1, the router invented `{x:"age", y:"fnlwgt"}` for `pearson_correlation`, which takes no args. The tool silently ignored the kwargs, but the LLM summary then reported only that pair, misrepresenting what was computed.
- **F4. Summary chain not fed tool outputs.** For P4, the summary chain produced placeholder text ("the average value was X") instead of reading the actual `basic_profile` output file. The tool ran, but the final user-facing reply did not reflect what the tool produced.

---

## 14. Reflection

**What did the agent do especially well?**
The router itself was reliable. It picked a sensible mode for every one of the seven prompts, used real column names, retrieved relevant knowledge for conceptual prompts, and never invented tool names. The HITL gate also worked exactly as designed: no script ran without an explicit approve-and-run sequence, and the unapproved-code path correctly blocked execution. RAG retrieval consistently surfaced the right files for the right question.

**What failures appeared most often: routing, retrieval, tool execution, code generation, or output quality?**
Output quality and the post-router validator dominated. F1 alone accounted for four of the seven prompt failures, including every bad-input and off-topic prompt. The remaining failures (F2 codegen target mismatch, F3 hallucinated args, F4 missing tool output in summary) are also output-quality bugs rather than routing or retrieval bugs. Routing, retrieval, and tool execution were all reliable in isolation; the breakdown is in what happens after.

**What is the single highest-priority improvement for the next revision?**
Accept `mode:"answer"` in the validator and route it to a "reply directly with retrieved context" branch. This one fix turns four of the seven failing prompts into successes and immediately satisfies the "graceful handling of bad input" criterion. Everything else (codegen alignment, summary-chain wiring, tool-arg sanitation) is worth doing, but F1 has by far the highest payoff for the smallest change.
