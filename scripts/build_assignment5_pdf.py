"""Build the Assignment 5 validation log PDF.

All content is embedded in this file. No external markdown source.

Usage:
    PYTHONPATH=. python scripts/build_assignment5_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = PROJECT_ROOT / "docs" / "Assignment5_Validation_Log.pdf"


CHECK = '<font color="#138a36"><b>[x]</b></font>'
PARTIAL = '<font color="#b8860b"><b>[~]</b></font>'
FAIL = '<font color="#b22222"><b>[&nbsp;]</b></font>'


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=20, leading=24, alignment=1, spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=11, leading=14, alignment=1,
            textColor=colors.HexColor("#444444"), spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=14, leading=17, spaceBefore=14, spaceAfter=6,
            textColor=colors.black,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=11, leading=14, spaceBefore=8, spaceAfter=3,
            textColor=colors.black,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontName="Helvetica",
            fontSize=10, leading=13, spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["BodyText"], fontName="Helvetica",
            fontSize=10, leading=13, leftIndent=14, bulletIndent=2,
            spaceAfter=2,
        ),
        "code": ParagraphStyle(
            "code", parent=base["Code"], fontName="Courier",
            fontSize=9, leading=12, leftIndent=8, rightIndent=8,
            backColor=colors.HexColor("#f5f5f5"),
            borderPadding=4, spaceBefore=4, spaceAfter=8,
        ),
        "small": ParagraphStyle(
            "small", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.5, leading=11,
        ),
    }


def p(text, st):
    return Paragraph(text, st)


def bullet(text, st):
    return Paragraph(text, st["bullet"], bulletText="-")


def make_table(rows, st, col_widths=None, header=True):
    body_st = ParagraphStyle("td", parent=st["small"], fontSize=8.5, leading=11)
    head_st = ParagraphStyle("th", parent=body_st, fontName="Helvetica-Bold")
    table_data = []
    for ri, row in enumerate(rows):
        s = head_st if (header and ri == 0) else body_st
        table_data.append([Paragraph(str(c), s) for c in row])
    n_cols = len(rows[0])
    if col_widths is None:
        avail = LETTER[0] - 1.2 * inch
        col_widths = [avail / n_cols] * n_cols
    t = Table(table_data, colWidths=col_widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6") if header else colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


CHECKLIST = [
    ("1. Core setup and environment checks", [
        (CHECK, "App starts without import or path errors"),
        (CHECK, "Environment variables and model settings load correctly (provider=openai, model=gpt-4o-mini)"),
        (CHECK, "Langfuse tracing is active and receiving runs"),
        (CHECK, "RAG index loads without errors (74 chunks loaded from knowledge/)"),
        (CHECK, "Tool registry loads correctly (16 tools available at startup)"),
        (CHECK, "Dataset file read operations work correctly (32,561 rows from data/adult.csv)"),
        (CHECK, "Schema text is extracted and passed into prompts (router decisions reference real columns)"),
        (CHECK, "Report, tool output, and figure directories are created correctly"),
        (CHECK, "Generated code and artifacts are written to the specified path"),
    ]),
    ("2. Router decision testing", [
        (CHECK, "Requests with available tools route to the correct tool (P1, P4)"),
        (CHECK, "Code-generation requests route to codegen (P2)"),
        (PARTIAL, "Knowledge-based questions use RAG appropriately (router routes to answer with retrieval, but the validator then rejects answer mode, see F1)"),
        (CHECK, "Ambiguous prompts still produce a reasonable choice"),
        (CHECK, "Router output is valid JSON for all 7 prompts"),
        (CHECK, "Router does not hallucinate tools that do not exist"),
        (CHECK, "Router does not default to codegen when an available tool fits"),
        (PARTIAL, "Router respects dataset schema (column names are real, but tool kwargs were hallucinated, see F3)"),
        (CHECK, "Prompts work even when the user is not highly specific"),
    ]),
    ("3. RAG retrieval testing", [
        (CHECK, "Top retrieved chunks are clearly relevant to the user request"),
        (CHECK, "Retrieved chunks come from the most appropriate knowledge files"),
        (CHECK, "Retrieved material contains enough detail to improve the answer"),
        (CHECK, "Different phrasings of the same question still retrieve useful context"),
        (CHECK, "The agent does not fabricate knowledge when retrieval is weak"),
        (CHECK, "Retrieved content does not push the router into the wrong mode"),
    ]),
    ("4. Tool execution testing", [
        (CHECK, "Tool name matches the user request (P1, P4)"),
        (CHECK, "Variable names passed to tools exist in the dataframe"),
        (CHECK, "Tool runs without crashing (2/2 tool calls succeeded)"),
        (CHECK, "Tool returns a standardized output object (text file written per tool)"),
        (CHECK, "Saved figures and files appear in the correct directories"),
        (PARTIAL, "Tool summaries are understandable to the end user (P4 summary contained placeholder values, see F4)"),
        (CHECK, "Missing or unexpected arguments are handled clearly (P1 sent extra kwargs, tool ignored them)"),
    ]),
    ("5. Code generation and HITL testing", [
        (PARTIAL, "Generated code matches the requested analysis (P2 produced wrong analysis, see F2)"),
        (CHECK, "Generated code uses valid dataframe and column names"),
        (CHECK, "Generated code is valid Python"),
        (CHECK, "Human-in-the-loop approval is required before execution"),
        (CHECK, "Unapproved code cannot be run"),
        (CHECK, "Approved code executes successfully (P2 ran in under a second, return code 0)"),
        (CHECK, "Errors, stdout, and stderr are captured to a run log"),
        (CHECK, "Outputs and artifacts are saved in the expected location"),
        (CHECK, "Generated code does not attempt unsafe operations"),
    ]),
    ("6. End-to-end workflow testing", [
        (PARTIAL, "Prompt flows cleanly from request to summary (works for P1, P2, P4; broken for P3, P5, P6, P7 due to F1)"),
        (PARTIAL, "Final response answers the actual question asked (P1 reports wrong scope; P2 ran wrong analysis; P4 had placeholders)"),
        (CHECK, "Outputs are interpretable for a novice user"),
        (CHECK, "Artifacts are visible and labeled clearly"),
        (PARTIAL, "Mixed requests behave correctly (P4 ran a tool but dropped the and then run it half)"),
        (FAIL, "Bad input cases produce helpful exception feedback (P6 detection works, but the validator blocks delivery)"),
    ]),
    ("7. Response quality validation", [
        (PARTIAL, "Response is statistically appropriate for the request"),
        (CHECK, "Response uses correct variable names"),
        (PARTIAL, "Response explains results clearly and accurately"),
        (CHECK, "Response avoids overclaiming or causal overinterpretation"),
        (CHECK, "Response acknowledges uncertainty or limitations when needed"),
        (PARTIAL, "Response makes appropriate use of retrieved knowledge"),
    ]),
    ("8. Error-handling and edge cases", [
        (CHECK, "Works with missing data in key variables"),
        (CHECK, "Handles small datasets and sparse categories gracefully"),
        (FAIL, "Handles misspelled or partially incorrect variable names (P6 detection right, delivery broken by F1)"),
        (PARTIAL, "Handles vague or overly broad requests gracefully (P5 right answer, blocked by validator)"),
        (FAIL, "Handles requests outside the app scope clearly (P7 right answer, blocked by validator)"),
        (CHECK, "Handles weak RAG results without hallucinating"),
        (FAIL, "Responds to unrelated content prompts with a notification (same root cause as F1)"),
    ]),
    ("9. Traceability and observability", [
        (CHECK, "Router decisions are visible in Langfuse traces"),
        (CHECK, "Retrieval steps and retrieved context are visible"),
        (CHECK, "Tool and codegen branches are easy to inspect in traces"),
        (CHECK, "Errors are captured clearly in traces"),
        (CHECK, "Session IDs, tags, and prompt versions are trackable"),
        (CHECK, "Successful and failed runs can be compared side by side"),
    ]),
    ("10. Tried at least one prompt from each category", [
        (CHECK, "Simple tool: compute pearson correlations between numeric columns (P1)"),
        (CHECK, "Simple codegen: write code that bins age into 5 quantiles and shows income rate per quantile (P2)"),
        (CHECK, "RAG-conceptual: according to the knowledge base when should I use multiple regression? (P3)"),
        (CHECK, "Mixed: use the knowledge base to recommend an analysis and then run it (P4)"),
        (CHECK, "Ambiguous: help me analyze this dataset (P5)"),
        (CHECK, "Bad input: analyze the column nonexistent_variable_xyz (P6)"),
        (CHECK, "Unrelated: how do I fix my kitchen sink? (P7)"),
    ]),
]


VALIDATION_ROWS = [
    ["ID", "Prompt", "Category", "Expected route", "Actual route", "Retrieval relevant?", "Execution OK?", "Quality (0-5)", "Notes"],
    ["1", "compute pearson correlations between numeric columns", "Simple tool",
     "tool: pearson_correlation",
     "tool: pearson_correlation, args:{x:'age', y:'fnlwgt'}",
     "n/a", "Yes", "2",
     "Router invented x and y for an arg-less tool. Tool ignored the kwargs and ran on all numeric columns, but the LLM summary reported the wrong scope."],
    ["2", "write code that bins age into 5 quantiles and shows income rate per quantile", "Simple codegen",
     "codegen", "codegen", "Partial", "Yes (return code 0)", "1",
     "Generated code is a boxplot of hours_per_week by income, not age-quantile income rate. Code executed cleanly but solves the wrong problem."],
    ["3", "according to the knowledge base when should I use multiple regression?", "RAG-conceptual",
     "answer", "answer", "Yes", "No (validator blocks)", "0",
     "Router gave a sensible plain-language answer; validator rejects mode=answer so the user sees only ERROR."],
    ["4", "use the knowledge base to recommend an analysis and then run it", "Mixed",
     "tool or chain", "tool: basic_profile", "Partial", "Yes (tool ran)", "2",
     "Tool executed. Final summary used placeholder values (X, Y) instead of the actual basic_profile output."],
    ["5", "help me analyze this dataset", "Ambiguous",
     "any reasonable", "answer", "Yes", "No (validator blocks)", "0",
     "Reasonable workflow guidance produced and discarded by validator."],
    ["6", "analyze the column nonexistent_variable_xyz", "Bad input",
     "refuse", "answer", "Yes", "No (validator blocks)", "0",
     "Correct detection of the missing column. Delivery broken by F1."],
    ["7", "how do I fix my kitchen sink?", "Unrelated",
     "refuse", "answer", "Yes", "No (validator blocks)", "0",
     "Polite off-topic refusal generated and discarded by validator."],
]


SCORECARD_ROWS = [
    ["Metric", "Target", "Observed", "Pass?"],
    ["Router accuracy", ">= 80%", "7/7 routed to a sensible mode", "Yes"],
    ["Relevant retrieval in top-k", ">= 80%", "6/7 with relevant chunks (P1 did not need RAG)", "Yes"],
    ["Tool execution success", ">= 90%", "2/2 tool calls succeeded", "Yes"],
    ["Approved code execution success", ">= 80%", "1/1 approved scripts executed", "Yes"],
    ["Average final response quality", ">= 4 / 5", "0.71 (scores: 2, 1, 0, 2, 0, 0, 0)", "No"],
    ["Graceful handling of bad input", ">= 90%", "0/2 (P6 and P7 both blocked by F1)", "No"],
]


FAILURES = [
    ("F1. Validator rejects mode:answer.",
     "The router can return three modes (tool, codegen, answer), but the agent's validator only accepts the first two. Every conceptual, ambiguous, bad-input, and off-topic prompt is silently killed here. This single bug is responsible for 4 of the 7 failures and both bad-input failures."),
    ("F2. Codegen target mismatch.",
     "For P2, the codegen chain produced a boxplot of hours_per_week by income instead of the requested age-quantile income-rate analysis. The retrieved context did not steer the generation toward the user's actual question."),
    ("F3. Tool-arg hallucination.",
     "For P1, the router invented {x:'age', y:'fnlwgt'} for pearson_correlation, which takes no args. The tool silently ignored the kwargs, but the LLM summary then reported only that pair, misrepresenting what was computed."),
    ("F4. Summary chain not fed tool outputs.",
     "For P4, the summary chain produced placeholder text such as the average value was X instead of reading the actual basic_profile output file. The tool ran, but the final user-facing reply did not reflect what the tool produced."),
]


REFLECTION = [
    ("What did the agent do especially well?",
     "The router itself was reliable. It picked a sensible mode for every one of the seven prompts, used real column names, retrieved relevant knowledge for conceptual prompts, and never invented tool names. The HITL gate also worked as designed: no script ran without explicit approval, and the unapproved-code path correctly blocked execution. RAG retrieval consistently surfaced the right files for the right question."),
    ("What failures appeared most often: routing, retrieval, tool execution, code generation, or output quality?",
     "Output quality and the post-router validator dominated. F1 alone accounted for four of the seven prompt failures, including every bad-input and off-topic prompt. The remaining failures (F2 codegen target mismatch, F3 hallucinated args, F4 missing tool output in summary) are also output-quality bugs rather than routing or retrieval bugs. Routing, retrieval, and tool execution were all reliable in isolation; the breakdown is in what happens after."),
    ("What is the single highest-priority improvement for the next revision?",
     'Accept mode:"answer" in the validator and route it to a "reply directly with retrieved context" branch. This one fix turns four of the seven failing prompts into successes and immediately satisfies the graceful handling of bad input criterion. Everything else (codegen alignment, summary-chain wiring, tool-arg sanitation) is worth doing, but F1 has by far the highest payoff for the smallest change.'),
]


def build():
    st = styles()
    flow = []

    flow.append(p("Assignment 5: Testing and Validation Log", st["title"]))
    flow.append(p("QAC387-01 Spring 2026 &nbsp;&nbsp;|&nbsp;&nbsp; Holt Young, Sam Penn &nbsp;&nbsp;|&nbsp;&nbsp; 2026-04-30", st["subtitle"]))

    flow.append(p("Agent under test", st["h2"]))
    flow.append(p("builds/build4_rag_router_agent.py (RAG + HITL + Tool Router). Dataset: UCI Adult Income (data/adult.csv, 32,561 rows x 15 columns). Knowledge corpus: 10 markdown files indexed as 74 FAISS chunks with text-embedding-3-small. Chat model: OpenAI gpt-4o-mini. Tracing: Langfuse Cloud.", st["body"]))

    flow.append(p("How to reproduce", st["h2"]))
    flow.append(p("Launch the agent once from the repo root with the .env file present:", st["body"]))
    flow.append(Preformatted(
        "python builds/build4_rag_router_agent.py \\\n"
        "    --data data/adult.csv \\\n"
        "    --knowledge_dir knowledge \\\n"
        "    --provider openai",
        st["code"],
    ))
    flow.append(p("When the &gt; prompt appears, type each of the seven prompts below (one at a time). When the agent asks Run tool now? or Approve and save this code?, type y. After approving generated code, type run, then y to execute it. Type exit when done.", st["body"]))
    flow.append(Preformatted(
        "ask compute pearson correlations between numeric columns\n"
        "ask write code that bins age into 5 quantiles and shows income rate per quantile\n"
        "ask according to the knowledge base when should I use multiple regression?\n"
        "ask use the knowledge base to recommend an analysis and then run it\n"
        "ask help me analyze this dataset\n"
        "ask analyze the column nonexistent_variable_xyz\n"
        "ask how do I fix my kitchen sink?\n"
        "exit",
        st["code"],
    ))

    flow.append(p("Checklist legend: " + CHECK + " pass &nbsp; " + PARTIAL + " partial &nbsp; " + FAIL + " fail", st["body"]))

    for section, items in CHECKLIST:
        flow.append(p(section, st["h1"]))
        for marker, text in items:
            flow.append(bullet(f"{marker} {text}", st))

    flow.append(p("Validation log", st["h1"]))
    avail = LETTER[0] - 1.2 * inch
    widths = [
        avail * 0.03, avail * 0.18, avail * 0.08, avail * 0.10,
        avail * 0.12, avail * 0.08, avail * 0.09, avail * 0.06,
        avail * 0.26,
    ]
    flow.append(make_table(VALIDATION_ROWS, st, col_widths=widths))

    flow.append(p("Success criteria scorecard", st["h1"]))
    avail2 = LETTER[0] - 1.2 * inch
    score_widths = [avail2 * 0.30, avail2 * 0.18, avail2 * 0.42, avail2 * 0.10]
    flow.append(make_table(SCORECARD_ROWS, st, col_widths=score_widths))

    flow.append(p("Failure modes observed", st["h1"]))
    for title, body in FAILURES:
        flow.append(bullet(f"<b>{title}</b> {body}", st))

    flow.append(p("Reflection", st["h1"]))
    for q, a in REFLECTION:
        flow.append(p(f"<b>{q}</b>", st["h2"]))
        flow.append(p(a, st["body"]))

    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=LETTER,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title="Assignment 5 Validation Log",
        author="Holt Young, Sam Penn",
    )
    doc.build(flow)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    build()
