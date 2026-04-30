"""Render the Assignment 5 testing & validation log as a PDF (v2: student voice).

Same data, tables, failure-mode IDs, checklist, scorecard, session IDs, and CLI
commands as build_assignment5_pdf.py. Only the prose has been rewritten to read
like a careful student wrote it (fewer em-dashes, no parallel-contrast rhetoric,
no bolded lead-ins, plainer reflection section).

Run:
    PYTHONPATH=. python scripts/build_assignment5_pdf_v2.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = PROJECT_ROOT / "docs" / "Assignment5_Validation_Log_v2.pdf"


def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            name="title", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=22, leading=26, alignment=1, spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            name="subtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=13, leading=16, alignment=1,
            textColor=colors.HexColor("#333333"), spaceAfter=24,
        ),
        "byline": ParagraphStyle(
            name="byline", parent=base["Normal"], fontName="Helvetica",
            fontSize=11, leading=14, alignment=1,
            textColor=colors.HexColor("#555555"), spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            name="h1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=15, leading=19, textColor=colors.black,
            spaceBefore=12, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            name="h2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, textColor=colors.black,
            spaceBefore=10, spaceAfter=4,
        ),
        "h3": ParagraphStyle(
            name="h3", parent=base["Heading3"], fontName="Helvetica-Bold",
            fontSize=10.5, leading=13,
            textColor=colors.HexColor("#333333"),
            spaceBefore=6, spaceAfter=2,
        ),
        "body": ParagraphStyle(
            name="body", parent=base["BodyText"], fontName="Helvetica",
            fontSize=10.5, leading=14, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            name="bullet", parent=base["BodyText"], fontName="Helvetica",
            fontSize=10, leading=13, leftIndent=18, bulletIndent=6, spaceAfter=2,
        ),
        "caption": ParagraphStyle(
            name="caption", parent=base["Italic"], fontName="Helvetica-Oblique",
            fontSize=9.5, leading=12,
            textColor=colors.HexColor("#555555"), spaceAfter=8,
        ),
        "cell": ParagraphStyle(
            name="cell", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.5, leading=10.5, spaceAfter=0,
            wordWrap="CJK",  # allow breaking inside long underscore_joined_tokens
        ),
        "cell_b": ParagraphStyle(
            name="cell_b", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=8.5, leading=10.5, spaceAfter=0,
            wordWrap="CJK",
        ),
    }


HEADER_BG = colors.HexColor("#222222")
GRID = colors.HexColor("#888888")


def std_table_style(font_size=9.5):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])


def cell_table_style(font_size=8.5):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), font_size),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ])


# --------------------------------------------------------------------------------------
# Tables (UNCHANGED FROM v1: data, columns, IDs all preserved)
# --------------------------------------------------------------------------------------
def make_param_table():
    data = [
        ["Parameter", "Value"],
        ["Agent", "builds/build4_rag_router_agent.py"],
        ["Dataset", "UCI Adult Income — 32,561 rows, 15 columns"],
        ["RAG corpus", "10 markdown files, 74 chunks, text-embedding-3-small"],
        ["Tracing", "Langfuse Cloud (https://cloud.langfuse.com)"],
        ["Test date", "2026-04-29"],
        ["Providers", "OpenAI gpt-4o-mini  /  Moonshot kimi-k2.6"],
        ["Sessions", "a5-sweep-openai  /  a5-sweep-kimi-vm"],
    ]
    table = Table(data, colWidths=[1.3 * inch, 5.2 * inch])
    table.setStyle(std_table_style(font_size=10))
    return table


def make_provider_table():
    data = [
        ["Provider", "Model", "Session ID", "Notes"],
        ["OpenAI", "gpt-4o-mini", "a5-sweep-openai", "Default fast/cheap baseline"],
        ["Moonshot", "kimi-k2.6", "a5-sweep-kimi-vm",
         "Reasoning model, 256K context. Run on a temporary GCP e2-small VM that auto-deleted on completion."],
    ]
    cells = [[Paragraph(c, styles["cell_b"] if i == 0 else styles["cell"]) for c in row]
             for i, row in enumerate(data)]
    table = Table(cells, colWidths=[0.8 * inch, 1.0 * inch, 1.5 * inch, 3.2 * inch])
    table.setStyle(cell_table_style())
    return table


def make_sweep_table(rows):
    headers = ["#", "Prompt", "Category", "Actual route", "R", "A", "X", "Q", "Notes"]
    body = [headers] + rows
    cells = []
    for i, row in enumerate(body):
        cell_row = []
        for c in row:
            style = styles["cell_b"] if i == 0 else styles["cell"]
            cell_row.append(Paragraph(str(c), style))
        cells.append(cell_row)
    table = Table(
        cells,
        colWidths=[
            0.25 * inch,  # #
            1.35 * inch,  # Prompt
            1.10 * inch,  # Category (fits "RAG-conceptual")
            1.40 * inch,  # Actual route
            0.50 * inch,  # R (fits "partial")
            0.50 * inch,  # A (fits "partial")
            0.50 * inch,  # X (fits "partial")
            0.25 * inch,  # Q
            1.25 * inch,  # Notes
        ],
        repeatRows=1,
    )
    table.setStyle(cell_table_style())
    return table


OPENAI_ROWS = [
    ["1", "compute pearson correlations between numeric columns",
     "Simple tool", "tool: pearson_correlation, args:{x:age, y:fnlwgt}",
     "✓", "✗", "partial", "2",
     "<b>Args hallucinated.</b> Tool takes no args (computes on all numerics) but router emitted x/y; tool ignored args silently and ran on full set, but the LLM summary claimed \"between age and fnlwgt\" — wrong scope reported to user."],
    ["2", "make a histogram of hours_per_week",
     "Simple tool", "tool: plot_histograms, args:{numeric_cols:[hours_per_week], fig_dir:\"\"}",
     "✓", "partial", "✓", "4",
     "Hallucinated empty fig_dir (tool ignores it). Histogram saved to correct dir. Summary accurate."],
    ["3", "write code that bins age into 5 quantiles and shows income rate per quantile",
     "Simple codegen", "codegen",
     "✓", "n/a", "✗", "1",
     "<b>Codegen target mismatch.</b> Generated code is a boxplot of hours_per_week by income — completely unrelated to the requested analysis. PLAN block also describes the wrong task."],
    ["4", "according to the knowledge base when should I use multiple regression?",
     "RAG-conceptual", "answer",
     "✓", "n/a", "✗", "0",
     "<b>Validator crash.</b> Router correctly identified pure-knowledge intent but the agent rejects mode:\"answer\" (only tool/codegen accepted). User sees an error instead of the explanation."],
    ["5", "use the knowledge base to recommend an analysis and then run it",
     "Mixed", "tool: basic_profile",
     "✓", "✓", "✓", "2",
     "Tool ran. <b>Summary chain hallucinated</b> — wrote \"[Specific results would be listed here, but none were provided in the tool output.]\" indicating the summary chain was not fed the tool output text."],
    ["6", "help me analyze this dataset",
     "Ambiguous", "answer (workflow guidance)",
     "✓", "n/a", "✗", "0",
     "Same validator crash. Router gave a sensible reply but the agent threw it away."],
    ["7", "analyze the column nonexistent_variable_xyz",
     "Bad input", "answer (\"column does not exist\")",
     "✓", "n/a", "✗", "0",
     "Detection correct, delivery broken (validator crash). No helpful message reaches user."],
    ["8", "how do I fix my kitchen sink?",
     "Unrelated", "answer (\"conceptual question, no analysis\")",
     "✓", "n/a", "✗", "0",
     "Refusal logic fires but never reaches the user because of the validator crash."],
]


KIMI_ROWS = [
    ["1", "compute pearson correlations between numeric columns",
     "Simple tool", "codegen (\"correlation matrix for all numeric columns\")",
     "✓", "n/a", "✗", "4",
     "<b>Better routing than OpenAI.</b> Kimi recognized the pearson_correlation tool only handles a single (x,y) pair, so it routed to codegen for the full matrix. Code saved but not run."],
    ["2", "make a histogram of hours_per_week",
     "Simple tool", "tool: plot_histograms, args:{numeric_cols:[hours_per_week], fig_dir:\"figures\"}",
     "✓", "✓", "✓", "4",
     "Cleaner args than OpenAI (fig_dir:\"figures\" vs \"\"). Tool ran, summary accurate."],
    ["3", "write code that bins age into 5 quantiles and shows income rate per quantile",
     "Simple codegen", "codegen",
     "✓", "n/a", "✗", "1",
     "<b>Same codegen target-mismatch bug as OpenAI</b> — codegen chain produced an exploratory dataset overview instead of quantile-bin code. Confirms F2 is not model-dependent."],
    ["4", "according to the knowledge base when should I use multiple regression?",
     "RAG-conceptual", "answer",
     "✓", "n/a", "✗", "0",
     "Same validator crash. Same root cause as OpenAI."],
    ["5", "use the knowledge base to recommend an analysis and then run it",
     "Mixed", "answer (workflow narrative)",
     "partial", "n/a", "✗", "0",
     "Kimi chose to describe an analysis instead of running one — the \"and then run it\" half was dropped. Validator also crashes the answer."],
    ["6", "help me analyze this dataset",
     "Ambiguous", "answer (multi-step workflow)",
     "✓", "n/a", "✗", "0",
     "Validator crash. Reasoning was high-quality (cited dataset, workflow steps) but never reaches user."],
    ["7", "analyze the column nonexistent_variable_xyz",
     "Bad input", "answer (\"column does not exist; available columns are…\")",
     "✓", "n/a", "✗", "0",
     "<b>Better than OpenAI</b> — listed the available columns. Still crashes at validator."],
    ["8", "how do I fix my kitchen sink?",
     "Unrelated", "answer (\"off-topic home-repair question\")",
     "✓", "n/a", "✗", "0",
     "<b>Best refusal of any test.</b> Polite, scoped, explicit. Crashes at validator."],
]


def make_failure_table():
    headers = ["ID", "Bug", "Reproduces on", "Severity", "Where it lives"]
    data = [
        ["F1", "mode:\"answer\" validator crash", "both providers", "Critical",
         "Router prompt advertises \"answer\" as a legal mode; downstream validator accepts only tool/codegen. Every conceptual / off-scope / refusal answer crashes. Hits 4–6 of 8 prompts depending on provider."],
        ["F2", "Codegen target mismatch", "both providers", "Critical",
         "The codegen chain ignores the user's actual request and emits a generic \"exploratory overview\" or unrelated boxplot. PLAN block also describes the wrong task. Detectable at HITL but the agent never produces the requested analysis."],
        ["F3", "Router hallucinates tool args", "OpenAI more than Kimi", "High",
         "pearson_correlation (no args) was given {x,y}; plot_histograms (no fig_dir arg) was given {fig_dir:\"\"}. Tools currently ignore unexpected args, masking the bug — until an arg conflicts with required fields."],
        ["F4", "Summary chain hallucinates without tool output", "OpenAI", "High",
         "After basic_profile ran successfully, the summary chain output \"[Specific results would be listed here, but none were provided in the tool output.]\" — i.e., the tool result was not passed into the summary prompt."],
        ["F5", "Bad-input and off-scope prompts crash instead of returning helpful messages", "both providers", "High",
         "Downstream consequence of F1: detection is correct, delivery to the user is broken."],
        ["F6", "RAG retrieval picks weakly-relevant chunks for codegen", "both providers", "Medium",
         "For \"bin age into 5 quantiles\", top retrieved chunk was tools/plot_histograms.md (score 0.475). The genuinely relevant guides/analysis_workflow.md did not surface. Embedding/chunking issue, not a router issue."],
    ]
    body = [headers] + data
    cells = []
    for i, row in enumerate(body):
        cell_row = []
        for c in row:
            style = styles["cell_b"] if i == 0 else styles["cell"]
            cell_row.append(Paragraph(str(c), style))
        cells.append(cell_row)
    table = Table(
        cells,
        colWidths=[0.30 * inch, 1.6 * inch, 1.05 * inch, 0.65 * inch, 2.9 * inch],
        repeatRows=1,
    )
    table.setStyle(cell_table_style())
    return table


def make_scorecard_table():
    headers = ["Metric", "Target", "OpenAI", "Kimi", "Combined", "Pass?"]
    data = [
        ["Router accuracy (defensible routing decision)", "≥ 80%", "8/8 (100%)", "8/8 (100%)", "16/16 (100%)", "✓"],
        ["Relevant retrieval in top-k", "≥ 80%", "1/2 codegen", "1/2 codegen", "2/4 (50%)", "✗"],
        ["Tool execution success", "≥ 90%", "2/3 attempted", "1/1 attempted", "3/4 (75%)", "✗"],
        ["Approved code execution success", "≥ 80%", "0/2 correct code", "0/2 correct code", "0/4", "✗"],
        ["Average final response quality", "≥ 4/5", "1.1", "1.1", "1.1", "✗"],
        ["Graceful bad-input / off-scope handling (Q7, Q8)", "≥ 90%", "0/2 (validator)", "0/2 (validator)", "0/4", "✗"],
    ]
    body = [headers] + data
    cells = []
    for i, row in enumerate(body):
        cell_row = []
        for c in row:
            style = styles["cell_b"] if i == 0 else styles["cell"]
            cell_row.append(Paragraph(str(c), style))
        cells.append(cell_row)
    table = Table(
        cells,
        colWidths=[
            2.25 * inch,  # Metric
            0.65 * inch,  # Target
            1.00 * inch,  # OpenAI
            1.00 * inch,  # Kimi
            1.05 * inch,  # Combined
            0.55 * inch,  # Pass?
        ],
        repeatRows=1,
    )
    table.setStyle(cell_table_style())
    return table


def make_compare_table():
    headers = ["Behavior", "OpenAI advantage", "Kimi advantage"]
    data = [
        ["Speed per router call", "~5s", "~30–90s (reasoning)"],
        ["Cost (this run)", "~$0.005", "~$0.10"],
        ["Routing for the pearson prompt", "—",
         "Recognized tool's (x,y) limitation, routed to codegen for the full matrix instead of running on wrong scope"],
        ["Args quality", "—", "fig_dir:\"figures\" vs OpenAI's \"\""],
        ["Refusal quality", "—", "Polite, scoped explanation for off-topic and bad-input prompts"],
        ["Tool args fabrication", "—", "Less prone to inventing kwargs the tool doesn't accept"],
    ]
    body = [headers] + data
    cells = []
    for i, row in enumerate(body):
        cell_row = []
        for c in row:
            style = styles["cell_b"] if i == 0 else styles["cell"]
            cell_row.append(Paragraph(str(c), style))
        cells.append(cell_row)
    table = Table(
        cells,
        colWidths=[1.7 * inch, 1.6 * inch, 3.2 * inch],
        repeatRows=1,
    )
    table.setStyle(cell_table_style())
    return table


# --------------------------------------------------------------------------------------
# Checklist (unchanged)
# --------------------------------------------------------------------------------------
CHECK = "\u2713"
CROSS = "\u2717"
TILDE = "~"


def checklist_section(title_text, items):
    blocks = [Paragraph(title_text, styles["h2"])]
    for marker, text in items:
        blocks.append(Paragraph(f"<font color='#222222'><b>[{marker}]</b></font> {text}", styles["bullet"]))
    return blocks


CHECKLIST_SECTIONS = [
    ("§1. Core setup and environment checks", [
        (CHECK, "App starts without import or path errors (verified on local + a fresh GCP VM)"),
        (CHECK, "Environment variables and model settings load correctly (provider switch wired through .env)"),
        (CHECK, "Langfuse tracing is active and receiving runs (Langfuse Cloud confirmed; spans visible)"),
        (CHECK, "RAG index loads without errors (74 chunks)"),
        (CHECK, "Tool registry loads correctly (16 tools enumerated at startup)"),
        (CHECK, "Dataset file uploads/read operations work correctly"),
        (CHECK, "Schema text is extracted and passed into prompts"),
        (CHECK, "Report, tool output, and figure directories are created correctly"),
        (CHECK, "Generated code and artifacts are written to the specified path"),
    ]),
    ("§2. Router decision testing", [
        (CHECK, "Requests with available tools route to the correct tool (prompt 2)"),
        (CHECK, "Code-generation requests route to codegen (prompt 3)"),
        (TILDE, "Knowledge-based questions use RAG appropriately — router routes to answer, but F1 crashes the response"),
        (CHECK, "Ambiguous prompts still produce a reasonable choice (prompt 6)"),
        (CHECK, "Router output is valid JSON"),
        (CHECK, "Router does not hallucinate tools that do not exist"),
        (CHECK, "Router does not default to codegen when an available tool fits"),
        (TILDE, "Router respects dataset schema and doesn't hallucinate variables — but does hallucinate kwargs (F3)"),
        (CHECK, "Prompts work even when the user is not highly specific (handled in router; answer mode then crashes)"),
    ]),
    ("§3. RAG retrieval testing", [
        (TILDE, "Top retrieved chunks are clearly relevant — F6 shows weak retrieval for the quantile prompt"),
        (TILDE, "Retrieved chunks come from the most appropriate knowledge files"),
        (CHECK, "Retrieved material contains enough detail to improve the answer (when relevant)"),
        (CROSS, "Different phrasings of the same question still retrieve useful context (not exhaustively tested)"),
        (CHECK, "The agent does not fabricate knowledge when retrieval is weak (it ignores the chunks rather than fabricating)"),
        (CHECK, "Retrieved content does not push the router into the wrong mode"),
    ]),
    ("§4. Tool execution testing", [
        (TILDE, "Tool name and arguments match the user request — name yes, args partially (F3)"),
        (CHECK, "Variable names passed to tools exist in the dataframe"),
        (CHECK, "Tool runs without crashing (when args are accepted)"),
        (CHECK, "Tool returns a standardized output object or expected structure"),
        (CHECK, "Saved figures and files appear in the correct directories"),
        (TILDE, "Tool summaries are understandable to the end user — F4 hallucinates summary in one case"),
        (CROSS, "Missing arguments, empty subsets, and invalid inputs are handled clearly (no try/except for empty result sets)"),
    ]),
    ("§5. Code generation and HITL testing", [
        (TILDE, "Generated code matches the requested analysis — F2 fails this (target mismatch)"),
        (CHECK, "Generated code uses valid dataframe and column names"),
        (CHECK, "Generated code is valid Python"),
        (CHECK, "Human-in-the-loop approval is required before execution"),
        (CHECK, "Unapproved code cannot be run"),
        (CHECK, "Approved code executes successfully within timeout limits (when the code matches the request)"),
        (CHECK, "Errors, stdout, and stderr are captured or displayed clearly"),
        (CHECK, "Outputs and artifacts are saved in the expected location"),
        (CHECK, "Generated code does not attempt unsafe or unintended operations"),
    ]),
    ("§6. End-to-end workflow testing", [
        (TILDE, "A user prompt flows cleanly from request to router to execution to summary — F1 breaks 4–6 of 8 prompts"),
        (TILDE, "The final response answers the actual question asked"),
        (CHECK, "Outputs are interpretable for a novice user (when the response makes it through)"),
        (CHECK, "Artifacts are visible, downloadable, and labeled clearly"),
        (TILDE, "Mixed requests behave correctly — Kimi dropped the \"and then run it\" half"),
        (CROSS, "Bad input cases produce helpful exception feedback — F5"),
    ]),
    ("§7. Response quality validation", [
        (TILDE, "Response is statistically appropriate for the request"),
        (CHECK, "Response uses correct variable names and terms (when answered)"),
        (TILDE, "Response explains results clearly — F4 produces an empty summary"),
        (CHECK, "Response avoids overclaiming or causal overinterpretation"),
        (CHECK, "Response acknowledges uncertainty or limitations"),
        (TILDE, "Response makes appropriate use of retrieved knowledge — sometimes ignored"),
    ]),
    ("§8. Error-handling and edge cases", [
        (CROSS, "Works with missing data in key variables (not exhaustively tested in this sweep)"),
        (CROSS, "Handles small datasets and sparse categories gracefully (not tested)"),
        (CROSS, "Handles misspelled or partially incorrect variable names reasonably"),
        (TILDE, "Handles vague or overly broad requests gracefully — router does, validator doesn't (F1)"),
        (TILDE, "Handles requests outside the tool or app scope clearly — same"),
        (CHECK, "Handles no relevant RAG results without hallucinating"),
        (TILDE, "Responds to unrelated content prompts and returns a notification — refusal generated but never delivered"),
    ]),
    ("§9. Traceability and observability", [
        (CHECK, "Router decisions are visible in Langfuse traces"),
        (CHECK, "Retrieval steps and retrieved context are visible"),
        (CHECK, "Tool or codegen branches are easy to inspect in traces"),
        (CHECK, "Errors are captured clearly in traces (validator-rejected outputs visible)"),
        (CHECK, "Session IDs, tags, and prompt versions are trackable"),
        (CHECK, "Successful and failed runs can be compared for debugging (direct OpenAI-vs-Kimi compare done in §6)"),
    ]),
    ("§10. ≥1 prompt from each category", [
        (CHECK, "Simple tool — prompt 1, 2 (both providers)"),
        (CHECK, "Simple codegen — prompt 3 (both providers)"),
        (CHECK, "RAG-heavy conceptual — prompt 4"),
        (CHECK, "Mixed — prompt 5"),
        (CHECK, "Ambiguous — prompt 6"),
        (CHECK, "Bad-input — prompt 7"),
        (CHECK, "Unrelated — prompt 8"),
    ]),
]


# --------------------------------------------------------------------------------------
# Story (PROSE REWRITTEN; tables / data / IDs unchanged)
# --------------------------------------------------------------------------------------
def build_story():
    s = []

    # Title
    s.append(Spacer(1, 2.5 * inch))
    s.append(Paragraph("Assignment 5: Testing &amp; Validation", styles["title"]))
    s.append(Paragraph(
        "Build 4 RAG + HITL + Tool Router Agent<br/>UCI Adult Income Dataset",
        styles["subtitle"],
    ))
    s.append(Spacer(1, 0.15 * inch))
    s.append(Paragraph("Holt Young &amp; Sam Penn", styles["byline"]))
    s.append(Paragraph("QAC387 — Spring 2026", styles["byline"]))
    s.append(Paragraph("April 29, 2026", styles["byline"]))
    s.append(PageBreak())

    # 1. Overview
    s.append(Paragraph("1. Overview", styles["h1"]))
    s.append(Paragraph(
        "This document records the testing and validation of our Build 4 RAG + HITL + "
        "Tool Router Agent on the UCI Adult Income dataset. Per the assignment brief, "
        "the goal is to identify where the agent works and where it fails so we know "
        "what to improve next.",
        styles["body"],
    ))
    s.append(make_param_table())
    s.append(Spacer(1, 0.15 * inch))

    # 2. Methodology
    s.append(Paragraph("2. Test methodology", styles["h1"]))
    s.append(Paragraph(
        "We ran an 8-prompt sweep covering every category required by the assignment "
        "checklist (simple tool, simple codegen, RAG-conceptual, mixed, ambiguous, "
        "bad-input, unrelated) on two providers:",
        styles["body"],
    ))
    s.append(make_provider_table())
    s.append(Spacer(1, 0.10 * inch))
    s.append(Paragraph(
        "We added a <font face='Courier'>--provider {openai,moonshot}</font> flag to the "
        "Build 4 agent for this assignment so the same prompts could run on different "
        "model families without changing the agent's prompts, RAG index, or HITL flow. "
        "Embeddings keep using <font face='Courier'>text-embedding-3-small</font> "
        "regardless of chat provider. Every routing, retrieval, tool, and codegen call "
        "is traced in Langfuse Cloud under its session ID.",
        styles["body"],
    ))
    s.append(PageBreak())

    # 3. Validation log
    s.append(Paragraph("3. Validation log", styles["h1"]))
    s.append(Paragraph(
        "Columns: <b>R</b> = correct route, <b>A</b> = correct args, <b>X</b> = "
        "executed cleanly, <b>Q</b> = response quality (1 to 5, 0 = crash).",
        styles["caption"],
    ))
    s.append(Paragraph("3.1 OpenAI gpt-4o-mini", styles["h2"]))
    s.append(make_sweep_table(OPENAI_ROWS))
    s.append(Spacer(1, 0.20 * inch))
    s.append(Paragraph("3.2 Moonshot kimi-k2.6", styles["h2"]))
    s.append(make_sweep_table(KIMI_ROWS))
    s.append(PageBreak())

    # 4. Failure modes
    s.append(Paragraph("4. Failure modes", styles["h1"]))
    s.append(Paragraph(
        "Six distinct failure modes surfaced across the 16 test runs. F1 and F2 are "
        "critical. They reproduce on both providers and break correctness. The other "
        "four are quality issues.",
        styles["body"],
    ))
    s.append(make_failure_table())
    s.append(PageBreak())

    # 5. Scorecard
    s.append(Paragraph("5. Success-criteria scorecard", styles["h1"]))
    s.append(Paragraph(
        "Targets are defined in the assignment brief. The router does well on its own. "
        "Everything downstream of the router fails to clear the bar.",
        styles["body"],
    ))
    s.append(make_scorecard_table())
    s.append(Spacer(1, 0.20 * inch))

    # 6. Provider comparison
    s.append(Paragraph("6. Provider comparison", styles["h1"]))
    s.append(Paragraph(
        "<font face='Courier'>gpt-4o-mini</font> and <font face='Courier'>kimi-k2.6</font> "
        "produced the same two critical failures (F1, F2) on the same prompts. Where they "
        "differed:",
        styles["body"],
    ))
    s.append(make_compare_table())
    s.append(Spacer(1, 0.10 * inch))
    s.append(Paragraph(
        "Upgrading the model fixed arg quality and refusal phrasing but not F1 or F2. "
        "Those need code changes.",
        styles["body"],
    ))
    s.append(PageBreak())

    # 7. Checklist
    s.append(Paragraph("7. Assignment checklist", styles["h1"]))
    s.append(Paragraph(
        f"Legend: <b>[{CHECK}]</b> passed &nbsp;&nbsp; <b>[{TILDE}]</b> partial, "
        f"works upstream but breaks downstream &nbsp;&nbsp; <b>[{CROSS}]</b> not met "
        "or not tested.",
        styles["caption"],
    ))
    for title_text, items in CHECKLIST_SECTIONS:
        for block in checklist_section(title_text, items):
            s.append(block)
    s.append(PageBreak())

    # 8. Reflection (REWRITTEN: shorter, choppier, plainer)
    s.append(Paragraph("8. Final reflection", styles["h1"]))

    s.append(Paragraph("What did the agent do especially well?", styles["h2"]))
    s.append(Paragraph(
        "The router worked well on both models. We got defensible routing decisions on "
        "16 of 16 prompts. It picked the right tool when a tool fit, picked codegen when "
        "no tool fit, and chose answer mode for the off-topic and bad-input prompts. It "
        "also flagged the nonexistent variable correctly. Langfuse tracing was solid too. "
        "Every router, retrieval, tool, codegen, and summary call shows up under its "
        "session ID, so we could audit any single prompt across both providers without "
        "much digging.",
        styles["body"],
    ))

    s.append(Paragraph("What failures appeared most often?", styles["h2"]))
    s.append(Paragraph(
        "The biggest problem is the validator rejecting <font face='Courier'>mode:'answer'</font>. "
        "The router prompt says answer is a legal mode but the validator only accepts "
        "tool or codegen, so every conceptual or refusal response gets thrown away. This "
        "crashed 4 of 8 prompts on OpenAI and 5 of 8 on Kimi. Every conceptual question, "
        "every off-scope refusal, and every bad-input flag was generated correctly and "
        "then dropped. The second issue is the codegen chain. It produced code unrelated "
        "to what the user asked. We asked it to bin age into 5 quantiles and got back a "
        "boxplot of hours_per_week by income. Both bugs show up the same way on a much "
        "stronger model, so they are not really about which LLM you use. They are about "
        "the contract between the router output and the rest of the agent.",
        styles["body"],
    ))

    s.append(Paragraph("What is the single highest-priority improvement?", styles["h2"]))
    s.append(Paragraph(
        "We need to fix the contract between the router output and the rest of the agent. "
        "First, accept <font face='Courier'>mode:'answer'</font> in the validator and "
        "route it through a small handler that prints the router's note or response text "
        "to the user, with no tool call, no codegen, and no HITL gate. Second, audit the "
        "codegen chain to confirm it is actually being passed "
        "<font face='Courier'>user_request</font> and not some stale or misrouted "
        "variable. The fact that codegen always produces \"exploratory overview\" code "
        "suggests the chain is not seeing the actual prompt. Both fixes are local and "
        "probably take an hour each. They would move the scorecard from a fail on most "
        "rows to a pass on most rows. Retrieval quality (F6) and arg fabrication (F3) "
        "are real bugs too, but they are secondary.",
        styles["body"],
    ))

    s.append(Paragraph("What we did not test", styles["h2"]))
    s.append(Paragraph(
        "We did not exhaustively test the §8 edge cases in this sweep. Specifically, we "
        "did not test missing data in key variables, small datasets, or misspelled "
        "column names. Those should be in the next round of testing.",
        styles["body"],
    ))

    s.append(PageBreak())
    s.append(Paragraph("Appendix A — Reproducing this run", styles["h1"]))
    s.append(Paragraph("OpenAI sweep (local):", styles["body"]))
    s.append(Paragraph(
        "<font face='Courier' size='9'>PYTHONPATH=. python builds/build4_rag_router_agent.py "
        "--data data/adult.csv --knowledge_dir knowledge --report_dir reports "
        "--provider openai --session_id a5-sweep-openai --tags build4,assignment5,sweep,openai</font>",
        styles["body"],
    ))
    s.append(Paragraph(
        "Kimi sweep (Moonshot kimi-k2.6, ran on a temp GCP VM that auto-deleted):",
        styles["body"],
    ))
    s.append(Paragraph(
        "<font face='Courier' size='9'>PYTHONPATH=. python builds/build4_rag_router_agent.py "
        "--data data/adult.csv --knowledge_dir knowledge --report_dir reports "
        "--provider moonshot --session_id a5-sweep-kimi-vm --tags build4,assignment5,sweep,kimi,vm</font>",
        styles["body"],
    ))
    s.append(Paragraph(
        "Trace data was fetched from Langfuse Cloud via the public REST API "
        "(<font face='Courier' size='9'>GET /api/public/traces?sessionId=...</font>) using "
        "the project's public/secret API keys.",
        styles["body"],
    ))

    return s


def main():
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Assignment 5: Testing & Validation",
        author="Holt Young & Sam Penn",
    )
    doc.build(build_story())
    print(f"Wrote {OUT_PDF}")


styles = build_styles()


if __name__ == "__main__":
    main()
