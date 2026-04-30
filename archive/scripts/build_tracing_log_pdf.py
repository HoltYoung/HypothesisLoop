"""Render the Build 4 tracing and error log as a PDF in the same style as the Build 3 PDF."""

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
OUT_PDF = PROJECT_ROOT / "docs" / "Langfuse_Tracing_Log_Build4.pdf"


def build_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            name="title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            alignment=1,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            name="subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=16,
            alignment=1,
            textColor=colors.HexColor("#333333"),
            spaceAfter=24,
        ),
        "byline": ParagraphStyle(
            name="byline",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            alignment=1,
            textColor=colors.HexColor("#555555"),
            spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            name="h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.black,
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            name="h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.black,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            name="body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            name="bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            leftIndent=18,
            bulletIndent=6,
            spaceAfter=2,
        ),
        "caption": ParagraphStyle(
            name="caption",
            parent=base["Italic"],
            fontName="Helvetica-Oblique",
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#555555"),
            spaceAfter=8,
        ),
    }
    return styles


def make_param_table(styles):
    data = [
        ["Parameter", "Value"],
        ["Model", "gpt-4o-mini (default)"],
        ["Tracing", "Langfuse (LangChain CallbackHandler + @observe)"],
        ["Dataset", "UCI Adult Income — 32,561 rows, 15 columns"],
        ["RAG corpus", "10 markdown files, 74 chunks, text-embedding-3-small"],
        ["Session IDs", "build4-smoke, build4-smoke-codegen, build4-traces"],
        ["Langfuse Host", "http://127.0.0.1:3000"],
    ]
    table = Table(data, colWidths=[1.5 * inch, 4.8 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222222")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#888888")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def make_scenarios_table():
    data = [
        ["Test", "Request", "Expected Mode", "Result"],
        ["1", "Basic profile of the dataset", "tool (basic_profile)", "Pass"],
        ["2", "Frequency table for education", "tool (summarize_categorical)", "Pass"],
        ["3", "Correlations among numeric columns", "tool (pearson_correlation)", "Pass"],
        ["4", "Histogram of age", "tool (plot_histograms)", "Pass"],
        [
            "5",
            "Logistic regression: income ~ age + hours_per_week",
            "codegen (RAG fires)",
            "Pass*",
        ],
        ["6", "Summarize missing values", "tool (missingness_table)", "Pass"],
    ]
    table = Table(
        data,
        colWidths=[0.5 * inch, 2.6 * inch, 1.9 * inch, 0.7 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#222222")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#888888")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def build_story(styles):
    story = []

    # ---- Title page ----
    story.append(Spacer(1, 2.5 * inch))
    story.append(Paragraph("Langfuse Tracing &amp; Error Log", styles["title"]))
    story.append(
        Paragraph(
            "Build 4: RAG + HITL + Tool Router Agent<br/>UCI Adult Income Dataset Analysis",
            styles["subtitle"],
        )
    )
    # separator-ish gap
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Holt Young &amp; Sam Penn", styles["byline"]))
    story.append(Paragraph("QAC387 — Spring 2026", styles["byline"]))
    story.append(Paragraph("April 24, 2026", styles["byline"]))
    story.append(PageBreak())

    # ---- Section 1: Overview ----
    story.append(Paragraph("1. Overview", styles["h1"]))
    story.append(
        Paragraph(
            "This document summarizes the Langfuse tracing output, errors encountered, and "
            "resolutions for our Build 4 RAG + HITL + Tool Router Agent running on the UCI "
            "Adult Income dataset (32,561 rows, 15 columns). Build 4 extends Build 3 by adding "
            "a FAISS-indexed markdown knowledge corpus; relevant chunks are retrieved and "
            "injected into the code-generation prompt on every codegen request. All traces are "
            "available in our local Langfuse instance.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(make_param_table(styles))
    story.append(Spacer(1, 0.15 * inch))

    # ---- Section 2: Test Scenarios ----
    story.append(Paragraph("2. Test Scenarios", styles["h1"]))
    story.append(
        Paragraph(
            "We ran six test scenarios across three sessions to exercise routing, tool "
            "execution, code generation with RAG retrieval, and the LLM summarization chain:",
            styles["body"],
        )
    )
    story.append(make_scenarios_table())
    story.append(Spacer(1, 0.05 * inch))
    story.append(
        Paragraph(
            "* The logistic-regression request was classified as codegen and generated a "
            "boxplot script after RAG-retrieved chunks flagged that <i>income</i> is binary. "
            "Details in §5.",
            styles["caption"],
        )
    )

    # ---- Section 3: Errors Encountered & Resolutions ----
    story.append(Paragraph("3. Errors Encountered &amp; Resolutions", styles["h1"]))

    story.append(Paragraph("Error 1: Python could not find our project code", styles["h2"]))
    story.append(
        Paragraph(
            "<b>What happened.</b> The very first time we tried to build the RAG index by running "
            "<font face='Courier'>python scripts/build_rag_index.py</font> from the project folder, it "
            "crashed immediately with the message <font face='Courier'>ModuleNotFoundError: No module named 'src'</font>. "
            "Our project keeps all the shared helper code (data readers, the FAISS utilities, etc.) in a "
            "folder called <font face='Courier'>src/</font>, and the build script needs to import from it.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Why.</b> When you launch a Python script, Python automatically searches for imports in "
            "the folder where the script itself lives — and only that folder, plus the standard library. "
            "Our build script lives in <font face='Courier'>scripts/</font>, so Python was looking inside "
            "<font face='Courier'>scripts/</font> for an <font face='Courier'>src</font> package. But "
            "<font face='Courier'>src/</font> is one level up at the project root, so Python had no way "
            "to see it. The teacher's script does compute the project root in a variable, but it never "
            "actually adds that path to Python's import search list, so the variable is unused for "
            "import purposes.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "<b>How we fixed it.</b> We started invoking the script with the prefix "
            "<font face='Courier'>PYTHONPATH=.</font> — that one extra piece tells Python \"also look in "
            "the current folder (the project root) when resolving imports.\" The full command becomes "
            "<font face='Courier'>PYTHONPATH=. python scripts/build_rag_index.py</font>. We did not "
            "change the teacher's reference script for this — we just documented the right invocation "
            "in the README so any grader or teammate runs it correctly the first time.",
            styles["body"],
        )
    )

    story.append(Paragraph("Error 2: OpenAI rejected our requests even though our key was valid", styles["h2"]))
    story.append(
        Paragraph(
            "<b>What happened.</b> Once Python could find <font face='Courier'>src/</font>, the build "
            "script reached the embedding step and immediately died with "
            "<font face='Courier'>openai.RateLimitError: 429 insufficient_quota</font>. That message "
            "normally means the OpenAI account is out of credits — but ours was fine, and the API key "
            "in our project's <font face='Courier'>.env</font> file was tested and working.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Why.</b> An API key is the password the agent uses to prove to OpenAI it is allowed to "
            "make requests. We keep ours in a file called <font face='Courier'>.env</font> at the project "
            "root, which Python loads using a library called <font face='Courier'>python-dotenv</font>. "
            "Separately, Windows has its own system-wide environment variables — values that any program "
            "on the computer can read. Holt's machine had an old, expired OpenAI key sitting in his "
            "Windows user environment from months ago. Two keys, same name "
            "(<font face='Courier'>OPENAI_API_KEY</font>), different values: one valid in the project's "
            "<font face='Courier'>.env</font>, one expired in the operating system.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "When the agent started, <font face='Courier'>python-dotenv</font> opened the "
            "<font face='Courier'>.env</font> file and noticed that <font face='Courier'>OPENAI_API_KEY</font> "
            "already existed in the operating system environment (the expired one). By default the "
            "library is conservative: if a variable is already set, it does not overwrite it — it assumes "
            "the existing value is intentional. So the project's good key was silently ignored and the "
            "expired one was used. OpenAI received requests signed with the dead key and answered with the "
            "<font face='Courier'>insufficient_quota</font> error, which made the underlying problem look "
            "like a billing issue instead of the key-collision it actually was.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "<b>How we fixed it.</b> We changed the two places in our codebase that load the "
            "<font face='Courier'>.env</font> file — <font face='Courier'>scripts/build_rag_index.py</font> "
            "and <font face='Courier'>builds/build4_rag_router_agent.py</font> — from "
            "<font face='Courier'>load_dotenv(...)</font> to "
            "<font face='Courier'>load_dotenv(..., override=True)</font>. The added "
            "<font face='Courier'>override=True</font> tells the library: \"always treat the value in "
            "<font face='Courier'>.env</font> as the source of truth, even if the same variable already "
            "exists in the system environment.\" After that one-word change the project's key wins every "
            "time, and the embedding step ran cleanly. This is the same fix we applied during Build 3 "
            "for the same root cause.",
            styles["body"],
        )
    )

    story.append(Paragraph("Error 3: We assumed approving generated code would auto-run it", styles["h2"]))
    story.append(
        Paragraph(
            "<b>What happened.</b> While testing the code-generation path, we triggered a query that the "
            "router classified as codegen, the model produced a script, and the agent asked "
            "<b>Approve and save this code? (y/n)</b>. We answered <font face='Courier'>y</font>, "
            "expecting the script to immediately execute. Instead the agent saved the script to disk and "
            "went back to the <font face='Courier'>&gt;</font> prompt waiting for our next command. Our "
            "follow-up <font face='Courier'>y</font> got rejected as <i>Unrecognized command</i>.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Why.</b> This is the agent's HITL design working as intended, not a defect. The "
            "approve-and-save step is treated as a separate human checkpoint from the run-the-code step. "
            "After approval the user is supposed to inspect the saved file (located under "
            "<font face='Courier'>reports/session_&lt;id&gt;/agent_generated_analysis.py</font>) and only "
            "then issue the literal command <font face='Courier'>run</font>, which triggers a final "
            "<i>Execute &lt;script&gt; now? (y/n)</i> prompt before the subprocess actually fires. "
            "There are deliberately two human gates: one for \"this code is safe to keep\" and a second "
            "for \"yes, run it on my machine.\"",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "<b>How we fixed it.</b> Nothing to fix in the agent itself — the two-gate flow is the right "
            "behavior for a tool that runs LLM-generated code on the user's machine. We updated the "
            "README's example session to make the approve-then-<font face='Courier'>run</font> sequence "
            "explicit, so anyone using the agent for the first time will not get tripped up the same way.",
            styles["body"],
        )
    )

    # ---- Section 4: Agent Performance Summary ----
    story.append(PageBreak())
    story.append(Paragraph("4. Agent Performance Summary", styles["h1"]))

    story.append(Paragraph("Routing Accuracy", styles["h2"]))
    story.append(
        Paragraph(
            "The router correctly classified all six scenarios. Tool mode was selected for "
            "requests matching the Build-0 tool catalog (profile, frequency tables, correlations, "
            "histograms, missingness). Codegen mode was selected for the logistic-regression "
            "request, which has no tool in the allow-list.",
            styles["body"],
        )
    )

    story.append(Paragraph("Tool Execution", styles["h2"]))
    story.append(
        Paragraph(
            "All five tool runs completed without error and wrote outputs to the report directory:",
            styles["body"],
        )
    )
    for b in [
        "<font face='Courier'>basic_profile</font> reported 32,561 rows × 15 columns with dtype breakdown.",
        "<font face='Courier'>summarize_categorical</font> produced accurate frequency counts for the "
        "education column.",
        "<font face='Courier'>pearson_correlation</font> returned pairwise r, 95% CIs, and p-values for "
        "all numeric columns.",
        "<font face='Courier'>plot_histograms</font> saved a histogram PNG for age.",
        "<font face='Courier'>missingness_table</font> correctly surfaced 5.66% missing in "
        "<font face='Courier'>occupation</font>, 5.64% in <font face='Courier'>workclass</font>, and "
        "1.79% in <font face='Courier'>native_country</font> — matching the "
        "<font face='Courier'>?</font>-encoding noted in our codebook.",
    ]:
        story.append(Paragraph(f"• {b}", styles["bullet"]))

    story.append(Paragraph("Code Generation and RAG Retrieval", styles["h2"]))
    story.append(
        Paragraph(
            "On the logistic-regression request, the agent retrieved four RAG chunks. The top two "
            "were both Caution sections — one from <font face='Courier'>adult_codebook.md</font> "
            "(score 0.601) warning that <font face='Courier'>income</font> is binary, and one from "
            "<font face='Courier'>tools/multiple_linear_regression.md</font> (score 0.595) warning not "
            "to use the regression tool on categorical outcomes. The codegen model reflected these "
            "cautions in its plan and produced a boxplot of <font face='Courier'>hours_per_week</font> "
            "by <font face='Courier'>income</font> instead — a safer analysis for the data at hand. "
            "The generated script used argparse, validated required columns, dropped rows with missing "
            "values, and saved the figure to <font face='Courier'>--report_dir</font>.",
            styles["body"],
        )
    )

    story.append(Paragraph("LLM Summarization", styles["h2"]))
    story.append(
        Paragraph(
            "The summarizer chain produced a five-part interpretation after each tool run: what "
            "analysis was performed, key numerical results, a plain-language interpretation, "
            "caveats and assumptions, and recommended next steps.",
            styles["body"],
        )
    )

    # ---- Section 5: RAG Impact Assessment ----
    story.append(Paragraph("5. RAG Impact Assessment", styles["h1"]))
    story.append(
        Paragraph(
            "<b>Verdict: RAG enhanced agent performance on the codegen path.</b>",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Evidence.</b> In Test 5 the codegen model was given four retrieved chunks before "
            "generating code. The top-scored chunks explicitly flagged that "
            "<font face='Courier'>income</font> is binary and that <font face='Courier'>multiple_linear_regression</font> "
            "is inappropriate for categorical outcomes. The generated script pivoted away from a "
            "strict linear regression and produced a visualization the data actually supports. This "
            "is the intended effect of RAG — the retrieved domain facts shaped the output in a way "
            "the bare LLM would not have done. Retrieved chunks also identified the "
            "<font face='Courier'>?</font> missing-value encoding and the available predictors, letting "
            "the generated code include explicit preprocessing without having to guess.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "<b>Where RAG did not matter.</b> The reference agent only injects retrieved chunks into "
            "the codegen prompt. For the four tool-routed requests (Tests 1–4) the router made "
            "correct decisions without any RAG context. There was no measurable RAG impact on the "
            "tool path in this test set.",
            styles["body"],
        )
    )
    story.append(Paragraph("Suggested improvements:", styles["body"]))
    for b in [
        "<b>Feed RAG into the router.</b> Indexed tool notes contain exact argument shapes; injecting "
        "them into the router prompt would reduce router argument-name mistakes on edge cases.",
        "<b>Feed RAG into the summarizer.</b> The interpretation step currently does not see the "
        "retrieved chunks. Letting it cite retrieved guidance would produce more grounded summaries.",
        "<b>Add a session-history corpus.</b> Indexing prior queries and router decisions would let "
        "the agent recognize when a requested analysis has already been run in the current session.",
    ]:
        story.append(Paragraph(f"• {b}", styles["bullet"]))

    # ---- Section 6: Key Takeaways ----
    story.append(Paragraph("6. Key Takeaways", styles["h1"]))
    for i, b in enumerate(
        [
            "<b>RAG works where it is wired in.</b> The codegen prompt was visibly shaped by retrieved "
            "caution chunks, producing safer output on an ambiguous request.",
            "<b>HITL is still essential.</b> Even with RAG-improved codegen, human approval remains "
            "the final gate before any tool or script runs.",
            "<b>Environment-level API keys will break your day.</b> Using "
            "<font face='Courier'>load_dotenv(override=True)</font> is a one-line fix that prevents a stale "
            "shell key from shadowing the project <font face='Courier'>.env</font>.",
            "<b>Langfuse traces capture everything worth auditing.</b> Router decisions, tool spans, "
            "codegen prompts, retrieved-chunk metadata, and summary output are all visible for review.",
        ],
        start=1,
    ):
        story.append(Paragraph(f"{i}. {b}", styles["body"]))

    return story


def main() -> None:
    styles = build_styles()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=LETTER,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title="Langfuse Tracing & Error Log — Build 4",
        author="Holt Young & Sam Penn",
    )
    doc.build(build_story(styles))
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
