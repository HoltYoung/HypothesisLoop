"""Render docs/Assignment5_Validation_Log.md to a PDF.

Usage:
    PYTHONPATH=. python scripts/build_assignment5_pdf.py
"""

from __future__ import annotations

import re
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
SRC_MD = PROJECT_ROOT / "docs" / "Assignment5_Validation_Log.md"
OUT_PDF = PROJECT_ROOT / "docs" / "Assignment5_Validation_Log.pdf"


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=20, leading=24, alignment=1, spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=15, leading=18, spaceBefore=14, spaceAfter=6,
            textColor=colors.black,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, spaceBefore=10, spaceAfter=4,
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
            fontSize=8.5, leading=11, leftIndent=8, rightIndent=8,
            backColor=colors.HexColor("#f5f5f5"),
            borderPadding=4, spaceBefore=4, spaceAfter=8,
        ),
        "small": ParagraphStyle(
            "small", parent=base["BodyText"], fontName="Helvetica",
            fontSize=8.5, leading=11,
        ),
    }


def md_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = text.replace("[x]", '<font color="#138a36"><b>[x]</b></font>')
    text = text.replace("[~]", '<font color="#b8860b"><b>[~]</b></font>')
    text = text.replace("[ ]", '<font color="#b22222"><b>[&nbsp;]</b></font>')
    return text


def parse_table(lines, i):
    rows = []
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        rows.append(row)
        i += 1
    if len(rows) >= 2 and all(set(c.replace(":", "").replace("-", "")) <= {""} for c in rows[1]):
        rows.pop(1)
    return rows, i


def render_table(rows, st):
    body_st = ParagraphStyle(
        "td", parent=st["small"], fontSize=8, leading=10, spaceAfter=0,
    )
    head_st = ParagraphStyle("th", parent=body_st, fontName="Helvetica-Bold")
    table_data = []
    for ri, row in enumerate(rows):
        s = head_st if ri == 0 else body_st
        table_data.append([Paragraph(md_inline(c), s) for c in row])
    n_cols = max(len(r) for r in table_data)
    avail = LETTER[0] - 1.2 * inch
    col_w = [avail / n_cols] * n_cols
    t = Table(table_data, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build():
    md = SRC_MD.read_text(encoding="utf-8")
    lines = md.splitlines()
    st = styles()
    flow = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flow.append(Spacer(1, 4))
            i += 1
            continue

        if stripped.startswith("# "):
            flow.append(Paragraph(md_inline(stripped[2:]), st["title"]))
            i += 1
            continue
        if stripped.startswith("## "):
            flow.append(Paragraph(md_inline(stripped[3:]), st["h1"]))
            i += 1
            continue
        if stripped.startswith("### "):
            flow.append(Paragraph(md_inline(stripped[4:]), st["h2"]))
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            flow.append(Preformatted("\n".join(buf), st["code"]))
            continue

        if stripped.startswith("|"):
            rows, i = parse_table(lines, i)
            flow.append(render_table(rows, st))
            flow.append(Spacer(1, 6))
            continue

        if stripped == "---":
            flow.append(Spacer(1, 8))
            i += 1
            continue

        if stripped.startswith("- "):
            flow.append(Paragraph(md_inline(stripped[2:]), st["bullet"], bulletText="-"))
            i += 1
            continue

        flow.append(Paragraph(md_inline(stripped), st["body"]))
        i += 1

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
