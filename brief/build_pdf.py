"""Builds brief/Agentic_Memory_Brief.pdf from brief_content.py and checks the submission limits."""

import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from brief_content import SECTIONS, SUBTITLE, TITLE  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = str(HERE / "Agentic_Memory_Brief.pdf")
FIG = str(HERE) + "/"


def words(markup: str) -> int:
    text = html.unescape(re.sub(r"<[^>]+>", " ", markup))
    return len([w for w in re.split(r"\s+", text) if re.search(r"[A-Za-z0-9]", w)])


counts = {}
for title, limit, blocks in SECTIONS:
    n = 0
    for b in blocks:
        if b[0] == "p":
            n += words(b[1])
        elif b[0] == "b":
            n += sum(words(x) for x in b[1])
        elif b[0] == "t":
            n += sum(words(x) for x in b[1]) + sum(words(c) for r in b[2] for c in r)
    counts[title] = n
    if title.startswith("1."):
        sentences = re.split(
            r"(?<=[.!?])\s+", html.unescape(re.sub(r"<[^>]+>", "", blocks[0][1])).strip()
        )
        assert len(sentences) == 1, sentences
    else:
        assert n <= limit, (title, n, limit)
total = (
    sum(counts.values()) + words(TITLE) + words(SUBTITLE) + sum(words(t) for t, _, _ in SECTIONS)
)
assert total <= 1200, total
print(counts, "total", total)

INK = colors.HexColor("#111827")
ACC = colors.HexColor("#1D4ED8")
base = ParagraphStyle(
    "b",
    fontName="Helvetica",
    fontSize=10,
    leading=13.0,
    textColor=INK,
    spaceAfter=5,
    alignment=TA_LEFT,
)
h1 = ParagraphStyle(
    "h1", parent=base, fontName="Helvetica-Bold", fontSize=17, leading=21, spaceAfter=2
)
sub = ParagraphStyle(
    "s", parent=base, fontSize=9.5, textColor=colors.HexColor("#4B5563"), spaceAfter=10
)
h2 = ParagraphStyle(
    "h2",
    parent=base,
    fontName="Helvetica-Bold",
    fontSize=12.5,
    leading=16,
    textColor=ACC,
    spaceBefore=9,
    spaceAfter=4,
)
cell = ParagraphStyle("c", parent=base, fontSize=9, leading=11.5, spaceAfter=0)
cap = ParagraphStyle(
    "cap",
    parent=base,
    fontSize=8.5,
    leading=11,
    textColor=colors.HexColor("#4B5563"),
    spaceBefore=2,
    spaceAfter=8,
)

doc = SimpleDocTemplate(
    OUT,
    pagesize=A4,
    leftMargin=20 * mm,
    rightMargin=20 * mm,
    topMargin=17 * mm,
    bottomMargin=17 * mm,
    title=TITLE,
    author="",
    subject="Agentic memory design brief",
)
W = A4[0] - 40 * mm
story = [Paragraph(TITLE, h1), Paragraph(SUBTITLE, sub)]
for title, _limit, blocks in SECTIONS:
    head = Paragraph(title, h2)
    first = True
    for b in blocks:
        if b[0] == "p":
            fl = Paragraph(b[1], base)
        elif b[0] == "b":
            fl = ListFlowable(
                [ListItem(Paragraph(x, base), leftIndent=12) for x in b[1]],
                bulletType="bullet",
                start="•",
                leftIndent=12,
                bulletFontSize=9,
            )
        elif b[0] == "t":
            hdr, rows, cw = b[1], b[2], b[3]
            data = [[Paragraph(f"<b>{h}</b>", cell) for h in hdr]] + [
                [Paragraph(c, cell) for c in r] for r in rows
            ]
            fl = Table(data, colWidths=[W * f for f in cw], repeatRows=1)
            fl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2FF")),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            fl = [fl, Spacer(1, 6)]
        elif b[0] == "fig":
            from PIL import Image as PI

            w, h = PI.open(FIG + b[1]).size
            iw = W * 1.0
            fl = KeepTogether(
                [Image(FIG + b[1], width=iw, height=iw * h / w), Paragraph(b[2], cap)]
            )
        items = fl if isinstance(fl, list) else [fl]
        if first:
            story.append(KeepTogether([head, items[0]]))
            story.extend(items[1:])
        else:
            story.extend(items)
        first = False


def footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#6B7280"))
    c.drawRightString(A4[0] - 20 * mm, 10 * mm, f"{d.page}")
    c.restoreState()


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("built", OUT)
