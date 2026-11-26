"""
Regenerates the Section 6 brief from the canonical results:
  1. figures.py      -> fig1_architecture.png, fig2_measurements.png (reads benchmarks/results.json and
                        eval/results/three_arm_offline.json)
  2. build_pdf.py    -> Agentic_Memory_Brief.pdf (asserts six sections, per-section word limits, <=1,200 words)
  3. brief.md        -> the same text as Markdown
Usage: python brief/generate_brief.py
"""

import html
import re
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _md(t: str) -> str:
    t = re.sub(r"</?b>", "**", t)
    t = re.sub(r"</?i>", "*", t)
    return html.unescape(t)


def write_markdown() -> None:
    import brief_content as b

    out = [f"# {b.TITLE}", "", _md(b.SUBTITLE), ""]
    for title, _limit, blocks in b.SECTIONS:
        out += [f"## {title}", ""]
        for blk in blocks:
            if blk[0] == "p":
                out += [_md(blk[1]), ""]
            elif blk[0] == "b":
                out += [f"- {_md(x)}" for x in blk[1]] + [""]
            elif blk[0] == "fig":
                out += [f"![{_md(blk[2])}]({blk[1]})", "", f"*{_md(blk[2])}*", ""]
    (HERE / "brief.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    runpy.run_path(str(HERE / "figures.py"), run_name="__main__")
    runpy.run_path(str(HERE / "build_pdf.py"), run_name="__main__")
    write_markdown()
