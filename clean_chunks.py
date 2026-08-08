"""
clean_chunks.py — post-hoc cleaner for the already-generated *_chunks.json
files: strips decorative/structural LaTeX (colors, fonts, FontAwesome
icons, tables, tikz diagrams, spacing) that chunk_notes.py currently
leaves untouched in the "content" field, while leaving real math
($...$ and \\[...\\]) completely untouched so nothing mathematical
changes meaning.

This is a downstream patch (works on the *_chunks.json already in the
repo, since the original .tex source isn't in this repo). The proper
long-term fix is to do the same stripping inside chunk_notes.py itself
(the box-title auto-generation there has already been patched — see
chunk_notes.py) and re-run it against the original .tex source.

STATUS (tested against the 21 *_chunks.json files in this repo):
  - Before: 57-100% of chunks per file contained raw LaTeX formatting
    commands (\\textcolor, \\begin{tabular}, \\begin{tikzpicture}, etc.)
    in "content" — this is what the AI was reading as "notes context"
    for every question, and the likely source of LaTeX leaking into
    shown answers.
  - After: ~72 of ~2260 chunks (~3%) still have some residual markup —
    mostly \\textcolor/\\cellcolor used *inside* real math ($...$), and
    a couple of tikzpicture blocks nested inside \\begin{center} in an
    order this script's regexes don't fully unwrap. Run this script,
    then grep the *.cleaned.json output for "textcolor|textbf|tikzpicture"
    to get the exact list before trusting it further — don't swap these
    into production un-reviewed.
"""
import re
import json
import glob

MATH_PLACEHOLDER = "\x00MATH{}\x00"

def _protect_math(text):
    """Pulls out $...$ and \\[...\\] blocks so later cleanup passes can't
    touch real math, returns (text_with_placeholders, saved_blocks)."""
    saved = []

    def _save(m):
        saved.append(m.group(0))
        return MATH_PLACEHOLDER.format(len(saved) - 1)

    text = re.sub(r"\\\[.*?\\\]", _save, text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\$(?:[^$\\]|\\.)*\$", _save, text)
    # Bare align/align* blocks (multi-step derivations) aren't wrapped in
    # $/\[ \] in the source, so Streamlit/KaTeX won't render them as math
    # at all otherwise — wrap in $$ $$ so they do, then protect them too.
    text = re.sub(
        r"\\begin\{align\*?\}.*?\\end\{align\*?\}",
        lambda m: _save_wrapped(m.group(0), saved),
        text, flags=re.DOTALL,
    )
    return text, saved


def _save_wrapped(block, saved):
    saved.append(f"$${block}$$")
    return MATH_PLACEHOLDER.format(len(saved) - 1)


def _restore_math(text, saved):
    def _r(m):
        return saved[int(m.group(1))]
    return re.sub(r"\x00MATH(\d+)\x00", _r, text)


BRACE_ARG = r"\{(?:[^{}]|\{[^{}]*\})*\}"  # one {...} allowing one level of nested {...}


def clean_latex_formatting(text: str) -> str:
    text, saved = _protect_math(text)

    # TikZ diagrams: pure drawing code, no textual value for a Q&A system.
    text = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", text, flags=re.DOTALL)

    # Tables -> markdown tables
    def _tabular_to_md(m):
        body = m.group(1)
        body = re.sub(r"\\hline", "", body)
        body = re.sub(r"\\rowcolor" + BRACE_ARG, "", body)
        body = re.sub(r"\\cellcolor" + BRACE_ARG, "", body)
        rows = [r.strip() for r in body.split(r"\\") if r.strip()]
        rows = [re.sub(r"\s+", " ", r).strip() for r in rows]
        if not rows:
            return ""
        table_rows = [[c.strip() for c in r.split("&")] for r in rows]
        ncols = max(len(r) for r in table_rows)
        table_rows = [r + [""] * (ncols - len(r)) for r in table_rows]
        lines = ["| " + " | ".join(table_rows[0]) + " |",
                 "|" + "|".join(["---"] * ncols) + "|"]
        for r in table_rows[1:]:
            lines.append("| " + " | ".join(r) + " |")
        return "\n" + "\n".join(lines) + "\n"

    text = re.sub(r"\\begin\{tabular\}" + BRACE_ARG + r"(.*?)\\end\{tabular\}", _tabular_to_md, text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{center\}\s*(.*?)\s*\\end\{center\}", r"\1", text, flags=re.DOTALL)

    # Lists -> markdown
    def _itemize_to_md(m):
        items = re.split(r"\\item(?:\[[^\]]*\])?", m.group(1))[1:]
        return "\n" + "\n".join(f"- {i.strip()}" for i in items if i.strip()) + "\n"

    def _enumerate_to_md(m):
        items = [i for i in re.split(r"\\item(?:\[[^\]]*\])?", m.group(1))[1:] if i.strip()]
        return "\n" + "\n".join(f"{idx+1}. {i.strip()}" for idx, i in enumerate(items)) + "\n"

    text = re.sub(r"\\begin\{itemize\}(?:\[[^\]]*\])?(.*?)\\end\{itemize\}", _itemize_to_md, text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{enumerate\}(?:\[[^\]]*\])?(.*?)\\end\{enumerate\}", _enumerate_to_md, text, flags=re.DOTALL)

    # Inline formatting/icons/color/spacing -> keep inner text only, or drop.
    # A few passes to unwrap nesting like \textcolor{x}{\textbf{y}}.
    textbf_re = re.compile(r"\\text(?:bf|it)\{((?:[^{}]|\{[^{}]*\})*)\}")
    textcolor_re = re.compile(r"\\textcolor\{[^}]*\}\{((?:[^{}]|\{[^{}]*\})*)\}")
    for _ in range(4):
        text = textbf_re.sub(r"\1", text)
        text = textcolor_re.sub(r"\1", text)
    text = re.sub(r"\\fa[A-Z][a-zA-Z]*\\?\s*", "", text)
    text = re.sub(r"\\vspace\{[^}]*\}", "", text)
    text = re.sub(r"\\newpage", "", text)
    text = re.sub(r"\\qed\b", "", text)
    text = re.sub(r"\\checkmark", "✓", text)

    text = _restore_math(text, saved)

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


if __name__ == "__main__":
    import sys
    report = []
    for fname in sorted(glob.glob("*_chunks.json")):
        with open(fname, encoding="utf-8") as f:
            chunks = json.load(f)
        before_leaked = sum(1 for c in chunks if re.search(
            r"\\(textcolor|textbf|begin\{tabular\}|begin\{tikzpicture\}|vspace|fa[A-Z]|rowcolor|cellcolor)",
            c.get("content", "")))
        for c in chunks:
            c["content"] = clean_latex_formatting(c["content"])
        after_leaked = sum(1 for c in chunks if re.search(
            r"\\(textcolor|textbf|begin\{tabular\}|begin\{tikzpicture\}|vspace|fa[A-Z]|rowcolor|cellcolor)",
            c.get("content", "")))
        out_name = fname.replace("_chunks.json", "_chunks.cleaned.json")
        with open(out_name, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        report.append((fname, len(chunks), before_leaked, after_leaked))

    print(f"{'file':35s} {'chunks':>7s} {'before':>7s} {'after':>7s}")
    for r in report:
        print(f"{r[0]:35s} {r[1]:7d} {r[2]:7d} {r[3]:7d}")
