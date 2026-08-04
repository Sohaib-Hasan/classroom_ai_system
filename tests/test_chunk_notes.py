"""
test_chunk_notes.py
----------------------
chunk_notes.py ki logic bilkul change NAHI ki gayi hai (koi confirmed bug
evidence nahi mila, aur asal .tex source files repo mein maujood nahi
hain jin se safely test kiya ja sake) — lekin isme pehle ZERO tests thay,
isliye ye basic sanity tests add kar rahe hain synthetic .tex snippets se,
taake aage koi bhi is file mein change kare to regression turant pakda jaye.

NOTE: is file mein ek known limitation hai jo maine code padh kar
identify ki (lekin live .tex data na hone ki wajah se confirm nahi kar
saka) — SAME TYPE ki nested boxes (jaise ek definitionbox ke andar
doosra definitionbox) shayad sahi tarah handle na hon, kyunke parser
flat hai (nesting-aware nahi). Agar aapke notes mein aisi nesting
istemal hoti hai, `test_same_type_nested_boxes_is_a_known_limitation`
test isko documents karta hai — agar future mein ye fix ho, is test ko
update kar dein.
"""

import json

from chunk_notes import parse_tex_notes


def write_tex(tmp_path, content):
    p = tmp_path / "sample.tex"
    p.write_text(content, encoding="utf-8")
    return str(p)


class TestChunkNotesBasics:
    def test_section_and_definitionbox(self, tmp_path):
        tex = r"""
\section{Vector Spaces}
\begin{definitionbox}{Vector Space}
A vector space is a set closed under addition and scalar multiplication.
\end{definitionbox}
"""
        path = write_tex(tmp_path, tex)
        chunks = parse_tex_notes(path, "Linear Algebra", "Chapter 1")
        assert len(chunks) == 1
        assert chunks[0]["title"] == "Vector Space"
        assert chunks[0]["section"] == "Vector Spaces"
        assert chunks[0]["course"] == "Linear Algebra"
        assert "closed under addition" in chunks[0]["content"]

    def test_multiple_boxes_same_section(self, tmp_path):
        tex = r"""
\section{Determinants}
\begin{definitionbox}{Determinant}
Some definition.
\end{definitionbox}
\begin{examplebox}{Worked Example}
Some example.
\end{examplebox}
"""
        path = write_tex(tmp_path, tex)
        chunks = parse_tex_notes(path, "Linear Algebra", "Chapter 2")
        assert len(chunks) == 2
        assert chunks[0]["type"] == "definition"
        assert chunks[1]["type"] == "example"

    def test_untitled_keypoint_gets_auto_title(self, tmp_path):
        tex = r"""
\section{Intro}
\begin{keypoint}
Always check your work carefully before submitting.
\end{keypoint}
"""
        path = write_tex(tmp_path, tex)
        chunks = parse_tex_notes(path, "Calculus", "Chapter 1")
        assert len(chunks) == 1
        assert chunks[0]["title"]  # auto-generated, non-empty

    def test_practice_section_without_boxes_splits_by_item(self, tmp_path):
        tex = r"""
\section{Practice Problems}
\begin{enumerate}
\item Compute the derivative of x^2.
\item Compute the integral of x.
\end{enumerate}
"""
        path = write_tex(tmp_path, tex)
        chunks = parse_tex_notes(path, "Calculus", "Chapter 1")
        practice_chunks = [c for c in chunks if c["type"] == "practice_problem"]
        assert len(practice_chunks) == 2

    def test_output_is_json_serializable(self, tmp_path):
        tex = r"""
\section{Test}
\begin{definitionbox}{Def}
Content here.
\end{definitionbox}
"""
        path = write_tex(tmp_path, tex)
        chunks = parse_tex_notes(path, "Test Course", "Chapter 1")
        json.dumps(chunks)  # should not raise


class TestSameTypeNestedBoxes:
    """Ye confirmed (test se, guess se nahi) behaviour hai: same-type
    nested boxes cleanly separate NAHI hote — Inner apna chunk nahi
    banata, aur uska raw markup Outer ke content mein leak ho jata hai.
    Koi text silently discard nahi hota (achi baat), lekin content
    'ganda' (raw LaTeX markup ke saath) ban jata hai. check_for_leaked_box_markup()
    isi ko loudly flag karta hai taake ye chup-chaap knowledge base mein
    na chala jaye."""

    def test_inner_box_does_not_become_its_own_chunk(self, tmp_path):
        tex = r"""
\section{Nested}
\begin{definitionbox}{Outer}
Outer content.
\begin{definitionbox}{Inner}
Inner content.
\end{definitionbox}
More outer content.
\end{definitionbox}
"""
        path = write_tex(tmp_path, tex)
        chunks = parse_tex_notes(path, "Test Course", "Chapter 1")
        titles = [c["title"] for c in chunks]
        assert "Inner" not in titles  # confirmed limitation, not silently assumed

    def test_no_text_is_silently_lost(self, tmp_path):
        tex = r"""
\section{Nested}
\begin{definitionbox}{Outer}
Outer content.
\begin{definitionbox}{Inner}
Inner content.
\end{definitionbox}
More outer content.
\end{definitionbox}
"""
        path = write_tex(tmp_path, tex)
        chunks = parse_tex_notes(path, "Test Course", "Chapter 1")
        outer = next((c for c in chunks if c["title"] == "Outer"), None)
        assert outer is not None
        assert "Outer content" in outer["content"]
        assert "Inner content" in outer["content"]
        assert "More outer content" in outer["content"]

    def test_leaked_markup_warning_fires(self, tmp_path, capsys):
        tex = r"""
\section{Nested}
\begin{definitionbox}{Outer}
Outer content.
\begin{definitionbox}{Inner}
Inner content.
\end{definitionbox}
More outer content.
\end{definitionbox}
"""
        path = write_tex(tmp_path, tex)
        parse_tex_notes(path, "Test Course", "Chapter 1")
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "Outer" in captured.out

    def test_non_nested_boxes_do_not_trigger_the_warning(self, tmp_path, capsys):
        tex = r"""
\section{Not Nested}
\begin{definitionbox}{A}
Content A.
\end{definitionbox}
\begin{examplebox}{B}
Content B.
\end{examplebox}
"""
        path = write_tex(tmp_path, tex)
        parse_tex_notes(path, "Test Course", "Chapter 1")
        captured = capsys.readouterr()
        assert "raw LaTeX box-markup" not in captured.out
