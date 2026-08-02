"""
chunk_notes.py
----------------
Ye script aapke LaTeX notes ko chota chota, meaningful tukdon (chunks) mein
todta hai, taake AI ko jab sawal ka jawab dena ho, to wo poora chapter nahi
balke sirf relevant tukda padhe. Har chunk ko "course" bhi tag hota hai,
taake alag-alag subjects ka content ek hi knowledge base mein merge ho sake
bina confuse huay.

Kaise use karna hai (agli baar khud):
    python3 chunk_notes.py chapter2.tex "Linear Algebra" "Chapter 2 - Vector Spaces" chapter2_chunks.json

Char cheezein deni hain:
    1. .tex file ka naam
    2. Course/subject ka naam (jaise "Linear Algebra" ya "Calculus and Analytic Geometry")
    3. Chapter ka naam
    4. Output file ka naam (jahan tukde save hongey)
"""

import re
import json
import sys


# Ye "known" content box-types hain jinke andar Title hota hai: \begin{examplebox}{Title}...
TITLED_BOX_TYPES = ["definitionbox", "examplebox", "theorembox", "proofbox"]
# Ye box-types bina title ke hote hain: \begin{keypoint}...\end{keypoint}
# (tcolorbox jaisi kuch boxes ke saath [styling options] bhi aa sakte hain,
# jinhe hum content mein shamil nahi karte)
UNTITLED_BOX_TYPES = ["keypoint", "tcolorbox"]

# Ye LaTeX ke built-in / formatting environments hain — inhe warning mein ignore karna hai
KNOWN_NON_CONTENT_ENVS = {
    "array", "bmatrix", "pmatrix", "vmatrix", "cases", "center", "enumerate",
    "itemize", "tabular", "align", "align*", "equation", "equation*",
    "split", "matrix", "document", "tikzpicture", "scope", "verbatim",
}


def check_for_unhandled_environments(text, handled_types):
    """Warn karta hai agar file mein koi aisa box-type mile jo humne handle nahi kiya."""
    all_envs = set(re.findall(r"\\begin\{([a-zA-Z*]+)\}", text))
    unhandled = all_envs - set(handled_types) - KNOWN_NON_CONTENT_ENVS
    if unhandled:
        print("⚠️  WARNING: Ye environment(s) file mein mile lekin chunk nahi huay:")
        for env in unhandled:
            count = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", text))
            print(f"   - {env}  ({count} baar aaya hai)")
        print("   Check karein ke inme important content to nahi — agar hai to script mein add karwa lein.\n")


def check_for_missed_practice_section(text, matched_title):
    """Warn karta hai agar koi 'practice' ya 'exercise' jaisa section mila
    jo humare regex ne pakda nahi."""
    all_sections = re.findall(r"\\section\{([^}]*)\}", text)
    for title in all_sections:
        if re.search(r"practice|exercise", title, re.IGNORECASE) and title.strip() != matched_title:
            print(f"⚠️  WARNING: Section '{title}' practice/exercise jaisa lagta hai lekin capture nahi hua — check kar lein.\n")


def split_top_level_items(text):
    """Splits on \\item, but only at the outermost list level —
    nested sub-items (like (i),(ii),(iii) inside Q1) stay merged with their
    parent question in a single chunk, instead of being split separately."""
    tokens = re.finditer(
        r"\\begin\{(?:enumerate|itemize)\}(?:\[[^\]]*\])?|\\end\{(?:enumerate|itemize)\}|\\item",
        text,
    )
    depth = 0
    item_positions = []
    for m in tokens:
        token = m.group(0)
        if token.startswith("\\begin"):
            depth += 1
        elif token.startswith("\\end"):
            depth -= 1
        elif token == "\\item" and depth == 1:
            # depth==1 matlab hum sirf sabse bahar wali list ke andar hain
            item_positions.append(m.end())

    items = []
    for i, start in enumerate(item_positions):
        end = item_positions[i + 1] if i + 1 < len(item_positions) else len(text)
        item_text = text[start:end]
        item_text = re.sub(r"\\end\{enumerate\}\s*$", "", item_text)  # aakhri item se outer-list ka closing tag hata do
        items.append(item_text.strip())
    return items


def parse_tex_notes(filepath, course_name, chapter_name):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Comment-only lines hata do (jo % se shuru hoti hain, sirf separators hain)
    text = re.sub(r"^%.*$", "", text, flags=re.MULTILINE)

    check_for_unhandled_environments(text, TITLED_BOX_TYPES + UNTITLED_BOX_TYPES)

    # Teen tarah ke "landmarks" dhoondo: \section{...}, titled boxes, aur untitled boxes
    section_pattern = re.compile(r"\\section\{([^}]*)\}")
    titled_box_pattern = re.compile(
        r"\\begin\{(" + "|".join(TITLED_BOX_TYPES) + r")\}\{([^}]*)\}(.*?)\\end\{\1\}",
        re.DOTALL,
    )
    untitled_box_pattern = re.compile(
        r"\\begin\{(" + "|".join(UNTITLED_BOX_TYPES) + r")\}(?:\[[^\]]*\])?(.*?)\\end\{\1\}",
        re.DOTALL,
    )

    landmarks = []
    for m in section_pattern.finditer(text):
        landmarks.append((m.start(), m.end(), "section", m.group(1).strip()))
    for m in titled_box_pattern.finditer(text):
        landmarks.append(
            (m.start(), m.end(), m.group(1), (m.group(2).strip(), m.group(3).strip()))
        )
    for m in untitled_box_pattern.finditer(text):
        # Untitled box ka title khud bana lete hain content ke pehle kuch lafzon se
        preview = re.sub(r"\\[a-zA-Z]+", "", m.group(2)).strip()
        auto_title = " ".join(preview.split()[:6]) or "Key Point"
        landmarks.append(
            (m.start(), m.end(), m.group(1), (auto_title, m.group(2).strip()))
        )
    landmarks.sort(key=lambda x: x[0])

    chunks = []
    current_section = "Introduction"
    pending = None

    def close_pending(trailing_text):
        if pending is not None:
            content = pending["content"]
            if trailing_text.strip():
                content += "\n\n" + trailing_text.strip()
            chunks.append(
                {
                    "course": course_name,
                    "chapter": chapter_name,
                    "section": pending["section"],
                    "type": pending["type"].replace("box", ""),
                    "title": pending["title"],
                    "content": content.strip(),
                }
            )

    last_end = 0
    for start, end, kind, data in landmarks:
        trailing = text[last_end:start]
        if kind == "section":
            close_pending(trailing)
            pending = None
            current_section = data
        else:
            close_pending(trailing)
            title, content = data
            pending = {
                "section": current_section,
                "type": kind,
                "title": title,
                "content": content,
            }
        last_end = end

    close_pending(text[last_end:])

    # Practice/Exercises section alag se handle karo (koi box nahi hoti wahan).
    # Naam alag ho sakta hai ("Practice Problems", "Practice Exercises", etc.)
    # isliye "Practice" se shuru hone wala koi bhi section pakadte hain, aur
    # sirf usi section tak mehdood rehte hain (agle \section tak, ya file ke end tak).
    match = re.search(
        r"\\section\{(Practice[^}]*)\}(.*?)(?=\\section\{|\Z)", text, re.DOTALL | re.IGNORECASE
    )
    matched_title = match.group(1).strip() if match else None
    check_for_missed_practice_section(text, matched_title)
    if match:
        section_title = match.group(1).strip()
        problems_text = match.group(2)
        already_has_boxes = re.search(
            r"\\begin\{(" + "|".join(TITLED_BOX_TYPES + UNTITLED_BOX_TYPES) + r")\}",
            problems_text,
        )
        if not already_has_boxes:
            # Ye section boxes use nahi kar raha (jaise Linear Algebra ki
            # \item wali list) — isliye ise manually split karte hain,
            # nested sub-items (i),(ii),(iii) ko parent question ke saath rakhte huay
            items = split_top_level_items(problems_text)
            for idx, item in enumerate(items):
                if not item:
                    continue
                chunks.append(
                    {
                        "course": course_name,
                        "chapter": chapter_name,
                        "section": section_title,
                        "type": "practice_problem",
                        "title": f"Practice Problem {idx}",
                        "content": item,
                    }
                )
        # Agar already_has_boxes True hai, to ye content upar hi main parser
        # ne (examplebox/theorembox ke tor pe) pakad liya hoga — dobara add
        # karne ki zarurat nahi.

    return chunks


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print('Usage: python3 chunk_notes.py <input.tex> "<course name>" "<chapter name>" <output.json>')
        sys.exit(1)

    input_file, course_name, chapter_name, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    chunks = parse_tex_notes(input_file, course_name, chapter_name)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Done! {len(chunks)} chunks bana kar {output_file} mein save kar diye.")
    print("\nType-wise breakdown:")
    types = {}
    for c in chunks:
        types[c["type"]] = types.get(c["type"], 0) + 1
    for t, count in types.items():
        print(f"  {t}: {count}")
