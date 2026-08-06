"""
core.py
---------
Ye module app.py ki saari "business logic" rakhta hai — Streamlit se
bilkul alag (koi `import streamlit` nahi yahan). Wajah simple hai:
agar logic Streamlit ke andar hi lipta rahe to use bina live app chalaye
test nahi kiya ja sakta. Yahan alag rakhne se:

  1. `tests/` folder mein har function directly test ho sakta hai
     (bina API key, bina browser, bina Streamlit server ke).
  2. app.py sirf "UI wiring" reh jata hai — chhota aur parhne mein aasan.

CHANGELOG (pichle review mein jo bugs mile, unke fixes yahan hain):
  - verify_computation(): pehle sirf positive numbers (1.5-4.5) se sample
    hota tha, isliye "sqrt(x**2) == x" jaisi GALAT simplification bhi
    "verified True" keh deta tha (kyunke positive x ke liye ye sach hai,
    lekin negative x ke liye jhoot). Ab negative, positive, aur zero ke
    paas — teenon regions se sample karte hain.
  - embed-safe text truncation add ki (MAX_EMBED_CHARS) taake koi chunk
    gemini-embedding-001 ki ~2048-token input limit se upar na jaye.
"""

import random
import re
from typing import Optional

import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr
from pydantic import BaseModel

# ------------------------------------------------------------------
# Config constants (yahan rakhe hain taake app.py aur scripts dono
# inhe import kar sakein aur kabhi out-of-sync na hon)
# ------------------------------------------------------------------
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.6-flash"

MAX_HISTORY_TURNS = 3
MAX_HISTORY_ANSWER_CHARS = 400
EMBEDDING_TIMEOUT_SECONDS = 15
GENERATION_TIMEOUT_SECONDS = 35
NOT_FOUND_THRESHOLD = 0.35
FALLBACK_THRESHOLD = 0.50
CACHE_SIMILARITY_THRESHOLD = 0.93
REPEAT_THRESHOLD = 0.75

# gemini-embedding-001 ki input limit ~2048 tokens hai. Rough estimate
# (English/LaTeX mix ke liye) ~1 token ≈ 3.3 characters — isliye safe
# margin ke saath character-budget rakhte hain, taake koi bada chunk
# silently truncate ya reject na ho.
MAX_EMBED_CHARS = 6500


class TutorAnswer(BaseModel):
    english: str
    roman_urdu: str
    grounding: str  # "direct_from_notes", "adapted_by_ai", ya "not_found"
    computation_expression: Optional[str] = None
    computation_result: Optional[str] = None
    visual_type: Optional[str] = None
    visual_title: Optional[str] = None
    visual_expressions: Optional[list[str]] = None
    visual_x_min: Optional[float] = None
    visual_x_max: Optional[float] = None
    visual_vectors: Optional[list[list[float]]] = None
    visual_nodes: Optional[list[str]] = None
    visual_edges: Optional[list[list[str]]] = None


SYSTEM_INSTRUCTION = """You are a patient teaching assistant for undergraduate
math courses (Linear Algebra, Calculus, Number Theory, Discrete Mathematics).
Answer using ONLY the course notes provided as context — do not use outside
knowledge, and do not invent formulas or examples not in the notes. If the
notes don't contain enough information, say so honestly in both languages
instead of guessing.

You will also be given the recent conversation history. If the student's new
question is a follow-up (e.g. "give an example", "simplify that", "explain
more"), use the history to understand what they are referring to.

Always respond with:
1. "english": a clear, simple English explanation. Do NOT use LaTeX/
   backslash commands here (like \\times, \\equiv, \\pmod, \\frac, \\buildrel)
   — this field is shown as plain text, not rendered as math. Use plain
   words or standard symbols instead (×, ÷, ≡, ², √, "mod"), e.g. write
   "2 × 3 ≡ 1 (mod 5)" instead of "2 \\times 3 \\equiv 1 \\pmod{5}".
2. "roman_urdu": the same explanation in Roman Urdu, mixing in English math
   terms naturally the way a Pakistani teacher would. Same rule as above —
   plain symbols only, no backslash/LaTeX commands. Keep math notation
   simple and readable, not typeset.
3. "grounding": exactly one of:
   - "direct_from_notes": your answer directly follows a definition, theorem,
     or worked example in the notes, with the same or very similar numbers.
   - "adapted_by_ai": the question uses different numbers/values/setup than
     the notes, so you had to compute the specific result yourself. Even a
     small change in numbers counts as "adapted_by_ai" — be strict and honest.
4. If grounding is "adapted_by_ai" AND the question is a well-defined
   calculation (a derivative, integral, determinant, solving an equation,
   simplifying an expression, a congruence, a counting/combinatorics result,
   etc.), also provide:
   - "computation_expression": the calculation as a valid SymPy-parseable
     Python expression, e.g. "diff(x**2*sin(x), x)" or
     "Matrix([[1,2],[3,4]]).det()".
   - "computation_result": your final answer as a valid SymPy-parseable
     expression, e.g. "2*x*sin(x) + x**2*cos(x)".
   CRITICAL: "computation_result" must be the exact same mathematical result
   you state as your final answer in the english/roman_urdu explanation — do
   not compute or phrase it separately. It will be shown to the student as
   the definitive final answer, so it must match what you claim in the
   explanation, not a re-derived or differently-simplified version of it.
   Only fill these two fields if you're confident they are valid syntax;
   otherwise leave both as null. Leave both null for conceptual/proof/
   definition questions.
5. If the student explicitly asks to see/show/graph/plot/visualize/draw
   something, OR the question is inherently visual (curve sketching, vector
   geometry, a graph-theory structure with vertices/edges), fill "visual_type"
   with exactly one of:
   - "function": provide "visual_expressions" (list of SymPy-parseable
     expressions in terms of x, e.g. ["x**2 - 3*x + 2"]), and "visual_x_min"/
     "visual_x_max" (a sensible domain, e.g. -10 to 10 unless the question
     implies otherwise).
   - "vector": provide "visual_vectors" as a list of [x, y] pairs.
   - "graph_network": provide "visual_nodes" (list of labels) and
     "visual_edges" (list of [node, node] pairs) for a graph-theory diagram.
   Always also provide "visual_title". You may generate a visual even if it
   is not present in the notes — a well-defined mathematical graph is not
   "invented content" the way a fabricated fact would be. If no visual is
   needed, leave "visual_type" as null."""


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------
def cosine_sim_matrix(query_vec, matrix):
    query_vec = np.array(query_vec, dtype=float)
    norm = np.linalg.norm(query_vec)
    if norm == 0:
        raise ValueError("query_vec ka norm zero hai — embedding khali/invalid lagti hai.")
    query_norm = query_vec / norm
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_norms[matrix_norms == 0] = 1e-12  # div-by-zero se bachne ke liye
    matrix_norm = matrix / matrix_norms
    return matrix_norm @ query_norm


def top_chunks_from_vector(query_vec, kb, embeddings_matrix, course_filter=None, top_k=4):
    """Pure computation, koi API call nahi yahan — isliye baar baar chalana sasta hai."""
    if course_filter:
        indices = np.array([i for i, c in enumerate(kb) if c["course"] == course_filter])
    else:
        indices = np.arange(len(kb))
    if len(indices) == 0:
        return []
    sims = cosine_sim_matrix(query_vec, embeddings_matrix[indices])
    order = np.argsort(sims)[::-1][:top_k]
    top_indices = indices[order]
    top_sims = sims[order]
    return [dict(kb[i], similarity=float(s)) for i, s in zip(top_indices, top_sims)]


def decide_retrieval_strategy(best_sim, cross_best_sim, cross_course=None):
    """FIX for a design bug from the review: pehle system har borderline
    sawal ke liye AI ko apne (kam-confidence) course ke chunks bhej deta
    tha, chahe doosre course mein match kitna hi behtar kyun na ho — matlab
    quota bhi waste hoti thi aur AI ko irrelevant context milta tha.

    Ab teen possible outcomes hain:
      - ("not_found", None): kahin bhi confidence itni kam hai ke AI ko
        call karne ka koi fayda nahi.
      - ("cross_course_redirect", course): apna course clearly galat
        lagta hai (doosre course mein match saaf tor pe kaafi behtar hai)
        — AI ko galat context ke saath call hi nahi karte, seedha redirect
        dikhate hain. Quota bachti hai, jawab bhi galat context se nahi
        aata.
      - ("answer", soft_suggestion): apne course se answer generate karo.
        `soft_suggestion` ya to None hoga, ya ek course ka naam (agar
        doosre course mein thoda behtar match mila lekin itna clear-cut
        nahi ke redirect kiya jaye) — is case mein purane behaviour ki
        tarah sirf ek info-box dikhayenge, context wahi apne course ka
        rahega.
    """
    if max(best_sim, cross_best_sim) < NOT_FOUND_THRESHOLD:
        return "not_found", None

    if (
        cross_course is not None
        and best_sim < FALLBACK_THRESHOLD
        and cross_best_sim >= FALLBACK_THRESHOLD
        and cross_best_sim > best_sim + 0.15
    ):
        return "cross_course_redirect", cross_course

    soft_suggestion = None
    if (
        cross_course is not None
        and best_sim < FALLBACK_THRESHOLD
        and cross_best_sim > best_sim + 0.1
    ):
        soft_suggestion = cross_course

    return "answer", soft_suggestion


# LaTeX/math command names that Gemini realistically might use in an
# undergrad Linear Algebra / Calculus / Discrete Math / Number Theory
# context. Deliberately a curated list (not a blanket "any 2+ letters
# after a backslash" regex) — a blanket regex would also incorrectly
# "fix" a LEGITIMATE `\n` (real newline) immediately followed by an
# ordinary word (e.g. "\nNext step..."), corrupting correct output.
# Matching only known command names avoids that false positive while
# still catching the realistic set of LaTeX macros that cause this bug.
_KNOWN_LATEX_COMMANDS = (
    "times|div|pm|mp|cdot|ldots|cdots|vdots|ddots|"
    "leq|geq|neq|ne|approx|equiv|pmod|bmod|mod|"
    "frac|sqrt|sum|int|prod|lim|infty|partial|nabla|binom|choose|"
    "alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|"
    "mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|"
    "Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|"
    "buildrel|over|overline|underline|bar|hat|vec|dot|ddot|widehat|widetilde|"
    "in|notin|subset|subseteq|supset|supseteq|cup|cap|emptyset|"
    "forall|exists|nexists|rightarrow|leftarrow|Rightarrow|Leftarrow|"
    "leftrightarrow|Leftrightarrow|to|mapsto|"
    "mathbb|mathcal|mathrm|mathbf|text|boxed|left|right|quad|qquad|"
    "det|dim|ker|deg|gcd|lcm|sin|cos|tan|cot|sec|csc|ln|log|exp"
)
_LATEX_ESCAPE_REPAIR_RE = re.compile(r"(?<!\\)\\(" + _KNOWN_LATEX_COMMANDS + r")\b")


def repair_json_escaping(raw_json_text):
    """PRODUCTION BUG FIX (Aug 2026): Gemini kabhi kabhi apne "english"/
    "roman_urdu" jawab mein LaTeX commands (jaise \\times, \\buildrel,
    \\pmod) likhta hai, lekin JSON string ke andar backslash ko DOUBLE
    likhna zaroori hota hai (\\\\times). Jab model ye bhool jata hai:
      - Agar backslash ke baad wala letter JSON ka valid escape-char ho
        (t, b, n, r, f) — to wo silently ek control-character (tab,
        backspace, waghera) ban jata hai, aur pehla letter GHAYAB ho jata
        hai jab display hota hai. Isi wajah se production mein "\\times"
        "imes" ban gaya tha, aur "\\buildrel" "uildrel" ban gaya tha.
      - Agar letter koi aur ho (jaise \\pmod ka 'p', \\over ka 'o') — to
        JSON parsing SEEDHA CRASH ho jati hai (poora jawab lost, student
        ko sirf "system busy" dikhta hai).

    Ye function raw JSON text ko PARSE karne se PEHLE fix karta hai:
    ek curated LaTeX-command list (dekhein _KNOWN_LATEX_COMMANDS) ke
    against match kar ke, un exact command-names ko double-escape kar
    deta hai. Jaan-boojh kar blanket "\\ + 2 ya zyada letters" regex NAHI
    use kiya — wo ek GENUINE `\\n` (real newline) ko bhi galat samajh
    leta agar uske turant baad koi ordinary word ho (jaise "\\nNext step
    dekhein..."), jo naya bug bana deta. Curated list se ye false-positive
    nahi hota, aur phir bhi wahi common LaTeX commands cover ho jate hain
    jo is context (undergrad math) mein realistically aane ka chance hai.
    """
    return _LATEX_ESCAPE_REPAIR_RE.sub(r"\\\\\1", raw_json_text)


def check_repeated_confusion(query_vec, history):
    """Check karta hai ke isi session mein pehle bhi (alag lafzon mein) yahi
    poocha ja chuka hai — repeated confusion ka signal."""
    query_vec = np.array(query_vec, dtype=float)
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return False
    for turn in history:
        prev_vec = turn.get("query_vec")
        if prev_vec is None:
            continue
        prev_vec = np.array(prev_vec, dtype=float)
        prev_norm = np.linalg.norm(prev_vec)
        if prev_norm == 0:
            continue
        sim = float(np.dot(query_vec, prev_vec) / (q_norm * prev_norm))
        if sim >= REPEAT_THRESHOLD:
            return True
    return False


# ------------------------------------------------------------------
# Visual-demand / cache-safety helpers
# ------------------------------------------------------------------
VISUAL_KEYWORDS = ["graph", "plot", "visuali", "draw", "diagram", "chart", "sketch", "picture", "show me"]


def wants_visual(text):
    """Check karta hai ke sawal mein graph/visual ki demand hai ya nahi —
    cache-safety ke liye zaroori, warna purana text-only jawab mil sakta hai
    jab student ne is baar graph maanga ho."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in VISUAL_KEYWORDS)


def math_signature(text):
    """Sawal ke saare numbers (order ke saath) nikalta hai. Cache hit dene
    se pehle ye match karna zaroori hai.

    NOTE (known limitation, dashboard mein bhi documented hai): ye sirf
    digits ki *sequence* dekhta hai, unki structural position (kaunsi row,
    kaunsa column) nahi. Isliye do alag-alag structured inputs jo same
    order mein same numbers use karein (rare) galti se same signature de
    sakte hain. CACHE_SIMILARITY_THRESHOLD (0.93) itna high hai ke aisa
    sirf tab hoga jab dono sawal already semantically bohat close hon —
    is liye risk chhota hai, lekin zero nahi. Agar aapke course mein matrix
    /system-of-equations wale sawal bohat aate hain, to
    `cache_store.py` ke `structural_signature()` helper ka use karein
    (neeche dekhein) jo bracket-nesting bhi capture karta hai.
    """
    return re.findall(r"-?\d+\.?\d*", text)


def structural_signature(text):
    """math_signature() se ek qadam aage — har number ke saath uska poora
    'sibling-path' record karta hai (kis bracket ke andar, kaunse comma-
    position par), taake [[1,2],[3,4]] aur [[1,3],[2,4]] (transpose) jaisi
    cheezein alag pehchani jayein — dashboard.py mein pehle isi exact
    limitation ka zikar tha ("caching safety-check compares numbers but
    not their row/column arrangement"), ab structural_signature() ise
    resolve karta hai.

    Har number ki 'path' ek tuple hai jaisे (0, 1, 0) — matlab: outermost
    bracket ke andar 2nd comma-group, uske andar 1st bracket, uske andar
    1st comma-group. Do numbers sirf tab match karenge jab unka path aur
    value dono same hon.
    """
    stack = [0]  # har nesting-level par "kaunsa comma-separated group chal raha hai"
    signature = []
    i = 0
    number_re = re.compile(r"-?\d+\.?\d*")
    while i < len(text):
        ch = text[i]
        if ch in "([{":
            stack.append(0)
            i += 1
            continue
        if ch in ")]}":
            if len(stack) > 1:
                stack.pop()
            i += 1
            continue
        if ch == ",":
            stack[-1] += 1
            i += 1
            continue
        m = number_re.match(text, i)
        if m:
            signature.append((tuple(stack), m.group(0)))
            i = m.end()
            continue
        i += 1
    return signature


# ------------------------------------------------------------------
# SymPy verification — FIXED to sample across negative + positive + near-zero
# ------------------------------------------------------------------
def _sample_value():
    """Ek random value deta hai jo teen regions mein se kisi ek se aata hai:
    negative, positive, ya zero ke qareeb (lekin exactly 0 nahi, taake
    1/x jaisi expressions crash na karein). Pehle sirf [1.5, 4.5] se
    sample hota tha — is se koi bhi identity jo sirf negative x ke liye
    galat ho (jaise sqrt(x**2) == x, ya log(x**2) == 2*log(x)) "verified
    True" ban jati thi. Ab teenon regions cover hote hain."""
    region = random.choice(["negative", "positive", "near_zero"])
    if region == "negative":
        return random.uniform(-4.5, -1.5)
    elif region == "positive":
        return random.uniform(1.5, 4.5)
    else:
        # 0 ke qareeb lekin 0 nahi — sign +ve ya -ve dono ho sakta hai
        magnitude = random.uniform(0.05, 0.4)
        return magnitude if random.random() < 0.5 else -magnitude


def verify_computation(expression_str, result_str, trials=8):
    """True/False/None laut ata hai — None matlab verify nahi ho saka
    (inconclusive). Pehle symbolic simplify try karta hai; agar wo fully
    resolve na ho paye, numerical sampling se doosri baar check karta hai.

    FIX: sampling ab negative, positive, aur near-zero — teenon regions se
    hoti hai (pehle sirf [1.5, 4.5], yaani sirf positive, jis se domain-
    sensitive galtiyan (sqrt, log, Abs waghera involve karne wali) pakdi
    nahi ja rahi thi). Trials bhi 5 se 8 kar diye hain taake mixed-region
    sampling ke bawajood confidence high rahe.
    """
    if not expression_str or not result_str:
        return None
    try:
        lhs = parse_expr(expression_str)
        rhs = parse_expr(result_str)
        diff = sympy.simplify(lhs - rhs)
        if diff == 0:
            return True
        free_vars = diff.free_symbols
        if not free_vars:
            return abs(complex(diff)) < 1e-9
        for _ in range(trials):
            subs = {v: _sample_value() for v in free_vars}
            try:
                val = diff.evalf(subs=subs)
                val_complex = complex(val)
            except (TypeError, ValueError):
                # Function domain mein sample point valid nahi tha
                # (jaise log(negative) real branch mein undefined) —
                # is trial ko skip karo, isse verification fail mat karo.
                continue
            if abs(val_complex) > 1e-6:
                return False
        return True
    except Exception:
        return None


def truncate_for_embedding(text, max_chars=MAX_EMBED_CHARS):
    """Embedding model ki input-limit se bachne ke liye text ko safe
    length tak truncate karta hai. Poori chunk-splitting karne ke bajaye
    (jo sirf ~0.05% content ko affect karti hai) simple truncation kaafi
    hai — bas silent na ho, isliye caller ko pata chalta hai (return value
    ka doosra hissa)."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


# ------------------------------------------------------------------
# Conversation history formatting
# ------------------------------------------------------------------
def format_history(history):
    if not history:
        return "(This is the first question of the conversation)"
    lines = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        answer_text = turn["answer"].english
        if len(answer_text) > MAX_HISTORY_ANSWER_CHARS:
            answer_text = answer_text[:MAX_HISTORY_ANSWER_CHARS] + "... (truncated)"
        lines.append(f"Student: {turn['question']}")
        lines.append(f"Assistant: {answer_text}")
    return "\n".join(lines)


def build_generation_prompt(question, chunks, history):
    context = "\n\n---\n\n".join(
        f"[{c['course']} | {c['chapter']} | {c['section']} | {c['title']}]\n{c['content']}"
        for c in chunks
    )
    return (
        f"Conversation so far:\n{format_history(history)}\n\n"
        f"Course notes context:\n{context}\n\n"
        f"Student's new question: {question}"
    )


# ------------------------------------------------------------------
# Visual rendering (matplotlib/sympy/networkx — no Streamlit dependency)
# ------------------------------------------------------------------
def render_visual(answer):
    """Answer mein diye visual_type ke hisaab se matplotlib figure banata
    hai. Koi arbitrary code execute nahi hoti — sirf sympy expressions
    safely evaluate hoti hain, isliye ye AI-generated input ke saath bhi
    safe hai. Matplotlib ko yahan lazily import karte hain taake is module
    ko import karna (tests ke liye) bhari na ho agar matplotlib chahiye
    hi na ho."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    if not answer.visual_type:
        return None
    try:
        if answer.visual_type == "function" and answer.visual_expressions:
            x_min = answer.visual_x_min if answer.visual_x_min is not None else -10
            x_max = answer.visual_x_max if answer.visual_x_max is not None else 10
            fig, ax = plt.subplots(figsize=(6, 4))
            x_vals = np.linspace(x_min, x_max, 400)
            x_sym = sympy.Symbol("x")
            for expr_str in answer.visual_expressions:
                expr = parse_expr(expr_str)
                f = sympy.lambdify(x_sym, expr, modules=["numpy"])
                y_vals = f(x_vals)
                ax.plot(x_vals, y_vals, label=f"${sympy.latex(expr)}$")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_title(answer.visual_title or "")
            return fig

        elif answer.visual_type == "vector" and answer.visual_vectors:
            fig, ax = plt.subplots(figsize=(5, 5))
            for i, v in enumerate(answer.visual_vectors):
                ax.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1, label=f"v{i+1}")
            flat = [c for v in answer.visual_vectors for c in v] or [1, -1]
            lim = max(abs(min(flat)), abs(max(flat))) * 1.3 or 5
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=0.5)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_aspect("equal")
            ax.set_title(answer.visual_title or "")
            return fig

        elif answer.visual_type == "graph_network" and answer.visual_nodes:
            G = nx.Graph()
            G.add_nodes_from(answer.visual_nodes)
            G.add_edges_from(answer.visual_edges or [])
            fig, ax = plt.subplots(figsize=(5, 5))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, ax=ax, with_labels=True, node_color="#a8d5ff", node_size=800, font_size=10)
            ax.set_title(answer.visual_title or "")
            return fig
    except Exception:
        return None
    return None
