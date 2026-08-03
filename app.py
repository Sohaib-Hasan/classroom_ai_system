"""
app.py
--------
Students ke liye chat screen. Features:
- Course selector — hard filter (default), soft cross-course suggestion agar
  selected course mein confidence kam ho
- Follow-up conversation memory
- English / Roman Urdu answer toggle (default English)
- Grounding transparency: "direct_from_notes" vs "adapted_by_ai"
- SymPy se AI ke apne calculations ki independent verification
- Q&A caching — same/bohat similar sawal dobara aaye to purana jawab reuse
  hota hai (API call bachti hai, rate-limit pressure kam hoti hai)
- Isi session mein "dobara wahi confusion, alag lafzon mein" detect karna

Chalane ka tareeqa:
    streamlit run app.py
"""

import json
import os
import csv
import re
import time
import random
import threading
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
import streamlit as st
import sympy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sympy.parsing.sympy_parser import parse_expr
from google import genai
from google.genai import types
from pydantic import BaseModel

try:
    # Streamlit Cloud pe deploy hone par yahan se milega (Secrets settings mein set karna hai)
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    CLASS_PIN = st.secrets["CLASS_PIN"]
except (FileNotFoundError, KeyError):
    # Apne computer pe local testing ke liye — config.py se milta hai
    from config import GEMINI_API_KEY, CLASS_PIN

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-3.6-flash"
MAX_HISTORY_TURNS = 3   # follow-up ka prompt halka rakhne ke liye — zyada purani history se koi khaas fayda nahi, sirf prompt bada hota hai
MAX_HISTORY_ANSWER_CHARS = 400  # purane answers ka sirf khulasa bhejte hain, poora nahi — taake conversation lambi ho to bhi prompt bara na ho
EMBEDDING_TIMEOUT_SECONDS = 15   # embeddings chhoti/fast calls hain
GENERATION_TIMEOUT_SECONDS = 35  # answers ka prompt bara ho sakta hai (follow-up history + notes context + kai fields generate karne hain), isliye zyada waqt dete hain
NOT_FOUND_THRESHOLD = 0.35 # is se neeche confidence pe seedha "notes mein nahi mila" bol dete hain, AI ko call hi nahi karte
FALLBACK_THRESHOLD = 0.50  # is se kam confidence pe cross-course suggestion dikhega
CACHE_SIMILARITY_THRESHOLD = 0.93  # itni similarity pe purana jawab reuse hoga
REPEAT_THRESHOLD = 0.75            # isi session mein "dobara wahi confusion" ka threshold
CACHE_FILE = "cache/qa_cache.json"
LOG_FILE = "logs/question_log.csv"

client = genai.Client(api_key=GEMINI_API_KEY)


def call_with_timeout(fn, timeout):
    """Kisi bhi function (API call) ko chalata hai, lekin agar itne second mein
    jawab na aaye, TimeoutError raise kar deta hai — chahe underlying library
    (google-genai) khud kuch bhi kare. Ye hang-proof guarantee deta hai.
    NOTE: shutdown(wait=False) zaroori hai — warna executor khud hi ruke huay
    thread ka intezar karta reh jata hai, aur timeout ka koi fayda nahi hota."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        raise TimeoutError(f"API call took longer than {timeout} seconds")
    finally:
        executor.shutdown(wait=False)


class TutorAnswer(BaseModel):
    english: str
    roman_urdu: str
    grounding: str  # "direct_from_notes" or "adapted_by_ai"
    computation_expression: Optional[str] = None
    computation_result: Optional[str] = None
    # Visual/graph generation — sirf tab bharte hain jab student ne demand ki ho
    # ya sawal ki nature inherently visual ho (curve, vector, graph-theory diagram)
    visual_type: Optional[str] = None  # "function", "vector", "graph_network", ya null
    visual_title: Optional[str] = None
    visual_expressions: Optional[list[str]] = None      # "function" ke liye
    visual_x_min: Optional[float] = None
    visual_x_max: Optional[float] = None
    visual_vectors: Optional[list[list[float]]] = None  # "vector" ke liye, jaise [[2,3],[-1,4]]
    visual_nodes: Optional[list[str]] = None             # "graph_network" ke liye
    visual_edges: Optional[list[list[str]]] = None       # jaise [["A","B"],["B","C"]]


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
1. "english": a clear, simple English explanation.
2. "roman_urdu": the same explanation in Roman Urdu, mixing in English math
   terms naturally the way a Pakistani teacher would. Keep math notation
   (like $A^{-1}$) exactly as written in the notes in both versions.
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
# Knowledge base
# ------------------------------------------------------------------
@st.cache_resource
def load_knowledge_base():
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        kb = json.load(f)
    embeddings_matrix = np.array([item["embedding"] for item in kb])
    courses = sorted(set(c["course"] for c in kb))
    return kb, embeddings_matrix, courses


def cosine_sim_matrix(query_vec, matrix):
    query_vec = np.array(query_vec)
    query_norm = query_vec / np.linalg.norm(query_vec)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix_norm @ query_norm


def embed_query(text, retries=2):
    def _call():
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return np.array(result.embeddings[0].values)

    last_error = None
    for attempt in range(retries):
        try:
            return call_with_timeout(_call, EMBEDDING_TIMEOUT_SECONDS)
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(2)
    raise last_error


def top_chunks_from_vector(query_vec, kb, embeddings_matrix, course_filter=None, top_k=4):
    """Pure computation, koi API call nahi yahan — isliye baar baar chalana sasta hai."""
    if course_filter:
        indices = np.array([i for i, c in enumerate(kb) if c["course"] == course_filter])
    else:
        indices = np.arange(len(kb))
    sims = cosine_sim_matrix(query_vec, embeddings_matrix[indices])
    order = np.argsort(sims)[::-1][:top_k]
    top_indices = indices[order]
    top_sims = sims[order]
    return [dict(kb[i], similarity=float(s)) for i, s in zip(top_indices, top_sims)]


def check_repeated_confusion(query_vec, history):
    """Check karta hai ke isi session mein pehle bhi (alag lafzon mein) yahi
    poocha ja chuka hai — repeated confusion ka signal."""
    for turn in history:
        prev_vec = turn.get("query_vec")
        if prev_vec is None:
            continue
        sim = float(
            np.dot(query_vec, prev_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(prev_vec))
        )
        if sim >= REPEAT_THRESHOLD:
            return True
    return False


# ------------------------------------------------------------------
# Q&A cache — same/similar sawal dobara aaye to fresh API call na karo
# ------------------------------------------------------------------
@st.cache_resource
def load_cache():
    if os.path.isfile(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


VISUAL_KEYWORDS = ["graph", "plot", "visuali", "draw", "diagram", "chart", "sketch", "picture", "show me"]


def wants_visual(text):
    """Check karta hai ke sawal mein graph/visual ki demand hai ya nahi —
    cache-safety ke liye zaroori, warna purana text-only jawab mil sakta hai
    jab student ne is baar graph maanga ho."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in VISUAL_KEYWORDS)


def math_signature(text):
    """Sawal ke saare numbers (negative sign aur order ke saath) nikalta hai.
    Cache hit dene se pehle ye match karna zaroori hai — sirf text-similarity
    kaafi nahi, kyunke 'differentiate x^2' aur 'differentiate x^3' embedding
    mein bohat close hote hain lekin jawab bilkul different — numbers alag
    hote hain isliye ye check unhe sahi tarah alag pehchan leta hai."""
    return re.findall(r"-?\d+\.?\d*", text)


def find_cached_answer(query_vec, cache, course, question):
    candidates = [c for c in cache if c["course"] == course]
    if not candidates:
        return None
    sims = [
        float(
            np.dot(query_vec, np.array(c["embedding"]))
            / (np.linalg.norm(query_vec) * np.linalg.norm(c["embedding"]))
        )
        for c in candidates
    ]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] < CACHE_SIMILARITY_THRESHOLD:
        return None
    # Semantic similarity high hone ke bawajood, numbers match nahi karte to
    # cache use nahi karte — math mein ye galti mehnga pad sakti hai
    if math_signature(question) != math_signature(candidates[best_idx]["question"]):
        return None
    # Agar is baar graph maanga gaya hai lekin cached jawab mein visual nahi
    # tha, cache use nahi karte — warna purana text-only jawab mil jayega
    if wants_visual(question) and not candidates[best_idx]["answer"].get("visual_type"):
        return None
    return candidates[best_idx]


cache_lock = threading.Lock()  # Streamlit Cloud ek hi process mein sab students serve karta hai —
                                # is lock ke bagair 2 students ka jawab ek hi waqt cache mein save
                                # hone se file corrupt ho sakti thi (local single-user testing mein
                                # ye nazar nahi aata, isliye pakadna mushkil bug hai)


def save_to_cache(cache, question, query_vec, course, answer, chunks):
    with cache_lock:
        cache.append(
            {
                "question": question,
                "embedding": query_vec.tolist(),
                "course": course,
                "answer": answer.model_dump(),
                "chunks": chunks,
            }
        )
        os.makedirs("cache", exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)


# ------------------------------------------------------------------
# SymPy verification — AI ke khud kiye calculation ko independently check karna
# ------------------------------------------------------------------
def verify_computation(expression_str, result_str):
    """True/False/None laut ata hai — None matlab verify nahi ho saka (inconclusive).
    Pehle symbolic simplify try karta hai; agar wo fully resolve na ho paye
    (bohat complex expressions mein aisa hota hai), numerical sampling se
    doosri baar check karta hai — dono hi bohat fast operations hain (koi
    API call nahi, sirf local computation, milliseconds mein hoti hain)."""
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
        for _ in range(5):
            subs = {v: random.uniform(1.5, 4.5) for v in free_vars}
            val = diff.evalf(subs=subs)
            if abs(complex(val)) > 1e-6:
                return False
        return True
    except Exception:
        return None


# ------------------------------------------------------------------
# Conversation + generation
# ------------------------------------------------------------------
def render_visual(answer):
    """Answer mein diye visual_type ke hisaab se matplotlib figure banata hai.
    Koi arbitrary code execute nahi hoti — sirf sympy expressions safely
    evaluate hoti hain (jaisa computation-verification mein bhi hota hai),
    isliye ye AI-generated input ke saath bhi safe hai."""
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


def generate_answer(question, chunks, history, retries=2):
    context = "\n\n---\n\n".join(
        f"[{c['course']} | {c['chapter']} | {c['section']} | {c['title']}]\n{c['content']}"
        for c in chunks
    )
    prompt = (
        f"Conversation so far:\n{format_history(history)}\n\n"
        f"Course notes context:\n{context}\n\n"
        f"Student's new question: {question}"
    )

    def _call():
        interaction = client.interactions.create(
            model=GENERATION_MODEL,
            system_instruction=SYSTEM_INSTRUCTION,
            input=prompt,
            response_format=TutorAnswer.model_json_schema(),
        )
        return TutorAnswer.model_validate_json(interaction.output_text)

    last_error = None
    for attempt in range(retries):
        try:
            return call_with_timeout(_call, GENERATION_TIMEOUT_SECONDS)
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(2)
    raise last_error


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
LOG_COLUMNS = [
    "timestamp", "course", "question", "matched_chapter", "matched_section",
    "similarity", "grounding", "verified", "repeated_confusion", "from_cache",
]


def ensure_log_file_schema():
    """Agar purani CSV ka header naye columns se match na kare (schema
    development ke dauran badal chuki hai), purani file ko rename kar ke
    nayi shuru karte hain — taake pandas parsing error na aaye. Purana data
    kho nahi jata, bas '_old_<timestamp>.csv' naam se safe ho jata hai."""
    if os.path.isfile(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if first_line != ",".join(LOG_COLUMNS):
            backup_name = LOG_FILE.replace(".csv", f"_old_{int(time.time())}.csv")
            os.rename(LOG_FILE, backup_name)


log_lock = threading.Lock()


def log_question(question, course, chunks, answer, verified, repeated, cached):
    with log_lock:
        os.makedirs("logs", exist_ok=True)
        file_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(LOG_COLUMNS)
            top = chunks[0] if chunks else {}
            writer.writerow(
                [
                    datetime.now().isoformat(),
                    course,
                    question,
                    top.get("chapter", ""),
                    top.get("section", ""),
                    round(top.get("similarity", 0), 3),
                    answer.grounding,
                    "" if verified is None else verified,
                    repeated,
                    cached,
                ]
            )


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------
def show_answer(answer, lang_pref):
    verified = None
    if answer.grounding == "not_found":
        st.info("ℹ️ This topic wasn't found in your course notes.")
    elif answer.grounding == "adapted_by_ai":
        verified = verify_computation(answer.computation_expression, answer.computation_result)
        if verified is True:
            st.caption("✅ AI calculated this using new numbers — independently verified with SymPy.")
        elif verified is False:
            st.warning("⚠️ This calculation could not be independently verified and may contain an error — please double-check with your teacher.")
        else:
            st.caption("⚠️ AI calculated this using new numbers — double-check the working.")
    else:
        st.caption("✅ Directly matches an example in your notes.")

    if lang_pref == "English":
        st.markdown(answer.english)
    else:
        st.markdown(answer.roman_urdu)

    fig = render_visual(answer)
    if fig:
        st.pyplot(fig)
        # Graph ka label bhi wahi grounding-signal reuse karta hai jo poore
        # answer ke liye already tay ho chuka hai — koi nayi detection nahi chahiye
        if answer.grounding == "direct_from_notes":
            st.caption("📘 Reconstructed from your notes.")
        else:
            st.caption("⚠️ AI-generated example — not from your notes. Please verify key features (roots, critical points, direction).")
    elif answer.visual_type:
        st.caption("(Couldn't render the graph for this one — try rephrasing your question.)")

    # Verified result ko hi "final answer" ke tor pe dikhate hain — koi alag
    # paraphrase nahi, taake jo verify hua wahi literally student ko nazar aaye
    if answer.computation_result and verified is not None:
        try:
            st.latex(f"\\text{{Final answer: }} {sympy.latex(parse_expr(answer.computation_result))}")
        except Exception:
            st.code(f"Final answer: {answer.computation_result}")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
st.set_page_config(page_title="Doubt Clearing Assistant", page_icon="📐")

# Halka access-gate — enterprise security nahi, sirf casual link-sharing se
# bachne ke liye (warna quota class se bahar bhi khatam ho sakti hai)
if "pin_ok" not in st.session_state:
    st.session_state.pin_ok = False

if not st.session_state.pin_ok:
    st.title("📐 Doubt Clearing Assistant")
    st.caption("Ask your teacher for the class PIN to continue.")
    pin = st.text_input("Class PIN:", type="password")
    if st.button("Enter"):
        if pin == CLASS_PIN:
            st.session_state.pin_ok = True
            st.rerun()
        else:
            st.error("Incorrect PIN.")
    st.stop()

st.title("📐 Doubt Clearing Assistant")
st.caption("Ask a question, and follow-up as much as you like — the assistant remembers the conversation.")

kb, embeddings_matrix, courses = load_knowledge_base()
cache = load_cache()
ensure_log_file_schema()

with st.sidebar:
    st.markdown("**Course**")
    selected_course = st.selectbox("Course", courses, label_visibility="collapsed")

    st.markdown("**Answer language**")
    lang_pref = st.radio(
        "Answer language", ["English", "Roman Urdu"], index=0, label_visibility="collapsed",
    )
    if st.button("Start a new topic"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state or st.session_state.get("course") != selected_course:
    st.session_state.messages = []
    st.session_state.course = selected_course

for turn in st.session_state.messages:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        show_answer(turn["answer"], lang_pref)
        if turn["answer"].grounding != "not_found":
            with st.expander("Sources used from notes"):
                for c in turn["chunks"]:
                    st.markdown(f"- **{c['title']}** — *{c['chapter']}, {c['section']}*")

question = st.chat_input("Type your question here...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your notes..."):
                history = st.session_state.messages
                search_text = question if not history else history[-1]["question"] + " " + question
                query_vec = embed_query(search_text)

                chunks = top_chunks_from_vector(query_vec, kb, embeddings_matrix, course_filter=selected_course)
                best_sim = chunks[0]["similarity"] if chunks else 0

                # Apne course mein confidence kam ho to doosre courses bhi check karo
                cross_suggestion = None
                cross_best_sim = 0
                if best_sim < FALLBACK_THRESHOLD:
                    all_chunks = top_chunks_from_vector(query_vec, kb, embeddings_matrix, course_filter=None)
                    other = next((c for c in all_chunks if c["course"] != selected_course), None)
                    if other:
                        cross_best_sim = other["similarity"]
                        if other["similarity"] > best_sim + 0.1:
                            cross_suggestion = other["course"]

                repeated = check_repeated_confusion(query_vec, history)

                if max(best_sim, cross_best_sim) < NOT_FOUND_THRESHOLD:
                    # Root-cause fix: confidence har jagah itni kam hai ke ye
                    # topic kahin bhi notes mein nahi — seedha bata dete hain,
                    # AI ko call hi nahi karte (na hang ka risk, na kharab jawab)
                    answer = TutorAnswer(
                        english="I couldn't find this in your course notes. Please check with your teacher, or try rephrasing your question.",
                        roman_urdu="Ye mujhe aapke course notes mein nahi mila. Apne teacher se poochein, ya sawal ko dobara likh kar try karein.",
                        grounding="not_found",
                    )
                    verified = None
                    cached_hit = None
                else:
                    cached_hit = None
                    if not history:  # follow-up par cache use nahi karte, context-dependent hota hai
                        cached_hit = find_cached_answer(query_vec, cache, selected_course, question)

                    if cached_hit:
                        answer = TutorAnswer(**cached_hit["answer"])
                        chunks = cached_hit["chunks"]
                    else:
                        answer = generate_answer(question, chunks, history)
                        save_to_cache(cache, question, query_vec, selected_course, answer, chunks)

                    verified = verify_computation(answer.computation_expression, answer.computation_result)

                log_question(question, selected_course, chunks, answer, verified, repeated, cached_hit is not None)

            show_answer(answer, lang_pref)
            if cross_suggestion:
                st.info(f"Related content found in **{cross_suggestion}** — you may want to check there too.")
            if answer.grounding != "not_found":
                with st.expander("Sources used from notes"):
                    for c in chunks:
                        st.markdown(f"- **{c['title']}** — *{c['chapter']}, {c['section']}*")
        except Exception:
            st.error("The system is busy right now — please wait a few seconds and try again.")
            st.stop()

    st.session_state.messages.append(
        {"question": question, "answer": answer, "chunks": chunks, "query_vec": query_vec}
    )