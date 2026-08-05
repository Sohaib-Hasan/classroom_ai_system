"""
app.py
--------
Students ke liye chat screen. Features:
- Course selector — hard filter (default), soft cross-course suggestion agar
  selected course mein confidence kam ho (aur agar galat course clearly
  saaf hai, to seedha redirect — AI ko galat context ke saath call nahi
  karte, dekhein core.decide_retrieval_strategy)
- Follow-up conversation memory
- English / Roman Urdu answer toggle (default English)
- Grounding transparency: "direct_from_notes" vs "adapted_by_ai"
- SymPy se AI ke apne calculations ki independent verification (ab negative
  + positive + near-zero domain se sample karta hai — dekhein core.py)
- Q&A caching (SQLite-backed — dekhein cache_store.py) — same/bohat similar
  sawal dobara aaye to purana jawab reuse hota hai
- Isi session mein "dobara wahi confusion, alag lafzon mein" detect karna
- PIN par bar-bar galat koshish pe temporary lockout (auth_guard.py)
- Har asli error `logs/error.log` mein likha jata hai (pehle silently
  gum ho jata tha)
- Generation ke liye optional fallback provider (jaise AgentRouter) —
  agar Gemini free-tier quota exam-week traffic mein khatam ho jaye, to
  automatically doosre provider par switch ho jata hai. Dekhein
  generation_backend.py aur README.md ("Zero-budget setup").

Chalane ka tareeqa:
    streamlit run app.py
"""

import csv
import os
import time
from datetime import datetime

import streamlit as st
import sympy
from google import genai
from sympy.parsing.sympy_parser import parse_expr

from auth_guard import AttemptState, is_locked_out, record_attempt, seconds_remaining
from cache_store import QACache
from core import (
    EMBEDDING_TIMEOUT_SECONDS,
    FALLBACK_THRESHOLD,
    SYSTEM_INSTRUCTION,
    TutorAnswer,
    build_generation_prompt,
    check_repeated_confusion,
    decide_retrieval_strategy,
    render_visual,
    top_chunks_from_vector,
    verify_computation,
)
from embedding_backend import get_backend as get_embedding_backend_impl
from generation_backend import get_generation_backend
from knowledge_base_loader import load_knowledge_base
from logging_setup import get_logger

try:
    # Streamlit Cloud pe deploy hone par yahan se milega (Secrets settings mein set karna hai)
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    CLASS_PIN = st.secrets["CLASS_PIN"]
    EMBEDDING_PROVIDER = st.secrets.get("EMBEDDING_PROVIDER", "gemini")
    GENERATION_FALLBACK_PROVIDER = st.secrets.get("GENERATION_FALLBACK_PROVIDER", None)
    GENERATION_FALLBACK_API_KEY = st.secrets.get("GENERATION_FALLBACK_API_KEY", None)
    GENERATION_FALLBACK_MODEL = st.secrets.get("GENERATION_FALLBACK_MODEL", None)
except (FileNotFoundError, KeyError):
    # Apne computer pe local testing ke liye — config.py se milta hai.
    # Naye optional settings (EMBEDDING_PROVIDER, GENERATION_FALLBACK_*)
    # purani config.py files mein nahi hongi, isliye har ek ko alag se,
    # getattr-jaisi soft-fallback ke saath import karte hain.
    from config import GEMINI_API_KEY, CLASS_PIN
    import config as _config

    EMBEDDING_PROVIDER = getattr(_config, "EMBEDDING_PROVIDER", "gemini")
    GENERATION_FALLBACK_PROVIDER = getattr(_config, "GENERATION_FALLBACK_PROVIDER", None)
    GENERATION_FALLBACK_API_KEY = getattr(_config, "GENERATION_FALLBACK_API_KEY", None)
    GENERATION_FALLBACK_MODEL = getattr(_config, "GENERATION_FALLBACK_MODEL", None)

CACHE_DB_FILE = "cache/qa_cache.db"
LOG_FILE = "logs/question_log.csv"

logger = get_logger()
client = genai.Client(api_key=GEMINI_API_KEY)


# ------------------------------------------------------------------
# Resources (Streamlit process ke liye ek hi baar banti hain)
# ------------------------------------------------------------------
@st.cache_resource
def get_embedding_backend():
    return get_embedding_backend_impl(EMBEDDING_PROVIDER, client=client)


@st.cache_resource
def get_primary_generation_backend():
    return get_generation_backend("gemini", client=client)


@st.cache_resource
def get_fallback_generation_backend():
    if not GENERATION_FALLBACK_PROVIDER:
        return None
    return get_generation_backend(
        GENERATION_FALLBACK_PROVIDER,
        api_key=GENERATION_FALLBACK_API_KEY,
        model=GENERATION_FALLBACK_MODEL,
    )


@st.cache_resource
def get_cache():
    return QACache(CACHE_DB_FILE)


def embed_query_safe(text: str):
    """embedding_backend ko call karta hai, retries + timeout ke saath."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

    backend = get_embedding_backend()

    def _call():
        return backend.embed_query(text)

    last_error = None
    for attempt in range(2):
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_call)
        try:
            return future.result(timeout=EMBEDDING_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            last_error = TimeoutError(f"Embedding call took longer than {EMBEDDING_TIMEOUT_SECONDS}s")
        except Exception as e:  # noqa: BLE001
            last_error = e
        finally:
            executor.shutdown(wait=False)
        if attempt == 0:
            time.sleep(2)
    raise last_error


def generate_answer(question, chunks, history):
    """Primary provider (Gemini) try karta hai. Agar wo fail ho jaye
    (quota khatam, timeout, ya koi bhi error) AUR ek fallback provider
    configure kiya gaya ho (jaise AgentRouter), to usse try karta hai —
    taake exam-week jaisi heavy-traffic situation mein poora system
    down na ho jaye. Fallback na configured ho to seedha primary ka
    error upar propagate ho jata hai (jaisa pehle hota tha)."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

    prompt = build_generation_prompt(question, chunks, history)
    primary = get_primary_generation_backend()

    def _call(backend, timeout):
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(backend.generate, SYSTEM_INSTRUCTION, prompt, TutorAnswer)
        try:
            return future.result(timeout=timeout)
        finally:
            executor.shutdown(wait=False)

    primary_error = None
    for attempt in range(2):
        try:
            return _call(primary, timeout=35)
        except FutureTimeoutError:
            primary_error = TimeoutError("Primary generation call timed out")
        except Exception as e:  # noqa: BLE001
            primary_error = e
        if attempt == 0:
            time.sleep(2)

    fallback = get_fallback_generation_backend()
    if fallback is None:
        raise primary_error

    logger.warning(f"Primary generation provider failed ({primary_error!r}) — trying fallback provider.")
    try:
        return _call(fallback, timeout=45)
    except Exception as fallback_error:  # noqa: BLE001
        logger.error(f"Fallback generation provider also failed: {fallback_error!r}")
        raise primary_error  # student ko wahi (primary) error dikhega, consistent messaging ke liye


# ------------------------------------------------------------------
# Logging (question activity — teacher dashboard ke liye)
# ------------------------------------------------------------------
LOG_COLUMNS = [
    "timestamp", "course", "question", "matched_chapter", "matched_section",
    "similarity", "grounding", "verified", "repeated_confusion", "from_cache",
]


def ensure_log_file_schema():
    if os.path.isfile(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        if first_line != ",".join(LOG_COLUMNS):
            backup_name = LOG_FILE.replace(".csv", f"_old_{int(time.time())}.csv")
            os.rename(LOG_FILE, backup_name)


def log_question(question, course, chunks, answer, verified, repeated, cached):
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
    elif answer.grounding == "cross_course_redirect":
        st.info(f"ℹ️ This looks like it belongs to **{answer.english}** rather than the course you have selected.")
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

    if answer.grounding != "cross_course_redirect":
        if lang_pref == "English":
            st.markdown(answer.english)
        else:
            st.markdown(answer.roman_urdu)

    fig = render_visual(answer)
    if fig:
        st.pyplot(fig)
        if answer.grounding == "direct_from_notes":
            st.caption("📘 Reconstructed from your notes.")
        else:
            st.caption("⚠️ AI-generated example — not from your notes. Please verify key features (roots, critical points, direction).")
    elif answer.visual_type:
        st.caption("(Couldn't render the graph for this one — try rephrasing your question.)")

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
# bachne ke liye (warna quota class se bahar bhi khatam ho sakti hai).
# Ab attempt-lockout bhi hai (dekhein auth_guard.py) taake koi script se
# PIN brute-force na kar sake.
if "pin_ok" not in st.session_state:
    st.session_state.pin_ok = False
if "pin_attempts" not in st.session_state:
    st.session_state.pin_attempts = AttemptState()

if not st.session_state.pin_ok:
    st.title("📐 Doubt Clearing Assistant")
    st.caption("Ask your teacher for the class PIN to continue.")

    if is_locked_out(st.session_state.pin_attempts):
        st.error(f"Too many incorrect attempts. Try again in {seconds_remaining(st.session_state.pin_attempts)}s.")
        st.stop()

    pin = st.text_input("Class PIN:", type="password")
    if st.button("Enter"):
        correct = pin == CLASS_PIN
        st.session_state.pin_attempts = record_attempt(st.session_state.pin_attempts, correct)
        if correct:
            st.session_state.pin_ok = True
            st.rerun()
        else:
            st.error("Incorrect PIN.")
            st.rerun()
    st.stop()

st.title("📐 Doubt Clearing Assistant")
st.caption("Ask a question, and follow-up as much as you like — the assistant remembers the conversation.")

kb, embeddings_matrix, courses = load_knowledge_base()
cache = get_cache()
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
        if turn["answer"].grounding not in ("not_found", "cross_course_redirect"):
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
                query_vec = embed_query_safe(search_text)

                chunks = top_chunks_from_vector(query_vec, kb, embeddings_matrix, course_filter=selected_course)
                best_sim = chunks[0]["similarity"] if chunks else 0

                # Apne course mein confidence kam ho to doosre courses bhi check karo
                cross_course = None
                cross_best_sim = 0
                if best_sim < FALLBACK_THRESHOLD:
                    all_chunks = top_chunks_from_vector(query_vec, kb, embeddings_matrix, course_filter=None)
                    other = next((c for c in all_chunks if c["course"] != selected_course), None)
                    if other:
                        cross_course = other["course"]
                        cross_best_sim = other["similarity"]

                repeated = check_repeated_confusion(query_vec, history)
                strategy, extra = decide_retrieval_strategy(best_sim, cross_best_sim, cross_course)

                cross_suggestion = None
                cached_hit = None

                if strategy == "not_found":
                    answer = TutorAnswer(
                        english="I couldn't find this in your course notes. Please check with your teacher, or try rephrasing your question.",
                        roman_urdu="Ye mujhe aapke course notes mein nahi mila. Apne teacher se poochein, ya sawal ko dobara likh kar try karein.",
                        grounding="not_found",
                    )
                    verified = None

                elif strategy == "cross_course_redirect":
                    # FIX (bug jo review mein mila tha): pehle is case mein
                    # bhi AI ko apne (kam-confidence) course ke chunks bhej
                    # diye jate the — quota waste hoti thi aur context bhi
                    # galat hota tha. Ab AI ko call hi nahi karte.
                    answer = TutorAnswer(
                        english=extra,
                        roman_urdu=extra,
                        grounding="cross_course_redirect",
                    )
                    verified = None
                    chunks = []

                else:  # "answer"
                    cross_suggestion = extra
                    if not history:  # follow-up par cache use nahi karte, context-dependent hota hai
                        cached_hit = cache.find_cached_answer(selected_course, question, query_vec)

                    if cached_hit:
                        answer = TutorAnswer(**cached_hit["answer"])
                        chunks = cached_hit["chunks"]
                    else:
                        answer = generate_answer(question, chunks, history)
                        cache.save_to_cache(selected_course, question, answer, chunks, query_vec)

                    verified = verify_computation(answer.computation_expression, answer.computation_result)

                log_question(question, selected_course, chunks, answer, verified, repeated, cached_hit is not None)

            show_answer(answer, lang_pref)
            if cross_suggestion:
                st.info(f"Related content found in **{cross_suggestion}** — you may want to check there too.")
            if answer.grounding not in ("not_found", "cross_course_redirect"):
                with st.expander("Sources used from notes"):
                    for c in chunks:
                        st.markdown(f"- **{c['title']}** — *{c['chapter']}, {c['section']}*")
        except Exception:
            # FIX (bug jo review mein mila tha): pehle yahan exception
            # silently gum ho jati thi — koi log, koi traceback nahi. Ab
            # poora traceback logs/error.log mein likha jata hai.
            logger.exception(f"Failed to answer question in course={selected_course!r}: {question!r}")
            st.error("The system is busy right now — please wait a few seconds and try again.")
            st.stop()

    st.session_state.messages.append(
        {"question": question, "answer": answer, "chunks": chunks, "query_vec": query_vec}
    )
