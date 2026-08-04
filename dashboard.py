"""
dashboard.py
--------------
Teacher-only view: topics, trends, gaps in notes, verification results,
cache effectiveness, aur repeated-confusion signals.

Chalane ka tareeqa:
    streamlit run dashboard.py
"""

import os

import pandas as pd
import streamlit as st

from auth_guard import AttemptState, is_locked_out, record_attempt, seconds_remaining
from cache_store import QACache

try:
    TEACHER_PASSWORD = st.secrets["TEACHER_PASSWORD"]
except (FileNotFoundError, KeyError):
    from config import TEACHER_PASSWORD

LOG_FILE = "logs/question_log.csv"
CACHE_DB_FILE = "cache/qa_cache.db"
WEAK_MATCH_THRESHOLD = 0.55

st.set_page_config(page_title="Teacher Dashboard", page_icon="📊")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "teacher_attempts" not in st.session_state:
    st.session_state.teacher_attempts = AttemptState()

if not st.session_state.authenticated:
    st.title("📊 Teacher Dashboard")

    if is_locked_out(st.session_state.teacher_attempts):
        st.error(f"Too many incorrect attempts. Try again in {seconds_remaining(st.session_state.teacher_attempts)}s.")
        st.stop()

    pwd = st.text_input("Password:", type="password")
    if st.button("Login"):
        correct = pwd == TEACHER_PASSWORD
        st.session_state.teacher_attempts = record_attempt(st.session_state.teacher_attempts, correct)
        if correct:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
            st.rerun()
    st.stop()

st.title("📊 Teacher Dashboard")
st.caption("A summary of student questions — see where the class is getting stuck.")

if not os.path.isfile(LOG_FILE):
    st.info("No questions recorded yet. Once students start using the app, data will appear here.")
    st.stop()

try:
    df = pd.read_csv(LOG_FILE)
except Exception:
    st.error(
        "Couldn't read the log file — it may be from an older version of the app with "
        "different columns. Rename or delete `logs/question_log.csv` and it will be "
        "recreated automatically the next time a question is asked."
    )
    st.stop()
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date

# ------------------------------------------------------------------
# Course filter — sab courses ya ek specific course
# ------------------------------------------------------------------
if "course" in df.columns:
    course_options = ["All courses"] + sorted(df["course"].dropna().unique().tolist())
    selected = st.selectbox("Course", course_options)
    if selected != "All courses":
        df = df[df["course"] == selected]

st.divider()

# ------------------------------------------------------------------
# Overall stats
# ------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total questions", len(df))
col2.metric("Active days", df["date"].nunique())
col3.metric("Busiest day", str(df["date"].value_counts().idxmax()) if len(df) else "-")

st.divider()

# ------------------------------------------------------------------
# 1. Topic heatmap
# ------------------------------------------------------------------
st.subheader("Most asked-about topics")
topic_counts = df["matched_section"].value_counts().head(10)
st.bar_chart(topic_counts)
st.caption("Shows which section generates the most doubts — a good place to focus your next lecture or revision session.")

st.divider()

# ------------------------------------------------------------------
# 2. Time trend
# ------------------------------------------------------------------
st.subheader("Questions per day")
daily_counts = df.groupby("date").size()
st.line_chart(daily_counts)
st.caption("A spike before an exam is a good signal it's time for a revision session.")

st.divider()

# ------------------------------------------------------------------
# 3. Gap alert
# ------------------------------------------------------------------
st.subheader("Possible gaps in the notes")
if "similarity" in df.columns:
    weak = df[df["similarity"] < WEAK_MATCH_THRESHOLD].sort_values("timestamp", ascending=False)
    if len(weak) == 0:
        st.success("No weak-match questions found — the notes are covering things well.")
    else:
        st.warning(f"{len(weak)} question(s) where the assistant couldn't find a confident answer in the notes. These topics may need more content:")
        st.dataframe(
            weak[["date", "question", "matched_section", "similarity"]],
            width='stretch',
            hide_index=True,
        )
else:
    st.info("This will populate once new questions are logged with the updated app.")

st.divider()

# ------------------------------------------------------------------
# 4. Grounding + verification transparency
# ------------------------------------------------------------------
if "grounding" in df.columns:
    st.subheader("Answer grounding & verification")
    adapted = df[df["grounding"] == "adapted_by_ai"]
    adapted_pct = (len(adapted) / len(df) * 100) if len(df) else 0

    col_a, col_b = st.columns(2)
    col_a.metric("Directly from notes", f"{100 - adapted_pct:.0f}%")
    col_b.metric("Calculated by AI (new numbers)", f"{adapted_pct:.0f}%")

    if "verified" in df.columns and len(adapted) > 0:
        v = adapted["verified"].astype(str)
        verified_true = (v == "True").sum()
        verified_false = (v == "False").sum()
        inconclusive = len(adapted) - verified_true - verified_false
        col_c, col_d, col_e = st.columns(3)
        col_c.metric("✅ Verified correct", verified_true)
        col_d.metric("⚠️ Verification failed", verified_false)
        col_e.metric("— Not auto-checkable", inconclusive)
        if verified_false > 0:
            st.warning(f"{verified_false} AI-calculated answer(s) failed SymPy verification — worth a manual spot-check:")
            st.dataframe(
                adapted[adapted["verified"].astype(str) == "False"][["date", "question", "matched_section"]],
                width='stretch',
                hide_index=True,
            )
    st.caption(
        "Recommended practice: each week, manually spot-check 20-30 'adapted by AI' answers yourself, "
        "especially ones marked 'not auto-checkable'. The verification sampling now checks negative, "
        "positive, and near-zero values (fixed from an earlier version that only sampled positive "
        "numbers), but SymPy verification is still a safety net, not a guarantee."
    )
    st.divider()

# ------------------------------------------------------------------
# 5. Cache effectiveness + repeated confusion
# ------------------------------------------------------------------
if "from_cache" in df.columns or "repeated_confusion" in df.columns:
    st.subheader("Efficiency & confusion signals")
    col_f, col_g = st.columns(2)
    if "from_cache" in df.columns:
        cache_pct = df["from_cache"].astype(str).eq("True").mean() * 100
        col_f.metric("Answered from cache", f"{cache_pct:.0f}%")
    if "repeated_confusion" in df.columns:
        repeat_pct = df["repeated_confusion"].astype(str).eq("True").mean() * 100
        col_g.metric("Rephrased repeat questions", f"{repeat_pct:.0f}%")
    st.caption("A high 'rephrased repeat' rate means students are asking the same thing in different words within one session — a strong signal that a topic wasn't clear the first time.")
    st.divider()

# ------------------------------------------------------------------
# 6. Cache size (quota-saving visibility — zero-budget setups care about this)
# ------------------------------------------------------------------
if os.path.isfile(CACHE_DB_FILE):
    st.subheader("Answer cache")
    try:
        cache = QACache(CACHE_DB_FILE)
        stats = cache.stats()
        col_h, col_i = st.columns(2)
        col_h.metric("Cached unique Q&A pairs", stats["total_entries"])
        col_i.metric("Courses represented", len(stats["by_course"]))
        st.caption(
            "Every cached entry is a generation API call that future students won't need to "
            "spend quota on. Stored in SQLite (cache/qa_cache.db) — no manual cleanup needed."
        )
    except Exception:
        pass
    st.divider()

# ------------------------------------------------------------------
# Raw data + export
# ------------------------------------------------------------------
with st.expander("View / download full data"):
    st.dataframe(df, width='stretch', hide_index=True)
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False),
        file_name="question_log.csv",
        mime="text/csv",
    )
