"""
dashboard.py
--------------
Teacher-only view: topics, trends, gaps in notes, AND now — verification
results, cache effectiveness, and repeated-confusion signals.

Chalane ka tareeqa:
    streamlit run dashboard.py
"""

import os
import pandas as pd
import streamlit as st

try:
    TEACHER_PASSWORD = st.secrets["TEACHER_PASSWORD"]
except (FileNotFoundError, KeyError):
    from config import TEACHER_PASSWORD

LOG_FILE = "logs/question_log.csv"
WEAK_MATCH_THRESHOLD = 0.55

st.set_page_config(page_title="Teacher Dashboard", page_icon="📊")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("📊 Teacher Dashboard")
    pwd = st.text_input("Password:", type="password")
    if st.button("Login"):
        if pwd == TEACHER_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
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
            use_container_width=True,
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
                use_container_width=True,
                hide_index=True,
            )
    st.caption("Recommended practice: each week, manually spot-check 20-30 'adapted by AI' answers yourself, especially ones marked 'not auto-checkable' — and specifically include a few matrix/determinant-type questions, since the caching safety-check compares numbers but not their row/column arrangement (e.g. [[1,2],[3,4]] vs [[1,3],[2,4]] would look identical to it).")
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
# Raw data + export
# ------------------------------------------------------------------
with st.expander("View / download full data"):
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False),
        file_name="question_log.csv",
        mime="text/csv",
    )