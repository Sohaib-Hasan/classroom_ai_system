"""
question_log_store.py
------------------------
Teacher dashboard ke liye question-activity log. PEHLE ye flat CSV
(`logs/question_log.csv`) mein tha — jo sirf usi Streamlit Cloud
container ke andar visible hota tha. Agar student app (`app.py`) aur
teacher dashboard (`dashboard.py`) ALAG apps ke tor par deploy hon
(jaisa is project mein hua — "plotlab-classroom" aur "plotlab-teacher"
do alag URLs/deployments hain), to har app ka apna, isolated container
hota hai, aur ek app ki likhi CSV file doosre app ko kabhi nazar nahi
aati — isi wajah se dashboard hamesha khali dikh raha tha, chahe
student app mein kitne bhi sawal poochhe gaye hon.

Fix: same shared connection (dekhein db_connection.py — local SQLite ya
Turso) use karte hain jo cache_store.py bhi use karta hai, taake dono
apps SAME database dekh sakein (agar Turso configure ho)."""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS question_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    course TEXT NOT NULL,
    question TEXT NOT NULL,
    matched_chapter TEXT,
    matched_section TEXT,
    similarity REAL,
    grounding TEXT,
    verified TEXT,
    repeated_confusion INTEGER,
    from_cache INTEGER
);
"""

COLUMNS = [
    "timestamp", "course", "question", "matched_chapter", "matched_section",
    "similarity", "grounding", "verified", "repeated_confusion", "from_cache",
]


class QuestionLogStore:
    def __init__(self, db_path: str = "cache/qa_cache.db", connection=None):
        if connection is not None:
            self._conn = connection
        else:
            import sqlite3

            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def log_question(self, timestamp, course, question, matched_chapter, matched_section,
                      similarity, grounding, verified, repeated_confusion, from_cache):
        with self._lock:
            self._conn.execute(
                "INSERT INTO question_log "
                "(timestamp, course, question, matched_chapter, matched_section, "
                "similarity, grounding, verified, repeated_confusion, from_cache) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    timestamp, course, question, matched_chapter, matched_section,
                    similarity, grounding,
                    "" if verified is None else str(verified),
                    int(bool(repeated_confusion)),
                    int(bool(from_cache)),
                ),
            )
            self._conn.commit()

    def get_dataframe(self) -> pd.DataFrame:
        """dashboard.py ke liye — pehle `pd.read_csv(LOG_FILE)` tha, ab
        yahan se aata hai. Empty ho to bhi sahi columns wala empty
        DataFrame deta hai (dashboard ka "no data yet" check chalta rahe)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT " + ", ".join(COLUMNS) + " FROM question_log ORDER BY timestamp"
            ).fetchall()
        return pd.DataFrame(list(rows), columns=COLUMNS)
