"""
cache_store.py
-----------------
Purana design: har naye Q&A pe POORI cache list (saare embeddings samet)
dobara JSON mein serialize ho kar disk pe likhi jati thi. Ek semester mein
agar 1000+ unique sawal cache ho jayein (har ek ~24-46 KB, embedding ke
saath), to har save ek 40-50 MB+ file rewrite karta — Streamlit Cloud
jaisi free hosting pe ye slow aur I/O-heavy ho jata hai, aur file corrupt
hone ka risk bhi badhta hai (agar rewrite ke beech app crash ho jaye).

Fix: SQLite. Ek naya row insert karna O(1) hai (poori file dobara nahi
likhni padti), aur embeddings ko binary BLOB mein store karte hain
(JSON floats se chhota aur tez). SQLite ek single file hi hai — koi
extra server/service nahi chahiye, Python ke saath built-in hai.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from core import (
    CACHE_SIMILARITY_THRESHOLD,
    TutorAnswer,
    cosine_sim_matrix,
    math_signature,
    structural_signature,
    wants_visual,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS qa_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course TEXT NOT NULL,
    question TEXT NOT NULL,
    answer_json TEXT NOT NULL,
    chunks_json TEXT NOT NULL,
    embedding BLOB NOT NULL,
    math_sig_json TEXT NOT NULL,
    struct_sig_json TEXT NOT NULL,
    had_visual INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_qa_cache_course ON qa_cache(course);
"""


def _vec_to_blob(vec) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    return struct.pack(f"{len(arr)}f", *arr.tolist())


def _blob_to_vec(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


class QACache:
    """Thread-safe SQL-backed QA cache. Ek hi instance poore app mein
    reuse karein (Streamlit ke `st.cache_resource` ke saath, jaisa
    app.py mein hota hai).

    Do tareeqon se banayi ja sakti hai:
      - `QACache("cache/qa_cache.db")` — purana behaviour, local SQLite
        file (single-app setups ke liye).
      - `QACache(connection=shared_conn)` — ek pehle se bani connection
        (local ya Turso, dekhein db_connection.py) pass karein, taake
        cache aur question-log dono SAME database share karein — teacher
        aur student apps alag deploy hon tab bhi."""

    def __init__(self, db_path: str = "cache/qa_cache.db", connection=None):
        if connection is not None:
            self._conn = connection
        else:
            self._db_path = Path(db_path)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self):
        self._conn.close()

    def find_cached_answer(
        self,
        course: str,
        question: str,
        query_vec,
        similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD,
    ) -> Optional[dict]:
        """Same-course rows load karta hai, cosine similarity + math
        signature dono check karta hai. Agar question mein visual maanga
        gaya hai lekin cached jawab mein visual nahi tha (ya opposite),
        to cache MISS treat karta hai (purana bug jo already fix tha,
        yahan preserve kiya gaya hai)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT question, answer_json, chunks_json, embedding, "
                "math_sig_json, struct_sig_json, had_visual "
                "FROM qa_cache WHERE course = ?",
                (course,),
            ).fetchall()

        if not rows:
            return None

        embeddings = np.stack([_blob_to_vec(r[3]) for r in rows])
        sims = cosine_sim_matrix(query_vec, embeddings)

        new_wants_visual = wants_visual(question)
        new_math_sig = math_signature(question)
        new_struct_sig = structural_signature(question)

        best_idx = None
        best_sim = -1.0
        for i, sim in enumerate(sims):
            if sim < similarity_threshold:
                continue
            row = rows[i]
            cached_wants_visual = bool(row[6])
            if new_wants_visual != cached_wants_visual:
                continue  # visual-demand mismatch -> is entry ko skip karo
            cached_math_sig = json.loads(row[4])
            if cached_math_sig != new_math_sig:
                continue  # numbers ka order match nahi hua
            cached_struct_sig = json.loads(row[5])
            # structural signature ek extra (zyada strict) check hai —
            # har element (path_tuple, number_str) hai; JSON round-trip
            # tuples ko lists mein convert kar deta hai isliye path wapas
            # tuple mein convert kar ke compare karte hain
            cached_struct_normalized = [(tuple(path), num) for path, num in cached_struct_sig]
            new_struct_normalized = [(tuple(path), num) for path, num in new_struct_sig]
            if cached_struct_normalized != new_struct_normalized:
                continue
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx is None:
            return None

        row = rows[best_idx]
        return {
            "question": row[0],
            "answer": json.loads(row[1]),
            "chunks": json.loads(row[2]),
            "similarity": float(best_sim),
        }

    def save_to_cache(self, course: str, question: str, answer: TutorAnswer, chunks: list[dict], query_vec) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO qa_cache "
                "(course, question, answer_json, chunks_json, embedding, "
                "math_sig_json, struct_sig_json, had_visual) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    course,
                    question,
                    answer.model_dump_json(),
                    json.dumps(chunks),
                    _vec_to_blob(query_vec),
                    json.dumps(math_signature(question)),
                    json.dumps(structural_signature(question)),
                    int(wants_visual(question)),
                ),
            )
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM qa_cache").fetchone()[0]
            by_course = self._conn.execute(
                "SELECT course, COUNT(*) FROM qa_cache GROUP BY course"
            ).fetchall()
        return {"total_entries": total, "by_course": dict(by_course)}

    def migrate_from_legacy_json(self, json_path: str) -> int:
        """Purani flat-JSON cache (agar maujood ho) ko ek dafa SQLite mein
        import karta hai, taake purana cached data zaya na ho. Ye sirf
        ek dafa chalayein (ya idempotent nahi hai — dobara chalane se
        duplicate rows ban sakti hain)."""
        p = Path(json_path)
        if not p.exists():
            return 0
        with open(p, encoding="utf-8") as f:
            legacy = json.load(f)
        count = 0
        for entry in legacy:
            try:
                answer = TutorAnswer(**entry["answer"])
                self.save_to_cache(
                    course=entry["course"],
                    question=entry["question"],
                    answer=answer,
                    chunks=entry.get("chunks", []),
                    query_vec=entry["embedding"],
                )
                count += 1
            except Exception:
                continue  # corrupt/incompatible purani entry — skip
        return count
