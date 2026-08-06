"""
db_connection.py
-------------------
Ek shared, DURABLE storage connection banata hai — local SQLite file
(jab sirf ek hi app instance ho) YA Turso (jab teacher + student apps
ALAG Streamlit Cloud deployments ke tor par chal rahe hon aur dono ko
SAME data dikhna chahiye, permanently, restarts ke bawajood).

KYUN ZAROORI HAI: Streamlit Cloud har app ko apna alag, isolated,
EPHEMERAL container deta hai. Agar app.py aur dashboard.py do alag apps
ke tor par deploy hon (jaisa "plotlab-teacher"/"plotlab-classroom" mein
hua), to local SQLite/CSV files kisi doosre app ko kabhi nazar nahi
aatin — aur restart/redeploy pe khud apne hi app ke liye bhi gum ho
jati hain. Turso (libsql-based hosted DB) is masle ko permanently solve
karta hai, free tier ke saath, aur ye is codebase ke SQLite schema/
queries se largely compatible hai.

Turso setup (free, ek dafa karna hai):
    1. https://turso.tech par signup karein (GitHub se, free tier)
    2. Turso CLI se: `turso db create classroom-ai`
    3. `turso db show classroom-ai --url` se DATABASE_URL milega
    4. `turso db tokens create classroom-ai` se AUTH_TOKEN milega
    5. DONO apps (teacher + student) ki Streamlit Secrets mein YE SAME
       do values daalein:
           TURSO_DATABASE_URL = "libsql://your-db-name.turso.io"
           TURSO_AUTH_TOKEN = "..."
       (Same values dono jagah — tabhi wo same data share karenge.)

Agar TURSO_DATABASE_URL/TURSO_AUTH_TOKEN set nahi hain, system
automatically local SQLite file par fall back karta hai — purana
behaviour, single-app ya local-testing setups ke liye abhi bhi theek hai
(bas is case mein teacher/student alag-deployed apps ek doosre ka data
nahi dekh sakenge, jaisa reported bug tha).

>>> IMPORTANT: Turso integration is sandbox mein LIVE test NAHI ho saki
>>> (turso.tech development sandbox ki allowed domains list mein nahi
>>> hai). `libsql_client` package ki documented API ke mutabiq likha
>>> gaya hai aur mocked tests se logic verify kiya hai, lekin deploy se
>>> pehle khud ek chhota manual test zaroor chalayein:
>>>     python3 verify_turso_connection.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class _TursoCursorLike:
    """`.execute(...).fetchall()` jaisi chaining chalti rahe iske liye —
    taake cache_store.py/question_log_store.py ka existing sqlite3-style
    code Turso ke sath bhi bina badle kaam kare."""

    def __init__(self, result_set):
        self._result_set = result_set

    def fetchall(self):
        return list(self._result_set.rows)

    def fetchone(self):
        rows = self._result_set.rows
        return rows[0] if rows else None

    @property
    def lastrowid(self):
        return self._result_set.last_insert_rowid


class TursoConnection:
    """sqlite3.Connection jaisa interface deta hai (execute,
    executescript, commit, close) — taake baaki codebase mein connection
    banane ke alawa kuch badalna na pade."""

    def __init__(self, url: str, auth_token: str):
        import libsql_client  # lazy import — sirf tab chahiye jab Turso use ho

        self._client = libsql_client.create_client_sync(url=url, auth_token=auth_token)

    def execute(self, sql: str, params=()) -> _TursoCursorLike:
        result_set = self._client.execute(sql, list(params) if params else None)
        return _TursoCursorLike(result_set)

    def executescript(self, sql: str) -> None:
        # Turso/libsql ek call mein multiple ';'-separated statements
        # accept nahi karta jaisa sqlite3.executescript karta hai —
        # isliye khud split kar ke ek-ek statement chalate hain.
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                self._client.execute(statement)

    def commit(self) -> None:
        # Turso har statement ko (explicit transaction ke bahar) apne aap
        # commit kar deta hai — is liye ye sirf ek no-op hai, taake
        # existing `.commit()` calls bina error ke chal sakein.
        pass

    def close(self) -> None:
        self._client.close()


def get_connection(
    local_path: str = "cache/qa_cache.db",
    turso_url: str | None = None,
    turso_auth_token: str | None = None,
):
    """Turso agar configured hai (dono values di gayi hon) to usse
    connect karta hai, warna local SQLite file par fall back karta hai."""
    if turso_url and turso_auth_token:
        return TursoConnection(turso_url, turso_auth_token)

    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path), check_same_thread=False)
