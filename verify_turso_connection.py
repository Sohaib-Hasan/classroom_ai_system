"""
verify_turso_connection.py
-----------------------------
Ye ek MANUAL smoke-test hai (pytest suite ka hissa nahi) — asli Turso
account aur internet chahiye. Maine (Claude) is integration ko
development sandbox mein live test NAHI kiya (turso.tech allowed
domains mein nahi tha) — code `libsql_client` ki documented API ke
mutabiq likha gaya hai, lekin deploy se pehle isse khud chalana zaroori
hai.

Chalane se pehle:
    pip install -r requirements-turso.txt
    config.py mein TURSO_DATABASE_URL aur TURSO_AUTH_TOKEN set karein

Chalana:
    python3 verify_turso_connection.py

Agar ye successfully chal jaye, to app.py aur dashboard.py dono mein
(Streamlit Secrets mein) SAME do values daal kar deploy karein.
"""

import sys

from config import TURSO_DATABASE_URL, TURSO_AUTH_TOKEN
from db_connection import get_connection


def main():
    if not TURSO_DATABASE_URL or not TURSO_AUTH_TOKEN:
        print("config.py mein TURSO_DATABASE_URL/TURSO_AUTH_TOKEN set nahi hain — test karne ko kuch nahi.")
        sys.exit(0)

    print(f"Connecting to: {TURSO_DATABASE_URL}")
    try:
        conn = get_connection(turso_url=TURSO_DATABASE_URL, turso_auth_token=TURSO_AUTH_TOKEN)

        print("Creating a test table...")
        conn.executescript("CREATE TABLE IF NOT EXISTS _connection_test (id INTEGER PRIMARY KEY, note TEXT)")
        conn.commit()

        print("Inserting a row...")
        conn.execute("INSERT INTO _connection_test (note) VALUES (?)", ("hello from verify_turso_connection.py",))
        conn.commit()

        print("Reading it back...")
        rows = conn.execute("SELECT id, note FROM _connection_test ORDER BY id DESC LIMIT 1").fetchall()
        print(f"   Got row: {tuple(rows[0])}")

        print("Cleaning up test table...")
        conn.executescript("DROP TABLE _connection_test")
        conn.commit()

        print("\n✅ Success! Turso connection works — safe to deploy with these credentials.")
    except Exception as e:
        print(f"\n❌ Failed: {e!r}")
        print(
            "\nCommon causes:\n"
            "  - TURSO_DATABASE_URL ya TURSO_AUTH_TOKEN galat hai\n"
            "  - `pip install -r requirements-turso.txt` nahi chalaya\n"
            "  - Turso database abhi tak actually bana nahi (turso db create)\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
