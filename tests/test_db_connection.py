"""
test_db_connection.py
------------------------
turso.tech is dev sandbox ki allowed-domains list mein NAHI hai, isliye
`libsql_client` ko yahan mock kiya gaya hai — ye tests confirm karte
hain ke TursoConnection sahi tareeqe se library ko call karti hai aur
sqlite3-jaisa interface deti hai. Live Turso ke against verify karne ke
liye `verify_turso_connection.py` chalayein (asli credentials ke saath).
"""

import sqlite3
from unittest.mock import MagicMock, patch

from db_connection import TursoConnection, get_connection


class TestGetConnectionFallback:
    def test_no_turso_config_returns_local_sqlite(self, tmp_path):
        conn = get_connection(local_path=str(tmp_path / "test.db"))
        assert isinstance(conn, sqlite3.Connection)

    def test_partial_turso_config_still_falls_back_to_local(self, tmp_path):
        # sirf URL hai, token nahi — Turso use nahi hona chahiye
        conn = get_connection(local_path=str(tmp_path / "test.db"), turso_url="libsql://x.turso.io")
        assert isinstance(conn, sqlite3.Connection)

    def test_local_connection_creates_parent_dir(self, tmp_path):
        nested_path = tmp_path / "nested" / "dir" / "test.db"
        get_connection(local_path=str(nested_path))
        assert nested_path.parent.is_dir()


class TestTursoConnection:
    def _make_mock_client(self):
        mock_client = MagicMock()
        with patch("libsql_client.create_client_sync", return_value=mock_client):
            conn = TursoConnection(url="libsql://test.turso.io", auth_token="fake-token")
        return conn, mock_client

    def test_get_connection_returns_turso_when_both_configured(self):
        with patch("libsql_client.create_client_sync", return_value=MagicMock()):
            conn = get_connection(turso_url="libsql://test.turso.io", turso_auth_token="fake-token")
        assert isinstance(conn, TursoConnection)

    def test_execute_calls_client_with_sql_and_params(self):
        conn, mock_client = self._make_mock_client()
        fake_result_set = MagicMock()
        fake_result_set.rows = [("row1",)]
        mock_client.execute.return_value = fake_result_set

        cursor = conn.execute("SELECT * FROM t WHERE x = ?", (5,))
        mock_client.execute.assert_called_once_with("SELECT * FROM t WHERE x = ?", [5])
        assert cursor.fetchall() == [("row1",)]

    def test_execute_with_no_params(self):
        conn, mock_client = self._make_mock_client()
        fake_result_set = MagicMock()
        fake_result_set.rows = []
        mock_client.execute.return_value = fake_result_set

        conn.execute("SELECT 1")
        mock_client.execute.assert_called_once_with("SELECT 1", None)

    def test_executescript_splits_and_runs_each_statement(self):
        conn, mock_client = self._make_mock_client()
        mock_client.execute.return_value = MagicMock(rows=[])

        conn.executescript("CREATE TABLE a (id INT); CREATE TABLE b (id INT);")
        assert mock_client.execute.call_count == 2
        called_sqls = [call.args[0] for call in mock_client.execute.call_args_list]
        assert "CREATE TABLE a (id INT)" in called_sqls
        assert "CREATE TABLE b (id INT)" in called_sqls

    def test_commit_is_a_safe_noop(self):
        conn, mock_client = self._make_mock_client()
        conn.commit()  # should not raise, should not call anything unexpected
        mock_client.execute.assert_not_called()

    def test_close_delegates_to_client(self):
        conn, mock_client = self._make_mock_client()
        conn.close()
        mock_client.close.assert_called_once()

    def test_lastrowid_from_result_set(self):
        conn, mock_client = self._make_mock_client()
        fake_result_set = MagicMock()
        fake_result_set.rows = []
        fake_result_set.last_insert_rowid = 42
        mock_client.execute.return_value = fake_result_set

        cursor = conn.execute("INSERT INTO t DEFAULT VALUES")
        assert cursor.lastrowid == 42
