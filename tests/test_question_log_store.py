import pandas as pd
import pytest

from question_log_store import QuestionLogStore, COLUMNS


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_log.db"
    return QuestionLogStore(str(db_path))


class TestQuestionLogStore:
    def test_empty_log_returns_empty_dataframe_with_correct_columns(self, store):
        df = store.get_dataframe()
        assert len(df) == 0
        assert list(df.columns) == COLUMNS

    def test_log_question_then_retrieve(self, store):
        store.log_question(
            timestamp="2026-08-05T10:00:00",
            course="Calculus",
            question="derivative of x^2",
            matched_chapter="Ch1",
            matched_section="Derivatives",
            similarity=0.92,
            grounding="direct_from_notes",
            verified=None,
            repeated_confusion=False,
            from_cache=False,
        )
        df = store.get_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["course"] == "Calculus"
        assert df.iloc[0]["question"] == "derivative of x^2"
        assert df.iloc[0]["from_cache"] == 0
        assert df.iloc[0]["repeated_confusion"] == 0

    def test_boolean_fields_stored_as_int(self, store):
        store.log_question(
            timestamp="2026-08-05T10:00:00", course="Calculus", question="q",
            matched_chapter="", matched_section="", similarity=0.5,
            grounding="adapted_by_ai", verified=True,
            repeated_confusion=True, from_cache=True,
        )
        df = store.get_dataframe()
        assert df.iloc[0]["from_cache"] == 1
        assert df.iloc[0]["repeated_confusion"] == 1
        assert df.iloc[0]["verified"] == "True"

    def test_multiple_entries_ordered_by_timestamp(self, store):
        for i, ts in enumerate(["2026-08-05T10:00:00", "2026-08-05T09:00:00", "2026-08-05T11:00:00"]):
            store.log_question(
                timestamp=ts, course="Calculus", question=f"q{i}",
                matched_chapter="", matched_section="", similarity=0.5,
                grounding="direct_from_notes", verified=None,
                repeated_confusion=False, from_cache=False,
            )
        df = store.get_dataframe()
        assert len(df) == 3
        timestamps = pd.to_datetime(df["timestamp"]).tolist()
        assert timestamps == sorted(timestamps)  # confirms ORDER BY timestamp worked

    def test_shared_connection_across_two_store_instances(self, tmp_path):
        # Simulates two apps (teacher + student) pointed at the SAME db file
        import sqlite3
        db_path = str(tmp_path / "shared.db")
        conn1 = sqlite3.connect(db_path, check_same_thread=False)
        conn2 = sqlite3.connect(db_path, check_same_thread=False)

        store1 = QuestionLogStore(connection=conn1)
        store1.log_question(
            timestamp="2026-08-05T10:00:00", course="Calculus", question="from store1",
            matched_chapter="", matched_section="", similarity=0.5,
            grounding="direct_from_notes", verified=None,
            repeated_confusion=False, from_cache=False,
        )

        store2 = QuestionLogStore(connection=conn2)
        df = store2.get_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["question"] == "from store1"
