import numpy as np
import pytest

from cache_store import QACache
from core import TutorAnswer


@pytest.fixture
def cache(tmp_path):
    db_path = tmp_path / "test_cache.db"
    c = QACache(str(db_path))
    yield c
    c.close()


def make_answer(**overrides):
    defaults = dict(
        english="The derivative of x^2 is 2x.",
        roman_urdu="x^2 ki derivative 2x hai.",
        grounding="adapted_by_ai",
        computation_expression="diff(x**2, x)",
        computation_result="2*x",
    )
    defaults.update(overrides)
    return TutorAnswer(**defaults)


class TestQACache:
    def test_miss_on_empty_cache(self, cache):
        result = cache.find_cached_answer("Calculus", "differentiate x^2", [1.0, 0.0, 0.0])
        assert result is None

    def test_hit_on_similar_question_same_numbers(self, cache):
        answer = make_answer()
        cache.save_to_cache("Calculus", "differentiate x^2", answer, chunks=[], query_vec=[1.0, 0.0, 0.0])

        # Almost-identical vector (high cosine similarity)
        result = cache.find_cached_answer("Calculus", "what's the derivative of x^2", [0.999, 0.001, 0.0])
        assert result is not None
        assert result["answer"]["computation_result"] == "2*x"

    def test_miss_when_similarity_too_low(self, cache):
        answer = make_answer()
        cache.save_to_cache("Calculus", "differentiate x^2", answer, chunks=[], query_vec=[1.0, 0.0, 0.0])

        result = cache.find_cached_answer("Calculus", "integrate sin(x)", [0.0, 1.0, 0.0])
        assert result is None

    def test_miss_when_different_course(self, cache):
        answer = make_answer()
        cache.save_to_cache("Calculus", "differentiate x^2", answer, chunks=[], query_vec=[1.0, 0.0, 0.0])

        result = cache.find_cached_answer("Linear Algebra", "differentiate x^2", [1.0, 0.0, 0.0])
        assert result is None

    def test_miss_when_numbers_differ_despite_high_similarity(self, cache):
        # 'differentiate x^2' aur 'differentiate x^3' embedding mein bohat
        # close ho sakte hain lekin jawab bilkul different hai
        answer = make_answer()
        cache.save_to_cache("Calculus", "differentiate x^2", answer, chunks=[], query_vec=[1.0, 0.0, 0.0])

        result = cache.find_cached_answer("Calculus", "differentiate x^3", [0.999, 0.001, 0.0])
        assert result is None

    def test_miss_when_visual_demanded_but_cached_answer_had_none(self, cache):
        answer = make_answer()  # visual_type=None by default
        cache.save_to_cache("Calculus", "what is x^2", answer, chunks=[], query_vec=[1.0, 0.0, 0.0])

        result = cache.find_cached_answer("Calculus", "plot x^2", [0.999, 0.001, 0.0])
        assert result is None

    def test_stats_reflects_saved_entries(self, cache):
        cache.save_to_cache("Calculus", "q1", make_answer(), chunks=[], query_vec=[1.0, 0.0])
        cache.save_to_cache("Linear Algebra", "q2", make_answer(), chunks=[], query_vec=[0.0, 1.0])
        stats = cache.stats()
        assert stats["total_entries"] == 2
        assert stats["by_course"]["Calculus"] == 1
        assert stats["by_course"]["Linear Algebra"] == 1

    def test_incremental_writes_do_not_lose_earlier_entries(self, cache):
        # FIX regression: purana design poori file rewrite karta tha —
        # ye test confirm karta hai ke sequential inserts sab preserve hote hain
        for i in range(50):
            vec = np.zeros(5)
            vec[i % 5] = 1.0
            cache.save_to_cache("Calculus", f"question {i}", make_answer(), chunks=[], query_vec=vec.tolist())
        assert cache.stats()["total_entries"] == 50
