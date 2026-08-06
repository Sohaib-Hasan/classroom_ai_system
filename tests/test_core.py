"""
test_core.py
--------------
Ye tests un exact bugs ko cover karte hain jo review mein manually
reproduce huay the — taake wo dobara chup-chaap wapas na aa sakein.

Chalane ka tareeqa:
    pip install pytest
    pytest tests/ -v
"""

import json

import numpy as np
import pytest

from core import (
    cosine_sim_matrix,
    decide_retrieval_strategy,
    math_signature,
    repair_json_escaping,
    structural_signature,
    top_chunks_from_vector,
    truncate_for_embedding,
    verify_computation,
    wants_visual,
)


# ------------------------------------------------------------------
# verify_computation — ye woh exact bug hai jo review mein mila tha
# ------------------------------------------------------------------
class TestRepairJsonEscaping:
    """FIX regression tests for the exact production bug reported: Gemini
    forgot to double-escape backslashes in LaTeX commands (\\times,
    \\buildrel, \\pmod), which silently corrupted text (missing first
    letters: "\\times" -> "imes") or crashed the whole answer."""

    def test_fixes_single_backslash_latex_command(self):
        raw = r'{"a": "x \times y"}'
        repaired = repair_json_escaping(raw)
        parsed = json.loads(repaired)
        assert parsed["a"] == r"x \times y"

    def test_fixes_multiple_latex_commands_in_one_string(self):
        raw = r'{"a": "a \times a^{-1} \buildrel p \over \equiv 1 \pmod{p}"}'
        repaired = repair_json_escaping(raw)
        parsed = json.loads(repaired)  # should not raise
        assert "\\times" in parsed["a"]
        assert "\\buildrel" in parsed["a"]
        assert "\\pmod" in parsed["a"]
        # crucially, no letters were silently eaten
        assert "imes" not in parsed["a"].replace("\\times", "")
        assert "uildrel" not in parsed["a"].replace("\\buildrel", "")

    def test_does_not_touch_already_correctly_escaped_json(self):
        raw = r'{"a": "x \\times y"}'  # already correct (double backslash)
        repaired = repair_json_escaping(raw)
        assert repaired == raw

    def test_does_not_touch_valid_single_char_json_escapes(self):
        raw = r'{"a": "line1\nline2\ttabbed"}'
        repaired = repair_json_escaping(raw)
        assert repaired == raw  # \n and \t alone (not followed by more letters) are untouched

    def test_does_not_touch_unicode_escapes(self):
        raw = r'{"a": "\u0041"}'
        repaired = repair_json_escaping(raw)
        assert repaired == raw

    def test_reproduces_and_fixes_the_exact_reported_bug(self):
        # Simulates the literal corruption pattern the user reported in
        # production: "\times" -> "imes", "\buildrel" -> "uildrel"
        broken_raw = r'{"a": "\times a \buildrel p \over \equiv 1"}'
        # Confirm this WOULD corrupt/crash without the fix
        import pytest as _pytest
        with _pytest.raises(json.JSONDecodeError):
            json.loads(broken_raw)
        # Confirm the fix resolves it
        repaired = repair_json_escaping(broken_raw)
        parsed = json.loads(repaired)  # should not raise now
        assert parsed["a"].startswith(r"\times")


class TestVerifyComputationDomainBug:
    """FIX regression test: pehle verify_computation sirf [1.5, 4.5]
    (sirf positive) se sample karta tha, isliye sqrt(x**2) == x jaisi
    GALAT simplification bhi "verified True" keh deta tha (sach hai sirf
    positive x ke liye, jhoot negative x ke liye). Ye tests confirm
    karte hain ke fix ke baad aisi galtiyan pakdi jati hain."""

    def test_sqrt_of_square_equals_x_is_correctly_flagged_false(self):
        # sqrt(x**2) == x sirf x>=0 ke liye sach hai, general mein GALAT
        # (sahi jawab Abs(x) hai). Kaafi trials ke saath, kam se kam kuch
        # negative samples aane chahiye jo ye pakad lein.
        results = [verify_computation("sqrt(x**2)", "x") for _ in range(20)]
        assert False in results, (
            "verify_computation ne kabhi bhi False nahi laut aya 20 runs mein — "
            "domain-sensitive bug wapas aa gaya lagta hai (sirf positive sampling ho rahi hai)."
        )

    def test_log_of_square_equals_2log_is_correctly_flagged_false(self):
        results = [verify_computation("log(x**2)", "2*log(x)") for _ in range(20)]
        assert False in results

    def test_sqrt_of_square_equals_abs_x_is_correct(self):
        # Ye mathematically SAHI hai (kisi bhi real x ke liye) — verified
        # True hona chahiye (ya kam se kam kabhi False nahi).
        results = [verify_computation("sqrt(x**2)", "Abs(x)") for _ in range(10)]
        assert False not in results

    def test_correct_derivative_verifies_true(self):
        assert verify_computation("diff(x**2, x)", "2*x") is True

    def test_incorrect_derivative_verifies_false(self):
        assert verify_computation("diff(x**2, x)", "3*x") is False

    def test_no_expression_returns_none(self):
        assert verify_computation(None, None) is None
        assert verify_computation("", "") is None

    def test_modular_arithmetic(self):
        assert verify_computation("15 % 7", "1") is True

    def test_combinatorics(self):
        assert verify_computation("binomial(5,2)", "10") is True

    def test_invalid_expression_returns_none_not_crash(self):
        assert verify_computation("this is not math )))", "x") is None


# ------------------------------------------------------------------
# Cache safety signatures
# ------------------------------------------------------------------
class TestSignatures:
    def test_math_signature_extracts_numbers_in_order(self):
        assert math_signature("differentiate 3x^2 + 5x") == ["3", "2", "5"]

    def test_math_signature_different_numbers_differ(self):
        assert math_signature("solve x + 2 = 3") != math_signature("solve x + 2 = 4")

    def test_structural_signature_distinguishes_transposed_matrix(self):
        # Ye exact case hai jo dashboard.py mein "known limitation" ke
        # tor par documented tha — transpose se numbers ka order text
        # mein same hi rehta hai lekin unki row/column position badal
        # jati hai
        original = structural_signature("[[1,2],[3,4]]")
        transposed = structural_signature("[[1,3],[2,4]]")
        assert original != transposed

    def test_structural_signature_distinguishes_different_grouping(self):
        a = structural_signature("[[1,2],[3,4]]")
        b = structural_signature("[[1,2,3],[4]]")
        assert a != b

    def test_structural_signature_same_structure_matches(self):
        a = structural_signature("[[1,2],[3,4]]")
        b = structural_signature("[[1,2],[3,4]]")
        assert a == b

    def test_wants_visual_detects_keywords(self):
        assert wants_visual("Can you plot this function?") is True
        assert wants_visual("What is the derivative of x^2?") is False


# ------------------------------------------------------------------
# Retrieval strategy decision (fix for the cross-course quota-waste bug)
# ------------------------------------------------------------------
class TestDecideRetrievalStrategy:
    def test_low_confidence_everywhere_is_not_found(self):
        strategy, extra = decide_retrieval_strategy(best_sim=0.1, cross_best_sim=0.2, cross_course="Calculus")
        assert strategy == "not_found"

    def test_clearly_wrong_course_redirects_without_calling_ai(self):
        # Apna course kamzor (0.2), doosra course bohat confident (0.8)
        strategy, extra = decide_retrieval_strategy(best_sim=0.2, cross_best_sim=0.8, cross_course="Linear Algebra")
        assert strategy == "cross_course_redirect"
        assert extra == "Linear Algebra"

    def test_own_course_confident_ignores_cross_course(self):
        strategy, extra = decide_retrieval_strategy(best_sim=0.9, cross_best_sim=0.95, cross_course="Calculus")
        assert strategy == "answer"

    def test_borderline_case_gives_soft_suggestion_not_redirect(self):
        # doosra course thoda behtar hai (diff > 0.1) lekin itna clear-cut
        # nahi ke redirect kiya jaye (diff < 0.15, ya cross khud < FALLBACK_THRESHOLD)
        strategy, extra = decide_retrieval_strategy(best_sim=0.45, cross_best_sim=0.58, cross_course="Calculus")
        assert strategy == "answer"
        assert extra == "Calculus"

    def test_small_difference_gives_no_suggestion_at_all(self):
        # diff <= 0.1 -> na redirect, na soft suggestion
        strategy, extra = decide_retrieval_strategy(best_sim=0.45, cross_best_sim=0.5, cross_course="Calculus")
        assert strategy == "answer"
        assert extra is None

    def test_no_cross_course_available(self):
        strategy, extra = decide_retrieval_strategy(best_sim=0.6, cross_best_sim=0, cross_course=None)
        assert strategy == "answer"
        assert extra is None


# ------------------------------------------------------------------
# Retrieval math
# ------------------------------------------------------------------
class TestRetrieval:
    def test_cosine_sim_identical_vector_is_one(self):
        v = np.array([1.0, 2.0, 3.0])
        matrix = np.array([[1.0, 2.0, 3.0]])
        sims = cosine_sim_matrix(v, matrix)
        assert sims[0] == pytest.approx(1.0)

    def test_cosine_sim_orthogonal_is_zero(self):
        v = np.array([1.0, 0.0])
        matrix = np.array([[0.0, 1.0]])
        sims = cosine_sim_matrix(v, matrix)
        assert sims[0] == pytest.approx(0.0, abs=1e-9)

    def test_cosine_sim_zero_query_raises(self):
        with pytest.raises(ValueError):
            cosine_sim_matrix(np.array([0.0, 0.0]), np.array([[1.0, 1.0]]))

    def test_top_chunks_course_filter(self):
        kb = [
            {"course": "A", "chapter": "1", "section": "s", "title": "t1"},
            {"course": "B", "chapter": "1", "section": "s", "title": "t2"},
        ]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        query = np.array([1.0, 0.0])
        result = top_chunks_from_vector(query, kb, embeddings, course_filter="A")
        assert len(result) == 1
        assert result[0]["course"] == "A"

    def test_top_chunks_empty_course_returns_empty(self):
        kb = [{"course": "A", "chapter": "1", "section": "s", "title": "t1"}]
        embeddings = np.array([[1.0, 0.0]])
        result = top_chunks_from_vector(np.array([1.0, 0.0]), kb, embeddings, course_filter="NoSuchCourse")
        assert result == []


# ------------------------------------------------------------------
# Embedding truncation safety
# ------------------------------------------------------------------
class TestTruncateForEmbedding:
    def test_short_text_unchanged(self):
        text, truncated = truncate_for_embedding("short text", max_chars=100)
        assert text == "short text"
        assert truncated is False

    def test_long_text_truncated(self):
        long_text = "a" * 10000
        text, truncated = truncate_for_embedding(long_text, max_chars=100)
        assert len(text) == 100
        assert truncated is True
