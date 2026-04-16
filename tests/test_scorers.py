#!/usr/bin/env python3
"""
test_scorers.py — Tests for all scorer types with edge cases.

Covers:
  - ExactMatch: exact, case, whitespace, unicode, empty
  - MCQScorer: all extraction patterns, multi-letter, no match
  - F1Scorer: full overlap, partial, no overlap, empty
  - NumericScorer: integers, floats, negatives, commas, units, no number
  - ReasoningScorer: CoT extraction patterns, inner scorer delegation
  - get_scorer() factory
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchwrap.core.scorer import (
    ExactMatch, MCQScorer, F1Scorer, NumericScorer, ReasoningScorer, get_scorer,
)


def run_test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        return False


# ===========================================================================
# ExactMatch
# ===========================================================================
def test_exact_perfect():
    s = ExactMatch().score("hello", "hello")
    assert s.exact_match == 1.0
    assert s.primary() == 1.0


def test_exact_case_insensitive():
    s = ExactMatch().score("Hello", "hello")
    assert s.exact_match == 1.0


def test_exact_whitespace():
    s = ExactMatch().score("  hello  ", "hello")
    assert s.exact_match == 1.0


def test_exact_mismatch():
    s = ExactMatch().score("hello", "world")
    assert s.exact_match == 0.0


def test_exact_empty_pred():
    s = ExactMatch().score("", "answer")
    assert s.exact_match == 0.0


def test_exact_empty_ref():
    s = ExactMatch().score("answer", "")
    assert s.exact_match == 0.0


def test_exact_both_empty():
    s = ExactMatch().score("", "")
    assert s.exact_match == 1.0


def test_exact_unicode():
    s = ExactMatch().score("café", "café")
    assert s.exact_match == 1.0


def test_exact_trailing_period():
    """Trailing period should NOT match — this is a known issue."""
    s = ExactMatch().score("Alice.", "Alice")
    assert s.exact_match == 0.0  # Strict by design


def test_exact_scoring_method():
    s = ExactMatch().score("a", "a")
    assert s.scoring_method == "exact_match"


# ===========================================================================
# MCQScorer
# ===========================================================================
def test_mcq_direct_letter():
    s = MCQScorer().score("A", "A")
    assert s.exact_match == 1.0
    assert s.accuracy == 1.0


def test_mcq_lowercase():
    s = MCQScorer().score("b", "B")
    assert s.exact_match == 1.0


def test_mcq_answer_is():
    s = MCQScorer().score("The answer is C", "C")
    assert s.exact_match == 1.0


def test_mcq_option():
    s = MCQScorer().score("Option D", "D")
    assert s.exact_match == 1.0


def test_mcq_parenthesized():
    s = MCQScorer().score("(B)", "B")
    assert s.exact_match == 1.0


def test_mcq_period():
    s = MCQScorer().score("A.", "A")
    assert s.exact_match == 1.0


def test_mcq_verbose():
    s = MCQScorer().score(
        "Based on the analysis, **Answer: B. The correct option**",
        "B",
    )
    assert s.exact_match == 1.0


def test_mcq_wrong_letter():
    s = MCQScorer().score("A", "B")
    assert s.exact_match == 0.0


def test_mcq_no_letter():
    s = MCQScorer().score("I don't know", "A")
    assert s.exact_match == 0.0
    assert s.matched == "'' vs 'A'"


def test_mcq_empty():
    s = MCQScorer().score("", "A")
    assert s.exact_match == 0.0


def test_mcq_multiple_letters():
    """Should extract the first standalone letter when multiple present."""
    s = MCQScorer().score("A is wrong, B is possible, but C is correct", "A")
    assert s.exact_match == 1.0


def test_mcq_extended_options():
    """MCQ with options beyond D (E, F)."""
    s = MCQScorer().score("Answer: E", "E")
    assert s.exact_match == 1.0


# ===========================================================================
# F1Scorer
# ===========================================================================
def test_f1_perfect():
    s = F1Scorer().score("the cat sat", "the cat sat")
    assert s.f1 == 1.0


def test_f1_partial():
    s = F1Scorer().score("the cat sat down", "the cat")
    # precision = 2/4, recall = 2/2, f1 = 2*0.5*1/(0.5+1) = 0.667
    assert abs(s.f1 - 0.6667) < 0.01


def test_f1_no_overlap():
    s = F1Scorer().score("dog runs", "cat sits")
    assert s.f1 == 0.0


def test_f1_empty_pred():
    s = F1Scorer().score("", "some answer")
    assert s.f1 == 0.0


def test_f1_empty_ref():
    s = F1Scorer().score("some answer", "")
    assert s.f1 == 0.0


def test_f1_case_insensitive():
    s = F1Scorer().score("The Cat", "the cat")
    assert s.f1 == 1.0


def test_f1_punctuation_stripped():
    s = F1Scorer().score("hello, world!", "hello world")
    assert s.f1 == 1.0


def test_f1_single_word():
    s = F1Scorer().score("Alice", "Alice")
    assert s.f1 == 1.0


# ===========================================================================
# NumericScorer
# ===========================================================================
def test_numeric_integer():
    s = NumericScorer().score("42", "42")
    assert s.exact_match == 1.0


def test_numeric_float():
    s = NumericScorer().score("3.14", "3.14")
    assert s.exact_match == 1.0


def test_numeric_negative():
    s = NumericScorer().score("-17", "-17")
    assert s.exact_match == 1.0


def test_numeric_with_commas():
    s = NumericScorer().score("$1,234", "1234")
    assert s.exact_match == 1.0


def test_numeric_in_text():
    s = NumericScorer().score("The answer is 42 dollars", "42")
    assert s.exact_match == 1.0


def test_numeric_wrong():
    s = NumericScorer().score("42", "43")
    assert s.exact_match == 0.0


def test_numeric_no_number_pred():
    s = NumericScorer().score("I don't know", "42")
    assert s.exact_match == 0.0


def test_numeric_no_number_ref():
    s = NumericScorer().score("42", "no number")
    assert s.exact_match == 0.0


def test_numeric_extract_last():
    """Should extract the LAST number (the final answer, not intermediate)."""
    s = NumericScorer().score("Step 1: 10. Step 2: 20. Step 3: 30. Answer: 30", "30")
    assert s.exact_match == 1.0


def test_numeric_percentage():
    s = NumericScorer().score("15%", "15")
    assert s.exact_match == 1.0


def test_numeric_zero():
    s = NumericScorer().score("0", "0")
    assert s.exact_match == 1.0


# ===========================================================================
# ReasoningScorer
# ===========================================================================
def test_reasoning_answer_is():
    inner = MCQScorer()
    s = ReasoningScorer(inner=inner)
    score = s.score("Let me think... The answer is B.", "B")
    assert score.exact_match == 1.0


def test_reasoning_boxed():
    inner = NumericScorer()
    s = ReasoningScorer(inner=inner)
    score = s.score("After calculation, \\boxed{42}", "42")
    assert score.exact_match == 1.0


def test_reasoning_fallback():
    inner = ExactMatch()
    s = ReasoningScorer(inner=inner)
    score = s.score("Some reasoning with no clear answer ending here", "here")
    # Falls back to last 1000 chars stripped
    assert score.scoring_method.startswith("reasoning")


def test_reasoning_custom_patterns():
    inner = ExactMatch()
    s = ReasoningScorer(inner=inner, patterns=[r'FINAL:\s*(\S+)'])
    score = s.score("Thinking... FINAL: success", "success")
    assert score.exact_match == 1.0


def test_reasoning_preserves_cot():
    inner = MCQScorer()
    s = ReasoningScorer(inner=inner)
    cot = "Long chain of thought... " * 50 + "Answer: C"
    score = s.score(cot, "C")
    assert score.raw_prediction == cot  # Full CoT preserved


# ===========================================================================
# get_scorer factory
# ===========================================================================
def test_factory_mcq():
    s = get_scorer("mcq")
    assert isinstance(s, MCQScorer)


def test_factory_numeric():
    s = get_scorer("numeric")
    assert isinstance(s, NumericScorer)


def test_factory_exact():
    s = get_scorer("exact")
    assert isinstance(s, ExactMatch)


def test_factory_f1():
    s = get_scorer("f1")
    assert isinstance(s, F1Scorer)


def test_factory_reasoning():
    s = get_scorer("reasoning")
    assert isinstance(s, ReasoningScorer)


def test_factory_reasoning_inner():
    s = get_scorer("reasoning", inner="numeric")
    assert isinstance(s, ReasoningScorer)
    assert isinstance(s.inner, NumericScorer)


def test_factory_unknown():
    try:
        get_scorer("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown scorer type" in str(e)


# ===========================================================================
# Run all
# ===========================================================================
if __name__ == "__main__":
    tests = [
        # ExactMatch
        ("exact/perfect", test_exact_perfect),
        ("exact/case", test_exact_case_insensitive),
        ("exact/whitespace", test_exact_whitespace),
        ("exact/mismatch", test_exact_mismatch),
        ("exact/empty_pred", test_exact_empty_pred),
        ("exact/empty_ref", test_exact_empty_ref),
        ("exact/both_empty", test_exact_both_empty),
        ("exact/unicode", test_exact_unicode),
        ("exact/trailing_period", test_exact_trailing_period),
        ("exact/scoring_method", test_exact_scoring_method),
        # MCQScorer
        ("mcq/direct", test_mcq_direct_letter),
        ("mcq/lowercase", test_mcq_lowercase),
        ("mcq/answer_is", test_mcq_answer_is),
        ("mcq/option", test_mcq_option),
        ("mcq/parenthesized", test_mcq_parenthesized),
        ("mcq/period", test_mcq_period),
        ("mcq/verbose", test_mcq_verbose),
        ("mcq/wrong", test_mcq_wrong_letter),
        ("mcq/no_letter", test_mcq_no_letter),
        ("mcq/empty", test_mcq_empty),
        ("mcq/multiple_letters", test_mcq_multiple_letters),
        ("mcq/extended", test_mcq_extended_options),
        # F1Scorer
        ("f1/perfect", test_f1_perfect),
        ("f1/partial", test_f1_partial),
        ("f1/no_overlap", test_f1_no_overlap),
        ("f1/empty_pred", test_f1_empty_pred),
        ("f1/empty_ref", test_f1_empty_ref),
        ("f1/case", test_f1_case_insensitive),
        ("f1/punctuation", test_f1_punctuation_stripped),
        ("f1/single_word", test_f1_single_word),
        # NumericScorer
        ("numeric/integer", test_numeric_integer),
        ("numeric/float", test_numeric_float),
        ("numeric/negative", test_numeric_negative),
        ("numeric/commas", test_numeric_with_commas),
        ("numeric/in_text", test_numeric_in_text),
        ("numeric/wrong", test_numeric_wrong),
        ("numeric/no_num_pred", test_numeric_no_number_pred),
        ("numeric/no_num_ref", test_numeric_no_number_ref),
        ("numeric/last", test_numeric_extract_last),
        ("numeric/percentage", test_numeric_percentage),
        ("numeric/zero", test_numeric_zero),
        # ReasoningScorer
        ("reasoning/answer_is", test_reasoning_answer_is),
        ("reasoning/boxed", test_reasoning_boxed),
        ("reasoning/fallback", test_reasoning_fallback),
        ("reasoning/custom_patterns", test_reasoning_custom_patterns),
        ("reasoning/preserves_cot", test_reasoning_preserves_cot),
        # Factory
        ("factory/mcq", test_factory_mcq),
        ("factory/numeric", test_factory_numeric),
        ("factory/exact", test_factory_exact),
        ("factory/f1", test_factory_f1),
        ("factory/reasoning", test_factory_reasoning),
        ("factory/reasoning_inner", test_factory_reasoning_inner),
        ("factory/unknown", test_factory_unknown),
    ]
    
    print(f"\n{'='*60}")
    print(f"  test_scorers.py — {len(tests)} tests")
    print(f"{'='*60}\n")
    
    passed = sum(run_test(name, fn) for name, fn in tests)
    failed = len(tests) - passed
    
    print(f"\n  Results: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(0 if failed == 0 else 1)
