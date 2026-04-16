#!/usr/bin/env python3
"""
test_types.py — Tests for core data types: Sample, Prompt, Prediction, Score, Result.

Covers:
  - Dataclass construction and defaults
  - Serialization (to_dict)
  - Score.primary() logic
  - Result.summary() formatting
  - Edge cases (empty strings, None values, unicode)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchwrap.core.types import Sample, Prompt, Prediction, Score, EvalResult, Result


def run_test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
def test_sample_basic():
    s = Sample(id="test_1", input="What is 2+2?", reference="4")
    assert s.id == "test_1"
    assert s.input == "What is 2+2?"
    assert s.reference == "4"
    assert s.choices is None
    assert s.metadata == {}


def test_sample_with_choices():
    s = Sample(id="mcq_1", input="Pick one", reference="B", choices=["A", "B", "C"])
    d = s.to_dict()
    assert d["choices"] == ["A", "B", "C"]
    assert d["reference"] == "B"


def test_sample_metadata():
    s = Sample(id="x", input="q", reference="a", metadata={"cat": "math", "idx": 3})
    assert s.metadata["cat"] == "math"
    assert s.metadata["idx"] == 3


def test_sample_unicode():
    s = Sample(id="uni", input="¿Qué es esto?", reference="这是一个测试")
    assert "¿" in s.input
    assert "测试" in s.reference


def test_sample_empty():
    s = Sample(id="", input="", reference="")
    d = s.to_dict()
    assert d["id"] == ""
    assert d["input"] == ""


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def test_prompt_basic():
    p = Prompt(
        system="You are helpful.",
        messages=[{"role": "user", "content": "Hello"}],
        raw_text="Hello",
    )
    assert p.system == "You are helpful."
    assert len(p.messages) == 1
    d = p.to_dict()
    assert d["raw_text"] == "Hello"


def test_prompt_no_system():
    p = Prompt(system=None, messages=[{"role": "user", "content": "Hi"}], raw_text="Hi")
    assert p.system is None


def test_prompt_multimessage():
    p = Prompt(
        system=None,
        messages=[
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Follow-up"},
        ],
        raw_text="First\nReply\nFollow-up",
    )
    assert len(p.messages) == 3


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def test_prediction_basic():
    p = Prediction(text="The answer is 42", model="test", backend="mock", latency_ms=100)
    assert p.text == "The answer is 42"
    assert p.latency_ms == 100
    d = p.to_dict()
    assert d["model"] == "test"


def test_prediction_tokens():
    p = Prediction(text="hello world", tokens_in=5, tokens_out=2)
    assert p.tokens_in == 5
    assert p.tokens_out == 2


def test_prediction_empty():
    p = Prediction(text="")
    assert p.text == ""
    assert p.tokens_in == 0


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------
def test_score_exact_match():
    s = Score(exact_match=1.0)
    assert s.exact_match == 1.0
    assert s.f1 is None
    assert s.accuracy is None


def test_score_primary_accuracy():
    """primary() returns accuracy when set."""
    s = Score(exact_match=0.0, accuracy=0.8)
    assert s.primary() == 0.8


def test_score_primary_exact_match():
    """primary() falls back to exact_match when accuracy is None."""
    s = Score(exact_match=1.0, accuracy=None)
    assert s.primary() == 1.0


def test_score_primary_f1_no_accuracy():
    """primary() returns exact_match even when f1 is set but accuracy is None."""
    s = Score(exact_match=0.0, f1=0.7, accuracy=None)
    assert s.primary() == 0.0


def test_score_to_dict():
    s = Score(exact_match=1.0, f1=0.9, accuracy=0.85, scoring_method="mcq")
    d = s.to_dict()
    assert d["exact_match"] == 1.0
    assert d["f1"] == 0.9
    assert d["accuracy"] == 0.85
    assert d["scoring_method"] == "mcq"


def test_score_custom():
    s = Score(exact_match=0.0, custom={"contains": 1.0, "token_f1": 0.5})
    assert s.custom["contains"] == 1.0


def test_score_raw_truncation():
    long_text = "x" * 500
    s = Score(exact_match=0.0, raw_prediction=long_text)
    d = s.to_dict()
    assert len(d["raw_prediction"]) == 200  # Truncated in to_dict()


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------
def test_eval_result():
    er = EvalResult(
        sample_id="test_1",
        score=Score(exact_match=1.0),
        prediction=Prediction(text="A"),
        prompt=Prompt(system=None, messages=[], raw_text="Q"),
    )
    d = er.to_dict()
    assert d["sample_id"] == "test_1"
    assert d["score"]["exact_match"] == 1.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
def test_result_summary():
    result = Result(
        adapter="test",
        model="mock",
        backend="mock",
        dataset="default",
        num_samples=10,
        metrics={"exact_match": 0.7, "n": 10, "avg_latency_ms": 150.5},
        per_category={},
        eval_results=[],
        config={},
        duration_s=10.0,
    )
    summary = result.summary()
    assert "test" in summary
    assert "70.0%" in summary
    assert "150.50" in summary  # latency not formatted as percentage


def test_result_summary_per_category():
    result = Result(
        adapter="test",
        model="mock",
        backend="mock",
        dataset="all",
        num_samples=6,
        metrics={"exact_match": 0.5, "n": 6},
        per_category={
            "math": {"exact_match": 0.8, "n": 3},
            "science": {"exact_match": 0.2, "n": 3},
        },
        eval_results=[],
        config={},
        duration_s=5.0,
    )
    summary = result.summary()
    assert "math" in summary
    assert "science" in summary


def test_result_int_category():
    """Integer categories should not crash summary()."""
    result = Result(
        adapter="test",
        model="mock",
        backend="mock",
        dataset="all",
        num_samples=4,
        metrics={"exact_match": 0.5, "n": 4},
        per_category={
            1: {"exact_match": 0.8, "n": 2},
            2: {"exact_match": 0.2, "n": 2},
        },
        eval_results=[],
        config={},
        duration_s=1.0,
    )
    summary = result.summary()
    assert "80.0%" in summary


def test_result_to_dict():
    result = Result(
        adapter="test",
        model="mock",
        backend="mock",
        dataset="default",
        num_samples=0,
        metrics={},
        per_category={},
        eval_results=[],
        config={},
        duration_s=0.0,
    )
    d = result.to_dict()
    assert d["adapter"] == "test"
    assert "timestamp" in d


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("sample/basic", test_sample_basic),
        ("sample/choices", test_sample_with_choices),
        ("sample/metadata", test_sample_metadata),
        ("sample/unicode", test_sample_unicode),
        ("sample/empty", test_sample_empty),
        ("prompt/basic", test_prompt_basic),
        ("prompt/no_system", test_prompt_no_system),
        ("prompt/multimessage", test_prompt_multimessage),
        ("prediction/basic", test_prediction_basic),
        ("prediction/tokens", test_prediction_tokens),
        ("prediction/empty", test_prediction_empty),
        ("score/exact_match", test_score_exact_match),
        ("score/primary_accuracy", test_score_primary_accuracy),
        ("score/primary_exact_match", test_score_primary_exact_match),
        ("score/primary_f1", test_score_primary_f1_no_accuracy),
        ("score/to_dict", test_score_to_dict),
        ("score/custom", test_score_custom),
        ("score/raw_truncation", test_score_raw_truncation),
        ("eval_result/basic", test_eval_result),
        ("result/summary", test_result_summary),
        ("result/per_category", test_result_summary_per_category),
        ("result/int_category", test_result_int_category),
        ("result/to_dict", test_result_to_dict),
    ]
    
    print(f"\n{'='*60}")
    print(f"  test_types.py — {len(tests)} tests")
    print(f"{'='*60}\n")
    
    passed = sum(run_test(name, fn) for name, fn in tests)
    failed = len(tests) - passed
    
    print(f"\n  Results: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(0 if failed == 0 else 1)
