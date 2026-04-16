#!/usr/bin/env python3
"""
test_engine.py — Tests for EvaluationEngine.

Covers:
  - End-to-end with mock adapter + mock model
  - Pre-evaluate hook called
  - Adapter scorer used (not generic exact_match)
  - Few-shot injection
  - Result aggregation
  - Per-category metrics
  - Save to disk
  - Empty samples
  - Adapter with memory backend
"""
import sys
import os
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchwrap.core.engine import EvaluationEngine, _AdapterScorer
from benchwrap.core.types import Sample, Prompt, Prediction, Score, Result
from tests.conftest import MockModelBackend, MockMemoryBackend, MockAdapter


def run_test(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def make_samples(n=5):
    return [
        Sample(
            id=f"s_{i}",
            input=f"Question {i}?",
            reference=f"Answer {i}",
            metadata={"category": "test"} if i < 3 else {},
        )
        for i in range(n)
    ]


# ===========================================================================
# Basic engine
# ===========================================================================
def test_engine_basic():
    samples = make_samples(3)
    model = MockModelBackend(response_fn=lambda p: "Answer 0")
    adapter = MockAdapter(samples=samples)
    
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run(dataset="default", limit=3)
    
    assert result.num_samples == 3
    assert result.adapter == "mock-bench"
    assert result.model == "mock-model"
    assert "exact_match" in result.metrics


def test_engine_correct_answers():
    """Model returns correct answer for each sample."""
    samples = make_samples(3)
    model = MockModelBackend(response_fn=lambda p: p.raw_text.split("Question ")[1].split("?")[0].strip())
    # Response will be "0", "1", "2" but reference is "Answer 0", etc. Won't match.
    # Let's use a better fn
    model = MockModelBackend(response_fn=lambda p: f"Answer {p.raw_text.split('Question ')[1].split('?')[0]}")
    adapter = MockAdapter(samples=samples)
    
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    
    assert result.metrics["exact_match"] == 1.0


def test_engine_all_wrong():
    samples = make_samples(3)
    model = MockModelBackend(default="WRONG")
    adapter = MockAdapter(samples=samples)
    
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    
    assert result.metrics["exact_match"] == 0.0


# ===========================================================================
# Pre-evaluate hook
# ===========================================================================
def test_engine_pre_evaluate():
    samples = make_samples(2)
    model = MockModelBackend(default="Answer 0")
    adapter = MockAdapter(samples=samples)
    
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    
    assert adapter._pre_evaluated is True


def test_engine_pre_evaluate_not_called_without_hook():
    """Adapters without pre_evaluate should not crash."""
    class MinimalAdapter(MockAdapter):
        # Remove pre_evaluate
        pass
    
    # Actually we need to not have the pre_evaluate attribute
    samples = make_samples(2)
    model = MockModelBackend(default="Answer 0")
    adapter = MockAdapter(samples=samples)
    # Delete pre_evaluate to simulate adapter without it
    del adapter.pre_evaluate
    
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    assert result.num_samples == 2


# ===========================================================================
# Adapter scorer
# ===========================================================================
def test_adapter_scorer_used():
    """Engine should use adapter's score() method, not generic scorer."""
    samples = [Sample(id="1", input="Q?", reference="A")]
    
    def custom_scorer(pred, ref, sample):
        # Always give 1.0 for testing
        return Score(exact_match=1.0, accuracy=1.0, scoring_method="custom_test")
    
    model = MockModelBackend(default="wrong answer")
    adapter = MockAdapter(samples=samples, score_fn=custom_scorer)
    
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    
    # Model returned wrong answer but custom scorer always returns 1.0
    assert result.metrics["exact_match"] == 1.0


def test_adapter_scorer_wrapper():
    """_AdapterScorer wraps adapter.score() correctly."""
    adapter = MockAdapter(samples=[])
    scorer = _AdapterScorer(adapter)
    
    score = scorer.score("hello", "hello", sample=None)
    assert score.exact_match == 1.0
    assert score.scoring_method == "mock_exact"


# ===========================================================================
# Result aggregation
# ===========================================================================
def test_engine_metrics():
    samples = make_samples(10)
    model = MockModelBackend(response_fn=lambda p: f"Answer {p.raw_text.split('Question ')[1].split('?')[0]}")
    adapter = MockAdapter(samples=samples)
    
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run(limit=5)
    
    assert result.num_samples == 5
    assert result.metrics["exact_match"] == 1.0
    assert result.metrics["n"] == 5
    assert "avg_latency_ms" in result.metrics


def test_engine_per_category():
    samples = [
        Sample(id="1", input="Q1?", reference="A", metadata={"category": "math"}),
        Sample(id="2", input="Q2?", reference="B", metadata={"category": "math"}),
        Sample(id="3", input="Q3?", reference="C", metadata={"category": "science"}),
    ]
    model = MockModelBackend(default="A")
    adapter = MockAdapter(samples=samples)
    
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    
    assert "math" in result.per_category
    assert "science" in result.per_category
    assert result.per_category["math"]["n"] == 2
    assert result.per_category["science"]["n"] == 1


# ===========================================================================
# Save to disk
# ===========================================================================
def test_engine_save():
    samples = make_samples(2)
    model = MockModelBackend(default="Answer 0")
    adapter = MockAdapter(samples=samples)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = EvaluationEngine(adapter=adapter, backend=model, save_dir=tmpdir)
        result = engine.run()
        
        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0].endswith(".json")
        
        with open(os.path.join(tmpdir, files[0])) as f:
            saved = json.load(f)
        assert saved["num_samples"] == 2


# ===========================================================================
# Empty / edge cases
# ===========================================================================
def test_engine_empty_samples():
    model = MockModelBackend(default="anything")
    adapter = MockAdapter(samples=[])
    
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    
    assert result.num_samples == 0
    assert result.metrics.get("n", 0) == 0


def test_engine_limit():
    samples = make_samples(10)
    model = MockModelBackend(default="Answer 0")
    adapter = MockAdapter(samples=samples)
    
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run(limit=3)
    
    assert result.num_samples == 3


# ===========================================================================
# Memory backend integration
# ===========================================================================
def test_engine_with_memory():
    """Adapter with memory_client should recall facts in prompts."""
    samples = [
        Sample(id="1", input="What is the budget?", reference="$50,000"),
        Sample(id="2", input="When is the deadline?", reference="March 15"),
    ]
    
    memory = MockMemoryBackend()
    memory.store("The budget was approved at $50,000.", label="budget")
    memory.store("Project deadline is March 15, 2026.", label="deadline")
    
    model = MockModelBackend(response_fn=lambda p: p.raw_text.split("Facts:\n")[1].split("\n\n")[0].split("- ")[1] if "Facts:" in p.raw_text else "no context")
    adapter = MockAdapter(samples=samples, memory_client=memory)
    
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    
    # Memory recall should have injected facts into prompts
    prompts = model.call_log()
    assert any("Facts:" in p for p in prompts)
    assert any("$50,000" in p for p in prompts)


# ===========================================================================
# Verbose mode
# ===========================================================================
def test_engine_verbose(capsys=None):
    samples = make_samples(2)
    model = MockModelBackend(default="Answer 0")
    adapter = MockAdapter(samples=samples)
    
    engine = EvaluationEngine(adapter=adapter, backend=model, verbose=True)
    result = engine.run()
    
    assert result.num_samples == 2


# ===========================================================================
# Run all
# ===========================================================================
if __name__ == "__main__":
    tests = [
        ("engine/basic", test_engine_basic),
        ("engine/correct", test_engine_correct_answers),
        ("engine/all_wrong", test_engine_all_wrong),
        ("engine/pre_evaluate", test_engine_pre_evaluate),
        ("engine/adapter_scorer", test_adapter_scorer_used),
        ("engine/scorer_wrapper", test_adapter_scorer_wrapper),
        ("engine/metrics", test_engine_metrics),
        ("engine/per_category", test_engine_per_category),
        ("engine/save", test_engine_save),
        ("engine/empty", test_engine_empty_samples),
        ("engine/limit", test_engine_limit),
        ("engine/memory", test_engine_with_memory),
        ("engine/verbose", test_engine_verbose),
    ]
    
    print(f"\n{'='*60}")
    print(f"  test_engine.py — {len(tests)} tests")
    print(f"{'='*60}\n")
    
    passed = sum(run_test(name, fn) for name, fn in tests)
    failed = len(tests) - passed
    
    print(f"\n  Results: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(0 if failed == 0 else 1)
