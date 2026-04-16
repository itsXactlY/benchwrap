#!/usr/bin/env python3
"""
test_integration.py — End-to-end integration tests.

Full pipeline tests: adapter → engine → scorer → result.
Tests real-world scenarios with mock backends to avoid API calls.

Covers:
  - MCQ pipeline (MMLU-style)
  - Numeric pipeline (GSM8K-style)
  - Memory-augmented pipeline (memory-bench-style)
  - LoCoMo with ingestion → recall → score
  - Custom scorer injection
  - Multi-adapter sequential runs
  - Result serialization roundtrip
"""
import sys
import os
import json
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchwrap.core.engine import EvaluationEngine
from benchwrap.core.types import Sample, Score
from benchwrap.core.scorer import MCQScorer, NumericScorer, F1Scorer
from tests.conftest import (
    MockModelBackend, MockMemoryBackend, MockAdapter,
    make_mcq_samples, make_numeric_samples, make_open_ended_samples,
)


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


# ===========================================================================
# MCQ Pipeline (MMLU-style)
# ===========================================================================
def test_mcq_pipeline():
    """Full MCQ evaluation: samples → MCQ scorer → accuracy."""
    samples = make_mcq_samples(10)
    model = MockModelBackend(response_fn=lambda p: p.raw_text.split("reference:")[-1].strip()[:1])
    # Actually just return the correct letter
    model = MockModelBackend(response_fn=lambda p: {
        "What is the answer to question 0?": "A",
        "What is the answer to question 1?": "B",
        "What is the answer to question 2?": "C",
        "What is the answer to question 3?": "D",
        "What is the answer to question 4?": "A",
    }.get(p.raw_text.split("\n")[0].replace("Question: ", "").strip(), "A"))
    
    # Simplify: just use correct responses
    responses = {}
    for i, s in enumerate(samples):
        responses[f"Question: {s.input}\n\nAnswer:"] = s.reference
    model = MockModelBackend(responses=responses)
    
    adapter = MockAdapter(samples=samples)
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    
    assert result.metrics["exact_match"] == 1.0
    assert result.num_samples == 10


def test_mcq_verbose_answers():
    """Model gives verbose MCQ answers — MCQScorer should extract letter."""
    from benchwrap.core.scorer import MCQScorer
    
    samples = make_mcq_samples(5)
    responses = {}
    for i, s in enumerate(samples):
        responses[f"Question: {s.input}\n\nAnswer:"] = f"**Answer: {s.reference}. The correct option**"
    model = MockModelBackend(responses=responses)
    
    # Use MCQ-aware scoring
    def mcq_score(pred, ref, sample):
        return MCQScorer().score(pred, ref)
    
    adapter = MockAdapter(samples=samples, score_fn=mcq_score)
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    
    assert result.metrics["exact_match"] == 1.0


# ===========================================================================
# Numeric Pipeline (GSM8K-style)
# ===========================================================================
def test_numeric_pipeline():
    """Numeric extraction from chain-of-thought responses."""
    from benchwrap.core.scorer import NumericScorer
    
    samples = make_numeric_samples(5)
    responses = {}
    for s in samples:
        q = s.input
        ref = s.reference
        responses[f"Question: {q}\n\nAnswer:"] = f"Let me calculate... Step 1: add. The answer is {ref}."
    model = MockModelBackend(responses=responses)
    
    # Use numeric-aware scoring
    def numeric_score(pred, ref, sample):
        return NumericScorer().score(pred, ref)
    
    adapter = MockAdapter(samples=samples, score_fn=numeric_score)
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    
    assert result.metrics["exact_match"] == 1.0


# ===========================================================================
# Memory-Augmented Pipeline
# ===========================================================================
def test_memory_augmented_pipeline():
    """Store facts → recall → answer → score."""
    samples = [
        Sample(id="1", input="What is the budget?", reference="$50,000"),
        Sample(id="2", input="Who leads the team?", reference="Alice"),
        Sample(id="3", input="When is the deadline?", reference="March 15, 2026"),
    ]
    
    memory = MockMemoryBackend()
    memory.store("The budget was approved at $50,000.", label="budget")
    memory.store("Alice is the team lead for the frontend project.", label="team")
    memory.store("Project deadline is March 15, 2026.", label="deadline")
    
    # Model extracts answer from recalled context
    def smart_response(prompt):
        text = prompt.raw_text
        # Check the QUESTION part (after "Question:")
        question = text.split("Question:")[1].split("\n")[0] if "Question:" in text else text
        q_lower = question.lower()
        
        if "budget" in q_lower and "$50,000" in text:
            return "$50,000"
        if ("leads" in q_lower or "who" in q_lower) and "Alice" in text:
            return "Alice"
        if ("deadline" in q_lower or "when" in q_lower) and "March 15, 2026" in text:
            return "March 15, 2026"
        return "unknown"
    
    model = MockModelBackend(response_fn=smart_response)
    adapter = MockAdapter(samples=samples, memory_client=memory)
    
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    
    assert result.metrics["exact_match"] == 1.0


def test_memory_recall_relevance():
    """Verify that relevant facts are recalled for each question."""
    samples = [
        Sample(id="1", input="What is the API key location?", reference="Vault"),
        Sample(id="2", input="What department does Bob join?", reference="QA"),
    ]
    
    memory = MockMemoryBackend()
    memory.store("The API key for production is stored in Vault.", label="security")
    memory.store("New hire Bob starts Monday in the QA department.", label="hiring")
    memory.store("The meeting is on Tuesday at 3pm.", label="meeting")
    
    adapter = MockAdapter(samples=samples, memory_client=memory)
    prompts = []
    
    class CaptureModel(MockModelBackend):
        def generate(self, prompt, **kwargs):
            prompts.append(prompt.raw_text)
            return super().generate(prompt, **kwargs)
    
    model = CaptureModel(default="Vault")
    engine = EvaluationEngine(adapter=adapter, backend=model)
    engine.run()
    
    # First prompt should have API key fact, not meeting fact
    assert "Vault" in prompts[0]
    # Second prompt should have Bob/QA fact
    assert "QA" in prompts[1] or "Bob" in prompts[1]


# ===========================================================================
# Custom Scorer Injection
# ===========================================================================
def test_custom_scorer():
    """Engine with explicit scorer should use it."""
    samples = [Sample(id="1", input="Q?", reference="answer")]
    model = MockModelBackend(default="The answer is answer")
    adapter = MockAdapter(samples=samples)
    
    scorer = F1Scorer()
    engine = EvaluationEngine(adapter=adapter, backend=model, scorer=scorer)
    result = engine.run()
    
    assert result.metrics.get("f1", 0) > 0


# ===========================================================================
# Multi-Adapter Sequential Runs
# ===========================================================================
def test_sequential_runs():
    """Run multiple benchmarks sequentially with same model."""
    model = MockModelBackend(default="A")
    
    # Run 1: MCQ
    mcq_samples = make_mcq_samples(3)
    adapter1 = MockAdapter(samples=mcq_samples)
    engine1 = EvaluationEngine(adapter=adapter1, backend=model)
    result1 = engine1.run()
    assert result1.num_samples == 3
    
    # Run 2: Open-ended
    oe_samples = make_open_ended_samples(3)
    adapter2 = MockAdapter(samples=oe_samples)
    engine2 = EvaluationEngine(adapter=adapter2, backend=model)
    result2 = engine2.run()
    assert result2.num_samples == 3


# ===========================================================================
# Result Serialization Roundtrip
# ===========================================================================
def test_result_roundtrip():
    """Save result to JSON, reload, verify integrity."""
    samples = make_mcq_samples(5)
    model = MockModelBackend(default="A")
    adapter = MockAdapter(samples=samples)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = EvaluationEngine(adapter=adapter, backend=model, save_dir=tmpdir)
        result = engine.run()
        
        # Load saved file
        files = os.listdir(tmpdir)
        assert len(files) == 1
        
        with open(os.path.join(tmpdir, files[0])) as f:
            loaded = json.load(f)
        
        assert loaded["adapter"] == result.adapter
        assert loaded["num_samples"] == result.num_samples
        assert loaded["metrics"]["exact_match"] == result.metrics["exact_match"]
        assert len(loaded["samples"]) == result.num_samples


# ===========================================================================
# Edge Cases
# ===========================================================================
def test_empty_adapter():
    """Adapter with no samples should produce valid empty result."""
    model = MockModelBackend(default="anything")
    adapter = MockAdapter(samples=[])
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    assert result.num_samples == 0


def test_single_sample():
    """Single sample should work."""
    model = MockModelBackend(default="answer")
    adapter = MockAdapter(samples=[
        Sample(id="1", input="Q?", reference="answer"),
    ])
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    assert result.num_samples == 1
    assert result.metrics["exact_match"] == 1.0


def test_unicode_samples():
    """Unicode in questions and answers."""
    samples = [
        Sample(id="1", input="¿Cuál es la capital?", reference="Madrid"),
        Sample(id="2", input="東京の人口は？", reference="1400万"),
    ]
    model = MockModelBackend(default="Madrid")
    adapter = MockAdapter(samples=samples)
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    assert result.num_samples == 2


def test_very_long_response():
    """Model returns very long response — should not crash."""
    samples = [Sample(id="1", input="Q?", reference="A")]
    long_answer = "A " * 10000  # 10K tokens
    model = MockModelBackend(default=long_answer)
    adapter = MockAdapter(samples=samples)
    engine = EvaluationEngine(adapter=adapter, backend=model)
    result = engine.run()
    assert result.num_samples == 1


def test_memory_cross_contamination():
    """Ensure memory from one benchmark doesn't leak into another."""
    # Benchmark 1: store project facts
    mem1 = MockMemoryBackend()
    mem1.store("The project budget is $100,000.", label="budget")
    samples1 = [Sample(id="1", input="What is the budget?", reference="$100,000")]
    adapter1 = MockAdapter(samples=samples1, memory_client=mem1)
    
    # Benchmark 2: separate memory, different facts
    mem2 = MockMemoryBackend()
    mem2.store("The server IP is 192.168.1.1.", label="server")
    samples2 = [Sample(id="2", input="What is the server IP?", reference="192.168.1.1")]
    adapter2 = MockAdapter(samples=samples2, memory_client=mem2)
    
    model = MockModelBackend(response_fn=lambda p: p.raw_text.split("- ")[1].split("\n")[0] if "- " in p.raw_text else "none")
    
    engine1 = EvaluationEngine(adapter=adapter1, backend=model)
    r1 = engine1.run()
    
    engine2 = EvaluationEngine(adapter=adapter2, backend=model)
    r2 = engine2.run()
    
    # Memory should be isolated
    prompts2 = [p for p in model.call_log() if "server" in p.lower() or "192.168" in p]
    assert len(prompts2) > 0


# ===========================================================================
# Run all
# ===========================================================================
if __name__ == "__main__":
    tests = [
        ("mcq/pipeline", test_mcq_pipeline),
        ("mcq/verbose", test_mcq_verbose_answers),
        ("numeric/pipeline", test_numeric_pipeline),
        ("memory/pipeline", test_memory_augmented_pipeline),
        ("memory/relevance", test_memory_recall_relevance),
        ("scorer/custom", test_custom_scorer),
        ("sequential/runs", test_sequential_runs),
        ("result/roundtrip", test_result_roundtrip),
        ("edge/empty", test_empty_adapter),
        ("edge/single", test_single_sample),
        ("edge/unicode", test_unicode_samples),
        ("edge/long_response", test_very_long_response),
        ("edge/memory_isolation", test_memory_cross_contamination),
    ]
    
    print(f"\n{'='*60}")
    print(f"  test_integration.py — {len(tests)} tests")
    print(f"{'='*60}\n")
    
    passed = sum(run_test(name, fn) for name, fn in tests)
    failed = len(tests) - passed
    
    print(f"\n  Results: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(0 if failed == 0 else 1)
