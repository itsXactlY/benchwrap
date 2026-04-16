#!/usr/bin/env python3
"""
test_custom_adapter.py — Framework for testing custom adapters.

Drop-in testing for any adapter that implements the BenchmarkAdapter interface.
Tests all 5 required methods + pre_evaluate + memory integration.

Usage:
  1. Create your adapter in benchwrap/adapters/custom/my_bench.py
  2. Run: python3 tests/test_custom_adapter.py my_bench
  3. Or test ALL custom adapters: python3 tests/test_custom_adapter.py --all
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchwrap.core.types import Sample, Prompt, Score
from benchwrap.core.adapter import BenchmarkAdapter
from benchwrap.core.engine import EvaluationEngine
from benchwrap.adapters import get_adapter, discover_adapters, list_adapters
from tests.conftest import MockModelBackend, MockMemoryBackend


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
# Generic Adapter Contract Tests
# ===========================================================================
def test_interface(adapter: BenchmarkAdapter):
    """Verify adapter implements all required methods."""
    assert hasattr(adapter, 'name'), "Missing name()"
    assert hasattr(adapter, 'datasets'), "Missing datasets()"
    assert hasattr(adapter, 'load'), "Missing load()"
    assert hasattr(adapter, 'format_prompt'), "Missing format_prompt()"
    assert hasattr(adapter, 'score'), "Missing score()"
    
    name = adapter.name()
    assert isinstance(name, str) and len(name) > 0, f"name() returned empty string"
    
    datasets = adapter.datasets()
    assert isinstance(datasets, list) and len(datasets) > 0, f"datasets() returned empty list"


def test_load(adapter: BenchmarkAdapter):
    """Verify load() returns valid Sample objects."""
    datasets = adapter.datasets()
    # Try first non-meta dataset
    dataset = next((d for d in datasets if d != "all"), datasets[0])
    
    samples = list(adapter.load(dataset, limit=3))
    assert len(samples) > 0, f"load() returned no samples for {dataset}"
    
    for s in samples:
        assert isinstance(s, Sample), f"load() yielded non-Sample: {type(s)}"
        assert s.id, "Sample has empty id"
        assert s.input, "Sample has empty input"
        assert s.reference is not None, "Sample has None reference"


def test_format_prompt(adapter: BenchmarkAdapter):
    """Verify format_prompt() returns valid Prompt."""
    datasets = adapter.datasets()
    dataset = next((d for d in datasets if d != "all"), datasets[0])
    
    samples = list(adapter.load(dataset, limit=1))
    if not samples:
        return  # Can't test without samples
    
    prompt = adapter.format_prompt(samples[0])
    assert isinstance(prompt, Prompt), f"format_prompt() returned {type(prompt)}"
    assert prompt.raw_text, "Prompt has empty raw_text"
    assert prompt.messages, "Prompt has no messages"


def test_score(adapter: BenchmarkAdapter):
    """Verify score() returns valid Score."""
    datasets = adapter.datasets()
    dataset = next((d for d in datasets if d != "all"), datasets[0])
    
    samples = list(adapter.load(dataset, limit=1))
    if not samples:
        return
    
    sample = samples[0]
    
    # Score correct answer
    score_correct = adapter.score(sample.reference, sample.reference, sample)
    assert isinstance(score_correct, Score), f"score() returned {type(score_correct)}"
    assert score_correct.exact_match == 1.0, "Correct answer should have exact_match=1.0"
    
    # Score wrong answer
    score_wrong = adapter.score("completely wrong answer xyz", sample.reference, sample)
    assert isinstance(score_wrong, Score)
    assert score_wrong.exact_match == 0.0, "Wrong answer should have exact_match=0.0"


def test_memory_integration(adapter: BenchmarkAdapter):
    """Test adapter with memory backend."""
    datasets = adapter.datasets()
    dataset = next((d for d in datasets if d != "all"), datasets[0])
    
    # Check if adapter accepts memory_client
    import inspect
    init_sig = inspect.signature(adapter.__init__)
    if 'memory_client' not in init_sig.parameters:
        return  # Adapter doesn't support memory
    
    # Create adapter with memory
    mem = MockMemoryBackend()
    adapter_class = type(adapter)
    adapter_with_mem = adapter_class(memory_client=mem)
    
    # Test pre_evaluate if available
    if hasattr(adapter_with_mem, 'pre_evaluate'):
        adapter_with_mem.pre_evaluate(dataset=dataset)
        assert mem.stats()["total_memories"] >= 0  # Should not crash
    
    # Test format_prompt with memory
    samples = list(adapter_with_mem.load(dataset, limit=1))
    if samples:
        prompt = adapter_with_mem.format_prompt(samples[0])
        assert isinstance(prompt, Prompt)


def test_default_dataset(adapter: BenchmarkAdapter):
    """Verify default_dataset() returns a valid dataset."""
    default = adapter.default_dataset()
    assert default in adapter.datasets(), f"default_dataset() '{default}' not in datasets()"


# ===========================================================================
# Run contract tests on a specific adapter
# ===========================================================================
def run_contract_tests(adapter_name: str):
    """Run all contract tests on a named adapter."""
    adapter = get_adapter(adapter_name)
    if not adapter:
        print(f"  ERROR: Adapter '{adapter_name}' not found")
        return 0, 1
    
    tests = [
        (f"{adapter_name}/interface", lambda: test_interface(adapter)),
        (f"{adapter_name}/load", lambda: test_load(adapter)),
        (f"{adapter_name}/format_prompt", lambda: test_format_prompt(adapter)),
        (f"{adapter_name}/score", lambda: test_score(adapter)),
        (f"{adapter_name}/default_dataset", lambda: test_default_dataset(adapter)),
        (f"{adapter_name}/memory", lambda: test_memory_integration(adapter)),
    ]
    
    return sum(run_test(n, f) for n, f in tests), len(tests)


# ===========================================================================
# Built-in adapter contract tests
# ===========================================================================
def test_builtin_contracts():
    """Run contract tests on all built-in adapters."""
    discover_adapters()
    adapters = list_adapters()
    
    total_pass = 0
    total_fail = 0
    
    for name in sorted(adapters):
        passed, total = run_contract_tests(name)
        total_pass += passed
        total_fail += total - passed
    
    return total_pass, total_fail + total_pass


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  test_custom_adapter.py — Adapter Contract Tests")
    print(f"{'='*60}\n")
    
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        # Test specific adapter
        name = sys.argv[1]
        passed, total = run_contract_tests(name)
        failed = total - passed
    else:
        # Test all
        passed, total = test_builtin_contracts()
        failed = total - passed
    
    print(f"\n  Results: {passed} passed, {failed} failed out of {total}")
    sys.exit(0 if failed == 0 else 1)
