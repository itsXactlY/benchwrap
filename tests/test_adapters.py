#!/usr/bin/env python3
"""
test_adapters.py — Tests for all benchmark adapters.

Tests each adapter's: name, datasets, load, format_prompt, score, pre_evaluate.
Uses synthetic data where possible to avoid external dependencies.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchwrap.core.types import Sample, Prompt, Score
from benchwrap.core.scorer import MCQScorer
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
# MMLU Adapter
# ===========================================================================
def test_mmlu_interface():
    from benchwrap.adapters.mmlu import MMLUAdapter
    a = MMLUAdapter()
    assert a.name() == "mmlu"
    assert "all" in a.datasets()
    assert "abstract_algebra" in a.datasets()
    assert a.default_dataset() == "all"


def test_mmlu_load_subject():
    """Load a single subject."""
    from benchwrap.adapters.mmlu import MMLUAdapter
    a = MMLUAdapter()
    samples = list(a.load("anatomy", limit=3))
    assert len(samples) == 3
    assert all(s.choices for s in samples)
    assert all(s.reference in ("A", "B", "C", "D", "E") for s in samples)


def test_mmlu_format_prompt():
    from benchwrap.adapters.mmlu import MMLUAdapter
    a = MMLUAdapter()
    sample = Sample(
        id="test",
        input="What is 2+2?",
        reference="B",
        choices=["A. 3", "B. 4", "C. 5", "D. 6"],
    )
    prompt = a.format_prompt(sample)
    assert "What is 2+2?" in prompt.raw_text
    assert "Answer:" in prompt.raw_text


def test_mmlu_score():
    from benchwrap.adapters.mmlu import MMLUAdapter
    a = MMLUAdapter()
    s = a.score("The answer is B", "B", Sample(id="", input="", reference="B"))
    assert s.exact_match == 1.0


# ===========================================================================
# GSM8K Adapter
# ===========================================================================
def test_gsm8k_interface():
    from benchwrap.adapters.gsm8k import GSM8KAdapter
    a = GSM8KAdapter()
    assert a.name() == "gsm8k"
    assert "main" in a.datasets()


def test_gsm8k_load():
    from benchwrap.adapters.gsm8k import GSM8KAdapter
    a = GSM8KAdapter()
    samples = list(a.load("main", limit=3))
    assert len(samples) == 3
    assert all(s.reference for s in samples)


def test_gsm8k_extract_answer():
    from benchwrap.adapters.gsm8k import GSM8KAdapter
    a = GSM8KAdapter()
    sample = Sample(id="test", input="Q?", reference="42")
    
    # #### pattern
    assert a.extract_answer("Some reasoning... #### 42", sample) == "42"
    
    # "the answer is" pattern
    assert a.extract_answer("So the answer is 42.", sample) == "42"


def test_gsm8k_score():
    from benchwrap.adapters.gsm8k import GSM8KAdapter
    a = GSM8KAdapter()
    s = a.score("#### 42", "42", Sample(id="", input="", reference="42"))
    assert s.exact_match == 1.0


# ===========================================================================
# Memory-Bench Adapter
# ===========================================================================
def test_memory_bench_interface():
    from benchwrap.adapters.memory_bench import MemoryBenchAdapter
    a = MemoryBenchAdapter()
    assert a.name() == "memory-bench"
    assert "recall-accuracy" in a.datasets()


def test_memory_bench_load_no_memory():
    """Without memory backend, queries have no context."""
    from benchwrap.adapters.memory_bench import MemoryBenchAdapter
    a = MemoryBenchAdapter()
    samples = list(a.load("recall-accuracy", limit=3))
    assert len(samples) == 3
    assert all(s.reference for s in samples)


def test_memory_bench_load_with_memory():
    """With memory backend, facts are ingested."""
    from benchwrap.adapters.memory_bench import MemoryBenchAdapter
    mem = MockMemoryBackend()
    a = MemoryBenchAdapter(memory_client=mem)
    samples = list(a.load("recall-accuracy", limit=3))
    
    # Facts should be stored in memory
    stats = mem.stats()
    assert stats["total_memories"] > 0


def test_memory_bench_format_no_memory():
    from benchwrap.adapters.memory_bench import MemoryBenchAdapter
    a = MemoryBenchAdapter()
    sample = Sample(id="1", input="When is the meeting?", reference="Tuesday")
    prompt = a.format_prompt(sample)
    assert "Question:" in prompt.raw_text
    assert "Facts:" not in prompt.raw_text


def test_memory_bench_format_with_memory():
    from benchwrap.adapters.memory_bench import MemoryBenchAdapter
    mem = MockMemoryBackend()
    mem.store("The meeting is on Tuesday at 3pm.", label="meeting")
    a = MemoryBenchAdapter(memory_client=mem)
    
    sample = Sample(id="1", input="When is the meeting?", reference="Tuesday")
    prompt = a.format_prompt(sample)
    assert "Facts:" in prompt.raw_text
    assert "Tuesday" in prompt.raw_text


def test_memory_bench_score_contains():
    from benchwrap.adapters.memory_bench import MemoryBenchAdapter
    a = MemoryBenchAdapter()
    
    # Contains match
    s = a.score("The budget is $50,000", "$50,000", Sample(id="", input="", reference=""))
    assert s.accuracy == 1.0  # contains
    assert s.exact_match == 0.0  # not exact


def test_memory_bench_score_exact():
    from benchwrap.adapters.memory_bench import MemoryBenchAdapter
    a = MemoryBenchAdapter()
    s = a.score("Alice", "Alice", Sample(id="", input="", reference=""))
    assert s.exact_match == 1.0
    assert s.accuracy == 1.0


# ===========================================================================
# LoCoMo Adapter
# ===========================================================================
def test_locomo_interface():
    from benchwrap.adapters.locomo import LoCoMoAdapter
    a = LoCoMoAdapter()
    assert a.name() == "locomo"
    assert "conv-0" in a.datasets()
    assert "multi-hop" in a.datasets()


def test_locomo_load():
    from benchwrap.adapters.locomo import LoCoMoAdapter
    a = LoCoMoAdapter()
    samples = list(a.load("conv-0", limit=5))
    assert len(samples) == 5
    assert all(s.reference for s in samples)
    assert all("category" in s.metadata for s in samples)


def test_locomo_format_no_memory():
    from benchwrap.adapters.locomo import LoCoMoAdapter
    a = LoCoMoAdapter()
    sample = Sample(id="1", input="When did X happen?", reference="May 2023")
    prompt = a.format_prompt(sample)
    assert "no information available" in prompt.raw_text.lower()


def test_locomo_format_with_memory():
    from benchwrap.adapters.locomo import LoCoMoAdapter
    mem = MockMemoryBackend()
    mem.store("[May 2023] Alice said, 'X happened yesterday.'", label="dialog")
    a = LoCoMoAdapter(memory_client=mem)
    
    sample = Sample(id="1", input="When did X happen?", reference="May 2023")
    prompt = a.format_prompt(sample)
    assert "Memories:" in prompt.raw_text
    assert "May 2023" in prompt.raw_text


def test_locomo_pre_evaluate():
    from benchwrap.adapters.locomo import LoCoMoAdapter
    mem = MockMemoryBackend()
    a = LoCoMoAdapter(memory_client=mem)
    a.pre_evaluate(dataset="conv-0")
    
    stats = mem.stats()
    assert stats["total_memories"] > 0


def test_locomo_score_f1():
    from benchwrap.adapters.locomo import LoCoMoAdapter
    a = LoCoMoAdapter()
    s = a.score("May 7, 2023", "7 May 2023", Sample(id="", input="", reference="", metadata={"category": 2}))
    assert s.f1 is not None
    assert s.f1 > 0.5  # Should have decent overlap
    assert s.scoring_method == "locomo_f1_cat2"


def test_locomo_score_adversarial():
    """Category 5 = adversarial. Should say 'no information'."""
    from benchwrap.adapters.locomo import LoCoMoAdapter
    a = LoCoMoAdapter()
    s = a.score(
        "No information available",
        "anything",
        Sample(id="", input="", reference="", metadata={"category": 5}),
    )
    assert s.exact_match == 1.0


# ===========================================================================
# EvoMem Adapter
# ===========================================================================
def test_evomem_interface():
    from benchwrap.adapters.evomem import EvoMemAdapter
    a = EvoMemAdapter()
    assert a.name() == "evomem"
    assert "mmlu-pro" in a.datasets()


def test_evomem_load_mmlu_pro():
    from benchwrap.adapters.evomem import EvoMemAdapter
    a = EvoMemAdapter()
    samples = list(a.load("mmlu-pro", limit=3))
    assert len(samples) == 3
    assert all(s.choices for s in samples)


def test_evomem_format():
    from benchwrap.adapters.evomem import EvoMemAdapter
    a = EvoMemAdapter()
    sample = Sample(
        id="test",
        input="What is X?",
        reference="B",
        choices=["A. Y", "B. X", "C. Z"],
    )
    prompt = a.format_prompt(sample)
    assert "Question:" in prompt.raw_text
    assert "Answer:" in prompt.raw_text


# ===========================================================================
# MemoryAgentBench Adapter
# ===========================================================================
def test_mab_interface():
    from benchwrap.adapters.memory_agent import MemoryAgentBenchAdapter
    a = MemoryAgentBenchAdapter()
    assert a.name() == "memory-agent-bench"
    assert "conflict-sh-6k" in a.datasets()


def test_mab_format_no_memory():
    from benchwrap.adapters.memory_agent import MemoryAgentBenchAdapter
    a = MemoryAgentBenchAdapter()
    sample = Sample(
        id="1",
        input="Who was born in London?",
        reference="Thomas Kyd",
        metadata={"context": "0. Thomas Kyd was born in London.\n1. X is Y"},
    )
    prompt = a.format_prompt(sample)
    assert "Context:" in prompt.raw_text
    assert "0. Thomas Kyd" in prompt.raw_text


def test_mab_format_with_memory():
    from benchwrap.adapters.memory_agent import MemoryAgentBenchAdapter
    mem = MockMemoryBackend()
    mem.store("Thomas Kyd was born in the city of London.", label="fact")
    a = MemoryAgentBenchAdapter(memory_client=mem)
    
    sample = Sample(id="1", input="Who was born in London?", reference="Thomas Kyd")
    prompt = a.format_prompt(sample)
    assert "Context:" in prompt.raw_text
    assert "- Thomas Kyd" in prompt.raw_text


def test_mab_parse_facts():
    from benchwrap.adapters.memory_agent import MemoryAgentBenchAdapter
    a = MemoryAgentBenchAdapter()
    context = "0. Fact one.\n1. Fact two.\n2. Fact three."
    facts = a._parse_context_facts(context)
    assert len(facts) == 3
    assert facts[0] == "Fact one."
    assert facts[1] == "Fact two."


# ===========================================================================
# Adapter Discovery
# ===========================================================================
def test_adapter_discovery():
    from benchwrap.adapters import discover_adapters, list_adapters
    discover_adapters()
    adapters = list_adapters()
    
    expected = ["mmlu", "gsm8k", "memory-bench", "locomo", "evomem", "memory-agent-bench"]
    for name in expected:
        assert name in adapters, f"Adapter '{name}' not discovered"


def test_adapter_get():
    from benchwrap.adapters import get_adapter
    adapter = get_adapter("mmlu")
    assert adapter is not None
    assert adapter.name() == "mmlu"


def test_adapter_get_unknown():
    from benchwrap.adapters import get_adapter
    adapter = get_adapter("nonexistent-benchmark")
    assert adapter is None


# ===========================================================================
# Run all
# ===========================================================================
if __name__ == "__main__":
    tests = [
        # MMLU
        ("mmlu/interface", test_mmlu_interface),
        ("mmlu/load_subject", test_mmlu_load_subject),
        ("mmlu/format_prompt", test_mmlu_format_prompt),
        ("mmlu/score", test_mmlu_score),
        # GSM8K
        ("gsm8k/interface", test_gsm8k_interface),
        ("gsm8k/load", test_gsm8k_load),
        ("gsm8k/extract_answer", test_gsm8k_extract_answer),
        ("gsm8k/score", test_gsm8k_score),
        # Memory-Bench
        ("memory-bench/interface", test_memory_bench_interface),
        ("memory-bench/load_no_mem", test_memory_bench_load_no_memory),
        ("memory-bench/load_with_mem", test_memory_bench_load_with_memory),
        ("memory-bench/format_no_mem", test_memory_bench_format_no_memory),
        ("memory-bench/format_with_mem", test_memory_bench_format_with_memory),
        ("memory-bench/score_contains", test_memory_bench_score_contains),
        ("memory-bench/score_exact", test_memory_bench_score_exact),
        # LoCoMo
        ("locomo/interface", test_locomo_interface),
        ("locomo/load", test_locomo_load),
        ("locomo/format_no_mem", test_locomo_format_no_memory),
        ("locomo/format_with_mem", test_locomo_format_with_memory),
        ("locomo/pre_evaluate", test_locomo_pre_evaluate),
        ("locomo/score_f1", test_locomo_score_f1),
        ("locomo/score_adversarial", test_locomo_score_adversarial),
        # EvoMem
        ("evomem/interface", test_evomem_interface),
        ("evomem/load_mmlu_pro", test_evomem_load_mmlu_pro),
        ("evomem/format", test_evomem_format),
        # MemoryAgentBench
        ("mab/interface", test_mab_interface),
        ("mab/format_no_mem", test_mab_format_no_memory),
        ("mab/format_with_mem", test_mab_format_with_memory),
        ("mab/parse_facts", test_mab_parse_facts),
        # Discovery
        ("discovery/all", test_adapter_discovery),
        ("discovery/get", test_adapter_get),
        ("discovery/unknown", test_adapter_get_unknown),
    ]
    
    print(f"\n{'='*60}")
    print(f"  test_adapters.py — {len(tests)} tests")
    print(f"{'='*60}\n")
    
    passed = sum(run_test(name, fn) for name, fn in tests)
    failed = len(tests) - passed
    
    print(f"\n  Results: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(0 if failed == 0 else 1)
