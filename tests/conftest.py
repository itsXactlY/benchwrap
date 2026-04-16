#!/usr/bin/env python3
"""
conftest.py — Shared fixtures for benchwrap tests.

Provides:
  - MockModelBackend: deterministic LLM for testing
  - MockMemoryBackend: in-memory store/recall
  - Synthetic sample generators for each benchmark type
  - Scorer fixtures
"""
import sys
import os
from pathlib import Path

# Ensure benchwrap is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchwrap.core.types import Sample, Prompt, Prediction, Score
from benchwrap.core.model import ModelBackend
from benchwrap.core.scorer import (
    Scorer, ExactMatch, MCQScorer, F1Scorer, NumericScorer, ReasoningScorer,
)
from benchwrap.core.adapter import BenchmarkAdapter
from benchwrap.core.engine import EvaluationEngine


# ---------------------------------------------------------------------------
# Mock Model Backend — returns deterministic responses
# ---------------------------------------------------------------------------
class MockModelBackend(ModelBackend):
    """Model backend that returns predefined responses for testing.
    
    Usage:
        mock = MockModelBackend(responses={"What is 2+2?": "4"})
        mock = MockModelBackend(response_fn=lambda prompt: "fixed answer")
    """
    
    def __init__(self, responses: dict = None, response_fn=None, default: str = "A"):
        self._responses = responses or {}
        self._response_fn = response_fn
        self._default = default
        self._call_log = []
    
    def generate(self, prompt: Prompt, **kwargs) -> Prediction:
        text = prompt.raw_text
        self._call_log.append(text)
        
        if self._response_fn:
            result = self._response_fn(prompt)
        elif text in self._responses:
            result = self._responses[text]
        else:
            # Try partial match
            result = self._default
            for key, val in self._responses.items():
                if key in text:
                    result = val
                    break
        
        return Prediction(
            text=result,
            model="mock",
            backend="mock",
            latency_ms=1.0,
            tokens_in=len(text.split()),
            tokens_out=len(result.split()),
        )
    
    def name(self) -> str:
        return "mock"
    
    def model_id(self) -> str:
        return "mock-model"
    
    def call_log(self) -> list:
        return list(self._call_log)


# ---------------------------------------------------------------------------
# Mock Memory Backend — in-memory store/recall
# ---------------------------------------------------------------------------
class MockMemoryBackend:
    """Memory backend that stores in a list, recalls by substring match."""
    
    def __init__(self):
        self._store = []
    
    def store(self, content: str, label: str = "", metadata: dict = None) -> str:
        idx = len(self._store)
        self._store.append({
            "id": str(idx),
            "content": content,
            "label": label,
            "metadata": metadata or {},
        })
        return str(idx)
    
    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        query_lower = query.lower()
        scored = []
        for item in self._store:
            content_lower = item["content"].lower()
            # Simple substring match scoring
            if any(word in content_lower for word in query_lower.split()):
                # Count matching words for ranking
                score = sum(1 for w in query_lower.split() if w in content_lower)
                scored.append({**item, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
    
    def ingest(self, items: list[dict]) -> int:
        count = 0
        for item in items:
            self.store(
                content=item.get("content", ""),
                label=item.get("label", ""),
                metadata=item.get("metadata"),
            )
            count += 1
        return count
    
    def clear(self):
        self._store.clear()
    
    def stats(self) -> dict:
        return {"total_memories": len(self._store)}


# ---------------------------------------------------------------------------
# Mock Adapter — configurable adapter for testing
# ---------------------------------------------------------------------------
class MockAdapter(BenchmarkAdapter):
    """Adapter that returns predefined samples for testing."""
    
    def __init__(self, samples: list[Sample] = None, memory_client=None, 
                 score_fn=None):
        self._samples = samples or []
        self.memory_client = memory_client
        self._score_fn = score_fn
        self._pre_evaluated = False
    
    def name(self) -> str:
        return "mock-bench"
    
    def datasets(self) -> list[str]:
        return ["default", "all"]
    
    def default_dataset(self) -> str:
        return "default"
    
    def load(self, dataset="default", split="test", limit=None) -> iter:
        for i, s in enumerate(self._samples):
            if limit and i >= limit:
                return
            yield s
    
    def format_prompt(self, sample, fewshot=None):
        context = ""
        if self.memory_client:
            try:
                results = self.memory_client.recall(sample.input, top_k=3)
                if results:
                    context = "\n".join(f"- {r['content']}" for r in results)
            except Exception:
                pass
        
        if context:
            text = f"Facts:\n{context}\n\nQuestion: {sample.input}\n\nAnswer:"
        else:
            text = f"Question: {sample.input}\n\nAnswer:"
        
        return Prompt(
            system=None,
            messages=[{"role": "user", "content": text}],
            raw_text=text,
        )
    
    def score(self, prediction, reference, sample):
        if self._score_fn:
            return self._score_fn(prediction, reference, sample)
        # Default: exact match
        em = 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0
        return Score(
            exact_match=em,
            raw_prediction=prediction,
            raw_reference=reference,
            scoring_method="mock_exact",
        )
    
    def pre_evaluate(self, dataset="default", backend=None):
        self._pre_evaluated = True


# ---------------------------------------------------------------------------
# Synthetic Sample Generators
# ---------------------------------------------------------------------------
def make_mcq_samples(n=5) -> list[Sample]:
    """Generate MCQ samples (like MMLU)."""
    choices = ["A", "B", "C", "D"]
    return [
        Sample(
            id=f"mcq_{i}",
            input=f"What is the answer to question {i}?",
            reference=choices[i % 4],
            choices=choices,
            metadata={"category": "test"},
        )
        for i in range(n)
    ]


def make_numeric_samples(n=5) -> list[Sample]:
    """Generate numeric QA samples (like GSM8K)."""
    return [
        Sample(
            id=f"num_{i}",
            input=f"What is {i} + {i}?",
            reference=str(i + i),
            metadata={"category": "math"},
        )
        for i in range(n)
    ]


def make_open_ended_samples(n=5) -> list[Sample]:
    """Generate open-ended QA samples."""
    facts = [
        ("Who leads the team?", "Alice"),
        ("What is the budget?", "$50,000"),
        ("When is the deadline?", "March 15, 2026"),
        ("Where is the key stored?", "Vault"),
        ("What department is Bob in?", "QA"),
    ]
    return [
        Sample(
            id=f"oe_{i}",
            input=facts[i % len(facts)][0],
            reference=facts[i % len(facts)][1],
            metadata={"category": "recall"},
        )
        for i in range(n)
    ]


def make_temporal_samples(n=5) -> list[Sample]:
    """Generate temporal ordering samples."""
    events = [
        ("When was the kickoff?", "January 15, 2026", "temporal"),
        ("What happened first?", "Project kickoff", "ordering"),
        ("What was the last event?", "Production launch", "temporal"),
        ("How many days between events?", "45", "arithmetic"),
        ("What happened after the audit?", "Beta release", "ordering"),
    ]
    return [
        Sample(
            id=f"temp_{i}",
            input=events[i % len(events)][0],
            reference=events[i % len(events)][1],
            metadata={"category": events[i % len(events)][2]},
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Scorer Fixtures
# ---------------------------------------------------------------------------
SCORERS = {
    "exact": ExactMatch,
    "mcq": MCQScorer,
    "f1": F1Scorer,
    "numeric": NumericScorer,
}


def get_all_scorers() -> dict[str, Scorer]:
    """Return instances of all built-in scorers."""
    return {name: cls() for name, cls in SCORERS.items()}
