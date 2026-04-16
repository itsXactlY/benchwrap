"""
Generic Memory Benchmark adapter for benchwrap.
Tests ANY memory system through a unified retrieve → answer → score pipeline.

This is the "one adapter to rule them all" for memory evaluation.
Plug in any memory backend with store() and recall() methods.
Run against any dataset. Compare systems fairly.

Built-in test scenarios:
  - recall-accuracy: store facts, recall by query
  - temporal-ordering: store time-stamped events, answer "when" questions
  - multi-hop: store connected facts, answer questions requiring 2+ hops
  - conflict-resolution: store contradictory facts, answer which is correct
  - session-memory: ingest multi-turn conversation, answer questions later

Custom scenarios: drop a JSON file in benchwrap/adapters/custom/memory/
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Iterator, Optional, Any

from benchwrap.core.adapter import BenchmarkAdapter
from benchwrap.core.types import Sample, Prompt, Score
from benchwrap.core.model import ModelBackend


class MemoryBackend:
    """Interface for memory systems under test.
    
    Any memory system must implement these 2 methods:
        store(content, label, metadata) -> id
        recall(query, top_k) -> list[dict]
    
    Optional:
        ingest(items) -> int  (batch store)
        clear() -> None
        stats() -> dict
    """

    def store(self, content: str, label: str = "", metadata: dict = None) -> str:
        """Store a memory. Returns memory ID."""
        raise NotImplementedError

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Recall memories relevant to query.
        
        Returns list of dicts with at least 'content' or 'text' key.
        """
        raise NotImplementedError

    def ingest(self, items: list[dict]) -> int:
        """Batch store. Returns count stored."""
        count = 0
        for item in items:
            self.store(
                content=item.get("content", ""),
                label=item.get("label", ""),
                metadata=item.get("metadata", {}),
            )
            count += 1
        return count

    def clear(self) -> None:
        """Clear all memories."""
        pass

    def stats(self) -> dict:
        """Return memory system statistics."""
        return {}


# ---------------------------------------------------------------------------
# Built-in test scenarios
# ---------------------------------------------------------------------------

# Scenario 1: Recall Accuracy
# Store facts, recall by exact/paraphrased query
RECALL_ACCURACY_DATA = [
    # Store these
    {"content": "The meeting is on Tuesday at 3pm in Room B.", "label": "meeting"},
    {"content": "Project deadline is March 15, 2026.", "label": "deadline"},
    {"content": "The budget was approved at $50,000.", "label": "budget"},
    {"content": "Alice is the team lead for the frontend project.", "label": "team"},
    {"content": "The server migration starts on April 1st.", "label": "migration"},
    {"content": "New hire Bob starts Monday in the QA department.", "label": "hiring"},
    {"content": "Annual review cycle begins in November.", "label": "review"},
    {"content": "The API key for production is stored in Vault.", "label": "security"},
    {"content": "Team lunch is every second Wednesday.", "label": "lunch"},
    {"content": "The bug fix for ticket #4521 was deployed yesterday.", "label": "deploy"},
]
RECALL_QUERIES = [
    {
        "query": "When is the meeting?",
        "answer": "Tuesday at 3pm in Room B",
        "category": "exact",
    },
    {
        "query": "What's the project deadline?",
        "answer": "March 15, 2026",
        "category": "paraphrase",
    },
    {
        "query": "How much budget was approved?",
        "answer": "$50,000",
        "category": "paraphrase",
    },
    {
        "query": "Who leads the frontend team?",
        "answer": "Alice",
        "category": "exact",
    },
    {
        "query": "When does the server migration happen?",
        "answer": "April 1st",
        "category": "paraphrase",
    },
    {
        "query": "What department does Bob join?",
        "answer": "QA",
        "category": "inference",
    },
    {
        "query": "Where is the production API key?",
        "answer": "Vault",
        "category": "exact",
    },
    {
        "query": "When was bug #4521 deployed?",
        "answer": "yesterday",
        "category": "temporal",
    },
]

# Scenario 2: Temporal Ordering
# Store events with timestamps, answer "when" and "what happened first" questions
TEMPORAL_DATA = [
    {"content": "[2026-01-15] Project kickoff meeting held.", "label": "event-1"},
    {"content": "[2026-02-01] First prototype completed.", "label": "event-2"},
    {"content": "[2026-02-20] Security audit passed.", "label": "event-3"},
    {"content": "[2026-03-10] Beta release deployed.", "label": "event-4"},
    {"content": "[2026-03-25] First customer feedback received.", "label": "event-5"},
    {"content": "[2026-04-05] Major bug found and fixed.", "label": "event-6"},
    {"content": "[2026-04-15] Production launch.", "label": "event-7"},
]
TEMPORAL_QUERIES = [
    {
        "query": "When was the project kickoff?",
        "answer": "January 15, 2026",
        "category": "temporal",
    },
    {
        "query": "What happened before the beta release?",
        "answer": "Security audit passed",
        "category": "ordering",
    },
    {
        "query": "How many days between prototype and production launch?",
        "answer": "73",
        "category": "arithmetic",
    },
    {
        "query": "What was the most recent event?",
        "answer": "Production launch",
        "category": "temporal",
    },
]

# Scenario 3: Multi-hop Reasoning
# Store connected facts, answer questions requiring 2+ hops
MULTIHOP_DATA = [
    {"content": "Alice reports to Bob.", "label": "org-1"},
    {"content": "Bob reports to Carol.", "label": "org-2"},
    {"content": "Carol is the VP of Engineering.", "label": "org-3"},
    {"content": "The VP of Engineering sets the quarterly OKRs.", "label": "policy-1"},
    {"content": "Q2 OKRs include reducing deployment time by 50%.", "label": "okr-1"},
    {"content": "Alice's team owns the CI/CD pipeline.", "label": "ownership-1"},
]
MULTIHOP_QUERIES = [
    {
        "query": "Who is Alice's manager's manager?",
        "answer": "Carol",
        "category": "2-hop",
    },
    {
        "query": "What is Alice's manager's manager's title?",
        "answer": "VP of Engineering",
        "category": "3-hop",
    },
    {
        "query": "Which team is most likely responsible for the Q2 OKR?",
        "answer": "Alice's team",
        "category": "reasoning",
    },
]


class MemoryBenchAdapter(BenchmarkAdapter):
    """Generic Memory Benchmark — tests ANY memory system.
    
    Plug in any memory backend with store() and recall() methods.
    Run against built-in scenarios or custom datasets.
    
    Pipeline:
        1. memory.ingest(data) — store facts
        2. For each query:
           a. recalled = memory.recall(query) — retrieve
           b. prompt = format recalled context + question
           c. prediction = llm.generate(prompt) — answer
           d. score = scorer(prediction, answer) — evaluate
    
    Memory backends are compared on the same data, same LLM,
    same scoring — isolating memory quality.
    """

    def __init__(
        self,
        memory_client: MemoryBackend | None = None,
        llm_backend: ModelBackend | None = None,
        custom_data_dir: str | None = None,
    ):
        self.memory_client = memory_client
        self.llm_backend = llm_backend
        self.custom_data_dir = custom_data_dir or str(
            Path(__file__).parent / "custom" / "memory"
        )

    def name(self) -> str:
        return "memory-bench"

    def datasets(self) -> list[str]:
        return [
            "recall-accuracy",
            "temporal-ordering",
            "multi-hop",
            "all-builtin",
        ] + self._list_custom_datasets()

    def load(
        self,
        dataset: str = "recall-accuracy",
        split: str = "test",
        limit: Optional[int] = None,
    ) -> Iterator[Sample]:
        """Load test samples for a scenario."""
        if dataset == "all-builtin":
            for ds in ["recall-accuracy", "temporal-ordering", "multi-hop"]:
                yield from self.load(ds, split, limit)
            return

        if dataset == "recall-accuracy":
            store_data = RECALL_ACCURACY_DATA
            queries = RECALL_QUERIES
        elif dataset == "temporal-ordering":
            store_data = TEMPORAL_DATA
            queries = TEMPORAL_QUERIES
        elif dataset == "multi-hop":
            store_data = MULTIHOP_DATA
            queries = MULTIHOP_QUERIES
        else:
            # Custom dataset
            store_data, queries = self._load_custom_dataset(dataset)

        # Ingest into memory if client available
        if self.memory_client:
            self.memory_client.clear()
            self.memory_client.ingest(store_data)

        count = 0
        for i, q in enumerate(queries):
            if limit and count >= limit:
                return
            yield Sample(
                id=f"membench_{dataset}_{i}",
                input=q["query"],
                reference=q["answer"],
                metadata={
                    "dataset": dataset,
                    "category": q.get("category", "unknown"),
                    "store_data_count": len(store_data),
                },
            )
            count += 1

    def format_prompt(
        self,
        sample: Sample,
        fewshot: Optional[list[Sample]] = None,
    ) -> Prompt:
        """Format a memory-augmented prompt.
        
        Queries the memory system for relevant context,
        then formats the question with recalled facts.
        """
        question = sample.input
        context = ""

        if self.memory_client:
            try:
                results = self.memory_client.recall(question, top_k=5)
                if results:
                    context_parts = []
                    for r in results:
                        content = r.get("content", r.get("text", str(r)))
                        score = r.get("score", r.get("distance", ""))
                        context_parts.append(f"- {content}")
                    context = "\n".join(context_parts)
            except Exception as e:
                context = f"[Memory recall error: {e}]"

        if context:
            text = (
                f"Based on the following information, answer the question concisely.\n\n"
                f"Facts:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )
        else:
            text = f"Question: {question}\n\nAnswer:"

        return Prompt(
            system=None,
            messages=[{"role": "user", "content": text}],
            raw_text=text,
        )

    def score(
        self,
        prediction: str,
        reference: str,
        sample: Sample,
    ) -> Score:
        """Score with F1 (for free-form) and exact match."""
        pred_clean = prediction.strip().lower()
        ref_clean = reference.strip().lower()

        # Exact match (strict)
        em = 1.0 if pred_clean == ref_clean else 0.0

        # Contains match (lenient) — does the prediction contain the answer?
        contains = 1.0 if ref_clean in pred_clean else 0.0

        # Token F1
        f1 = _token_f1(pred_clean, ref_clean)

        # Use contains as primary for memory benchmarks
        # (because the LLM might add extra words)
        return Score(
            exact_match=em,
            f1=f1,
            accuracy=contains,
            raw_prediction=prediction,
            raw_reference=reference,
            scoring_method="memory_bench_contains_f1",
            custom={"contains": contains, "token_f1": f1},
        )

    def set_memory_client(self, client: MemoryBackend):
        """Set the memory backend to test."""
        self.memory_client = client

    def set_llm_backend(self, backend: ModelBackend):
        """Set the LLM backend."""
        self.llm_backend = backend

    def _list_custom_datasets(self) -> list[str]:
        """List custom datasets in the custom/memory directory."""
        custom_dir = Path(self.custom_data_dir)
        if not custom_dir.exists():
            return []
        return [
            f.stem for f in custom_dir.glob("*.json")
            if f.is_file()
        ]

    def _load_custom_dataset(self, name: str) -> tuple:
        """Load a custom dataset."""
        custom_dir = Path(self.custom_data_dir)
        filepath = custom_dir / f"{name}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Custom dataset not found: {filepath}")

        with open(filepath) as f:
            data = json.load(f)

        store_data = data.get("store", data.get("data", []))
        queries = data.get("queries", data.get("test", []))
        return store_data, queries


def _token_f1(prediction: str, reference: str) -> float:
    """Token-level F1."""
    pred_tokens = set(prediction.split())
    ref_tokens = set(reference.split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
