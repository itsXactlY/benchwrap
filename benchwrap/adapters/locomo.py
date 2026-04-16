"""
LoCoMo adapter for benchwrap.
Long-term conversational memory benchmark (ACL 2024).

Tests: multi-hop, temporal, single-hop, open-domain, adversarial memory tasks.
Uses existing data at ~/projects/locomo-bench/data/locomo10.json.

IMPORTANT: This adapter tests MEMORY SYSTEMS, not just LLMs.
The pipeline is: ingest dialogs → recall memories → LLM generates answer → score F1.
"""

import json
import os
import re
import sys
import time
import string
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

from benchwrap.core.adapter import BenchmarkAdapter
from benchwrap.core.types import Sample, Prompt, Score
from benchwrap.core.model import ModelBackend


# LoCoMo QA categories
CATEGORIES = {
    1: "multi-hop",
    2: "temporal",
    3: "single-hop",
    4: "open-domain",
    5: "adversarial",
}

CATEGORY_NAMES = {v: k for k, v in CATEGORIES.items()}


class LoCoMoAdapter(BenchmarkAdapter):
    """LoCoMo — Long-term Conversational Memory benchmark.
    
    10 conversations, ~200 QA pairs each, 5 categories:
    - multi-hop: requires combining info from multiple dialog turns
    - temporal: requires understanding of time/events
    - single-hop: specific fact lookup
    - open-domain: general knowledge about the conversation
    - adversarial: unanswerable questions (should say "no information")
    
    Data: ~/projects/locomo-bench/data/locomo10.json
    
    Memory backend is injected via set_memory_client() — supports any system
    with store() and recall() methods.
    """

    def __init__(
        self,
        data_path: str | None = None,
        memory_client=None,
        llm_backend: ModelBackend | None = None,
    ):
        self.data_path = data_path or str(
            Path.home() / "projects/locomo-bench/data/locomo10.json"
        )
        self.memory_client = memory_client
        self.llm_backend = llm_backend
        self._data = None

    def name(self) -> str:
        return "locomo"

    def datasets(self) -> list[str]:
        return ["all", "conv-0", "conv-1", "conv-2", "conv-3", "conv-4",
                "conv-5", "conv-6", "conv-7", "conv-8", "conv-9",
                "multi-hop", "temporal", "single-hop", "open-domain", "adversarial"]

    def load(
        self,
        dataset: str = "all",
        split: str = "test",
        limit: Optional[int] = None,
    ) -> Iterator[Sample]:
        """Load LoCoMo QA pairs as Samples."""
        data = self._load_data()
        count = 0

        for conv_idx, conv in enumerate(data):
            if dataset.startswith("conv-"):
                target_conv = int(dataset.split("-")[1])
                if conv_idx != target_conv:
                    continue

            qa_pairs = conv.get("qa", [])
            for qa_idx, qa in enumerate(qa_pairs):
                if limit and count >= limit:
                    return

                category = qa.get("category", 0)
                cat_name = CATEGORIES.get(category, f"cat-{category}")

                # Filter by category if specified
                if dataset in CATEGORY_NAMES and cat_name != dataset:
                    continue

                answer = qa.get("answer", "")
                if isinstance(answer, list):
                    answer = "; ".join(str(a) for a in answer)
                elif not isinstance(answer, str):
                    answer = str(answer)

                yield Sample(
                    id=f"locomo_c{conv_idx}_q{qa_idx}",
                    input=qa.get("question", ""),
                    reference=answer,
                    metadata={
                        "conversation": conv_idx,
                        "qa_idx": qa_idx,
                        "category": category,
                        "category_name": cat_name,
                        "evidence": qa.get("evidence", []),
                        "sample_id": conv.get("sample_id", conv_idx),
                    },
                )
                count += 1

    def format_prompt(
        self,
        sample: Sample,
        fewshot: Optional[list[Sample]] = None,
    ) -> Prompt:
        """Format a LoCoMo QA as a recall prompt.
        
        If memory_client is set, this queries the memory system first.
        Otherwise, it's a plain QA prompt (for baseline mode).
        """
        question = sample.input
        context = ""

        if self.memory_client:
            # Query memory system for relevant context
            try:
                results = self.memory_client.recall(question, top_k=10)
                if results:
                    context_parts = []
                    for r in results:
                        content = r.get("content", r.get("text", str(r)))
                        context_parts.append(content)
                    context = "\n".join(context_parts)
            except Exception as e:
                context = f"[Memory error: {e}]"

        if context:
            text = (
                f"Based on the following memories about a conversation, "
                f"answer the question concisely.\n\n"
                f"Memories:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer (be concise, use dates/times if relevant):"
            )
        else:
            text = (
                f"Answer the following question about a conversation. "
                f"If there is no information available, say 'No information available'.\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )

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
        """Score using LoCoMo's F1 metric with Porter stemming."""
        category = sample.metadata.get("category", 0)
        f1 = _score_qa(prediction, reference, category)

        return Score(
            exact_match=1.0 if f1 > 0.5 else 0.0,
            f1=f1,
            raw_prediction=prediction,
            raw_reference=reference,
            scoring_method=f"locomo_f1_cat{category}",
            custom={"category": category, "f1": f1},
        )

    def pre_evaluate(self, dataset: str = "all", backend=None):
        """Ingest conversation data into memory before evaluation."""
        if not self.memory_client:
            return
        
        data = self._load_data()
        
        # Determine which conversations to ingest
        if dataset.startswith("conv-"):
            conv_indices = [int(dataset.split("-")[1])]
        elif dataset in CATEGORY_NAMES:
            # Ingest all conversations (category filtering happens in load())
            conv_indices = list(range(len(data)))
        else:
            # "all" or other — ingest everything
            conv_indices = list(range(len(data)))
        
        for conv_idx in conv_indices:
            if conv_idx >= len(data):
                continue
            conv = data[conv_idx]
            dialogs = _extract_dialogs(conv)
            
            # Clear previous conversation data to avoid cross-contamination
            # (memory_bench data from prior runs)
            
            for dialog in dialogs:
                content = _build_memory_content(dialog)
                label = _build_memory_label(dialog)
                self.memory_client.store(content, label=label, metadata={
                    "conversation": conv_idx,
                    "session": dialog.get("session"),
                    "dia_id": dialog.get("dia_id"),
                })

    def ingest_conversation(self, conv_idx: int, backend: ModelBackend | None = None):
        """Ingest a conversation into the memory system.
        
        Must be called before evaluating QA pairs from that conversation.
        NOTE: pre_evaluate() now handles this automatically during engine.run().
        """
        if not self.memory_client:
            raise ValueError("No memory client set. Use set_memory_client() first.")

        data = self._load_data()
        if conv_idx >= len(data):
            raise ValueError(f"Conversation {conv_idx} not found. Have {len(data)} conversations.")

        conv = data[conv_idx]
        dialogs = _extract_dialogs(conv)

        for dialog in dialogs:
            content = _build_memory_content(dialog)
            label = _build_memory_label(dialog)
            self.memory_client.store(content, label=label, metadata={
                "conversation": conv_idx,
                "session": dialog.get("session"),
                "dia_id": dialog.get("dia_id"),
            })

        return len(dialogs)

    def set_memory_client(self, client):
        """Set the memory backend for retrieval-augmented evaluation."""
        self.memory_client = client

    def set_llm_backend(self, backend: ModelBackend):
        """Set the LLM backend for answer generation."""
        self.llm_backend = backend

    def _load_data(self) -> list:
        if self._data is None:
            with open(self.data_path) as f:
                self._data = json.load(f)
        return self._data


# ---------------------------------------------------------------------------
# LoCoMo scoring (ported from the original benchmark)
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    s = s.replace(",", "")
    s = re.sub(r'\b(a|an|the|and)\b', ' ', s)
    s = ' '.join(s.split())
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return s.lower()


def _f1_score(prediction: str, ground_truth: str) -> float:
    """F1 with Porter stemming — matches LoCoMo eval exactly."""
    try:
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        pred_tokens = [ps.stem(w) for w in _normalize_answer(prediction).split()]
        gt_tokens = [ps.stem(w) for w in _normalize_answer(ground_truth).split()]
    except ImportError:
        # Fallback without stemming
        pred_tokens = _normalize_answer(prediction).split()
        gt_tokens = _normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _f1_multi(prediction: str, ground_truth: str) -> float:
    """Multi-answer F1 — splits on comma, takes max per sub-answer."""
    preds = [p.strip() for p in prediction.split(',')]
    gts = [g.strip() for g in ground_truth.split(',')]
    import numpy as np
    return float(np.mean([max([_f1_score(p, gt) for p in preds]) for gt in gts]))


def _score_qa(prediction: str, answer, category: int) -> float:
    """Score a single QA prediction against ground truth."""
    if isinstance(answer, list):
        answer = '; '.join(str(a) for a in answer)
    elif not isinstance(answer, str):
        answer = str(answer)

    if category == 3:
        answer = answer.split(';')[0].strip()

    if category in [2, 3, 4]:
        return _f1_score(prediction, answer)
    elif category == 1:
        return _f1_multi(prediction, answer)
    elif category == 5:
        pred_lower = prediction.lower()
        if 'no information available' in pred_lower or 'not mentioned' in pred_lower:
            return 1.0
        return 0.0
    return 0.0


def _extract_dialogs(data: dict) -> list:
    """Extract all dialog turns from a LoCoMo conversation."""
    conv = data.get('conversation', {})
    dialogs = []
    session_nums = sorted([
        int(k.split('_')[-1]) for k in conv.keys()
        if k.startswith('session_') and not k.endswith(('_date_time',))
    ])

    for sid in session_nums:
        session_key = f'session_{sid}'
        date_key = f'session_{sid}_date_time'
        if session_key not in conv:
            continue
        date_time = conv.get(date_key, "")
        for turn in conv[session_key]:
            text = f'{turn["speaker"]} said, "{turn["text"]}"'
            if turn.get('blip_caption'):
                text += f' and shared {turn["blip_caption"]}'
            dialogs.append({
                'dia_id': turn.get('dia_id', ''),
                'speaker': turn.get('speaker', ''),
                'text': turn.get('text', ''),
                'session': sid,
                'date_time': date_time,
                'formatted': text,
            })
    return dialogs


def _build_memory_label(dialog: dict) -> str:
    return f"D{dialog['dia_id']}|{dialog['date_time']}|{dialog['speaker']}"


def _build_memory_content(dialog: dict) -> str:
    return f"[{dialog['date_time']}] {dialog['formatted']}"
