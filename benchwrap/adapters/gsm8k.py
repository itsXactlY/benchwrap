"""
GSM8K (Grade School Math 8K) adapter.
Standalone implementation — no external dependencies.

8.5K grade school math word problems requiring multi-step reasoning.
Free-form numeric answer. Standard evaluation: exact match on final number.

Source: https://huggingface.co/datasets/openai/gsm8k
"""

import os
import json
import re
import urllib.request
from typing import Iterator, Optional

from benchwrap.core.adapter import BenchmarkAdapter
from benchwrap.core.types import Sample, Prompt, Score
from benchwrap.core.scorer import NumericScorer


class GSM8KAdapter(BenchmarkAdapter):
    """GSM8K (Grade School Math 8K).
    
    8,500 linguistically diverse grade school math word problems.
    Requires 2-8 steps of basic arithmetic to solve.
    Free-form numeric answer (e.g., "42", "3.5", "-17").
    
    Source: OpenAI, released with chain-of-thought rationales.
    """

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir or os.path.expanduser(
            "~/.cache/benchwrap/gsm8k"
        )
        self._scorer = NumericScorer()

    def name(self) -> str:
        return "gsm8k"

    def default_eval_config(self) -> dict:
        # Wei et al. canonical CoT eval: 8-shot with reasoning traces.
        return {"fewshot": 8}

    def datasets(self) -> list[str]:
        return ["main", "socratic"]

    def load(
        self,
        dataset: str = "main",
        split: str = "test",
        limit: Optional[int] = None,
    ) -> Iterator[Sample]:
        """Load GSM8K samples.
        
        dataset: 'main' or 'socratic'
        split: 'train' or 'test'
        """
        data = self._load_data(dataset, split)
        count = 0

        for item in data:
            if limit and count >= limit:
                return
            yield self._item_to_sample(item, dataset, count)
            count += 1

    def format_prompt(
        self,
        sample: Sample,
        fewshot: Optional[list[Sample]] = None,
    ) -> Prompt:
        """Format GSM8K as a math word problem prompt.
        
        The model should work through the problem and provide
        the final numeric answer.
        """
        lines = []

        # Few-shot examples
        if fewshot:
            for fs in fewshot:
                lines.append(f"Question: {fs.input}")
                # Include rationale if available in metadata
                if "rationale" in fs.metadata:
                    lines.append(fs.metadata["rationale"])
                lines.append(f"The answer is {fs.reference}.")
                lines.append("")

        # The actual question
        lines.append(f"Question: {sample.input}")
        lines.append("")
        lines.append("Let me work through this step by step.")

        text = "\n".join(lines)
        return Prompt(
            system=None,
            messages=[{"role": "user", "content": text}],
            fewshot_ids=[s.id for s in (fewshot or [])],
            raw_text=text,
        )

    def score(
        self,
        prediction: str,
        reference: str,
        sample: Sample,
    ) -> Score:
        """Score numeric answer — extract final number from CoT first."""
        extracted = self.extract_answer(prediction, sample)
        return self._scorer.score(extracted, reference)

    def fewshot_pool(self, dataset: str, split: str = "train") -> list[Sample]:
        """Load train split for few-shot examples (Wei et al. uses 8)."""
        data = self._load_data(dataset, "train")
        return [self._item_to_sample(item, dataset, i)
                for i, item in enumerate(data[:8])]

    def extract_answer(self, response: str, sample: Sample) -> str:
        """Extract the FINAL numeric answer from a (possibly verbose) response.

        Reasoning models write many intermediate '=' equations before the
        final answer. We prefer the last match of strong markers, falling
        through to the last number in the text.
        """
        num = r'-?\d{1,}(?:[,]\d{3})*(?:\.\d+)?'

        # 1) GSM8K-style: '#### NUMBER' (always wins; take last if multiple)
        ms = list(re.finditer(rf'####\s*({num})', response))
        if ms:
            return ms[-1].group(1).replace(",", "")

        # 2) LaTeX \boxed{NUMBER} — common in reasoning-model output
        ms = list(re.finditer(rf'\\boxed\{{\s*\$?\s*({num})\s*\}}', response))
        if ms:
            return ms[-1].group(1).replace(",", "")

        # 3) 'the answer is NUMBER' / 'Answer: NUMBER' / 'Final answer: NUMBER'
        ms = list(re.finditer(
            rf'(?ix) (?: the\s+answer\s+is | answer\s*[:=] | final\s+answer\s*[:=] )'
            rf'\s*\$?\s*({num})',
            response,
        ))
        if ms:
            return ms[-1].group(1).replace(",", "")

        # 4) Last number anywhere — works for "...so it's 42." style endings
        nums = re.findall(num, response)
        if nums:
            return nums[-1].replace(",", "")

        return response.strip()

    def _load_data(self, dataset: str, split: str) -> list[dict]:
        """Load GSM8K JSONL data."""
        filepath = self._get_or_download(dataset, split)
        items = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def _item_to_sample(self, item: dict, dataset: str, idx: int) -> Sample:
        """Convert a JSONL item to a Sample."""
        question = item.get("question", "")
        answer_text = item.get("answer", "")

        # Extract the final numeric answer from #### format
        m = re.search(r'####\s*(-?\d+(?:\.\d+)?)', answer_text)
        reference = m.group(1) if m else answer_text.strip()

        # Extract rationale (everything before ####)
        rationale = answer_text
        if m:
            rationale = answer_text[:m.start()].strip()

        return Sample(
            id=f"gsm8k_{dataset}_{idx}",
            input=question,
            reference=reference,
            metadata={
                "dataset": dataset,
                "rationale": rationale,
                "full_answer": answer_text,
            },
        )

    def _get_or_download(self, dataset: str, split: str) -> str:
        """Download GSM8K data if not cached."""
        os.makedirs(self.cache_dir, exist_ok=True)
        filename = f"{dataset}_{split}.jsonl"
        filepath = os.path.join(self.cache_dir, filename)

        if os.path.exists(filepath):
            return filepath

        # Download from HuggingFace
        if dataset == "main":
            url = (
                f"https://raw.githubusercontent.com/openai/grade-school-math/master/"
                f"grade_school_math/data/{split}.jsonl"
            )
        else:
            url = (
                f"https://raw.githubusercontent.com/openai/grade-school-math/master/"
                f"grade_school_math/data/{split}_socratic.jsonl"
            )

        print(f"[gsm8k] Downloading {dataset} ({split})...")
        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download GSM8K data. URL: {url}. Error: {e}"
            )

        return filepath
