"""
EvoMem adapter for benchwrap.
Streaming memory benchmark with Dream Engine consolidation.

Tests memory-augmented knowledge retrieval:
  - MMLU-Pro: 57 academic subjects with memory assistance
  - GPQA Diamond: graduate-level science questions
  - AIME 2024: math competition (0% for small models)

Source: ~/projects/evo_mem/
"""

import json
import re
from pathlib import Path
from typing import Iterator, Optional

from benchwrap.core.adapter import BenchmarkAdapter
from benchwrap.core.types import Sample, Prompt, Score
from benchwrap.core.model import ModelBackend


EVOMEM_DIR = Path.home() / "projects/evo_mem"

# Available datasets
DATASETS = {
    "mmlu-pro": {
        "description": "MMLU-Pro academic subjects (57 subjects)",
        "scorer": "mcq",
    },
    "gpqa-diamond": {
        "description": "GPQA Diamond graduate-level science",
        "scorer": "mcq",
    },
    "aime-2024": {
        "description": "AIME 2024 math competition",
        "scorer": "numeric",
    },
}


class EvoMemAdapter(BenchmarkAdapter):
    """EvoMem — Streaming memory with Dream Engine consolidation.
    
    Tests knowledge retrieval from a memory system that has been
    pre-loaded with reference material. The memory system stores
    facts, and the LLM uses recalled facts to answer questions.
    
    Data: ~/projects/evo_mem/data/
    
    Memory backend is injected via set_memory_client().
    Knowledge is pre-loaded into memory before evaluation.
    """

    def __init__(
        self,
        memory_client=None,
        llm_backend: ModelBackend | None = None,
    ):
        self.memory_client = memory_client
        self.llm_backend = llm_backend
        self._data_cache = {}

    def name(self) -> str:
        return "evomem"

    def datasets(self) -> list[str]:
        return ["all"] + list(DATASETS.keys())

    def load(
        self,
        dataset: str = "all",
        split: str = "test",
        limit: Optional[int] = None,
    ) -> Iterator[Sample]:
        """Load EvoMem evaluation samples."""
        if dataset == "all":
            dataset_list = list(DATASETS.keys())
        elif dataset in DATASETS:
            dataset_list = [dataset]
        else:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Available: {', '.join(DATASETS.keys())}"
            )

        count = 0
        for ds_name in dataset_list:
            samples = self._load_dataset(ds_name)
            for sample in samples:
                if limit and count >= limit:
                    return
                sample.metadata["dataset_name"] = ds_name
                sample.metadata["scorer_type"] = DATASETS[ds_name]["scorer"]
                yield sample
                count += 1

    def format_prompt(
        self,
        sample: Sample,
        fewshot: Optional[list[Sample]] = None,
    ) -> Prompt:
        """Format as a memory-augmented prompt.
        
        Queries the memory system for relevant context, then
        formats the question with recalled facts.
        """
        question = sample.input
        choices = sample.choices
        context = ""

        if self.memory_client:
            try:
                results = self.memory_client.recall(question, top_k=5)
                if results:
                    context_parts = []
                    for r in results:
                        content = r.get("content", r.get("text", str(r)))
                        context_parts.append(content)
                    context = "\n".join(context_parts)
            except Exception:
                pass

        # Build the prompt
        parts = []
        if context:
            parts.append(f"Relevant information:\n{context}")
            parts.append("")

        if choices:
            parts.append(f"Question: {question}")
            for i, choice in enumerate(choices):
                letter = chr(65 + i)  # A, B, C, D
                parts.append(f"{letter}. {choice}")
            parts.append("")
            parts.append("Answer:")
        else:
            parts.append(f"Question: {question}")
            parts.append("")
            parts.append("Answer:")

        text = "\n".join(parts)
        return Prompt(
            system=None,
            messages=[{"role": "user", "content": text}],
            raw_text=text[:500],
        )

    def score(
        self,
        prediction: str,
        reference: str,
        sample: Sample,
    ) -> Score:
        """Score based on dataset type."""
        scorer_type = sample.metadata.get("scorer_type", "mcq")

        if scorer_type == "mcq":
            return self._score_mcq(prediction, reference)
        elif scorer_type == "numeric":
            return self._score_numeric(prediction, reference)
        else:
            return self._score_exact(prediction, reference)

    def _score_mcq(self, prediction: str, reference: str) -> Score:
        """Score MCQ answer."""
        # Extract letter from prediction
        pred_letter = ""
        pred_upper = prediction.strip().upper()
        patterns = [
            r'(?:answer|Answer|ANSWER)[\s:]*([A-F])\b',
            r'(?:option|Option)\s*([A-F])\b',
            r'\(([A-F])\)',
            r'^([A-F])[.)\s]',
            r'\b([A-F])\b',
        ]
        for pat in patterns:
            m = re.search(pat, pred_upper)
            if m:
                pred_letter = m.group(1)
                break
        if not pred_letter and pred_upper in ("A", "B", "C", "D", "E", "F"):
            pred_letter = pred_upper

        ref_letter = reference.strip().upper()
        em = 1.0 if pred_letter == ref_letter else 0.0

        return Score(
            exact_match=em,
            accuracy=em,
            raw_prediction=prediction,
            raw_reference=reference,
            scoring_method="evomem_mcq",
            matched=f"'{pred_letter}' vs '{ref_letter}'",
        )

    def _score_numeric(self, prediction: str, reference: str) -> Score:
        """Score numeric answer (AIME-style)."""
        # Extract number from prediction
        pred_num = re.findall(r'-?\d+(?:\.\d+)?', prediction.replace(",", ""))
        ref_num = re.findall(r'-?\d+(?:\.\d+)?', str(reference).replace(",", ""))

        if not pred_num or not ref_num:
            return Score(
                exact_match=0.0,
                raw_prediction=prediction,
                raw_reference=reference,
                scoring_method="evomem_numeric",
            )

        em = 1.0 if abs(float(pred_num[-1]) - float(ref_num[-1])) < 1e-6 else 0.0
        return Score(
            exact_match=em,
            raw_prediction=prediction,
            raw_reference=reference,
            scoring_method="evomem_numeric",
            matched=f"{pred_num[-1]} vs {ref_num[-1]}",
        )

    def _score_exact(self, prediction: str, reference: str) -> Score:
        """Score exact match."""
        em = 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0
        return Score(
            exact_match=em,
            raw_prediction=prediction,
            raw_reference=reference,
            scoring_method="evomem_exact",
        )

    def _load_dataset(self, ds_name: str) -> list[Sample]:
        """Load samples from an EvoMem dataset."""
        cache_key = ds_name
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        # Look for data files in the EvoMem project
        data_dir = EVOMEM_DIR / "data"
        samples = []

        if ds_name == "mmlu-pro":
            samples = self._load_mmlu_pro(data_dir)
        elif ds_name == "gpqa-diamond":
            samples = self._load_gpqa(data_dir)
        elif ds_name == "aime-2024":
            samples = self._load_aime(data_dir)

        self._data_cache[cache_key] = samples
        return samples

    def _load_mmlu_pro(self, data_dir: Path) -> list[Sample]:
        """Load MMLU-Pro samples from EvoMem data.
        
        Handles both:
        - Directory of per-subject JSON files: data/mmlu_pro/*.json
        - Single flat JSON file: data/mmlu_pro.json
        """
        samples = []

        def _parse_items(data, source_name):
            parsed = []
            if isinstance(data, list):
                for i, item in enumerate(data[:50]):  # Cap per file
                    question = item.get("question", "")
                    choices = item.get("choices", item.get("options", []))
                    answer = item.get("answer", "")
                    if isinstance(answer, int):
                        answer = chr(65 + answer)  # 0 -> A, 1 -> B, etc.
                    # Handle answer-as-index into choices
                    if isinstance(answer, str) and answer.isdigit() and choices:
                        idx = int(answer)
                        if 0 <= idx < len(choices):
                            answer = chr(65 + idx)
                    parsed.append(Sample(
                        id=f"evomem_mmlu_{source_name}_{i}",
                        input=question,
                        reference=str(answer),
                        choices=choices if choices else None,
                        metadata={"subject": source_name},
                    ))
            return parsed

        # Try directory of per-subject files first
        mmlu_dir = data_dir / "mmlu_pro"
        if not mmlu_dir.exists():
            mmlu_dir = data_dir / "mmlu-pro"
        if mmlu_dir.exists() and mmlu_dir.is_dir():
            for f in mmlu_dir.glob("*.json*"):
                with open(f) as fh:
                    try:
                        data = json.load(fh)
                    except json.JSONDecodeError:
                        continue
                samples.extend(_parse_items(data, f.stem))
            if samples:
                return samples

        # Fallback: flat JSON file (e.g. data/mmlu_pro.json)
        for candidate in ["mmlu_pro.json", "mmlu-pro.json"]:
            flat_file = data_dir / candidate
            if flat_file.exists():
                with open(flat_file) as fh:
                    try:
                        data = json.load(fh)
                    except json.JSONDecodeError:
                        continue
                samples = _parse_items(data, flat_file.stem)
                break

        return samples

    def _load_gpqa(self, data_dir: Path) -> list[Sample]:
        """Load GPQA Diamond samples."""
        samples = []
        gpqa_file = data_dir / "gpqa" / "gpqa_diamond.json"
        if not gpqa_file.exists():
            gpqa_file = data_dir / "gpqa_diamond.json"
        if not gpqa_file.exists():
            return samples

        with open(gpqa_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data[:50]):
                question = item.get("question", "")
                choices = item.get("choices", [])
                answer = item.get("answer", "")
                samples.append(Sample(
                    id=f"evomem_gpqa_{i}",
                    input=question,
                    reference=str(answer),
                    choices=choices if choices else None,
                ))
        return samples

    def _load_aime(self, data_dir: Path) -> list[Sample]:
        """Load AIME 2024 samples."""
        samples = []
        aime_file = data_dir / "aime" / "aime_2024.json"
        if not aime_file.exists():
            aime_file = data_dir / "aime_2024.json"
        if not aime_file.exists():
            return samples

        with open(aime_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data[:50]):
                question = item.get("problem", item.get("question", ""))
                answer = item.get("answer", "")
                samples.append(Sample(
                    id=f"evomem_aime_{i}",
                    input=question,
                    reference=str(answer),
                    metadata={"type": "math"},
                ))
        return samples

    def set_memory_client(self, client):
        """Set the memory backend."""
        self.memory_client = client

    def set_llm_backend(self, backend: ModelBackend):
        """Set the LLM backend."""
        self.llm_backend = backend
