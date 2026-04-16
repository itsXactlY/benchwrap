"""
MMLU (Massive Multitask Language Understanding) adapter.
Standalone implementation — no lm-eval dependency.

57 subjects across STEM, humanities, social sciences, and more.
Multiple choice (A/B/C/D). Standard 5-shot evaluation.

Source: https://huggingface.co/datasets/cais/mmlu
"""

import os
import json
import urllib.request
from typing import Iterator, Optional

from benchwrap.core.adapter import BenchmarkAdapter
from benchwrap.core.types import Sample, Prompt, Score
from benchwrap.core.scorer import MCQScorer


MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

MMLU_CATEGORIES = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics",
        "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history",
        "high_school_us_history", "high_school_world_history",
        "philosophy", "prehistory", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_aging", "human_sexuality", "international_law",
        "jurisprudence", "logical_fallacies", "professional_psychology",
        "public_relations", "security_studies", "sociology",
        "us_foreign_policy",
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "professional_accounting", "professional_law",
        "professional_medicine", "virology",
    ],
}

LETTERS = ["A", "B", "C", "D"]


class MMLUAdapter(BenchmarkAdapter):
    """MMLU (Massive Multitask Language Understanding).
    
    57 academic subjects, multiple choice (A/B/C/D).
    Zero or few-shot evaluation.
    
    Data source: GitHub raw CSV (cais/mmlu repository).
    Falls back to HuggingFace datasets if GitHub unavailable.
    """

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir or os.path.expanduser(
            "~/.cache/benchwrap/mmlu"
        )
        self._scorer = MCQScorer()

    def name(self) -> str:
        return "mmlu"

    def datasets(self) -> list[str]:
        return ["all"] + MMLU_SUBJECTS + list(MMLU_CATEGORIES.keys())

    def default_dataset(self) -> str:
        return "all"  # MMLU default is 'all' — run across all subjects

    def load(
        self,
        dataset: str = "all",
        split: str = "test",
        limit: Optional[int] = None,
    ) -> Iterator[Sample]:
        """Load MMLU samples.
        
        dataset: 'all', a subject name, or a category name (STEM, Humanities, etc.)
        split: 'test', 'dev', or 'validation'
        """
        subjects = self._resolve_subjects(dataset)
        count = 0

        for subject in subjects:
            data = self._load_subject(subject, split)
            for row in data:
                if limit and count >= limit:
                    return
                yield self._row_to_sample(row, subject)
                count += 1

    def format_prompt(
        self,
        sample: Sample,
        fewshot: Optional[list[Sample]] = None,
    ) -> Prompt:
        """Format MMLU sample as a multiple-choice prompt.
        
        IMPORTANT: The answer is NOT included in the prompt.
        The model must choose from A/B/C/D.
        """
        lines = []

        # Few-shot examples (if provided)
        if fewshot:
            for fs in fewshot:
                lines.append(f"The following is a multiple choice question.")
                lines.append(f"Question: {fs.input}")
                for i, choice in enumerate(fs.choices or []):
                    lines.append(f"{LETTERS[i]}. {choice}")
                lines.append(f"Answer: {fs.reference}")
                lines.append("")

        # The actual question
        lines.append("The following is a multiple choice question.")
        lines.append(f"Question: {sample.input}")
        for i, choice in enumerate(sample.choices or []):
            lines.append(f"{LETTERS[i]}. {choice}")
        lines.append("Answer:")

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
        """Score MCQ answer (A/B/C/D extraction)."""
        return self._scorer.score(prediction, reference)

    def fewshot_pool(self, dataset: str, split: str = "train") -> list[Sample]:
        """Load dev split for few-shot examples.
        
        MMLU uses a separate 'dev' split (5 examples per subject) for few-shot.
        Falls back to test split if dev is unavailable.
        """
        subjects = self._resolve_subjects(dataset)
        pool = []
        for subject in subjects:
            try:
                data = self._load_subject(subject, "dev")
            except Exception:
                # Dev split not available, use test (with warning)
                try:
                    data = self._load_subject(subject, "test")
                except Exception:
                    continue
            for row in data[:5]:  # Max 5 per subject
                pool.append(self._row_to_sample(row, subject))
        return pool

    def extract_answer(self, response: str, sample: Sample) -> str:
        """Extract letter answer from response."""
        return response.strip()

    def _resolve_subjects(self, dataset: str) -> list[str]:
        """Resolve dataset name to list of subjects."""
        if dataset == "all":
            return MMLU_SUBJECTS
        if dataset in MMLU_CATEGORIES:
            return MMLU_CATEGORIES[dataset]
        if dataset in MMLU_SUBJECTS:
            return [dataset]
        raise ValueError(
            f"Unknown MMLU dataset '{dataset}'. "
            f"Use 'all', a category ({', '.join(MMLU_CATEGORIES.keys())}), "
            f"or a subject name."
        )

    def _load_subject(self, subject: str, split: str) -> list[dict]:
        """Load data for a subject from HuggingFace dataset API."""
        return self._fetch_from_hf(subject, split)

    def _row_to_sample(self, row: dict, subject: str) -> Sample:
        """Convert a row dict to a Sample."""
        question = row.get("question", "").strip()
        choices = [c.strip() for c in row.get("choices", [])]
        answer = row.get("answer", "")

        # MMLU answer is an index (0-3) → map to letter
        if isinstance(answer, int) or (isinstance(answer, str) and answer.isdigit()):
            idx = int(answer)
            answer = LETTERS[idx] if 0 <= idx < len(LETTERS) else str(answer)
        elif isinstance(answer, str) and answer.lower() in ("a", "b", "c", "d"):
            answer = answer.upper()

        # Determine category
        category = "Other"
        for cat, subjects in MMLU_CATEGORIES.items():
            if subject in subjects:
                category = cat
                break

        return Sample(
            id=f"mmlu_{subject}_{hash(question) % 100000}",
            input=question,
            reference=str(answer),
            choices=choices,
            metadata={
                "subject": subject,
                "category": category,
            },
        )

    def _fetch_from_hf(self, subject: str, split: str) -> list[dict]:
        """Fetch MMLU data from HuggingFace dataset server API.
        
        Uses the /rows endpoint — stdlib-only, no pip dependencies.
        Paginates through all rows if needed.
        """
        cache_file = os.path.join(self.cache_dir, f"{subject}_{split}.jsonl")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Check cache
        if os.path.exists(cache_file):
            rows = []
            with open(cache_file, "r") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            return rows

        print(f"[mmlu] Fetching {subject} ({split}) from HuggingFace...")
        all_rows = []
        offset = 0
        batch_size = 100

        while True:
            url = (
                f"https://datasets-server.huggingface.co/rows"
                f"?dataset=cais/mmlu&config={subject}&split={split}"
                f"&offset={offset}&length={batch_size}"
            )
            req = urllib.request.Request(
                url, headers={"User-Agent": "benchwrap/0.1.0"}
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
            except Exception as e:
                if offset == 0:
                    raise RuntimeError(
                        f"Failed to fetch MMLU data for {subject}. Error: {e}"
                    )
                break

            rows = data.get("rows", [])
            if not rows:
                break

            for r in rows:
                row_data = r.get("row", {})
                all_rows.append(row_data)

            offset += len(rows)
            if len(rows) < batch_size:
                break

        # Cache to disk
        with open(cache_file, "w") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")

        print(f"[mmlu] Cached {len(all_rows)} rows for {subject} ({split})")
        return all_rows
