"""
Scorers for benchwrap.
Honest scoring — no manipulation, no weighting, no "adjustments."
All metrics reported separately.
"""

import re
from abc import ABC, abstractmethod
from benchwrap.core.types import Score


class Scorer(ABC):
    """Base scorer interface."""

    @abstractmethod
    def score(self, prediction: str, reference: str, **context) -> Score:
        """Score a prediction against reference."""
        ...


class ExactMatch(Scorer):
    """Exact string match after normalization.
    
    Normalization: strip whitespace, lowercase.
    No fuzzy matching. No partial credit. Either it matches or it doesn't.
    """

    def score(self, prediction: str, reference: str, **context) -> Score:
        pred_norm = prediction.strip().lower()
        ref_norm = reference.strip().lower()
        em = 1.0 if pred_norm == ref_norm else 0.0
        return Score(
            exact_match=em,
            raw_prediction=prediction,
            raw_reference=reference,
            scoring_method="exact_match",
            matched=f"'{pred_norm}' vs '{ref_norm}'",
        )


class MCQScorer(Scorer):
    """Multiple-choice question scorer.
    
    Extracts the first letter (A/B/C/D) from the prediction and compares.
    Handles various answer formats: "A", "The answer is A", "(A)", "A.", etc.
    """

    def score(self, prediction: str, reference: str, **context) -> Score:
        pred_letter = self._extract_letter(prediction)
        ref_letter = self._extract_letter(reference)
        em = 1.0 if pred_letter == ref_letter else 0.0
        return Score(
            exact_match=em,
            accuracy=em,
            raw_prediction=prediction,
            raw_reference=reference,
            scoring_method="mcq_letter",
            matched=f"'{pred_letter}' vs '{ref_letter}'",
        )

    @staticmethod
    def _extract_letter(text: str) -> str:
        """Extract the first valid MCQ letter from text."""
        text = text.strip().upper()
        # Direct single letter
        if text in ("A", "B", "C", "D", "E", "F"):
            return text
        # Common patterns
        patterns = [
            r'(?:answer|Answer|ANSWER)[\s:]*([A-F])\b',
            r'(?:option|Option)\s*([A-F])\b',
            r'\(([A-F])\)',
            r'^([A-F])[.)\s]',
            r'\b([A-F])\b(?:\s*[:.]\s|\s*$)',
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1)
        # Last resort: first standalone letter
        letters = re.findall(r'\b([A-F])\b', text)
        return letters[-1] if letters else ""


class F1Scorer(Scorer):
    """Token-level F1 score (SQuAD-style).
    
    Compares overlap of normalized tokens between prediction and reference.
    Useful for open-ended QA where exact match is too strict.
    """

    def score(self, prediction: str, reference: str, **context) -> Score:
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        if not pred_tokens or not ref_tokens:
            f1 = 0.0
        else:
            common = set(pred_tokens) & set(ref_tokens)
            if not common:
                f1 = 0.0
            else:
                precision = len(common) / len(pred_tokens)
                recall = len(common) / len(ref_tokens)
                f1 = 2 * precision * recall / (precision + recall)

        pred_norm = prediction.strip().lower()
        ref_norm = reference.strip().lower()
        em = 1.0 if pred_norm == ref_norm else 0.0

        return Score(
            exact_match=em,
            f1=f1,
            raw_prediction=prediction,
            raw_reference=reference,
            scoring_method="f1_token",
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normalize and tokenize text."""
        text = text.lower().strip()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()


class NumericScorer(Scorer):
    """Numeric answer scorer for math benchmarks (GSM8K, AIME, etc.).
    
    Extracts the final number from both prediction and reference,
    then compares numerically. Handles commas, dollar signs, units, etc.
    """

    def score(self, prediction: str, reference: str, **context) -> Score:
        pred_num = self._extract_number(prediction)
        ref_num = self._extract_number(reference)

        if pred_num is None or ref_num is None:
            em = 0.0
        else:
            em = 1.0 if abs(pred_num - ref_num) < 1e-6 else 0.0

        return Score(
            exact_match=em,
            raw_prediction=prediction,
            raw_reference=reference,
            scoring_method="numeric",
            matched=f"{pred_num} vs {ref_num}",
        )

    @staticmethod
    def _extract_number(text: str) -> float | None:
        """Extract the last number from text."""
        # Remove commas, dollar signs, percent signs
        text = text.replace(",", "").replace("$", "").replace("%", "")
        # Find all numbers (including decimals and negatives)
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                return None
        return None


class ReasoningScorer(Scorer):
    """Scorer for reasoning models with chain-of-thought output.
    
    Extracts the final answer from CoT using configurable patterns,
    then scores using a wrapped scorer (MCQ, numeric, or exact match).
    
    This is EXPLICIT — no hidden extraction. The patterns are logged.
    """

    def __init__(
        self,
        inner: Scorer,
        patterns: list[str] | None = None,
        lookback_chars: int = 1000,
    ):
        self.inner = inner
        self.lookback = lookback_chars
        self.patterns = patterns or [
            r'(?:answer|Answer|ANSWER)[\s:]*([A-Fa-f0-9]+)\b',
            r'\\boxed\{([^}]+)\}',
            r'(?:option|Option)\s*([A-Fa-f])\b',
            r'(?:the answer is|The answer is)\s*([^\n.]+)',
        ]

    def score(self, prediction: str, reference: str, **context) -> Score:
        extracted = self.extract(prediction)
        score = self.inner.score(extracted, reference, **context)
        score.scoring_method = f"reasoning({self.inner.__class__.__name__})"
        score.raw_prediction = prediction  # Keep full CoT for auditing
        score.custom["extracted_answer"] = extracted
        return score

    def extract(self, response: str) -> str:
        """Extract final answer from chain-of-thought response."""
        tail = response[-self.lookback:]
        for pattern in self.patterns:
            m = re.search(pattern, tail)
            if m:
                return m.group(1).strip()
        # Fallback: return tail stripped
        return tail.strip()


def get_scorer(benchmark_type: str = "mcq", **kwargs) -> Scorer:
    """Factory for scorers based on benchmark type.
    
    Types:
        mcq        — Multiple choice (MMLU, HellaSwag, ARC)
        numeric    — Numeric answer (GSM8K, AIME)
        exact      — Exact string match
        f1         — Token-level F1 (SQuAD-style)
        reasoning  — CoT extraction + inner scorer
    """
    scorers = {
        "mcq": MCQScorer,
        "numeric": NumericScorer,
        "exact": ExactMatch,
        "f1": F1Scorer,
    }

    if benchmark_type == "reasoning":
        inner_type = kwargs.pop("inner", "mcq")
        inner = get_scorer(inner_type, **kwargs)
        return ReasoningScorer(inner=inner, **kwargs)

    if benchmark_type not in scorers:
        raise ValueError(
            f"Unknown scorer type '{benchmark_type}'. "
            f"Supported: {', '.join(scorers.keys())}, reasoning"
        )
    return scorers[benchmark_type]()
