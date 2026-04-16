"""
Core data types for benchwrap.
Every type is transparent and inspectable — no hidden state.
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class Sample:
    """A single benchmark sample. The atomic unit of evaluation.
    
    reference is NEVER shown to the model — it's only used for scoring.
    """
    id: str
    input: str
    reference: str
    choices: Optional[list[str]] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "input": self.input,
            "reference": self.reference,
            "choices": self.choices,
            "metadata": self.metadata,
        }


@dataclass
class Prompt:
    """A formatted prompt ready to send to a model.
    
    raw_text is the EXACT text that will be sent — fully inspectable.
    No hidden injection. No surprise few-shot. What you see is what the model gets.
    """
    system: Optional[str]
    messages: list[dict]  # [{"role": "user", "content": "..."}]
    fewshot_ids: list[str] = field(default_factory=list)  # IDs of fewshot samples used
    raw_text: str = ""  # Reconstructed full prompt for logging

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "messages": self.messages,
            "fewshot_ids": self.fewshot_ids,
            "raw_text": self.raw_text,
        }


@dataclass
class Prediction:
    """Raw model output + metadata."""
    text: str
    model: str = ""
    backend: str = ""
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    raw_response: dict = field(default_factory=dict)  # Full API response for debugging

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model": self.model,
            "backend": self.backend,
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }


@dataclass
class Score:
    """Honest, unmanipulated score for a single sample.
    
    All metrics reported separately — no blending, no weighting.
    scoring_method describes HOW it was scored — fully transparent.
    """
    exact_match: float  # 0.0 or 1.0
    f1: Optional[float] = None  # Token-level F1 if applicable
    accuracy: Optional[float] = None  # MCQ accuracy if applicable
    custom: dict = field(default_factory=dict)  # Benchmark-specific metrics
    raw_prediction: str = ""
    raw_reference: str = ""
    scoring_method: str = "exact_match"
    matched: str = ""  # What was actually compared (after normalization)

    def primary(self) -> float:
        """Return the primary metric for this score."""
        if self.accuracy is not None:
            return self.accuracy
        return self.exact_match

    def to_dict(self) -> dict:
        return {
            "exact_match": self.exact_match,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "custom": self.custom,
            "scoring_method": self.scoring_method,
            "matched": self.matched,
            "raw_prediction": self.raw_prediction[:200],  # Truncate for storage
            "raw_reference": self.raw_reference,
        }


@dataclass
class EvalResult:
    """Result of evaluating a single sample."""
    sample_id: str
    score: Score
    prediction: Prediction
    prompt: Prompt
    dataset: str = ""
    adapter: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "dataset": self.dataset,
            "adapter": self.adapter,
            "score": self.score.to_dict(),
            "prediction": self.prediction.to_dict(),
            "prompt": self.prompt.to_dict(),
            "timestamp": self.timestamp,
        }


@dataclass
class Result:
    """Aggregated results for a full benchmark run."""
    adapter: str
    model: str
    backend: str
    dataset: str
    num_samples: int
    metrics: dict  # {"exact_match": 0.75, "f1": 0.82, ...}
    per_category: dict  # {"category_a": {"exact_match": 0.8, "n": 50}, ...}
    eval_results: list[EvalResult]
    config: dict  # Full config used for this run
    timestamp: float = field(default_factory=time.time)
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "adapter": self.adapter,
            "model": self.model,
            "backend": self.backend,
            "dataset": self.dataset,
            "num_samples": self.num_samples,
            "metrics": self.metrics,
            "per_category": self.per_category,
            "config": self.config,
            "timestamp": self.timestamp,
            "duration_s": self.duration_s,
            "samples": [r.to_dict() for r in self.eval_results],
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"{'=' * 60}",
            f"  {self.adapter} — {self.dataset}",
            f"  Model: {self.model} ({self.backend})",
            f"  Samples: {self.num_samples}",
            f"  Duration: {self.duration_s:.1f}s",
            f"{'=' * 60}",
        ]
        # Metrics that are rates (0-1) shown as percentages
        rate_metrics = {"exact_match", "accuracy", "f1"}
        for metric, value in self.metrics.items():
            if isinstance(value, float):
                if metric in rate_metrics:
                    lines.append(f"  {metric:20s}: {value:.4f} ({value * 100:.1f}%)")
                else:
                    lines.append(f"  {metric:20s}: {value:.2f}")
            else:
                lines.append(f"  {metric:20s}: {value}")
        if self.per_category:
            lines.append(f"\n  {'Category':20s} | {'EM':>8s} | {'N':>6s}")
            lines.append(f"  {'-' * 20}-+-{'-' * 8}-+-{'-' * 6}")
            for cat, vals in sorted(self.per_category.items()):
                em = vals.get("exact_match", vals.get("accuracy", 0))
                n = vals.get("n", 0)
                lines.append(f"  {str(cat):20s} | {em:7.1%} | {n:6d}")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)
