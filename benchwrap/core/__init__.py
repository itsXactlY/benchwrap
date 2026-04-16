"""
benchwrap.core — Core engine components.
"""

from benchwrap.core.types import Sample, Prompt, Prediction, Score, EvalResult, Result
from benchwrap.core.adapter import BenchmarkAdapter
from benchwrap.core.model import ModelBackend, OllamaBackend, OpenAICompatBackend, parse_backend
from benchwrap.core.scorer import Scorer, ExactMatch, MCQScorer, F1Scorer, NumericScorer, get_scorer
from benchwrap.core.engine import EvaluationEngine
from benchwrap.core.reporter import Reporter
