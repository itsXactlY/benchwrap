"""
BenchmarkAdapter — the plug-in contract.
Any benchmark implements these 5 methods. That's the entire API.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional
from benchwrap.core.types import Sample, Prompt, Score


class BenchmarkAdapter(ABC):
    """Base class for all benchmark adapters.

    To create a new adapter, subclass and implement:
        name()          — human-readable name
        datasets()      — available dataset names
        load()          — yield Sample objects
        format_prompt() — turn a Sample into a Prompt
        score()         — evaluate a prediction against reference
    """

    @abstractmethod
    def name(self) -> str:
        """Human-readable benchmark name. e.g. 'MMLU', 'GSM8K', 'LoCoMo'"""
        ...

    @abstractmethod
    def datasets(self) -> list[str]:
        """List available dataset/subset names.
        
        Return ['default'] if there's only one dataset.
        """
        ...

    @abstractmethod
    def load(
        self,
        dataset: str = "default",
        split: str = "test",
        limit: Optional[int] = None,
    ) -> Iterator[Sample]:
        """Load samples from the dataset.
        
        Args:
            dataset: Dataset name from datasets()
            split: Data split (train/test/validation)
            limit: Max samples to load (None = all)
        
        Yields:
            Sample objects with id, input, reference, and metadata.
            
        IMPORTANT: reference must NEVER be included in the prompt.
        The adapter's format_prompt() method controls what the model sees.
        """
        ...

    @abstractmethod
    def format_prompt(
        self,
        sample: Sample,
        fewshot: Optional[list[Sample]] = None,
    ) -> Prompt:
        """Format a sample into a prompt for the model.
        
        Args:
            sample: The sample to format
            fewshot: Optional list of few-shot examples.
                     These are EXPLICIT — the caller decides whether to use them.
                     No hidden injection. No baked-in few-shot.
        
        Returns:
            Prompt with system, messages, and raw_text for logging.
            
        CRITICAL: raw_text must be the EXACT text the model will see.
        This is the transparency guarantee.
        """
        ...

    @abstractmethod
    def score(
        self,
        prediction: str,
        reference: str,
        sample: Sample,
    ) -> Score:
        """Score a prediction against the reference answer.
        
        Args:
            prediction: What the model generated
            reference: The correct answer (from Sample.reference)
            sample: The original sample (for context/metadata)
        
        Returns:
            Score with exact_match, f1, accuracy, and scoring_method.
            
        CRITICAL: No manipulation. No weighting. No "adjustments."
        Score exactly what the model outputted. Report the method used.
        """
        ...

    def fewshot_pool(self, dataset: str, split: str = "train") -> list[Sample]:
        """Optional: return samples that can be used for few-shot examples.
        
        Default: empty (no few-shot). Override to enable.
        These samples must NOT overlap with test samples.
        """
        return []

    def extract_answer(self, response: str, sample: Sample) -> str:
        """Optional: extract the final answer from a verbose response.
        
        Default: return the response as-is.
        Override for benchmarks where models generate reasoning before the answer.
        This is EXPLICIT — no hidden extraction logic.
        """
        return response.strip()
