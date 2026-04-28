"""
EvaluationEngine — the orchestrator.
Loads adapter → loads samples → formats prompts → generates predictions → scores → reports.
Every step is logged. Every prompt is saved. Full transparency.
"""

import time
import json
import os
from typing import Optional

from benchwrap.core.types import Sample, EvalResult, Result
from benchwrap.core.adapter import BenchmarkAdapter
from benchwrap.core.model import ModelBackend
from benchwrap.core.scorer import Scorer, get_scorer


class _AdapterScorer(Scorer):
    """Wraps adapter.score() as a Scorer so the engine uses adapter-specific scoring."""
    def __init__(self, adapter):
        self.adapter = adapter
    def score(self, prediction: str, reference: str, sample=None, **kwargs) -> 'Score':
        return self.adapter.score(prediction, reference, sample)


class EvaluationEngine:
    """Runs a benchmark evaluation end-to-end.
    
    Usage:
        engine = EvaluationEngine(adapter, backend, scorer)
        result = engine.run(dataset="mmlu", limit=100, fewshot=5)
        print(result.summary())
    """

    def __init__(
        self,
        adapter: BenchmarkAdapter,
        backend: ModelBackend,
        scorer: Scorer | None = None,
        verbose: bool = False,
        save_dir: str | None = None,
    ):
        self.adapter = adapter
        self.backend = backend
        self.scorer = scorer
        self.verbose = verbose
        self.save_dir = save_dir

    def run(
        self,
        dataset: str | None = None,
        split: str = "test",
        limit: Optional[int] = None,
        fewshot: int | None = None,
        **gen_kwargs,
    ) -> Result:
        """Run the full evaluation pipeline.
        
        Args:
            dataset: Dataset name from adapter.datasets(). None = adapter default.
            split: Data split (train/test/validation)
            limit: Max samples to evaluate
            fewshot: Number of few-shot examples (0 = zero-shot)
            **gen_kwargs: Passed to backend.generate() (temperature, max_tokens, etc.)
        
        Returns:
            Result with aggregated metrics and per-sample details.
        """
        start_time = time.time()
        adapter_name = self.adapter.name()
        model_name = self.backend.model_id()
        backend_name = self.backend.name()

        # Honor each adapter's canonical eval protocol (fewshot count, temp,
        # etc.). Caller-provided values still win — adapter only fills gaps.
        defaults = self.adapter.default_eval_config() if hasattr(
            self.adapter, "default_eval_config"
        ) else {}
        if fewshot is None:
            fewshot = defaults.get("fewshot", 0)
        for k, v in defaults.items():
            if k == "fewshot":
                continue
            gen_kwargs.setdefault(k, v)

        # Resolve dataset — use adapter default if not specified
        if dataset is None:
            dataset = self.adapter.default_dataset()

        if self.verbose:
            print(f"[benchwrap] Starting {adapter_name} on {dataset}")
            print(f"[benchwrap] Model: {model_name} ({backend_name})")
            print(f"[benchwrap] Few-shot: {fewshot}")

        # Load few-shot pool if needed
        fewshot_pool = []
        if fewshot > 0:
            fewshot_pool = self.adapter.fewshot_pool(dataset, split="train")
            if not fewshot_pool:
                # Fall back to test split (warning: data leakage — logged)
                if self.verbose:
                    print(f"[benchwrap] WARNING: No train split for few-shot. "
                          f"Using test split (potential data leakage).")

        # Pre-evaluation hook: adapters can do ingestion/setup before eval
        if hasattr(self.adapter, 'pre_evaluate'):
            self.adapter.pre_evaluate(dataset=dataset, backend=self.backend)
            if self.verbose:
                print(f"[benchwrap] Pre-evaluation hook completed")

        # Load samples
        samples = list(self.adapter.load(dataset, split=split, limit=limit))
        if self.verbose:
            print(f"[benchwrap] Loaded {len(samples)} samples")

        # Run evaluation
        eval_results = []
        correct = 0
        # Always use adapter's score() method — it knows its own scoring best
        scorer = self.scorer or _AdapterScorer(self.adapter)

        for i, sample in enumerate(samples):
            # Show progress: every sample for small runs, every 10th for larger.
            n_total = len(samples)
            log_every = 1 if n_total <= 20 else 10
            if (i + 1) % log_every == 0 or (i + 1) == n_total:
                print(f"[benchwrap] {adapter_name} {i + 1}/{n_total}", flush=True)

            # Select few-shot examples (from pool, not from test set)
            fs_samples = fewshot_pool[:fewshot] if fewshot_pool else []

            # Format prompt
            prompt = self.adapter.format_prompt(sample, fewshot=fs_samples)

            # Log prompt if verbose
            if self.verbose:
                print(f"\n[benchwrap] Sample {sample.id}:")
                print(f"  Prompt: {prompt.raw_text[:200]}...")

            # Generate prediction
            prediction = self.backend.generate(prompt, **gen_kwargs)
            # Smoking-gun warning: if the model burned tokens but produced
            # NO visible text, the eventual score is going to be 0.0 not
            # because the answer was wrong but because we got nothing to
            # score. Surface that loudly so it isn't read as a real result.
            if not prediction.text.strip() and prediction.tokens_out > 0:
                print(
                    f"[benchwrap] WARN sample {sample.id}: empty prediction "
                    f"despite {prediction.tokens_out} output tokens — likely "
                    f"hidden in thinking blocks; score will be 0.0",
                    flush=True,
                )

            if self.verbose:
                print(f"  Response: {prediction.text[:100]}...")

            # Extract answer (adapter can override)
            extracted = self.adapter.extract_answer(prediction.text, sample)

            # Score
            score = scorer.score(extracted, sample.reference, sample=sample)

            if score.primary() == 1.0:
                correct += 1

            if self.verbose:
                status = "✓" if score.primary() == 1.0 else "✗"
                print(f"  {status} Extracted: '{extracted}' vs Ref: '{sample.reference}' "
                      f"(method: {score.scoring_method})")

            eval_results.append(EvalResult(
                sample_id=sample.id,
                score=score,
                prediction=prediction,
                prompt=prompt,
                dataset=dataset,
                adapter=adapter_name,
            ))

        # Aggregate results
        duration = time.time() - start_time
        metrics = self._aggregate_metrics(eval_results)
        per_category = self._aggregate_per_category(eval_results, samples)

        result = Result(
            adapter=adapter_name,
            model=model_name,
            backend=backend_name,
            dataset=dataset,
            num_samples=len(samples),
            metrics=metrics,
            per_category=per_category,
            eval_results=eval_results,
            config={
                "dataset": dataset,
                "split": split,
                "limit": limit,
                "fewshot": fewshot,
                "scorer": scorer.__class__.__name__,
                "gen_kwargs": gen_kwargs,
            },
            duration_s=duration,
        )

        # Save if requested
        if self.save_dir:
            self._save_result(result)

        return result

    def _infer_scorer(self, samples: list[Sample]) -> Scorer:
        """Infer the appropriate scorer from sample characteristics."""
        # Check if it looks like MCQ
        has_choices = sum(1 for s in samples if s.choices) > len(samples) * 0.5
        if has_choices:
            return get_scorer("mcq")

        # Check if references look numeric
        numeric_refs = sum(
            1 for s in samples
            if re.match(r'^-?\d+\.?\d*$', s.reference.strip())
        ) > len(samples) * 0.5
        if numeric_refs:
            return get_scorer("numeric")

        # Default to exact match
        return get_scorer("exact")

    def _aggregate_metrics(self, results: list[EvalResult]) -> dict:
        """Compute aggregate metrics across all results."""
        n = len(results)
        if n == 0:
            return {}

        em_sum = sum(r.score.exact_match for r in results)
        metrics = {
            "exact_match": em_sum / n,
            "n": n,
        }

        # Aggregate F1 if present
        f1_scores = [r.score.f1 for r in results if r.score.f1 is not None]
        if f1_scores:
            metrics["f1"] = sum(f1_scores) / len(f1_scores)

        # Aggregate accuracy if present
        acc_scores = [r.score.accuracy for r in results if r.score.accuracy is not None]
        if acc_scores:
            metrics["accuracy"] = sum(acc_scores) / len(acc_scores)

        # Timing
        latencies = [r.prediction.latency_ms for r in results if r.prediction.latency_ms > 0]
        if latencies:
            metrics["avg_latency_ms"] = sum(latencies) / len(latencies)
            metrics["total_tokens_in"] = sum(r.prediction.tokens_in for r in results)
            metrics["total_tokens_out"] = sum(r.prediction.tokens_out for r in results)

        return metrics

    def _aggregate_per_category(
        self, results: list[EvalResult], samples: list[Sample]
    ) -> dict:
        """Compute per-category metrics if samples have category metadata."""
        # Build sample lookup
        sample_map = {s.id: s for s in samples}

        # Group by category
        categories: dict[str, list[EvalResult]] = {}
        for r in results:
            sample = sample_map.get(r.sample_id)
            if sample and "category" in sample.metadata:
                cat = sample.metadata["category"]
                categories.setdefault(cat, []).append(r)

        if not categories:
            return {}

        per_cat = {}
        for cat, cat_results in categories.items():
            n = len(cat_results)
            em = sum(r.score.exact_match for r in cat_results) / n
            per_cat[cat] = {"exact_match": em, "n": n}

        return per_cat

    def _save_result(self, result: Result):
        """Save result to JSON file."""
        os.makedirs(self.save_dir, exist_ok=True)
        filename = (
            f"{result.adapter}_{result.dataset}_{result.model}"
            f"_{int(result.timestamp)}.json"
        ).replace(":", "_").replace("/", "_")
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        if self.verbose:
            print(f"[benchwrap] Saved results to {filepath}")


# Need re for _infer_scorer
import re
