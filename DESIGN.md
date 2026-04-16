# benchwrap — Unified LLM Benchmark Wrapper

## THE PROBLEM

We've run 5+ benchmark frameworks. Each one:
- Different config format (YAML vs CLI vs Python dicts)
- Different model backend (NIM vs Ollama vs OpenAI vs raw HF)
- Different scoring (EM vs F1 vs acc_norm vs custom)
- Different result format (JSON vs CSV vs stdout-only)
- Different bugs (AIME first-char, MAB force dupes, EvoMem broken imports, lm-eval opaque prompts)

Worse: some frameworks have baked-in assumptions that make honest evaluation hard.
Pre-injected few-shot answers. Weighted scoring that masks failures. Opaque prompt templates
you can't inspect or override.

## THE SOLUTION

ONE wrapper. ONE interface. ANY benchmark plugs in. ZERO cheating.

### Core Principles

1. **TRANSPARENT** — Every prompt sent to the model is logged. Every scoring decision is visible.
   No black-box prompt templates. No hidden few-shot injection.

2. **HONEST** — No pre-injected answers. No answer weight manipulation. No "adjusted" scores.
   The model sees the prompt. The model generates the answer. The scorer evaluates it. That's it.

3. **PLUGGABLE** — Any benchmark is an adapter implementing `BenchmarkAdapter`.
   5 methods: `name()`, `datasets()`, `load()`, `format_prompt()`, `score()`.
   That's the entire contract.

4. **MODEL-AGNOSTIC** — Ollama, OpenAI, HF, vLLM, custom — all through `ModelBackend`.
   1 method: `generate(prompt, **kwargs) -> str`. That's it.

5. **STDLIB-CORE** — The engine, scoring, and reporting use only Python stdlib.
   Adapters can use external libs (datasets, pandas, etc.) but the core never requires them.

6. **REPRODUCIBLE** — Same config + same model = same results. Every run is a self-contained
   artifact with full prompt logs, raw outputs, and scoring breakdowns.

## Architecture

```
benchwrap/
├── __init__.py              # Version, public API
├── cli.py                   # argparse CLI — benchwrap run/compare/list/diagnose
├── core/
│   ├── __init__.py
│   ├── adapter.py           # BenchmarkAdapter ABC
│   ├── engine.py            # EvaluationEngine — orchestrates everything
│   ├── model.py             # ModelBackend ABC + built-in backends
│   ├── scorer.py            # Scorer ABC + built-in scorers (EM, F1, acc, custom)
│   ├── reporter.py          # Result aggregation + terminal/JSON/CSV output
│   └── types.py             # Dataclasses: Sample, Prompt, Prediction, Score, Result
├── adapters/
│   ├── __init__.py          # Auto-discovery via entry_points or directory scan
│   ├── lm_eval.py           # Wraps lm-evaluation-harness tasks
│   ├── locomo.py            # LoCoMo conversational memory benchmark
│   ├── memory_agent.py      # MemoryAgentBench adapter
│   ├── evomem.py            # EvoMem streaming memory benchmark
│   ├── mmlu.py              # Standalone MMLU (no lm-eval dependency)
│   ├── gsm8k.py             # Standalone GSM8K
│   ├── humaneval.py         # HumanEval code generation
│   └── custom/              # User's custom benchmarks go here
│       └── README.md
├── backends/
│   ├── __init__.py
│   ├── ollama.py            # Ollama local inference
│   ├── openai_compat.py     # OpenAI-compatible APIs (NIM, Together, etc.)
│   ├── hf_local.py          # HuggingFace transformers local
│   └── vllm.py              # vLLM high-throughput
└── results/                 # Default output directory
    └── .gitkeep
```

## Key Interfaces

### BenchmarkAdapter (the plug-in contract)

```python
class BenchmarkAdapter(ABC):
    @abstractmethod
    def name(self) -> str:
        """Human-readable benchmark name."""

    @abstractmethod
    def datasets(self) -> list[str]:
        """Available dataset/subset names."""

    @abstractmethod
    def load(self, dataset: str, split: str = "test", limit: int | None = None) -> Iterator[Sample]:
        """Yield Sample objects. No pre-processing, no answer injection."""

    @abstractmethod
    def format_prompt(self, sample: Sample, fewshot: list[Sample] | None = None) -> Prompt:
        """Format a sample into a prompt. Fewshot are EXPLICIT — never hidden."""

    @abstractmethod
    def score(self, prediction: str, reference: str, sample: Sample) -> Score:
        """Score a prediction against reference. Returns Score with metric breakdown."""
```

### ModelBackend (the inference contract)

```python
class ModelBackend(ABC):
    @abstractmethod
    def generate(self, prompt: Prompt, **kwargs) -> Prediction:
        """Generate a response. Returns Prediction with raw text + metadata."""

    def generate_batch(self, prompts: list[Prompt], **kwargs) -> list[Piction]:
        """Batch generation. Default: sequential. Backends can override for parallel."""
```

### Sample (the atomic unit)

```python
@dataclass
class Sample:
    id: str                          # Unique identifier
    input: str                       # The question/context
    reference: str                   # The expected answer (NEVER shown to model)
    choices: list[str] | None        # MCQ options if applicable
    metadata: dict                   # Dataset-specific metadata (category, difficulty, etc.)
```

### Prompt (transparent, inspectable)

```python
@dataclass
class Prompt:
    system: str | None               # System prompt (None if not used)
    messages: list[dict]             # [{"role": "user", "content": "..."}]
    fewshot_samples: list[str] | None  # IDs of fewshot samples used (for reproducibility)
    raw_text: str                    # The actual text sent to the model (for logging)
```

### Score (honest, unmanipulated)

```python
@dataclass
class Score:
    exact_match: float               # 0.0 or 1.0
    f1: float | None                 # Token-level F1 if applicable
    accuracy: float | None           # MCQ accuracy if applicable
    custom: dict[str, float]         # Benchmark-specific metrics
    raw_prediction: str              # What the model actually said
    raw_reference: str               # What the answer should be
    scoring_method: str              # How it was scored (transparent)
```

## CLI Interface

```bash
# List available benchmarks
benchwrap list

# Diagnose backends
benchwrap diagnose

# Run a benchmark
benchwrap run mmlu --model ollama:openhermes:7b-v2.5 --dataset all --limit 100
benchwrap run locomo --model ollama:openhermes:7b-v2.5 --dataset all
benchwrap run mmlu --model nim:meta/llama-3.3-70b-instruct --fewshot 5

# Compare models
benchwrap compare --models ollama:openhermes:7b,ollama:qwen2.5:7b --benchmarks mmlu,gsm8k

# Run with full transparency
benchwrap run mmlu --model ollama:openhermes:7b --verbose --save-prompts prompts/

# Custom benchmark
benchwrap run custom/my_bench.py --model ollama:openhermes:7b
```

## What Makes This Different

### vs lm-evaluation-harness
- lm-eval has opaque prompt templates — you can't easily see what's sent
- lm-eval bundles everything — we separate core from adapters
- lm-eval requires HF models by default — we support any backend
- lm-eval is a monolith — we're a framework with a plugin system

### vs HELM
- HELM focuses on fairness/calibration — we focus on raw benchmark execution
- HELM is heavy (Stanford infrastructure) — we're stdlib-core + adapters

### vs custom scripts (what we've been doing)
- Custom scripts have hardcoded paths, model connections, scoring bugs
- We've fixed the same bugs in 3 different frameworks
- ONE fix here fixes it for ALL benchmarks

### vs "just use lm-eval"
- lm-eval can't run LoCoMo, MemoryAgentBench, EvoMem, or custom benchmarks
- lm-eval prompt templates are fixed — can't test different prompting strategies
- lm-eval scoring is fixed — can't add custom extraction for reasoning models

## Scoring Philosophy

### What We DO
- Score exactly what the model outputs
- Show every prompt sent
- Log every raw response
- Report exact_match, f1, accuracy separately (no blending)
- Support custom extractors (for reasoning models with CoT)

### What We DON'T DO
- Pre-inject answers into prompts
- Weight scores to "normalize" across benchmarks
- Filter out "unfair" samples post-hoc
- Adjust scores based on difficulty
- Hide any part of the evaluation pipeline

### Handling Reasoning Models
Reasoning models (o1, DeepSeek-R1, QwQ) bury answers in chain-of-thought.
We handle this with EXPLICIT extractors, not hidden logic:

```python
class ReasoningExtractor:
    """Extract final answer from CoT response. Configurable, not hardcoded."""
    def __init__(self, patterns: list[str], fallback: str = "last_letter"):
        self.patterns = patterns
        self.fallback = fallback

    def extract(self, response: str) -> str:
        # Try each pattern in order
        for pattern in self.patterns:
            match = re.search(pattern, response[-1000:])  # Last 1000 chars
            if match:
                return match.group(1).upper()
        # Fallback strategy (configurable)
        if self.fallback == "last_letter":
            letters = re.findall(r'\b([A-D])\b', response)
            return letters[-1].upper() if letters else ""
        return ""
```

This is EXPLICIT in the config. No hidden behavior.

## Adapter Development Guide

Creating a new adapter requires 5 methods:

```python
from benchwrap.core.adapter import BenchmarkAdapter
from benchwrap.core.types import Sample, Prompt, Score

class MyBenchmark(BenchmarkAdapter):
    def name(self) -> str:
        return "my-benchmark"

    def datasets(self) -> list[str]:
        return ["dataset-a", "dataset-b"]

    def load(self, dataset: str, split: str = "test", limit: int | None = None):
        # Load from wherever — HuggingFace, local files, API
        # Yield Sample objects
        for i, item in enumerate(data):
            if limit and i >= limit:
                break
            yield Sample(
                id=item["id"],
                input=item["question"],
                reference=item["answer"],
                choices=item.get("choices"),
                metadata={"category": item.get("category")}
            )

    def format_prompt(self, sample: Sample, fewshot: list[Sample] | None = None) -> Prompt:
        # Build the prompt — YOU control exactly what's sent
        text = f"Question: {sample.input}\nAnswer:"
        if fewshot:
            fewshot_text = "\n".join(
                f"Q: {s.input}\nA: {s.reference}" for s in fewshot
            )
            text = fewshot_text + "\n" + text
        return Prompt(
            system=None,
            messages=[{"role": "user", "content": text}],
            fewshot_samples=[s.id for s in (fewshot or [])],
            raw_text=text
        )

    def score(self, prediction: str, reference: str, sample: Sample) -> Score:
        # Honest scoring — no manipulation
        pred_clean = prediction.strip().upper()
        ref_clean = reference.strip().upper()
        em = 1.0 if pred_clean == ref_clean else 0.0
        return Score(
            exact_match=em,
            f1=None,
            accuracy=em if sample.choices else None,
            custom={},
            raw_prediction=prediction,
            raw_reference=reference,
            scoring_method="exact_match"
        )
```

That's it. Register it, and `benchwrap run my-benchmark --model ollama:openhermes:7b` works.

## Lessons From Our Benchmark History

1. **AIME scoring** — first-char extraction wrong for integer answers. Solution: dataset-aware scoring.
2. **MAB force flag** — loaded old results without clearing. Solution: force = skip load_existing.
3. **EvoMem broken imports** — AgentType/DatasetType don't exist. Solution: standalone scripts only.
4. **NIM rate limiting** — 0.01 it/s. Solution: Ollama for benchmarks, NIM for quality comparison.
5. **Hardcoded paths** — break on other machines. Solution: Path.home() everywhere.
6. **Memory ceiling** — 75% MMLU regardless of model. Solution: report both baseline and memory.
7. **Reasoning models** — empty content via /v1 API. Solution: native Ollama API with think:false.
8. **qwen3.5:4b** — thinking mode incompatible. Solution: skip or patch per-model.

ALL of these become non-issues with benchwrap. The adapter handles model quirks.
The scorer handles dataset quirks. The engine handles reproducibility.
