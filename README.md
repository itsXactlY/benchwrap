# benchwrap

ONE adapter to rule them all.

A unified LLM benchmark wrapper where ANY benchmark plugs in. Zero cheating. Full transparency.

## The Problem

We've run 5+ benchmark frameworks. Each one: different config format, different model connection, different scoring, different result format, different bugs we had to fix. Worse — some have baked-in assumptions that make honest evaluation hard.

## The Solution

**ONE wrapper. ONE interface. ANY benchmark plugs in. ZERO cheating.**

### Principles

1. **TRANSPARENT** — Every prompt sent to the model is logged. Every scoring decision is visible. No black-box templates.
2. **HONEST** — No pre-injected answers. No weight manipulation. No "adjusted" scores. The model sees the prompt. The model generates. The scorer evaluates. That's it.
3. **PLUGGABLE** — Any benchmark is an adapter: 5 methods. `name()`, `datasets()`, `load()`, `format_prompt()`, `score()`. That's the entire contract.
4. **MODEL-AGNOSTIC** — Ollama, OpenAI, NIM, vLLM, custom — all through one interface: `generate(prompt) -> text`.
5. **STDLIB-CORE** — Engine, scoring, and reporting use only Python stdlib. Adapters can use external libs.

## Quick Start

```bash
cd ~/projects/benchwrap

# List available benchmarks
python3 benchwrap.py list

# Diagnose backends
python3 benchwrap.py diagnose

# Run MMLU (5 samples, zero-shot, Ollama)
python3 benchwrap.py run mmlu --model ollama:openhermes:7b-v2.5 --limit 5 --verbose

# Run GSM8K with few-shot
python3 benchwrap.py run gsm8k --model ollama:qwen2.5:7b --fewshot 5 --limit 10

# Use NVIDIA NIM (70B)
python3 benchwrap.py run mmlu --model nim:meta/llama-3.3-70b-instruct --limit 20

# JSON output
python3 benchwrap.py run mmlu --model ollama:openhermes:7b --emit json --limit 3

# Save results
python3 benchwrap.py run mmlu --model ollama:openhermes:7b --save-dir results/
```

## Model Backends

| Format | Backend | Example |
|--------|---------|---------|
| `ollama:MODEL` | Ollama local | `ollama:openhermes:7b-v2.5` |
| `ollama:MODEL@HOST` | Ollama custom | `ollama:qwen2.5:7b@http://192.168.0.2:11434` |
| `nim:MODEL` | NVIDIA NIM | `nim:meta/llama-3.3-70b-instruct` |
| `openai:MODEL` | OpenAI API | `openai:gpt-4` |
| `api:MODEL@URL` | Custom API | `api:my-model@http://localhost:8000/v1` |

## Built-in Adapters

| Adapter | Datasets | Type | Scoring |
|---------|----------|------|---------|
| **mmlu** | 57 subjects + 4 categories | MCQ (A/B/C/D) | MCQ letter extraction |
| **gsm8k** | main, socratic | Numeric | Numeric extraction |

## Custom Adapters

Drop a `.py` file in `benchwrap/adapters/custom/`. Implement 5 methods:

```python
from benchwrap.core.adapter import BenchmarkAdapter
from benchwrap.core.types import Sample, Prompt, Score

class MyBenchmark(BenchmarkAdapter):
    def name(self):
        return "my-benchmark"

    def datasets(self):
        return ["default"]

    def load(self, dataset="default", split="test", limit=None):
        yield Sample(id="q1", input="What is 2+2?", reference="4")

    def format_prompt(self, sample, fewshot=None):
        text = f"Q: {sample.input}\nA:"
        return Prompt(system=None, messages=[{"role": "user", "content": text}], raw_text=text)

    def score(self, prediction, reference, sample):
        em = 1.0 if prediction.strip() == reference.strip() else 0.0
        return Score(exact_match=em, raw_prediction=prediction, raw_reference=reference)
```

Auto-discovered on next `benchwrap list`.

## Scoring

All metrics reported separately. No blending. No weighting.

| Scorer | When | Method |
|--------|------|--------|
| MCQ | Multiple choice (A/B/C/D) | Letter extraction + match |
| Numeric | Math benchmarks (GSM8K, AIME) | Number extraction + comparison |
| Exact | Open-ended | Normalized string match |
| F1 | SQuAD-style QA | Token overlap |
| Reasoning | CoT models (o1, DeepSeek-R1) | Configurable pattern extraction + inner scorer |

### Reasoning Models

Models with chain-of-thought output use explicit extractors:

```python
ReasoningScorer(
    inner=MCQScorer(),
    patterns=[
        r'(?:answer|Answer|ANSWER)[\s:]*([A-F])\b',
        r'\\boxed\{([^}]+)\}',
    ],
    lookback_chars=1000
)
```

Patterns are LOGGED. Extraction is AUDITABLE. No hidden behavior.

## Architecture

```
benchwrap/
├── benchwrap.py           # CLI entry point
├── benchwrap/
│   ├── core/
│   │   ├── types.py       # Sample, Prompt, Prediction, Score, Result
│   │   ├── adapter.py     # BenchmarkAdapter ABC (the plug-in contract)
│   │   ├── model.py       # ModelBackend ABC + Ollama, OpenAI backends
│   │   ├── scorer.py      # MCQ, Numeric, F1, Reasoning scorers
│   │   ├── engine.py      # EvaluationEngine orchestrator
│   │   └── reporter.py    # Terminal/JSON/CSV output
│   └── adapters/
│       ├── mmlu.py        # MMLU (57 subjects, HuggingFace API)
│       ├── gsm8k.py       # GSM8K (grade school math)
│       └── custom/        # Drop your adapters here
```

## What Makes This Different

**vs lm-evaluation-harness**: lm-eval has opaque prompt templates, bundles everything, requires HF models by default. We separate core from adapters, support any backend, and show every prompt.

**vs HELM**: HELM focuses on fairness/calibration — we focus on raw benchmark execution with full transparency.

**vs custom scripts**: We've fixed the same bugs (AIME first-char scoring, MAB force duplicates, hardcoded paths) in 3+ different frameworks. ONE fix here fixes it for ALL benchmarks.

**vs "just use lm-eval"**: lm-eval can't run LoCoMo, MemoryAgentBench, EvoMem, or custom benchmarks. lm-eval prompt templates are fixed. We're a framework with a plugin system.

## What We DON'T Do

- Pre-inject answers into prompts
- Weight scores to "normalize" across benchmarks
- Filter out "unfair" samples post-hoc
- Adjust scores based on difficulty
- Hide any part of the evaluation pipeline

## Lessons From Our Benchmark History

1. **AIME scoring** — first-char extraction wrong for integer answers → dataset-aware scoring
2. **MAB force flag** — loaded old results without clearing → force = skip load_existing
3. **EvoMem broken imports** — AgentType/DatasetType don't exist → standalone only
4. **NIM rate limiting** — 0.01 it/s → Ollama for benchmarks, NIM for quality
5. **Hardcoded paths** — break on other machines → Path.home() everywhere
6. **Memory ceiling** — 75% MMLU regardless of model → report both baseline and memory
7. **Reasoning models** — empty content via /v1 API → native Ollama API
8. **qwen3.5:4b** — thinking mode incompatible → skip or patch per-model

ALL of these become non-issues with benchwrap. The adapter handles model quirks. The scorer handles dataset quirks. The engine handles reproducibility.

## License

MIT
