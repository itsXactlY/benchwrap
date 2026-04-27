# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project has no install step — it runs in place from the repo root via `benchwrap.py`, which inserts the project root onto `sys.path` before delegating to `benchwrap.cli:main`.

```bash
# CLI (from repo root)
python3 benchwrap.py list                               # list discovered adapters
python3 benchwrap.py diagnose                           # check Ollama + adapter discovery
python3 benchwrap.py run mmlu --model ollama:openhermes:7b-v2.5 --limit 5 --verbose
python3 benchwrap.py run gsm8k --model ollama:qwen2.5:7b --fewshot 5 --limit 10
python3 benchwrap.py run mmlu --model ollama:m@http://192.168.0.2:11434
python3 benchwrap.py run mmlu --model nim:meta/llama-3.3-70b-instruct      # uses NIM_API_KEY
python3 benchwrap.py run BENCH --model ... --emit json|csv --save-dir results/

# Tests (subprocess-based runner — does NOT use pytest)
python3 tests/run_tests.py                              # full suite
python3 tests/run_tests.py --quick                      # types + scorers + engine only (no network)
python3 tests/run_tests.py types scorers                # named suites: types, scorers, engine, adapters, integration, custom
python3 tests/test_engine.py                            # any test file is also runnable directly

# Visualize saved results (ASCII bar charts; auto-pairs no-mem vs with-mem runs)
python3 benchview.py results/
python3 benchview.py results/ --json
```

Backend spec format (parsed by `benchwrap.core.model.parse_backend`):
`ollama:MODEL[@HOST]` · `nim:MODEL` · `openai:MODEL` · `api:MODEL@URL[#KEY]`. NIM/OpenAI keys come from `NIM_API_KEY` / `OPENAI_API_KEY` env vars.

## Architecture

The whole system is **five interfaces orchestrated by one engine**. To work productively, internalize the data flow:

```
EvaluationEngine.run()
    ├── adapter.default_dataset() / adapter.fewshot_pool()
    ├── adapter.pre_evaluate(...)        # OPTIONAL hook — used by memory adapters to ingest before eval
    ├── adapter.load(...)        → Sample
    ├── adapter.format_prompt()  → Prompt          (raw_text MUST be exactly what the model sees)
    ├── backend.generate()       → Prediction
    ├── adapter.extract_answer() → str             (default: strip; override for CoT models)
    └── (scorer or _AdapterScorer(adapter)).score() → Score
```

### The five-method adapter contract — `benchwrap/core/adapter.py`

Every adapter subclasses `BenchmarkAdapter` and implements: `name`, `datasets`, `load`, `format_prompt`, `score`. Optional overrides: `default_dataset`, `fewshot_pool`, `extract_answer`, and (for memory benchmarks) a `pre_evaluate(dataset, backend)` hook the engine calls if present.

Adapters are **auto-discovered** by `benchwrap/adapters/__init__.py:discover_adapters` — it scans `benchwrap/adapters/` and `benchwrap/adapters/custom/` for `.py` files (skipping `_*`), imports them, and registers any non-abstract `BenchmarkAdapter` subclass it can instantiate with no args. Drop a file in either directory and it appears in `benchwrap list`.

### Scorer choice is implicit

`EvaluationEngine` does NOT use `--scorer auto` to infer a scorer at runtime. If `scorer is None` (the default when `--scorer auto` is passed), it wraps `adapter.score()` in `_AdapterScorer` and uses that. This means **the adapter's own `score()` is authoritative** unless the user passes an explicit `--scorer mcq|numeric|exact|f1`. (`engine._infer_scorer` exists but is currently unused dead code.)

The `--reasoning` flag wraps the chosen inner scorer in `ReasoningScorer`, which extracts an answer from chain-of-thought text using configurable regex patterns. Patterns are **logged as part of the score**, never hidden.

### Core types (`benchwrap/core/types.py`) are deliberate

- `Prompt.raw_text` is the transparency contract — it must equal the exact text sent to the model. Never construct a `Prompt` whose `raw_text` differs from what `format_prompt` actually serializes.
- `Score.primary()` returns `accuracy` if set, else `exact_match` — this is what the engine uses to count "correct" samples.
- `Score.to_dict()` truncates `raw_prediction` to 200 chars when serializing (full text is kept in memory only).
- `Result.to_dict()` is what gets persisted; filenames are `{adapter}_{dataset}_{model}_{timestamp}.json` with `:` and `/` replaced by `_`.

### Backends (`benchwrap/core/model.py`)

Two built-in backends, both stdlib-only (`urllib.request`, no `requests`/`httpx`):
- `OllamaBackend` hits **`/api/chat`** (native), not `/v1`. This is intentional — reasoning models return empty content via the OpenAI-compatible endpoint.
- `OpenAICompatBackend` covers OpenAI, NIM, vLLM, and any `--model api:...@URL` target via `/v1/chat/completions`.

`generate_batch` defaults to sequential; backends override for parallelism.

### The MemoryBackend extension

Memory benchmarks (`memory_bench.py`, `locomo.py`, `memory_agent.py`, `evomem.py`) use a **second interface beyond the 5-method contract**: a `MemoryBackend` (defined in `benchwrap/adapters/memory_bench.py`) with `store/recall/ingest/clear/stats`. The flow is: `pre_evaluate` ingests facts into the memory backend → `format_prompt` calls `recall` to inject retrieved context → the LLM answers → the scorer compares. `neural_memory.py` is the showcase wrapper around an external Neural Memory system (it adds `sys.path` for `~/projects/neural-memory-adapter/python` at import time — that path is hardcoded).

## Project conventions

- **stdlib-core**: Engine, scoring, reporting, and backends use only Python stdlib. Adapters MAY use external libs (`datasets`, `pandas`, `nltk`) — but if you add an adapter dep, gate the import inside the adapter file so missing deps don't break `benchwrap list`.
- **No answer leakage**: `Sample.reference` is only used by `score()`. It must never appear in `format_prompt` output. Few-shot examples come from `fewshot_pool()` (typically the train split) and must not overlap with test samples.
- **Transparency over convenience**: prefer explicit, logged extraction (regex patterns in `ReasoningScorer`, `scoring_method` strings on every `Score`) over silent normalization. If you add a new scoring path, set `scoring_method` to describe it.
- **Tests are not pytest**: `tests/run_tests.py` shells out to each `test_*.py` and parses `"N passed, M failed"` from stdout. Each test file is a standalone script — preserve that pattern when adding tests.
