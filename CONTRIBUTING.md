# Contributing to benchwrap

## Adding a New Adapter

1. Create `benchwrap/adapters/your_benchmark.py`
2. Subclass `BenchmarkAdapter` and implement 5 methods:
   - `name()` — benchmark name
   - `datasets()` — available datasets
   - `load()` — yield Sample objects
   - `format_prompt()` — Sample → Prompt
   - `score()` — prediction vs reference → Score
3. Auto-discovered on next `benchwrap list`

See `benchwrap/adapters/custom/README.md` for examples.

## Adding a New Model Backend

1. Subclass `ModelBackend` in `benchwrap/core/model.py`
2. Implement `generate()`, `name()`, `model_id()`
3. Add to `parse_backend()` for CLI support

## Design Principles

- **stdlib-core**: Engine, scoring, reporting use only Python stdlib
- **Transparent**: Every prompt logged, every score auditable
- **Honest**: No pre-injected answers, no weight manipulation
- **Pluggable**: 5-method adapter contract, nothing more

## Running Tests

```bash
python3 benchwrap.py list
python3 benchwrap.py diagnose
python3 benchwrap.py run mmlu --model ollama:test --limit 3 --verbose
```
