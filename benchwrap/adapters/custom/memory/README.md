# Custom Memory Benchmarks

Place your custom memory benchmark datasets here as `.json` files.

## Format

```json
{
  "store": [
    {"content": "Fact 1", "label": "fact-1"},
    {"content": "Fact 2", "label": "fact-2"}
  ],
  "queries": [
    {"query": "Question about fact 1?", "answer": "Answer 1", "category": "exact"},
    {"query": "Question about fact 2?", "answer": "Answer 2", "category": "paraphrase"}
  ]
}
```

## Usage

After placing your file here, run:

```bash
benchwrap run memory-bench --model ollama:openhermes:7b --dataset your_file_name
```

(The `.json` extension is stripped automatically.)
