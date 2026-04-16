# Custom Benchmark Adapters

Place your custom benchmark adapters here as `.py` files.

## Creating an Adapter

1. Create a `.py` file (e.g., `my_benchmark.py`)
2. Define a class that inherits from `BenchmarkAdapter`
3. Implement the 5 required methods:
   - `name()` — benchmark name
   - `datasets()` — available datasets
   - `load()` — yield Sample objects
   - `format_prompt()` — turn Sample into Prompt
   - `score()` — evaluate prediction vs reference

## Example

```python
from benchwrap.core.adapter import BenchmarkAdapter
from benchwrap.core.types import Sample, Prompt, Score

class MyBenchmark(BenchmarkAdapter):
    def name(self):
        return "my-benchmark"

    def datasets(self):
        return ["default"]

    def load(self, dataset="default", split="test", limit=None):
        # Load your data and yield Sample objects
        yield Sample(
            id="q1",
            input="What is 2+2?",
            reference="4",
            metadata={"category": "math"}
        )

    def format_prompt(self, sample, fewshot=None):
        text = f"Q: {sample.input}\nA:"
        return Prompt(
            system=None,
            messages=[{"role": "user", "content": text}],
            raw_text=text
        )

    def score(self, prediction, reference, sample):
        em = 1.0 if prediction.strip() == reference.strip() else 0.0
        return Score(
            exact_match=em,
            raw_prediction=prediction,
            raw_reference=reference,
            scoring_method="exact_match"
        )
```

After placing your file here, `benchwrap list` will automatically discover it.
