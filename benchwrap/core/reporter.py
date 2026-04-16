"""
Reporter — formats and outputs results.
Terminal table, JSON, CSV. No interpretation, no "adjustments."
"""

import json
import os
from benchwrap.core.types import Result


class Reporter:
    """Format and display benchmark results."""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir

    def terminal(self, result: Result) -> str:
        """Format result as a terminal-friendly table."""
        return result.summary()

    def compare(self, results: list[Result]) -> str:
        """Compare multiple results side-by-side."""
        if not results:
            return "No results to compare."

        lines = [
            "",
            "=" * 80,
            "  BENCHMARK COMPARISON",
            "=" * 80,
        ]

        # Header
        metrics_set = set()
        for r in results:
            metrics_set.update(r.metrics.keys())
        # Filter to meaningful metrics
        display_metrics = [
            m for m in ["exact_match", "accuracy", "f1"]
            if m in metrics_set
        ]

        # Column widths
        model_width = max(len(r.model) for r in results) + 2
        bench_width = max(len(r.dataset) for r in results) + 2

        header = f"  {'Model':<{model_width}} {'Benchmark':<{bench_width}}"
        for m in display_metrics:
            header += f" {m:>10}"
        header += f" {'N':>6} {'Time':>8}"
        lines.append(header)
        lines.append(f"  {'-' * model_width} {'-' * bench_width}"
                     + "".join(f" {'-' * 10}" for _ in display_metrics)
                     + f" {'-' * 6} {'-' * 8}")

        for r in sorted(results, key=lambda x: x.model):
            row = f"  {r.model:<{model_width}} {r.dataset:<{bench_width}}"
            for m in display_metrics:
                v = r.metrics.get(m, 0)
                row += f" {v:>9.1%}"
            row += f" {r.num_samples:>6} {r.duration_s:>7.1f}s"
            lines.append(row)

        lines.append("=" * 80)
        return "\n".join(lines)

    def json_out(self, result: Result) -> str:
        """Format result as JSON."""
        return json.dumps(result.to_dict(), indent=2)

    def save_json(self, result: Result, filename: str | None = None) -> str:
        """Save result to a JSON file."""
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            if not filename:
                filename = (
                    f"{result.adapter}_{result.dataset}_{result.model}"
                    f"_{int(result.timestamp)}.json"
                ).replace(":", "_").replace("/", "_")
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            return filepath
        return ""

    def csv_out(self, results: list[Result]) -> str:
        """Format results as CSV."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "adapter", "model", "backend", "dataset",
            "exact_match", "f1", "accuracy", "n", "duration_s",
        ])
        for r in results:
            writer.writerow([
                r.adapter, r.model, r.backend, r.dataset,
                r.metrics.get("exact_match", ""),
                r.metrics.get("f1", ""),
                r.metrics.get("accuracy", ""),
                r.num_samples,
                f"{r.duration_s:.1f}",
            ])
        return output.getvalue()
