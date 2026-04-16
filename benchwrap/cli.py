"""
benchwrap CLI — Unified benchmark evaluation.
"""

import argparse
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchwrap.core.model import parse_backend
from benchwrap.core.engine import EvaluationEngine
from benchwrap.core.reporter import Reporter
from benchwrap.core.scorer import get_scorer
from benchwrap.adapters import get_adapter, list_adapters, discover_adapters


def cmd_list(args):
    """List available benchmarks."""
    discover_adapters()
    adapters = list_adapters()
    if not adapters:
        print("No adapters found.")
        print("Place adapter .py files in benchwrap/adapters/ or benchwrap/adapters/custom/")
        return

    print(f"\n  {'Name':<25} {'Datasets':<40} {'Description'}")
    print(f"  {'-' * 25} {'-' * 40} {'-' * 40}")
    for name, info in sorted(adapters.items()):
        datasets = ", ".join(info.get("datasets", []))
        desc = info.get("description", "")
        print(f"  {name:<25} {datasets:<40} {desc[:40]}")
    print()


def cmd_diagnose(args):
    """Diagnose available backends."""
    print("\n  Backend Diagnostics")
    print("  " + "=" * 50)

    # Check Ollama
    try:
        from benchwrap.core.model import OllamaBackend
        ollama = OllamaBackend(model=args.model or "openhermes:7b-v2.5")
        status = ollama.diagnose()
        print(f"\n  Ollama: {status['status']}")
        if status.get("available_models"):
            print(f"  Models: {', '.join(status['available_models'][:5])}")
    except Exception as e:
        print(f"\n  Ollama: error ({e})")

    # Check installed adapters
    discover_adapters()
    adapters = list_adapters()
    print(f"\n  Adapters: {len(adapters)} available")
    for name in sorted(adapters):
        print(f"    - {name}")

    print()


def cmd_run(args):
    """Run a benchmark evaluation."""
    # Load adapter
    adapter = get_adapter(args.benchmark)
    if not adapter:
        print(f"Error: Unknown benchmark '{args.benchmark}'")
        print(f"Available: {', '.join(list_adapters().keys())}")
        sys.exit(1)

    # Parse backend
    backend = parse_backend(args.model)

    # Parse scorer
    scorer_type = args.scorer or "auto"
    if scorer_type == "auto":
        scorer = None  # Will be inferred
    else:
        scorer_kwargs = {}
        if args.reasoning:
            scorer_type = "reasoning"
            scorer_kwargs["inner"] = args.scorer or "mcq"
        scorer = get_scorer(scorer_type, **scorer_kwargs)

    # Create engine
    engine = EvaluationEngine(
        adapter=adapter,
        backend=backend,
        scorer=scorer,
        verbose=args.verbose,
        save_dir=args.save_dir,
    )

    # Run
    try:
        result = engine.run(
            dataset=args.dataset or "default",
            split=args.split,
            limit=args.limit,
            fewshot=args.fewshot,
            temperature=args.temperature,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Output
    reporter = Reporter(output_dir=args.save_dir)

    if args.emit == "json":
        print(reporter.json_out(result))
    elif args.emit == "csv":
        print(reporter.csv_out([result]))
    else:
        print(reporter.terminal(result))

    # Save if directory specified
    if args.save_dir:
        path = reporter.save_json(result)
        if path:
            print(f"\nSaved to: {path}")


def cmd_compare(args):
    """Compare models on benchmarks."""
    print("Compare mode — run benchmarks individually, then:")
    print("  benchwrap run BENCH1 --model MODEL_A --save-dir results/")
    print("  benchwrap run BENCH1 --model MODEL_B --save-dir results/")
    print("  benchwrap compare results/")
    print()
    print("Or use the engine programmatically for parallel runs.")


def main():
    parser = argparse.ArgumentParser(
        prog="benchwrap",
        description="benchwrap — Unified LLM Benchmark Wrapper. ONE adapter to rule them all.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  benchwrap list                                    # List available benchmarks
  benchwrap diagnose                                # Check backends
  benchwrap run mmlu --model ollama:openhermes:7b   # Run MMLU
  benchwrap run gsm8k --model nim:meta/llama-3.3-70b-instruct --fewshot 5
  benchwrap run mmlu --model ollama:qwen2.5:7b --limit 50 --verbose
  benchwrap run custom/my_bench.py --model ollama:openhermes:7b

Model format:
  ollama:model_name                     Ollama local
  ollama:model_name@host                Ollama custom host
  nim:model_name                        NVIDIA NIM
  openai:model_name                     OpenAI API
  api:model_name@url                    Custom OpenAI-compatible API
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # list
    sub_list = subparsers.add_parser("list", help="List available benchmarks")
    sub_list.set_defaults(func=cmd_list)

    # diagnose
    sub_diag = subparsers.add_parser("diagnose", help="Diagnose backends")
    sub_diag.add_argument("--model", help="Model to check")
    sub_diag.set_defaults(func=cmd_diagnose)

    # run
    sub_run = subparsers.add_parser("run", help="Run a benchmark")
    sub_run.add_argument("benchmark", help="Benchmark name (use 'list' to see available)")
    sub_run.add_argument("--model", required=True, help="Model backend (e.g. ollama:openhermes:7b)")
    sub_run.add_argument("--dataset", help="Dataset/subset name")
    sub_run.add_argument("--split", default="test", help="Data split (default: test)")
    sub_run.add_argument("--limit", type=int, help="Max samples to evaluate")
    sub_run.add_argument("--fewshot", type=int, default=0, help="Number of few-shot examples")
    sub_run.add_argument("--scorer", choices=["mcq", "numeric", "exact", "f1", "auto"],
                         default="auto", help="Scoring method (default: auto-detect)")
    sub_run.add_argument("--reasoning", action="store_true",
                         help="Model uses chain-of-thought (extract answer from CoT)")
    sub_run.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    sub_run.add_argument("--emit", choices=["terminal", "json", "csv"], default="terminal",
                         help="Output format")
    sub_run.add_argument("--save-dir", help="Save results to directory")
    sub_run.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    sub_run.set_defaults(func=cmd_run)

    # compare
    sub_cmp = subparsers.add_parser("compare", help="Compare results")
    sub_cmp.add_argument("results_dir", nargs="?", default="results/",
                         help="Directory with saved results")
    sub_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
