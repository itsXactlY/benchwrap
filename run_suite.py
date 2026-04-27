#!/usr/bin/env python3
"""
run_suite.py — Full benchwrap suite: ALL benchmarks × 3 memory modes.

Modes
-----
  baseline       — LLM only. memory_client=None. Memory adapters run zero-shot
                   (no recalled context). Confabulation baseline for memory tasks.
  fresh          — Fresh temp Neural Memory DB. Each memory adapter clears and
                   ingests its own fixtures, then queries. Standard memory eval.
  prod-readonly  — Snapshot of the LIVE production DB. Recall-only — store(),
                   ingest(), and clear() are dropped. Measures how the system
                   behaves with whatever production happens to surface.

The prod DB is NEVER opened for writing — see
`benchwrap/adapters/neural_memory_readonly.py` for the safety model.

Usage
-----
    # Run all benchmarks in all 3 modes (default: 10 samples each)
    python3 run_suite.py

    # Single mode, more samples
    python3 run_suite.py --modes fresh --limit 50

    # Skip benchmarks that need external data dirs
    python3 run_suite.py --modes baseline,fresh --skip locomo,evomem,memory-agent-bench

    # Different model
    python3 run_suite.py --model ollama:qwen2.5:7b
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

# Make the repo importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchwrap.core.engine import EvaluationEngine
from benchwrap.core.model import parse_backend
from benchwrap.adapters import discover_adapters, get_adapter

MODES = ("baseline", "fresh", "prod-readonly")

# Benchmarks the suite knows how to run, in display order.
# Each entry: (adapter_name, default_dataset_or_None, uses_memory)
BENCHMARKS = [
    ("mmlu",                "STEM",              False),
    ("gsm8k",               "main",              False),
    ("memory-bench",        "recall-accuracy",   True),
    ("memory-bench",        "temporal-ordering", True),
    ("memory-bench",        "multi-hop",         True),
    ("locomo",              "single-hop",        True),
    ("evomem",              "mmlu-pro",          True),
    ("memory-agent-bench",  None,                True),
]


def build_memory_backend(mode: str):
    """Construct the memory backend for a given mode (or None for baseline)."""
    if mode == "baseline":
        return None
    if mode == "fresh":
        from benchwrap.adapters.neural_memory import NeuralMemoryBackend
        return NeuralMemoryBackend(db_path=None)  # tempfile under /tmp/
    if mode == "prod-readonly":
        from benchwrap.adapters.neural_memory_readonly import (
            ReadOnlyNeuralMemoryBackend,
        )
        return ReadOnlyNeuralMemoryBackend()
    raise ValueError(f"unknown mode: {mode}")


def configure_adapter(adapter_name: str, memory_backend, llm_backend):
    """Get a fresh adapter instance and inject memory/LLM backends if it accepts them."""
    adapter = get_adapter(adapter_name)
    if adapter is None:
        return None
    # Memory-aware adapters all expose .memory_client and .llm_backend
    if hasattr(adapter, "memory_client"):
        adapter.memory_client = memory_backend
    if hasattr(adapter, "llm_backend"):
        adapter.llm_backend = llm_backend
    return adapter


def run_one(
    adapter_name: str,
    dataset: str | None,
    memory_backend,
    llm_backend,
    limit: int,
    save_dir: Path,
    verbose: bool,
) -> dict:
    """Run a single (adapter, dataset) and return a summary dict."""
    adapter = configure_adapter(adapter_name, memory_backend, llm_backend)
    if adapter is None:
        return {"status": "missing_adapter", "adapter": adapter_name}

    engine = EvaluationEngine(
        adapter=adapter,
        backend=llm_backend,
        scorer=None,           # use adapter.score()
        verbose=verbose,
        save_dir=str(save_dir),
    )

    t0 = time.time()
    try:
        result = engine.run(dataset=dataset, limit=limit)
    except FileNotFoundError as e:
        return {
            "status": "skipped_missing_data",
            "adapter": adapter_name,
            "dataset": dataset,
            "error": str(e),
            "duration_s": time.time() - t0,
        }
    except Exception as e:
        return {
            "status": "error",
            "adapter": adapter_name,
            "dataset": dataset,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "duration_s": time.time() - t0,
        }

    return {
        "status": "ok",
        "adapter": result.adapter,
        "dataset": result.dataset,
        "n": result.num_samples,
        "metrics": result.metrics,
        "duration_s": result.duration_s,
    }


def print_table(rows: list[dict], modes: list[str]):
    """Print a (benchmark/dataset) × mode table of primary scores."""
    by_key: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        key = (r["adapter"], r.get("dataset") or "-")
        by_key.setdefault(key, {})[r["mode"]] = r

    name_w = max(28, max(len(f"{a}/{d}") for a, d in by_key.keys()) + 2)
    header = f"  {'benchmark/dataset':<{name_w}}"
    for m in modes:
        header += f" {m:>16}"
    header += f" {'n':>6}"

    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for (adapter, dataset), per_mode in sorted(by_key.items()):
        line = f"  {adapter+'/'+dataset:<{name_w}}"
        n_seen = 0
        for m in modes:
            r = per_mode.get(m)
            if r is None or r["status"] != "ok":
                marker = (
                    "skip"  if r and r["status"].startswith("skipped") else
                    "err"   if r and r["status"] == "error" else
                    "—"
                )
                line += f" {marker:>16}"
                continue
            metrics = r["metrics"]
            primary = metrics.get("accuracy",
                       metrics.get("exact_match",
                       metrics.get("f1", 0)))
            line += f" {primary*100:>15.1f}%"
            n_seen = max(n_seen, r["n"])
        line += f" {n_seen:>6}"
        print(line)
    print("=" * len(header))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",   default="ollama:openhermes:7b-v2.5",
                    help="LLM backend spec (default: ollama:openhermes:7b-v2.5)")
    ap.add_argument("--modes",   default=",".join(MODES),
                    help=f"Comma-separated modes from: {','.join(MODES)}")
    ap.add_argument("--skip",    default="",
                    help="Comma-separated adapter names to skip")
    ap.add_argument("--limit",   type=int, default=10,
                    help="Max samples per benchmark (default: 10)")
    ap.add_argument("--save-dir", default="results/suite",
                    help="Output directory (default: results/suite)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            ap.error(f"unknown mode {m!r}; choose from {MODES}")
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    discover_adapters()
    llm_backend = parse_backend(args.model)

    # Diagnose Ollama up front so we fail fast rather than per-sample.
    if hasattr(llm_backend, "diagnose"):
        d = llm_backend.diagnose()
        print(f"[suite] LLM backend: {d.get('backend')}/{d.get('model')} → {d.get('status')}")
        if d.get("status") not in ("ok", "unknown"):
            print(f"[suite] WARNING: {d}")

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t_suite = time.time()

    for mode in modes:
        print(f"\n{'#' * 70}\n#  MODE: {mode}\n{'#' * 70}")
        try:
            memory_backend = build_memory_backend(mode)
        except Exception as e:
            print(f"[suite] failed to build memory backend for {mode}: {e}")
            for adapter_name, dataset, _ in BENCHMARKS:
                if adapter_name in skip:
                    continue
                rows.append({
                    "mode": mode, "adapter": adapter_name, "dataset": dataset or "-",
                    "status": "error", "error": f"backend init: {e}",
                })
            continue

        mode_dir = save_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        for adapter_name, dataset, _uses_memory in BENCHMARKS:
            if adapter_name in skip:
                continue
            label = f"{adapter_name}/{dataset or '-'}"
            print(f"\n[suite][{mode}] >> {label} (limit={args.limit})")

            r = run_one(
                adapter_name=adapter_name,
                dataset=dataset,
                memory_backend=memory_backend,
                llm_backend=llm_backend,
                limit=args.limit,
                save_dir=mode_dir,
                verbose=args.verbose,
            )
            r["mode"] = mode
            r.setdefault("adapter", adapter_name)
            r.setdefault("dataset", dataset or "-")
            rows.append(r)

            status = r["status"]
            if status == "ok":
                m = r["metrics"]
                primary = m.get("accuracy", m.get("exact_match", m.get("f1", 0)))
                print(f"[suite][{mode}] {label}: {primary*100:.1f}%  "
                      f"n={r['n']}  ({r['duration_s']:.1f}s)")
            elif status == "skipped_missing_data":
                print(f"[suite][{mode}] {label}: SKIP — {r['error']}")
            else:
                print(f"[suite][{mode}] {label}: {status.upper()} — "
                      f"{r.get('error', '?')}")

        # Tear down between modes — release embedder/CUDA/snapshot.
        if hasattr(memory_backend, "close"):
            try: memory_backend.close()
            except Exception: pass
        del memory_backend

    # Final report
    summary_path = save_root / "suite_summary.json"
    summary_path.write_text(json.dumps({
        "model": args.model,
        "limit": args.limit,
        "modes": modes,
        "duration_s": time.time() - t_suite,
        "rows": rows,
    }, indent=2, default=str))
    print(f"\n[suite] Wrote {summary_path}")

    print_table(rows, modes)

    failed = [r for r in rows if r["status"] == "error"]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
