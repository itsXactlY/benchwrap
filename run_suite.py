#!/usr/bin/env python3
"""
run_suite.py — benchwrap suite: plain LLM vs LLM + Neural Memory.

Modes
-----
  baseline   — LLM only, no memory. Memory adapters get memory_client=None
               and run zero-shot. Confabulation floor for memory tasks.
  neural     — LLM + Neural Memory with the full toolset (remember, recall,
               recall_multihop, recall_temporal, think, connections, graph).
               Each memory adapter dispatches to the right tool for its
               task type (multi-hop vs temporal vs plain recall).

Each adapter declares its canonical eval protocol via default_eval_config()
— MMLU runs 5-shot, GSM8K runs 8-shot CoT, etc. The harness doesn't impose
arbitrary defaults; it honors the protocol the benchmark was designed for.

Usage
-----
    python3 run_suite.py                               # both modes, 10 samples
    python3 run_suite.py --full                        # all sub-tasks, all samples
    python3 run_suite.py --modes neural --limit 50
    python3 run_suite.py --skip evomem
    python3 run_suite.py --model ollama:qwen2.5:7b
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Make the repo importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Environment setup — pull HF token from ~/.hermes/.env (or HF's own cache),
# pin caches so models aren't re-downloaded each run.
# ---------------------------------------------------------------------------

def _parse_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().lstrip("export ").strip()
        val = val.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def load_env() -> None:
    """Pull HF_TOKEN/cache settings from ~/.hermes/.env, then HF's own cache."""
    hermes_env = _parse_dotenv(Path.home() / ".hermes" / ".env")

    # Promote anything Hermes set, without clobbering caller's env.
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HOME",
              "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE",
              "MINIMAX_API_KEY", "MINIMAX_CN_API_KEY",
              "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NIM_API_KEY"):
        if k in hermes_env and not os.environ.get(k):
            os.environ[k] = hermes_env[k]

    # Fallback: HF CLI token cache (~/.cache/huggingface/token).
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        tok_path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
        if tok_path.is_file():
            tok = tok_path.read_text().strip()
            if tok:
                os.environ["HF_TOKEN"] = tok
                os.environ["HUGGINGFACE_HUB_TOKEN"] = tok

    # Mirror token to both common names so every lib finds it.
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    if os.environ.get("HUGGINGFACE_HUB_TOKEN") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_HUB_TOKEN"]

    # Pin caches so cached snapshots (BAAI/bge-m3 etc.) get reused, not re-fetched.
    cache_root = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME",
                          str(Path.home() / ".neural_memory" / "models"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


load_env()

from benchwrap.core.engine import EvaluationEngine
from benchwrap.core.model import parse_backend
from benchwrap.adapters import discover_adapters, get_adapter

MODES = ("baseline", "neural")

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

# When --full is set, we expand each benchmark to its broadest dataset.
FULL_DATASETS = {
    "mmlu":               "all",
    "gsm8k":              "main",
    "locomo":             "all",
    "evomem":             "all",
    "memory-agent-bench": "all",
}


def expand_for_full(benchmarks):
    """Substitute each entry's dataset with its FULL_DATASETS variant if any."""
    return [
        (a, FULL_DATASETS.get(a, d), u) for (a, d, u) in benchmarks
    ]


def build_memory_backend(mode: str):
    """Construct the memory backend for a given mode (or None for baseline).

    For 'neural': we always create a tempdir-isolated SQLite DB. The backend
    asserts at construction time that this path is NOT the production DB.
    """
    if mode == "baseline":
        return None
    if mode == "neural":
        import tempfile
        from benchwrap.adapters.neural_memory import (
            NeuralMemoryBackend, PROD_DB_PATH,
        )
        tmpdir = tempfile.mkdtemp(prefix="benchwrap_nm_")
        db = Path(tmpdir) / "memory.db"
        print(f"[suite] neural mode: isolated DB at {db}")
        print(f"[suite]              PROD DB ({PROD_DB_PATH}) is OFF-LIMITS — "
              f"asserted at backend init.")
        return NeuralMemoryBackend(db_path=db)
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
    limit: int | None,
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
                    help="Max samples per benchmark (default: 10; 0 = no limit)")
    ap.add_argument("--full",    action="store_true",
                    help="Full coverage: every sub-task, every sample (overrides --limit)")
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

    benchmarks = expand_for_full(BENCHMARKS) if args.full else list(BENCHMARKS)
    limit = None if (args.full or args.limit <= 0) else args.limit

    print(f"[suite] HF_HOME={os.environ.get('HF_HOME')}")
    print(f"[suite] HF_TOKEN={'set' if os.environ.get('HF_TOKEN') else 'unset'}")
    print(f"[suite] coverage={'FULL (all sub-tasks, all samples)' if args.full else f'limited to {limit} samples each'}")

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
            for adapter_name, dataset, _ in benchmarks:
                if adapter_name in skip:
                    continue
                rows.append({
                    "mode": mode, "adapter": adapter_name, "dataset": dataset or "-",
                    "status": "error", "error": f"backend init: {e}",
                })
            continue

        mode_dir = save_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        for adapter_name, dataset, _uses_memory in benchmarks:
            if adapter_name in skip:
                continue
            label = f"{adapter_name}/{dataset or '-'}"
            print(f"\n[suite][{mode}] >> {label} (limit={'∞' if limit is None else limit})")

            r = run_one(
                adapter_name=adapter_name,
                dataset=dataset,
                memory_backend=memory_backend,
                llm_backend=llm_backend,
                limit=limit,
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
