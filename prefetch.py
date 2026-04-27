#!/usr/bin/env python3
"""prefetch.py — warm all benchmark caches before a suite run.

Hits each adapter's loader so the data lands on disk under
~/.cache/benchwrap/ (mmlu, gsm8k) and ~/.cache/huggingface/datasets/
(memory-agent-bench), so the actual suite run is offline-fast and not
rate-limited.

Usage:
    python3 prefetch.py                  # everything
    python3 prefetch.py mmlu gsm8k       # named adapters only
    python3 prefetch.py --skip mmlu      # all except those

Adapters with external data (locomo, evomem) are reported as MISSING /
PRESENT — we don't know how to clone them, but we tell you which paths
to populate.
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_suite import load_env  # picks up HF token + cache pinning
load_env()

from benchwrap.adapters.mmlu import MMLUAdapter, MMLU_SUBJECTS
from benchwrap.adapters.gsm8k import GSM8KAdapter


def prefetch_mmlu() -> tuple[int, int]:
    """Hit every (subject, split) pair so the JSONL cache is fully populated."""
    a = MMLUAdapter()
    splits = ("test", "dev")
    ok = fail = 0
    total = len(MMLU_SUBJECTS) * len(splits)
    for i, subject in enumerate(MMLU_SUBJECTS, 1):
        for split in splits:
            cache_file = Path(a.cache_dir) / f"{subject}_{split}.jsonl"
            if cache_file.exists():
                ok += 1
                continue
            print(f"  [{i}/{len(MMLU_SUBJECTS)}] mmlu/{subject}/{split} → fetch")
            try:
                a._fetch_from_hf(subject, split)
                ok += 1
            except Exception as e:
                fail += 1
                print(f"    FAIL: {e}")
                # be polite — back off a bit before the next subject
                time.sleep(2)
    print(f"[prefetch][mmlu] {ok}/{total} cached, {fail} failed")
    return ok, fail


def prefetch_gsm8k() -> tuple[int, int]:
    a = GSM8KAdapter()
    ok = fail = 0
    targets = [("main", "train"), ("main", "test"),
               ("socratic", "train"), ("socratic", "test")]
    for ds, split in targets:
        try:
            path = a._get_or_download(ds, split)
            print(f"  gsm8k/{ds}/{split} → {path}")
            ok += 1
        except Exception as e:
            print(f"  gsm8k/{ds}/{split} FAIL: {e}")
            fail += 1
    print(f"[prefetch][gsm8k] {ok}/{len(targets)} cached, {fail} failed")
    return ok, fail


def prefetch_locomo() -> tuple[int, int]:
    """Download locomo10.json from snap-research/locomo to the path the adapter expects."""
    import urllib.request
    target = Path.home() / "projects" / "locomo-bench" / "data" / "locomo10.json"
    if target.exists() and target.stat().st_size > 1000:
        print(f"  locomo10.json already at {target} ({target.stat().st_size} bytes)")
        return 1, 0
    target.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
    print(f"  downloading {url}")
    try:
        urllib.request.urlretrieve(url, target)
        print(f"  → {target} ({target.stat().st_size} bytes)")
        return 1, 0
    except Exception as e:
        print(f"  FAIL: {e}")
        print(f"  → manually fetch into {target}")
        return 0, 1


def prefetch_memory_agent_bench() -> tuple[int, int]:
    """Walk MemoryAgentBench's dataset list so HF datasets cache is warm."""
    try:
        from benchwrap.adapters.memory_agent import (
            MemoryAgentBenchAdapter, DATASETS,
        )
    except Exception as e:
        print(f"[prefetch][memory-agent-bench] adapter unavailable: {e}")
        return 0, 0

    a = MemoryAgentBenchAdapter()
    ok = fail = 0
    for ds_name, info in DATASETS.items():
        try:
            print(f"  memory-agent-bench/{ds_name} → fetch")
            samples = list(a.load(ds_name, limit=1))  # triggers full HF download
            print(f"    OK ({len(samples)}+ samples cached)")
            ok += 1
        except Exception as e:
            print(f"    FAIL: {type(e).__name__}: {e}")
            fail += 1
    print(f"[prefetch][memory-agent-bench] {ok}/{len(DATASETS)} cached, {fail} failed")
    return ok, fail


def report_external() -> None:
    """Tell the user about benchmarks that need manually-placed data."""
    targets = {
        "evomem": Path.home() / "projects" / "evo_mem" / "data",
    }
    print("\n[prefetch] external-data benchmarks:")
    for name, path in targets.items():
        status = "PRESENT" if path.exists() else "MISSING"
        print(f"  {name:<8} {status:<8} {path}")
        if status == "MISSING":
            print(f"           populate manually — no canonical public source")


def warm_embedder() -> None:
    """Optional: pre-load BAAI/bge-m3 so the first memory run doesn't pay for it."""
    try:
        sys.path.insert(0, str(Path.home() / "projects" / "neural-memory-adapter" / "python"))
        from embed_provider import EmbeddingProvider
        print("[prefetch] warming bge-m3 embedder ...")
        EmbeddingProvider(backend="auto")  # loads model into ~/.neural_memory/models/
        print("[prefetch] embedder ready")
    except Exception as e:
        print(f"[prefetch] embedder warm-up skipped: {e}")


ADAPTERS = {
    "mmlu":               prefetch_mmlu,
    "gsm8k":              prefetch_gsm8k,
    "locomo":             prefetch_locomo,
    "memory-agent-bench": prefetch_memory_agent_bench,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*",
                    help=f"adapters to prefetch (default: all). choices: {list(ADAPTERS)}")
    ap.add_argument("--skip", default="", help="comma-separated adapters to skip")
    ap.add_argument("--no-embedder", action="store_true",
                    help="don't warm bge-m3 embedder")
    args = ap.parse_args()

    print(f"[prefetch] HF_HOME={os.environ.get('HF_HOME')}")
    print(f"[prefetch] HF_TOKEN={'set' if os.environ.get('HF_TOKEN') else 'unset (rate-limit risk)'}")

    chosen = args.targets or list(ADAPTERS)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    chosen = [c for c in chosen if c not in skip]
    bad = [c for c in chosen if c not in ADAPTERS]
    if bad:
        ap.error(f"unknown targets: {bad}; choose from {list(ADAPTERS)}")

    t0 = time.time()
    totals = {}
    for name in chosen:
        print(f"\n=== {name} ===")
        try:
            totals[name] = ADAPTERS[name]()
        except Exception:
            traceback.print_exc()
            totals[name] = (0, -1)

    if not args.no_embedder:
        print("\n=== embedder ===")
        warm_embedder()

    report_external()

    print(f"\n[prefetch] done in {time.time() - t0:.1f}s")
    for name, (ok, fail) in totals.items():
        print(f"  {name:<22} ok={ok}  fail={fail}")
    sys.exit(0 if all(f == 0 for _, f in totals.values()) else 1)


if __name__ == "__main__":
    main()
