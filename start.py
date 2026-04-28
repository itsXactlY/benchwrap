#!/usr/bin/env python3
"""start.py — interactive launcher for run_suite.py.

Examples
--------
    python3 start.py                                          # asks everything
    python3 start.py ollama:openhermes:7b-v2.5 --full
    python3 start.py ollama:openhermes:7b-v2.5 --preview
    python3 start.py ollama:openhermes:7b-v2.5 --prod
    python3 start.py ollama:openhermes:7b-v2.5 --prod-readonly
    python3 start.py --prefetch                               # warm caches first
    python3 start.py ollama:openhermes:7b-v2.5 --full --prefetch

Modes
-----
    --preview       limit=3,   modes=baseline,fresh,prod-readonly   (smoke test)
    --full          limit=100, modes=baseline,fresh,prod-readonly   (the works)
    --prod          limit=50,  modes=fresh,prod-readonly            (production-like)
    --prod-readonly limit=50,  modes=prod-readonly                  (live DB only)
"""
import subprocess
import sys

PRESETS = {
    # 2 modes only: baseline (no memory) vs neural (full Neural Memory tooling).
    "preview": {"limit": 5,   "modes": "baseline,neural", "extras": []},
    "quick":   {"limit": 25,  "modes": "baseline,neural", "extras": []},
    "full":    {"limit": 0,   "modes": "baseline,neural", "extras": ["--full"]},
}
DEFAULT_MODEL = "minimax:MiniMax-M2.7"


def ask(prompt: str, default: str) -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val or default


def parse_argv(argv: list[str]) -> tuple[str | None, str | None, bool]:
    model, preset, prefetch = None, None, False
    for a in argv:
        if a == "--prefetch":
            prefetch = True
        elif a.startswith("--") and a[2:] in PRESETS:
            preset = a[2:]
        elif not a.startswith("-"):
            model = a
    return model, preset, prefetch


def _caches_missing() -> bool:
    """Return True if MMLU dev splits or GSM8K main are not on disk yet."""
    import os.path
    cache = os.path.expanduser("~/.cache/benchwrap")
    if not os.path.exists(f"{cache}/gsm8k/main_train.jsonl"):
        return True
    # at least the STEM subset's dev splits should exist
    stem = ["abstract_algebra", "college_physics", "machine_learning"]
    if any(not os.path.exists(f"{cache}/mmlu/{s}_dev.jsonl") for s in stem):
        return True
    return False


def main():
    model, preset, prefetch = parse_argv(sys.argv[1:])

    if not prefetch and _caches_missing():
        print("[start] benchmark caches not found locally — running prefetch first.")
        prefetch = True

    if prefetch:
        rc = subprocess.call([sys.executable, "prefetch.py"])
        if rc != 0:
            print(f"[start] prefetch exited {rc}; continuing anyway.")

    if model is None:
        model = ask("Model", DEFAULT_MODEL)
    if preset is None:
        preset = ask(f"Preset ({'/'.join(PRESETS)})", "preview")
    if preset not in PRESETS:
        print(f"unknown preset: {preset!r}. choose from {list(PRESETS)}")
        sys.exit(2)

    cfg = PRESETS[preset]
    cmd = [
        sys.executable, "run_suite.py",
        "--model", model,
        "--modes", cfg["modes"],
        "--limit", str(cfg["limit"]),
        *cfg["extras"],
    ]

    print(f"\n  model:  {model}")
    print(f"  preset: {preset}")
    print(f"  modes:  {cfg['modes']}")
    print(f"  limit:  {'∞ (--full)' if '--full' in cfg['extras'] else cfg['limit']}")
    print(f"  →  {' '.join(cmd)}\n")

    if ask("Run now?", "y").lower() not in ("y", "yes"):
        print("aborted.")
        sys.exit(0)

    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
