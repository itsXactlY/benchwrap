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
    # full = every sub-task, every sample. extras gets --full forwarded to run_suite.
    "preview":       {"limit": 3,   "modes": "baseline,fresh,prod-readonly", "extras": []},
    "full":          {"limit": 0,   "modes": "baseline,fresh,prod-readonly", "extras": ["--full"]},
    "prod":          {"limit": 50,  "modes": "fresh,prod-readonly",          "extras": []},
    "prod-readonly": {"limit": 50,  "modes": "prod-readonly",                "extras": []},
}
DEFAULT_MODEL = "ollama:openhermes:7b-v2.5"


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


def main():
    model, preset, prefetch = parse_argv(sys.argv[1:])

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
