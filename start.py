#!/usr/bin/env python3
"""start.py — interactive launcher for run_suite.py.

Examples
--------
    python3 start.py                                          # asks everything
    python3 start.py ollama:openhermes:7b-v2.5 --full
    python3 start.py ollama:openhermes:7b-v2.5 --preview
    python3 start.py ollama:openhermes:7b-v2.5 --prod
    python3 start.py ollama:openhermes:7b-v2.5 --prod-readonly

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
    "preview":       {"limit": 3,   "modes": "baseline,fresh,prod-readonly"},
    "full":          {"limit": 100, "modes": "baseline,fresh,prod-readonly"},
    "prod":          {"limit": 50,  "modes": "fresh,prod-readonly"},
    "prod-readonly": {"limit": 50,  "modes": "prod-readonly"},
}
DEFAULT_MODEL = "ollama:openhermes:7b-v2.5"


def ask(prompt: str, default: str) -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val or default


def parse_argv(argv: list[str]) -> tuple[str | None, str | None]:
    model, preset = None, None
    for a in argv:
        if a.startswith("--") and a[2:] in PRESETS:
            preset = a[2:]
        elif not a.startswith("-"):
            model = a
    return model, preset


def main():
    model, preset = parse_argv(sys.argv[1:])

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
    ]

    print(f"\n  model:  {model}")
    print(f"  preset: {preset}")
    print(f"  modes:  {cfg['modes']}")
    print(f"  limit:  {cfg['limit']}")
    print(f"  →  {' '.join(cmd)}\n")

    if ask("Run now?", "y").lower() not in ("y", "yes"):
        print("aborted.")
        sys.exit(0)

    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
