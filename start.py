#!/usr/bin/env python3
"""start.py — one-shot launcher: runs the full suite with defaults."""
import subprocess, sys
sys.exit(subprocess.call([sys.executable, "run_suite.py", *sys.argv[1:]]))
