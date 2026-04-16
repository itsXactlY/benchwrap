#!/usr/bin/env python3
"""
benchwrap — Unified LLM Benchmark Wrapper
ONE adapter to rule them all. Zero cheating. Full transparency.

Usage:
    python3 benchwrap.py list
    python3 benchwrap.py diagnose
    python3 benchwrap.py run mmlu --model ollama:openhermes:7b-v2.5
    python3 benchwrap.py run gsm8k --model ollama:qwen2.5:7b --fewshot 5
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchwrap.cli import main

if __name__ == "__main__":
    main()
