#!/usr/bin/env python3
"""
run_tests.py — Master test runner for benchwrap.

Runs all test suites, collects results, prints summary.

Usage:
    python3 tests/run_tests.py              # All tests
    python3 tests/run_tests.py types        # Just test_types.py
    python3 tests/run_tests.py scorers      # Just test_scorers.py
    python3 tests/run_tests.py engine       # Just test_engine.py
    python3 tests/run_tests.py adapters     # Just test_adapters.py
    python3 tests/run_tests.py integration  # Just test_integration.py
    python3 tests/run_tests.py custom       # Custom adapter contracts
    python3 tests/run_tests.py --quick      # Smoke test (fast, no network)
"""
import sys
import os
import time
import subprocess
from pathlib import Path

TESTS_DIR = Path(__file__).parent
PROJECT_DIR = TESTS_DIR.parent

SUITES = {
    "types": "test_types.py",
    "scorers": "test_scorers.py",
    "engine": "test_engine.py",
    "adapters": "test_adapters.py",
    "integration": "test_integration.py",
    "custom": "test_custom_adapter.py",
}


def run_suite(name: str, script: str) -> tuple[int, int, float, str]:
    """Run a test suite and return (passed, total, duration, output)."""
    path = TESTS_DIR / script
    if not path.exists():
        return 0, 0, 0.0, f"File not found: {path}"
    
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_DIR),
        )
        output = result.stdout + result.stderr
        duration = time.time() - t0
        
        # Parse results from output
        passed = 0
        total = 0
        for line in output.split("\n"):
            if "passed," in line and "failed" in line:
                parts = line.strip().split()
                for i, p in enumerate(parts):
                    if p == "passed,":
                        passed = int(parts[i - 1])
                    elif p == "failed":
                        total = passed + int(parts[i - 1])
        
        if total == 0:
            total = passed  # All passed, no explicit count
        
        return passed, total, duration, output
    
    except subprocess.TimeoutExpired:
        return 0, 0, 120.0, "TIMEOUT after 120s"
    except Exception as e:
        return 0, 0, 0.0, f"ERROR: {e}"


def print_header():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  BENCHWRAP TEST SUITE".center(68) + "║")
    print("╚" + "═" * 68 + "╝")


def print_suite_result(name, passed, total, duration, output):
    status = "✓" if passed == total and total > 0 else "✗"
    pct = (passed / total * 100) if total > 0 else 0
    
    print(f"\n  {status} {name:<20} {passed:>3}/{total:<3}  ({duration:.1f}s)")
    
    if passed < total or total == 0:
        # Show failures
        for line in output.split("\n"):
            if "FAIL" in line or "ERROR" in line or "Traceback" in line:
                print(f"    {line.strip()}")


def print_summary(results):
    total_pass = sum(r[1] for r in results)
    total_tests = sum(r[2] for r in results)
    total_time = sum(r[3] for r in results)
    total_fail = total_tests - total_pass
    
    print(f"\n{'─' * 70}")
    print(f"  {'TOTAL':<20} {total_pass:>3}/{total_tests:<3}  ({total_time:.1f}s)")
    
    if total_fail == 0:
        print(f"\n  ✓ ALL {total_tests} TESTS PASSED")
    else:
        print(f"\n  ✗ {total_fail} FAILED out of {total_tests}")
    
    print()


def main():
    print_header()
    
    # Determine which suites to run
    if "--quick" in sys.argv:
        suites_to_run = ["types", "scorers", "engine"]
    elif len(sys.argv) > 1 and sys.argv[1] not in ("--quick",):
        suites_to_run = [s for s in sys.argv[1:] if s in SUITES]
        if not suites_to_run:
            print(f"  Unknown suite. Available: {', '.join(SUITES.keys())}")
            sys.exit(1)
    else:
        suites_to_run = list(SUITES.keys())
    
    results = []
    
    for suite_name in suites_to_run:
        script = SUITES[suite_name]
        passed, total, duration, output = run_suite(suite_name, script)
        print_suite_result(suite_name, passed, total, duration, output)
        results.append((suite_name, passed, total, duration))
    
    print_summary(results)
    
    # Exit code
    all_passed = all(p == t and t > 0 for _, p, t, _ in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
