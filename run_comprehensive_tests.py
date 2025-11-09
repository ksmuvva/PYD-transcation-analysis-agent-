"""
Comprehensive Test Suite Runner
Runs all tests systematically and generates detailed report
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

def run_tests(test_pattern, description):
    """Run tests and return results"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")

    start_time = time.time()
    result = subprocess.run(
        [".venv/bin/python", "-m", "pytest", test_pattern, "-v", "--tb=short", "--json-report", "--json-report-file=test_results.json"],
        capture_output=True,
        text=True,
        timeout=600
    )
    elapsed = time.time() - start_time

    return {
        "description": description,
        "pattern": test_pattern,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed": elapsed,
        "passed": result.returncode == 0
    }

def main():
    print("=" * 80)
    print("COMPREHENSIVE TEST EXECUTION")
    print("=" * 80)
    print(f"Start Time: {datetime.now()}")

    test_suites = [
        ("tests/test_models.py", "Data Model Validation Tests"),
        ("tests/test_tools.py::TestCurrencyConversion", "Currency Conversion Tests"),
        ("tests/test_mcts.py", "MCTS Algorithm Tests"),
        ("tests/test_prompts.py", "Prompt Engineering Tests"),
    ]

    results = []
    for pattern, desc in test_suites:
        try:
            result = run_tests(pattern, desc)
            results.append(result)
        except Exception as e:
            print(f"ERROR running {desc}: {e}")
            results.append({
                "description": desc,
                "pattern": pattern,
                "error": str(e),
                "passed": False
            })

    # Generate summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))

    for r in results:
        status = "✅ PASS" if r.get("passed", False) else "❌ FAIL"
        print(f"{status} - {r['description']}")
        if 'elapsed' in r:
            print(f"   Time: {r['elapsed']:.2f}s")

    print(f"\nOverall: {passed}/{total} test suites passed ({100*passed/total:.1f}%)")
    print(f"End Time: {datetime.now()}")

    # Save detailed results
    with open("comprehensive_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: comprehensive_test_results.json")

if __name__ == "__main__":
    main()
