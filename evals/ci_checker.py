#!/usr/bin/env python3
"""
CI checker for evaluation thresholds (REQ-EVAL-029).

Checks if evaluation metrics meet required thresholds for PR merge.
"""

import json
import sys
from pathlib import Path

from evals.models import MetricThresholds


def check_thresholds(report_path: str) -> bool:
    """
    Check if evaluation report meets all thresholds.

    Args:
        report_path: Path to evaluation report JSON

    Returns:
        True if all thresholds met, False otherwise
    """
    # Load report
    with open(report_path) as f:
        report = json.load(f)

    # Load thresholds
    thresholds = MetricThresholds()

    # Check overall metrics
    checks = []

    # Overall accuracy must not drop >2% from baseline
    if report.get("baseline_comparison"):
        baseline_deltas = report["baseline_comparison"]
        accuracy_delta = baseline_deltas.get("overall_accuracy", 0.0)
        if accuracy_delta < -0.02:
            checks.append((False, f"Overall accuracy dropped {abs(accuracy_delta):.2%} (threshold: -2%)"))
        else:
            checks.append((True, f"Overall accuracy delta: {accuracy_delta:+.2%}"))

    # Tool-specific checks
    for tool_name, tool_result in report.get("tool_results", {}).items():
        tool_accuracy = tool_result["accuracy"]

        # Check if any tool accuracy drops >5%
        if report.get("baseline_comparison"):
            tool_delta = baseline_deltas.get(f"{tool_name}_accuracy", 0.0)
            if tool_delta < -0.05:
                checks.append((False, f"{tool_name} accuracy dropped {abs(tool_delta):.2%} (threshold: -5%)"))
            else:
                checks.append((True, f"{tool_name} accuracy delta: {tool_delta:+.2%}"))

    # Cost per transaction must not increase >10%
    cost = report.get("cost_per_transaction", 0.0)
    if report.get("baseline_comparison"):
        cost_delta_abs = baseline_deltas.get("cost_per_transaction", 0.0)
        # Calculate percentage increase
        baseline_cost = cost - cost_delta_abs
        if baseline_cost > 0:
            cost_delta_pct = cost_delta_abs / baseline_cost
            if cost_delta_pct > 0.10:
                checks.append((False, f"Cost per transaction increased {cost_delta_pct:.2%} (threshold: +10%)"))
            else:
                checks.append((True, f"Cost per transaction delta: {cost_delta_pct:+.2%}"))

    # MCTS efficiency must not drop >5%
    if report.get("baseline_comparison"):
        efficiency_delta = baseline_deltas.get("mcts_efficiency", 0.0)
        if efficiency_delta < -0.05:
            checks.append((False, f"MCTS efficiency dropped {abs(efficiency_delta):.2%} (threshold: -5%)"))
        else:
            checks.append((True, f"MCTS efficiency delta: {efficiency_delta:+.2%}"))

    # Print results
    print("\n" + "="*80)
    print("CI THRESHOLD CHECKS (REQ-EVAL-029)")
    print("="*80)

    for passed, message in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {message}")

    all_passed = all(passed for passed, _ in checks)

    print("="*80)
    if all_passed:
        print("✅ All CI checks passed - PR can be merged")
    else:
        print("❌ Some CI checks failed - PR blocked")
    print("="*80)

    return all_passed


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python evals/ci_checker.py <report.json>")
        sys.exit(1)

    report_path = sys.argv[1]

    if not Path(report_path).exists():
        print(f"Error: Report file not found: {report_path}")
        sys.exit(1)

    passed = check_thresholds(report_path)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
