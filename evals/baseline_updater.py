#!/usr/bin/env python3
"""
Baseline metrics updater (REQ-EVAL-030).

Extracts metrics from evaluation report and updates baseline file.
"""

import json
import sys
from pathlib import Path


def update_baseline(report_path: str, baseline_path: str = "evals/baselines/main_branch_metrics.json"):
    """
    Update baseline metrics from evaluation report.

    Args:
        report_path: Path to evaluation report JSON
        baseline_path: Path to baseline metrics file
    """
    # Load report
    with open(report_path) as f:
        report = json.load(f)

    # Extract all metrics
    baseline_metrics = {}

    # Tool metrics
    for tool_name, tool_result in report.get("tool_results", {}).items():
        for metric_name, value in tool_result["metrics"].items():
            baseline_metrics[f"{tool_name}.{metric_name}"] = value

        # Also store tool accuracy
        baseline_metrics[f"{tool_name}_accuracy"] = tool_result["accuracy"]

    # Composite metrics
    baseline_metrics["overall_accuracy"] = report["overall_accuracy"]
    baseline_metrics["mcts_efficiency"] = report["mcts_efficiency"]
    baseline_metrics["cost_per_transaction"] = report["cost_per_transaction"]
    baseline_metrics["fraud_fpr"] = report["fraud_fpr"]

    # Add metadata
    baseline_metrics["_metadata"] = {
        "updated_at": report["timestamp"],
        "git_commit": report.get("git_commit"),
        "dataset_version": report.get("dataset_version"),
        "random_seed": report.get("random_seed"),
    }

    # Ensure baseline directory exists
    baseline_file = Path(baseline_path)
    baseline_file.parent.mkdir(parents=True, exist_ok=True)

    # Write baseline
    with open(baseline_file, "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    print(f"✅ Baseline updated: {baseline_path}")
    print(f"   Metrics count: {len(baseline_metrics) - 1}")  # Exclude metadata
    print(f"   Overall accuracy: {baseline_metrics['overall_accuracy']:.2%}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python evals/baseline_updater.py <report.json> [baseline.json]")
        sys.exit(1)

    report_path = sys.argv[1]
    baseline_path = sys.argv[2] if len(sys.argv) > 2 else "evals/baselines/main_branch_metrics.json"

    if not Path(report_path).exists():
        print(f"Error: Report file not found: {report_path}")
        sys.exit(1)

    update_baseline(report_path, baseline_path)


if __name__ == "__main__":
    main()
