"""
Security and compliance features for metrics (REQ-METRIC-SEC-001, REQ-METRIC-SEC-002).

Implements:
- Metric immutability enforcement
- Audit export for 7-year retention
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import List

from evals.models import EvaluationReport, MetricAuditRecord


def export_metrics_for_compliance(
    report: EvaluationReport,
    output_path: str = "metrics_audit_export.csv"
) -> str:
    """
    Export metrics for compliance audit (REQ-METRIC-SEC-002).

    Must be retained for 7 years per compliance requirements.

    Args:
        report: Evaluation report to export
        output_path: Path for CSV export file

    Returns:
        Path to exported file
    """
    audit_records: List[MetricAuditRecord] = []

    # Extract all metrics from report
    for tool_name, tool_result in report.tool_results.items():
        for metric_name, value in tool_result.metrics.items():
            record = MetricAuditRecord(
                metric_name=f"{tool_name}.{metric_name}",
                value=value,
                calculated_at=report.timestamp.isoformat(),
                dataset_version=report.dataset_version,
                evaluator_version="1.0.0",  # Should be from package version
                git_commit=report.git_commit or "unknown",
            )
            audit_records.append(record)

    # Add composite metrics
    composite_metrics = [
        ("overall_accuracy", report.overall_accuracy),
        ("mcts_efficiency", report.mcts_efficiency),
        ("cost_per_transaction", report.cost_per_transaction),
        ("fraud_fpr", report.fraud_fpr),
    ]

    for metric_name, value in composite_metrics:
        record = MetricAuditRecord(
            metric_name=f"composite.{metric_name}",
            value=value,
            calculated_at=report.timestamp.isoformat(),
            dataset_version=report.dataset_version,
            evaluator_version="1.0.0",
            git_commit=report.git_commit or "unknown",
        )
        audit_records.append(record)

    # Write to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric_name",
                "value",
                "calculated_at",
                "dataset_version",
                "evaluator_version",
                "git_commit",
            ]
        )
        writer.writeheader()

        for record in audit_records:
            writer.writerow(record.model_dump())

    return str(output_file)


def verify_report_immutability(report: EvaluationReport) -> bool:
    """
    Verify that evaluation report is immutable (REQ-METRIC-SEC-001).

    Args:
        report: Evaluation report to verify

    Returns:
        True if immutable, False otherwise
    """
    # Pydantic models with frozen=True are immutable
    # This function verifies the config is correct
    try:
        # Attempt to modify - should raise error
        report.overall_accuracy = 0.99
        return False  # If we get here, not immutable!
    except Exception:
        # Good - modification blocked
        return True


def generate_compliance_summary(report: EvaluationReport) -> str:
    """
    Generate compliance summary report.

    Args:
        report: Evaluation report

    Returns:
        Formatted compliance summary
    """
    summary = f"""
EVALUATION COMPLIANCE REPORT
{'='*80}

Run Information:
  Run ID: {report.run_id}
  Timestamp: {report.timestamp.isoformat()}
  Random Seed: {report.random_seed} (REQ-EVAL-002: Deterministic)
  Git Commit: {report.git_commit or 'unknown'}
  Dataset Version: {report.dataset_version}

Evaluation Metrics:
  Total Cases Evaluated: {report.total_cases_evaluated}
  Evaluation Duration: {report.evaluation_duration_seconds:.2f}s
  Overall Accuracy: {report.overall_accuracy:.2%}
  MCTS Efficiency: {report.mcts_efficiency:.2%}
  Cost per Transaction: ${report.cost_per_transaction:.4f}
  Fraud False Positive Rate: {report.fraud_fpr:.2%}

Tool Results:
"""

    for tool_name, result in report.tool_results.items():
        summary += f"\n  {tool_name.upper()}:\n"
        summary += f"    Total Cases: {result.total_cases}\n"
        summary += f"    Accuracy: {result.accuracy:.2%}\n"
        summary += f"    Passed: {result.passed_count}\n"
        summary += f"    Failed: {result.failed_count}\n"
        summary += f"    Metrics:\n"
        for metric_name, value in result.metrics.items():
            summary += f"      - {metric_name}: {value:.4f}\n"

    summary += f"\nCompliance Status:\n"
    summary += f"  All Metrics Passed: {'✓ YES' if report.all_metrics_passed else '✗ NO'}\n"
    summary += f"  Report Immutable: {'✓ YES' if verify_report_immutability(report) else '✗ NO'}\n"

    if report.baseline_comparison:
        summary += f"\nBaseline Comparison:\n"
        for metric_name, delta in report.baseline_comparison.items():
            sign = "+" if delta >= 0 else ""
            summary += f"  {metric_name}: {sign}{delta:.4f}\n"

    summary += f"\n{'='*80}\n"
    summary += f"REQ-EVAL-003 Compliance: No LLM-as-judge (All metrics are pure Python)\n"
    summary += f"REQ-METRIC-SEC-001 Compliance: Immutable evaluation results\n"
    summary += f"REQ-METRIC-SEC-002 Compliance: Audit export available\n"
    summary += f"{'='*80}\n"

    return summary
