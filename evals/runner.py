"""
Evaluation runner with Logfire integration (REQ-EVAL-002, REQ-LOGFIRE-METRIC-001).

Main entry point for running pure mathematical evaluations.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from uuid import uuid4

import logfire
import numpy as np
import pandas as pd
from pydantic_ai import RunContext

from evals.dataset_generator import generate_full_dataset
from evals.models import (
    EvaluationReport,
    ToolEvaluationResult,
    EvaluationConfig,
    MetricThresholds,
    MetricResult,
    Case,
)
from evals.metrics import (
    filter_accuracy,
    conversion_precision,
    filter_iteration_efficiency,
    classification_accuracy,
    confidence_calibration,
    path_diversity,
    fraud_risk_accuracy,
    critical_classification_strictness,
    fraud_reward_convergence,
    fraud_indicator_coverage,
    csv_column_completeness,
    csv_row_count_accuracy,
    explanation_validity,
    overall_agent_accuracy,
    mcts_efficiency_score,
    cost_per_transaction,
    fraud_false_positive_rate,
    calculate_mean_metric,
)
from src.tools_spec_compliant import (
    filter_above_250,
    classify_transaction,
    detect_fraud,
    generate_enhanced_csv,
)


class EvaluationRunner:
    """
    Main evaluation runner (REQ-EVAL-002).

    Runs deterministic evaluations with fixed random seed and Logfire logging.
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluation runner.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.thresholds = MetricThresholds()

        # REQ-EVAL-002: Set random seed for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Configure Logfire
        if config.enable_logfire:
            logfire.configure()

        self.run_id = str(uuid4())
        self.start_time = time.time()

    def run_full_evaluation(self) -> EvaluationReport:
        """
        Run full evaluation across all tools (REQ-EVAL-002).

        Returns:
            EvaluationReport with all results
        """
        with logfire.span(
            "evaluation_run",
            random_seed=self.config.random_seed,
            run_id=self.run_id,
            evaluation_type="full"
        ) as span:
            # Generate dataset
            dataset = generate_full_dataset()

            # Apply subset if configured
            if self.config.subset_size:
                dataset = self._subsample_dataset(dataset, self.config.subset_size)

            # Evaluate each tool
            tool_results = {}

            tool_results["filter"] = self._evaluate_filter_tool(dataset["filter"])
            tool_results["classify"] = self._evaluate_classify_tool(dataset["classify"])
            tool_results["fraud"] = self._evaluate_fraud_tool(dataset["fraud"])
            tool_results["csv"] = self._evaluate_csv_tool(dataset["csv"])

            # Calculate composite metrics
            all_metrics = self._aggregate_metrics(tool_results)

            overall_acc = overall_agent_accuracy(all_metrics)
            mcts_eff = mcts_efficiency_score(all_metrics)
            cost_per_tx = cost_per_transaction(0.0, len(dataset["filter"]))  # Cost tracking TBD
            fraud_fpr = fraud_false_positive_rate(
                self._extract_fraud_predictions(tool_results["fraud"])
            )

            # Get baseline comparison
            baseline_comparison = None
            if self.config.compare_to_baseline:
                baseline_comparison = self._compare_to_baseline(all_metrics)

            # Get git commit
            git_commit = self._get_git_commit()

            # Calculate total cases
            total_cases = sum(len(dataset[tool]) for tool in dataset)

            # Check if all metrics passed
            all_passed = self._check_all_thresholds(all_metrics)

            # Build report
            report = EvaluationReport(
                run_id=self.run_id,
                timestamp=datetime.now(),
                random_seed=self.config.random_seed,
                dataset_version="1.0.0",
                git_commit=git_commit,
                tool_results=tool_results,
                overall_accuracy=overall_acc,
                mcts_efficiency=mcts_eff,
                cost_per_transaction=cost_per_tx,
                fraud_fpr=fraud_fpr,
                baseline_comparison=baseline_comparison,
                total_cases_evaluated=total_cases,
                evaluation_duration_seconds=time.time() - self.start_time,
                all_metrics_passed=all_passed,
            )

            # Log final report to Logfire
            span.set_attribute("overall_accuracy", overall_acc)
            span.set_attribute("mcts_efficiency", mcts_eff)
            span.set_attribute("all_passed", all_passed)

            return report

    def _evaluate_filter_tool(self, cases: List[Case]) -> ToolEvaluationResult:
        """Evaluate Tool 1: Filter transactions."""
        with logfire.span("evaluate_filter_tool", total_cases=len(cases)):
            results = []
            metrics_sum = {"filter_accuracy": 0.0, "conversion_precision": 0.0, "filter_iteration_efficiency": 0.0}

            for case in cases:
                # Create DataFrame from case inputs
                df = pd.DataFrame([case.inputs])

                # Create RunContext
                ctx = RunContext(deps=df, retry=0, tool_name="filter_above_250")

                # Execute tool
                try:
                    output = filter_above_250(ctx, case.inputs["tx_id"])

                    # Calculate metrics
                    acc = filter_accuracy(output, case.expected_output)
                    prec = conversion_precision(output, case.expected_output)
                    eff = filter_iteration_efficiency(output, case.expected_output)

                    metrics_sum["filter_accuracy"] += acc
                    metrics_sum["conversion_precision"] += prec
                    metrics_sum["filter_iteration_efficiency"] += eff

                    # Log metric to Logfire
                    self._log_metric("filter_accuracy", acc, self.thresholds.filter_accuracy)

                except Exception as e:
                    logfire.error("Filter tool error", error=str(e), tx_id=case.inputs["tx_id"])

            # Calculate averages
            n = len(cases)
            avg_metrics = {k: v / n for k, v in metrics_sum.items()}

            passed = sum(1 for m in avg_metrics.values() if m >= 0.9)
            failed = len(avg_metrics) - passed

            return ToolEvaluationResult(
                tool_name="filter",
                total_cases=n,
                metrics=avg_metrics,
                passed_count=passed,
                failed_count=failed,
                accuracy=avg_metrics["filter_accuracy"],
            )

    def _evaluate_classify_tool(self, cases: List[Case]) -> ToolEvaluationResult:
        """Evaluate Tool 2: Classify transactions."""
        with logfire.span("evaluate_classify_tool", total_cases=len(cases)):
            results = []
            metrics_sum = {
                "classification_accuracy": 0.0,
                "confidence_calibration": 0.0,
                "path_diversity": 0.0
            }

            for case in cases:
                df = pd.DataFrame([case.inputs])
                ctx = RunContext(deps=df, retry=0, tool_name="classify_transaction")

                try:
                    output = classify_transaction(ctx, case.inputs["tx_id"])

                    # Calculate metrics
                    acc = classification_accuracy(output, case.expected_output)
                    cal = confidence_calibration(output, case.expected_output)
                    div = path_diversity(output, case.expected_output)

                    metrics_sum["classification_accuracy"] += acc
                    metrics_sum["confidence_calibration"] += cal
                    metrics_sum["path_diversity"] += div

                    self._log_metric("classification_accuracy", acc, self.thresholds.classification_accuracy)

                except Exception as e:
                    logfire.error("Classify tool error", error=str(e), tx_id=case.inputs["tx_id"])

            n = len(cases)
            avg_metrics = {k: v / n for k, v in metrics_sum.items()}

            passed = sum(1 for m in avg_metrics.values() if m >= 0.7)
            failed = len(avg_metrics) - passed

            return ToolEvaluationResult(
                tool_name="classify",
                total_cases=n,
                metrics=avg_metrics,
                passed_count=passed,
                failed_count=failed,
                accuracy=avg_metrics["classification_accuracy"],
            )

    def _evaluate_fraud_tool(self, cases: List[Case]) -> ToolEvaluationResult:
        """Evaluate Tool 3: Detect fraud."""
        with logfire.span("evaluate_fraud_tool", total_cases=len(cases)):
            metrics_sum = {
                "fraud_risk_accuracy": 0.0,
                "critical_classification_strictness": 0.0,
                "fraud_reward_convergence": 0.0,
                "fraud_indicator_coverage": 0.0,
            }

            for case in cases:
                df = pd.DataFrame([case.inputs])
                ctx = RunContext(deps=df, retry=0, tool_name="detect_fraud")

                try:
                    output = detect_fraud(ctx, case.inputs["tx_id"])

                    # Calculate metrics
                    acc = fraud_risk_accuracy(output, case.expected_output)
                    strict = critical_classification_strictness(output, case.expected_output)
                    conv = fraud_reward_convergence(output, case.expected_output)
                    cov = fraud_indicator_coverage(output, case.expected_output)

                    metrics_sum["fraud_risk_accuracy"] += acc
                    metrics_sum["critical_classification_strictness"] += strict
                    metrics_sum["fraud_reward_convergence"] += conv
                    metrics_sum["fraud_indicator_coverage"] += cov

                    self._log_metric("fraud_risk_accuracy", acc, self.thresholds.fraud_risk_accuracy)

                except Exception as e:
                    logfire.error("Fraud tool error", error=str(e), tx_id=case.inputs["tx_id"])

            n = len(cases)
            avg_metrics = {k: v / n for k, v in metrics_sum.items()}

            passed = sum(1 for m in avg_metrics.values() if m >= 0.85)
            failed = len(avg_metrics) - passed

            return ToolEvaluationResult(
                tool_name="fraud",
                total_cases=n,
                metrics=avg_metrics,
                passed_count=passed,
                failed_count=failed,
                accuracy=avg_metrics["fraud_risk_accuracy"],
            )

    def _evaluate_csv_tool(self, cases: List[Case]) -> ToolEvaluationResult:
        """Evaluate Tool 4: Generate CSV."""
        with logfire.span("evaluate_csv_tool", total_cases=len(cases)):
            metrics_sum = {
                "csv_column_completeness": 0.0,
                "csv_row_count_accuracy": 0.0,
                "explanation_validity": 0.0,
            }

            for case in cases:
                df = pd.DataFrame([case.inputs])
                ctx = RunContext(deps=df, retry=0, tool_name="generate_enhanced_csv")

                try:
                    output = generate_enhanced_csv(
                        ctx,
                        [case.inputs["tx_id"]],
                        f"/tmp/eval_csv_{self.run_id}_{case.inputs['tx_id']}.csv"
                    )

                    # Calculate metrics
                    comp = csv_column_completeness(output, case.expected_output)
                    row_acc = csv_row_count_accuracy(output, case.expected_output)
                    expl = explanation_validity(output, case.expected_output, case.inputs["tx_id"])

                    metrics_sum["csv_column_completeness"] += comp
                    metrics_sum["csv_row_count_accuracy"] += row_acc
                    metrics_sum["explanation_validity"] += expl

                    self._log_metric("csv_column_completeness", comp, self.thresholds.csv_column_completeness)

                except Exception as e:
                    logfire.error("CSV tool error", error=str(e), tx_id=case.inputs["tx_id"])

            n = len(cases)
            avg_metrics = {k: v / n for k, v in metrics_sum.items()}

            passed = sum(1 for m in avg_metrics.values() if m >= 0.8)
            failed = len(avg_metrics) - passed

            return ToolEvaluationResult(
                tool_name="csv",
                total_cases=n,
                metrics=avg_metrics,
                passed_count=passed,
                failed_count=failed,
                accuracy=avg_metrics["csv_column_completeness"],
            )

    def _log_metric(self, metric_name: str, value: float, threshold: float):
        """Log metric to Logfire (REQ-LOGFIRE-METRIC-001)."""
        if not self.config.enable_logfire:
            return

        passes = value >= threshold
        logfire.info(
            "metric_calculation",
            metric_name=metric_name,
            value=value,
            passes_threshold=passes,
            threshold=threshold,
        )

    def _aggregate_metrics(self, tool_results: Dict[str, ToolEvaluationResult]) -> Dict[str, float]:
        """Aggregate all metrics from tool results."""
        all_metrics = {}
        for tool_name, result in tool_results.items():
            all_metrics.update(result.metrics)
        return all_metrics

    def _extract_fraud_predictions(self, fraud_result: ToolEvaluationResult) -> List[Dict[str, Any]]:
        """Extract fraud predictions for FPR calculation."""
        # This would need actual predictions, returning empty for now
        return []

    def _subsample_dataset(self, dataset: Dict, size: int) -> Dict:
        """Subsample dataset for quick evaluation runs."""
        subsampled = {}
        for tool_name, cases in dataset.items():
            # Take first N cases
            subsampled[tool_name] = cases[:size]
        return subsampled

    def _compare_to_baseline(self, metrics: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Compare metrics to baseline (REQ-EVAL-030)."""
        baseline_path = Path(self.config.baseline_path)
        if not baseline_path.exists():
            logfire.warn("Baseline file not found", path=str(baseline_path))
            return None

        try:
            with open(baseline_path) as f:
                baseline = json.load(f)

            deltas = {}
            for metric_name, current_value in metrics.items():
                baseline_value = baseline.get(metric_name)
                if baseline_value is not None:
                    delta = current_value - baseline_value
                    deltas[metric_name] = delta

            logfire.info("baseline_comparison", deltas=deltas)
            return deltas

        except Exception as e:
            logfire.error("Failed to load baseline", error=str(e))
            return None

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception:
            return None

    def _check_all_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check if all metrics pass their thresholds."""
        thresholds = self.thresholds.model_dump()

        for metric_name, threshold in thresholds.items():
            if metric_name in metrics:
                # For cost_per_transaction and fraud_fpr, lower is better
                if metric_name in ["cost_per_transaction", "fraud_false_positive_rate"]:
                    if metrics[metric_name] > threshold:
                        return False
                else:
                    if metrics[metric_name] < threshold:
                        return False

        return True

    def save_report(self, report: EvaluationReport, output_path: str):
        """Save evaluation report to JSON file."""
        report_dict = report.model_dump(mode="json")

        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        logfire.info("report_saved", path=output_path)


def main():
    """CLI entry point for evaluation runner."""
    parser = argparse.ArgumentParser(description="Run pure mathematical evaluations")
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Run on subset of N cases per tool (for quick pre-commit checks)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_report.json",
        help="Output path for evaluation report"
    )
    parser.add_argument(
        "--no-logfire",
        action="store_true",
        help="Disable Logfire logging"
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Disable baseline comparison"
    )

    args = parser.parse_args()

    # Create config
    config = EvaluationConfig(
        random_seed=args.seed,
        subset_size=args.subset,
        enable_logfire=not args.no_logfire,
        compare_to_baseline=not args.no_baseline,
    )

    # Run evaluation
    runner = EvaluationRunner(config)
    report = runner.run_full_evaluation()

    # Save report
    runner.save_report(report, args.output)

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION REPORT SUMMARY")
    print("="*80)
    print(f"Run ID: {report.run_id}")
    print(f"Random Seed: {report.random_seed}")
    print(f"Total Cases: {report.total_cases_evaluated}")
    print(f"Duration: {report.evaluation_duration_seconds:.2f}s")
    print(f"\nOverall Accuracy: {report.overall_accuracy:.2%}")
    print(f"MCTS Efficiency: {report.mcts_efficiency:.2%}")
    print(f"Cost per Transaction: ${report.cost_per_transaction:.4f}")
    print(f"Fraud FPR: {report.fraud_fpr:.2%}")
    print(f"\nAll Metrics Passed: {'✓ YES' if report.all_metrics_passed else '✗ NO'}")
    print("="*80)

    # Exit with error code if metrics failed
    sys.exit(0 if report.all_metrics_passed else 1)


if __name__ == "__main__":
    main()
