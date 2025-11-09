"""
Pure mathematical metric functions (REQ-EVAL-003).

All metrics are deterministic Python functions. NO LLM-as-judge allowed.

Implements:
- REQ-METRIC-001 through REQ-METRIC-013: Individual tool metrics
- REQ-METRIC-014 through REQ-METRIC-017: Composite metrics
"""

import numpy as np
from typing import Dict, Any, List

from src.models import (
    FilterResult,
    ClassificationResult,
    FraudResult,
    CSVResult,
)
from evals.models import ExpectedOutput


# ==============================================================================
# Tool 1: Filter Transactions Above 250 GBP Metrics
# ==============================================================================

def filter_accuracy(output: FilterResult, expected: ExpectedOutput) -> float:
    """
    REQ-METRIC-001: Binary accuracy for filter decision.

    Formula: 1.0 if filter decision matches expected, else 0.0
    Target: >98% accuracy on dataset

    Args:
        output: FilterResult from tool execution
        expected: ExpectedOutput ground truth

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    return 1.0 if output.is_above_threshold == expected.tool_1_filtered else 0.0


def conversion_precision(output: FilterResult, expected: ExpectedOutput) -> float:
    """
    REQ-METRIC-002: Currency conversion precision.

    Formula: 1.0 - absolute_error / expected_amount, clamped to [0,1]
    Target: Mean precision >0.95 across all conversion cases

    Args:
        output: FilterResult from tool execution
        expected: ExpectedOutput ground truth

    Returns:
        Precision score (0.0 to 1.0)
    """
    # If direct GBP, no conversion error
    if output.conversion_path_used == "Direct (GBP)":
        return 1.0

    # If no expected amount, assume correct
    if expected.expected_gbp_amount is None:
        return 1.0

    # Calculate relative error
    error = abs(output.amount_gbp - expected.expected_gbp_amount)
    precision = max(0.0, 1.0 - (error / expected.expected_gbp_amount))
    return precision


def filter_iteration_efficiency(output: FilterResult, expected: ExpectedOutput) -> float:
    """
    REQ-METRIC-003: MCTS iteration efficiency.

    Formula: 1.0 if MCTS used <= budget (100 for filter), 0.0 if exceeded
    Target: 100% efficiency (no budget overruns)

    Args:
        output: FilterResult from tool execution
        expected: ExpectedOutput ground truth

    Returns:
        1.0 if efficient, 0.0 if over budget
    """
    iterations_used = output.mcts_metadata.root_node_visits
    budget = expected.expected_min_iterations  # Should be 100 for filter
    return 1.0 if iterations_used <= budget else 0.0


# ==============================================================================
# Tool 2: Classify Transactions Metrics
# ==============================================================================

def classification_accuracy(output: ClassificationResult, expected: ExpectedOutput) -> float:
    """
    REQ-METRIC-004: Classification accuracy.

    Formula: Exact match on category
    Target: >90% accuracy (multi-class is harder)

    Args:
        output: ClassificationResult from tool execution
        expected: ExpectedOutput ground truth

    Returns:
        1.0 if correct category, 0.0 if incorrect
    """
    return 1.0 if output.category == expected.tool_2_classification else 0.0


def confidence_calibration(output: ClassificationResult, expected: ExpectedOutput) -> float:
    """
    REQ-METRIC-005: Confidence calibration error.

    Formula: 1.0 - |confidence - accuracy|
    Measures if confidence matches reality
    Target: Mean calibration error <0.10 (well-calibrated)

    Args:
        output: ClassificationResult from tool execution
        expected: ExpectedOutput ground truth

    Returns:
        Calibration score (0.0 to 1.0, higher is better)
    """
    is_correct = (output.category == expected.tool_2_classification)
    accuracy = 1.0 if is_correct else 0.0
    calibration = max(0.0, 1.0 - abs(output.confidence - accuracy))
    return calibration


def path_diversity(output: ClassificationResult, expected: ExpectedOutput) -> float:
    """
    REQ-METRIC-006: MCTS path diversity.

    Formula: unique_actions / total_actions_in_space
    Rewards exploring distinct actions
    Target: >0.60 (explores >60% of action space on average)

    Args:
        output: ClassificationResult from tool execution
        expected: ExpectedOutput ground truth

    Returns:
        Diversity score (0.0 to 1.0)
    """
    unique_actions = len(set(output.mcts_path))
    total_possible = 5  # Based on action space size for classification
    diversity = min(unique_actions / total_possible, 1.0) if total_possible > 0 else 0.0
    return diversity


# ==============================================================================
# Tool 3: Detect Fraudulent Transactions Metrics
# ==============================================================================

def fraud_risk_accuracy(output: FraudResult, expected: ExpectedOutput) -> float:
    """
    REQ-METRIC-007: Fraud risk level accuracy.

    Formula: Exact match on risk level
    Target: >92% accuracy

    Args:
        output: FraudResult from tool execution
        expected: ExpectedOutput ground truth

    Returns:
        1.0 if correct risk level, 0.0 if incorrect
    """
    return 1.0 if output.risk_level.value == expected.tool_3_fraud_risk else 0.0


def critical_classification_strictness(output: FraudResult, expected: ExpectedOutput) -> float:
    """
    REQ-METRIC-008: Critical vs non-critical separation.

    Formula:
    - If expected CRITICAL: 1.0 if predicted CRITICAL, 0.0 otherwise
    - If expected not CRITICAL: 1.0 if not predicted CRITICAL, 0.5 if false positive
    Target: Zero false positives for CRITICAL (safety-critical)

    Args:
        output: FraudResult from tool execution
        expected: ExpectedOutput ground truth

    Returns:
        Strictness score (0.0, 0.5, or 1.0)
    """
    expected_critical = (expected.tool_3_fraud_risk == "CRITICAL")
    predicted_critical = (output.risk_level.value == "CRITICAL")

    if expected_critical:
        # Must catch all CRITICAL cases
        return 1.0 if predicted_critical else 0.0
    else:
        # Penalize false positives for CRITICAL
        return 1.0 if not predicted_critical else 0.5


def fraud_reward_convergence(output: FraudResult, expected: ExpectedOutput) -> float:
    """
    REQ-METRIC-009: MCTS reward convergence.

    Formula: 1.0 if final reward variance < 0.05, else 0.0
    Target: >95% of cases converge

    Args:
        output: FraudResult from tool execution
        expected: ExpectedOutput ground truth

    Returns:
        1.0 if converged, 0.0 if not
    """
    variance = output.mcts_metadata.final_reward_variance
    max_variance = expected.max_acceptable_variance  # Should be 0.05
    return 1.0 if variance <= max_variance else 0.0


def fraud_indicator_coverage(output: FraudResult, expected: ExpectedOutput) -> float:
    """
    REQ-METRIC-010: Fraud indicator coverage.

    Formula: TP / (TP + FN) - did MCTS find all expected indicators?
    Target: >85% coverage (finds most true indicators)

    Args:
        output: FraudResult from tool execution
        expected: ExpectedOutput ground truth

    Returns:
        Coverage ratio (0.0 to 1.0)
    """
    expected_indicators = set(expected.expected_fraud_indicators)
    found_indicators = set(output.fraud_indicators)

    if len(expected_indicators) == 0:
        # No indicators expected, perfect coverage
        return 1.0

    true_positives = len(expected_indicators & found_indicators)
    false_negatives = len(expected_indicators - found_indicators)

    coverage = true_positives / (true_positives + false_negatives)
    return coverage


# ==============================================================================
# Tool 4: Generate Enhanced CSV Metrics
# ==============================================================================

def csv_column_completeness(output: CSVResult, expected: ExpectedOutput) -> float:
    """
    REQ-METRIC-011: Column completeness.

    Formula: +0.25 per required column present
    Target: 100% completeness (1.0)

    Args:
        output: CSVResult from tool execution
        expected: ExpectedOutput ground truth

    Returns:
        Completeness score (0.0 to 1.0)
    """
    required_columns = ["classification", "fraud_risk", "confidence", "mcts_explanation"]
    present = sum(1 for col in required_columns if col in output.columns_included)
    return present / len(required_columns)


def csv_row_count_accuracy(output: CSVResult, expected: ExpectedOutput) -> float:
    """
    REQ-METRIC-012: Row count accuracy.

    Formula: 1.0 if row count matches expected, else 0.0
    Target: 100% accuracy

    Args:
        output: CSVResult from tool execution
        expected: ExpectedOutput ground truth

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    if expected.expected_row_count is None:
        # No expected row count specified
        return 1.0

    return 1.0 if output.row_count == expected.expected_row_count else 0.0


def explanation_validity(output: CSVResult, expected: ExpectedOutput, tx_id: str) -> float:
    """
    REQ-METRIC-013: Explanation validity.

    Formula: Check if MCTS explanation contains expected keywords
    Target: >80% of keywords present

    Args:
        output: CSVResult from tool execution
        expected: ExpectedOutput ground truth
        tx_id: Transaction ID to check

    Returns:
        Keyword coverage ratio (0.0 to 1.0)
    """
    explanation = output.mcts_explanations.get(tx_id, "")
    required_keywords = expected.expected_keywords_in_explanation

    if len(required_keywords) == 0:
        # No keywords expected
        return 1.0

    found = sum(1 for keyword in required_keywords if keyword.lower() in explanation.lower())
    return found / len(required_keywords)


# ==============================================================================
# Aggregation & Composite Metrics
# ==============================================================================

def overall_agent_accuracy(scores: Dict[str, float]) -> float:
    """
    REQ-METRIC-014: Overall agent accuracy.

    Formula: Macro-average of all tool accuracies
    Target: >0.90 for production release

    Args:
        scores: Dictionary of all metric scores

    Returns:
        Overall accuracy (0.0 to 1.0)
    """
    tool_accuracies = [
        scores.get("filter_accuracy", 0.0),
        scores.get("classification_accuracy", 0.0),
        scores.get("fraud_risk_accuracy", 0.0),
        scores.get("csv_column_completeness", 0.0),
    ]
    return sum(tool_accuracies) / len(tool_accuracies)


def mcts_efficiency_score(scores: Dict[str, float]) -> float:
    """
    REQ-METRIC-015: MCTS efficiency score.

    Formula: Geometric mean of iteration efficiency and convergence rates
    Target: >0.75 (efficient and robust)

    Args:
        scores: Dictionary of all metric scores

    Returns:
        Efficiency score (0.0 to 1.0)
    """
    efficiency_metrics = [
        scores.get("filter_iteration_efficiency", 0.0),
        scores.get("fraud_reward_convergence", 0.0),
        scores.get("path_diversity", 0.0),
    ]

    # Geometric mean penalizes low performance in any area
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    safe_metrics = [max(m, epsilon) for m in efficiency_metrics]
    product = np.prod(safe_metrics)
    geometric_mean = product ** (1.0 / len(safe_metrics))

    return float(geometric_mean)


def cost_per_transaction(total_cost: float, case_count: int) -> float:
    """
    REQ-METRIC-016: Cost per transaction.

    Formula: total_cost / case_count
    Target: <$0.01 per transaction for POC, <$0.005 for production

    Args:
        total_cost: Total cost in dollars
        case_count: Number of transactions processed

    Returns:
        Cost per transaction in dollars
    """
    if case_count == 0:
        return 0.0
    return total_cost / case_count


def fraud_false_positive_rate(scores: List[Dict[str, Any]]) -> float:
    """
    REQ-METRIC-017: False positive rate for CRITICAL fraud risk.

    Formula: FP / (FP + TN) for CRITICAL risk level
    Target: <2% (critical for compliance)

    Args:
        scores: List of score dictionaries with 'expected' and 'actual' risk levels

    Returns:
        False positive rate (0.0 to 1.0)
    """
    false_positives = 0
    true_negatives = 0

    for score in scores:
        expected_risk = score.get("expected", "LOW")
        actual_risk = score.get("actual", "LOW")

        is_expected_critical = (expected_risk == "CRITICAL")
        is_actual_critical = (actual_risk == "CRITICAL")

        if not is_expected_critical:
            if is_actual_critical:
                false_positives += 1
            else:
                true_negatives += 1

    total = false_positives + true_negatives
    if total == 0:
        return 0.0

    return false_positives / total


# ==============================================================================
# Helper Functions for Batch Evaluation
# ==============================================================================

def calculate_mean_metric(
    results: List[tuple[Any, ExpectedOutput]],
    metric_func,
    **kwargs
) -> float:
    """
    Calculate mean value of a metric across multiple test cases.

    Args:
        results: List of (output, expected) tuples
        metric_func: Metric function to apply
        **kwargs: Additional arguments to pass to metric_func

    Returns:
        Mean metric value
    """
    if not results:
        return 0.0

    values = []
    for output, expected in results:
        try:
            value = metric_func(output, expected, **kwargs)
            values.append(value)
        except Exception as e:
            # Log error but continue
            print(f"Error calculating metric {metric_func.__name__}: {e}")
            values.append(0.0)

    return sum(values) / len(values) if values else 0.0


def calculate_pass_rate(
    results: List[tuple[Any, ExpectedOutput]],
    metric_func,
    threshold: float,
    **kwargs
) -> float:
    """
    Calculate pass rate (percentage of cases meeting threshold).

    Args:
        results: List of (output, expected) tuples
        metric_func: Metric function to apply
        threshold: Threshold value to meet
        **kwargs: Additional arguments to pass to metric_func

    Returns:
        Pass rate (0.0 to 1.0)
    """
    if not results:
        return 0.0

    passed = 0
    for output, expected in results:
        try:
            value = metric_func(output, expected, **kwargs)
            if value >= threshold:
                passed += 1
        except Exception as e:
            print(f"Error calculating metric {metric_func.__name__}: {e}")

    return passed / len(results)
