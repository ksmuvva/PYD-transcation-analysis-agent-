"""
Pure Mathematical Evaluations & Metrics Framework (REQ-EVAL-001).

This package implements deterministic, LLM-free evaluation metrics for the
transaction analysis agent. All metrics are pure Python functions.

Implements:
- REQ-EVAL-001: pydantic-evals integration
- REQ-EVAL-002: Deterministic evaluation mode
- REQ-EVAL-003: No LLM-as-judge policy
"""

from evals.models import Case, ExpectedOutput, EvaluationReport
from evals.metrics import (
    # Tool 1 metrics
    filter_accuracy,
    conversion_precision,
    filter_iteration_efficiency,
    # Tool 2 metrics
    classification_accuracy,
    confidence_calibration,
    path_diversity,
    # Tool 3 metrics
    fraud_risk_accuracy,
    critical_classification_strictness,
    fraud_reward_convergence,
    fraud_indicator_coverage,
    # Tool 4 metrics
    csv_column_completeness,
    csv_row_count_accuracy,
    explanation_validity,
    # Composite metrics
    overall_agent_accuracy,
    mcts_efficiency_score,
    cost_per_transaction,
    fraud_false_positive_rate,
)

__all__ = [
    "Case",
    "ExpectedOutput",
    "EvaluationReport",
    "filter_accuracy",
    "conversion_precision",
    "filter_iteration_efficiency",
    "classification_accuracy",
    "confidence_calibration",
    "path_diversity",
    "fraud_risk_accuracy",
    "critical_classification_strictness",
    "fraud_reward_convergence",
    "fraud_indicator_coverage",
    "csv_column_completeness",
    "csv_row_count_accuracy",
    "explanation_validity",
    "overall_agent_accuracy",
    "mcts_efficiency_score",
    "cost_per_transaction",
    "fraud_false_positive_rate",
]
