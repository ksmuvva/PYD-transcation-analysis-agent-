"""
Data models for pure mathematical evaluations (REQ-EVAL-004, REQ-EVAL-005).

All models are frozen Pydantic models for immutability (REQ-METRIC-SEC-001).
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


# ==============================================================================
# Expected Output Models (REQ-EVAL-004)
# ==============================================================================

class ExpectedOutput(BaseModel):
    """
    Ground truth expected output for a test case (REQ-EVAL-004).

    Contains all expected values across all four tools.
    """
    model_config = ConfigDict(frozen=True)  # REQ-METRIC-SEC-001: Immutability

    # Tool 1: Filter expectations
    tool_1_filtered: bool = Field(..., description="Expected filter result (>= 250 GBP)")
    expected_gbp_amount: Optional[float] = Field(None, description="Expected GBP amount after conversion")

    # Tool 2: Classification expectations
    tool_2_classification: Literal["Business", "Personal", "Investment", "Gambling"] = Field(
        ..., description="Expected classification category"
    )

    # Tool 3: Fraud detection expectations
    tool_3_fraud_risk: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        ..., description="Expected fraud risk level"
    )
    tool_3_confidence: float = Field(..., ge=0, le=1, description="Expected confidence threshold")
    expected_fraud_indicators: list[str] = Field(
        default_factory=list, description="Expected fraud indicators to be found"
    )

    # Tool 4: CSV expectations
    tool_4_columns_complete: bool = Field(..., description="Expected column completeness")
    expected_row_count: Optional[int] = Field(None, description="Expected row count in CSV")
    expected_keywords_in_explanation: list[str] = Field(
        default_factory=list, description="Expected keywords in MCTS explanation"
    )

    # MCTS ground truth metrics (REQ-EVAL-005)
    expected_min_iterations: int = Field(100, description="Minimum MCTS iterations expected")
    expected_min_reward: float = Field(0.80, description="Minimum average reward expected")
    expected_min_path_length: int = Field(10, description="Minimum exploration path length")
    max_acceptable_variance: float = Field(0.05, description="Maximum convergence variance allowed")


class CaseMetadata(BaseModel):
    """
    Metadata for a test case (REQ-EVAL-004).

    Contains categorization and labeling information.
    """
    model_config = ConfigDict(frozen=True)

    category: str = Field(..., description="Test category (e.g., 'adversarial_threshold')")
    labeled_by: str = Field(..., description="Who labeled this test case")
    is_fraud: bool = Field(False, description="Ground truth fraud label")
    is_adversarial: bool = Field(False, description="Is this an adversarial/edge case")
    mcc_code: Optional[str] = Field(None, description="Merchant Category Code if applicable")


class Case(BaseModel):
    """
    Complete test case for evaluation (REQ-EVAL-004).

    Combines inputs, expected outputs, and metadata.
    """
    model_config = ConfigDict(frozen=True)

    inputs: dict = Field(..., description="Transaction input data")
    expected_output: ExpectedOutput = Field(..., description="Expected results")
    metadata: CaseMetadata = Field(..., description="Test case metadata")


# ==============================================================================
# Evaluation Result Models
# ==============================================================================

class MetricResult(BaseModel):
    """
    Result from a single metric calculation.

    Implements REQ-LOGFIRE-METRIC-001: Structured metric logging.
    """
    model_config = ConfigDict(frozen=True)

    metric_name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Calculated metric value")
    passes_threshold: bool = Field(..., description="Whether metric passes its threshold")
    threshold: float = Field(..., description="Threshold value for this metric")
    calculated_at: datetime = Field(default_factory=datetime.now, description="When metric was calculated")


class ToolEvaluationResult(BaseModel):
    """
    Results from evaluating a single tool across all test cases.
    """
    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(..., description="Name of the tool")
    total_cases: int = Field(..., ge=0, description="Total test cases evaluated")
    metrics: dict[str, float] = Field(..., description="All metric values")
    passed_count: int = Field(..., ge=0, description="Number of passing cases")
    failed_count: int = Field(..., ge=0, description="Number of failing cases")
    accuracy: float = Field(..., ge=0, le=1, description="Overall accuracy")


class EvaluationReport(BaseModel):
    """
    Complete evaluation report (REQ-METRIC-SEC-001: Immutable).

    Contains all results from a full evaluation run.
    """
    model_config = ConfigDict(frozen=True)

    run_id: str = Field(..., description="Unique evaluation run ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="When evaluation ran")
    random_seed: int = Field(42, description="Random seed used (REQ-EVAL-002)")
    dataset_version: str = Field(..., description="Dataset version identifier")
    git_commit: Optional[str] = Field(None, description="Git commit hash")

    # Tool-specific results
    tool_results: dict[str, ToolEvaluationResult] = Field(..., description="Results per tool")

    # Composite metrics
    overall_accuracy: float = Field(..., ge=0, le=1, description="Overall agent accuracy")
    mcts_efficiency: float = Field(..., ge=0, le=1, description="MCTS efficiency score")
    cost_per_transaction: float = Field(..., ge=0, description="Cost per transaction")
    fraud_fpr: float = Field(..., ge=0, le=1, description="Fraud false positive rate")

    # Regression comparison (if baseline exists)
    baseline_comparison: Optional[dict[str, float]] = Field(
        None, description="Deltas from baseline metrics"
    )

    # Metadata
    total_cases_evaluated: int = Field(..., ge=0, description="Total test cases")
    evaluation_duration_seconds: float = Field(..., ge=0, description="Total runtime")
    all_metrics_passed: bool = Field(..., description="Whether all thresholds passed")


# ==============================================================================
# Configuration Models
# ==============================================================================

class EvaluationConfig(BaseModel):
    """
    Configuration for evaluation runs (REQ-EVAL-002).
    """
    random_seed: int = Field(42, description="Random seed for reproducibility")
    subset_size: Optional[int] = Field(None, description="Subset size for quick runs (None = full)")
    parallel_workers: int = Field(1, description="Number of parallel workers")
    enable_logfire: bool = Field(True, description="Enable Logfire logging")
    compare_to_baseline: bool = Field(True, description="Compare to baseline metrics")
    baseline_path: str = Field(
        "evals/baselines/main_branch_metrics.json",
        description="Path to baseline metrics file"
    )


class MetricThresholds(BaseModel):
    """
    Thresholds for all metrics (REQ-METRIC-001 through REQ-METRIC-017).
    """
    # Tool 1: Filter
    filter_accuracy: float = Field(0.98, description="REQ-METRIC-001: >98%")
    conversion_precision: float = Field(0.95, description="REQ-METRIC-002: >0.95")
    filter_iteration_efficiency: float = Field(1.0, description="REQ-METRIC-003: 100%")

    # Tool 2: Classify
    classification_accuracy: float = Field(0.90, description="REQ-METRIC-004: >90%")
    confidence_calibration: float = Field(0.90, description="REQ-METRIC-005: <0.10 error")
    path_diversity: float = Field(0.60, description="REQ-METRIC-006: >60%")

    # Tool 3: Fraud
    fraud_risk_accuracy: float = Field(0.92, description="REQ-METRIC-007: >92%")
    critical_classification_strictness: float = Field(1.0, description="REQ-METRIC-008: Zero FP")
    fraud_reward_convergence: float = Field(0.95, description="REQ-METRIC-009: >95%")
    fraud_indicator_coverage: float = Field(0.85, description="REQ-METRIC-010: >85%")

    # Tool 4: CSV
    csv_column_completeness: float = Field(1.0, description="REQ-METRIC-011: 100%")
    csv_row_count_accuracy: float = Field(1.0, description="REQ-METRIC-012: 100%")
    explanation_validity: float = Field(0.80, description="REQ-METRIC-013: >80%")

    # Composite
    overall_agent_accuracy: float = Field(0.90, description="REQ-METRIC-014: >0.90")
    mcts_efficiency_score: float = Field(0.75, description="REQ-METRIC-015: >0.75")
    cost_per_transaction: float = Field(0.01, description="REQ-METRIC-016: <$0.01")
    fraud_false_positive_rate: float = Field(0.02, description="REQ-METRIC-017: <2%")


# ==============================================================================
# Compliance Export Models (REQ-METRIC-SEC-002)
# ==============================================================================

class MetricAuditRecord(BaseModel):
    """
    Audit record for compliance export (REQ-METRIC-SEC-002).

    Must be retained for 7 years.
    """
    metric_name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Calculated value")
    calculated_at: str = Field(..., description="ISO timestamp")
    dataset_version: str = Field(..., description="Dataset version")
    evaluator_version: str = Field(..., description="Evaluator version")
    git_commit: str = Field(..., description="Git commit hash")
