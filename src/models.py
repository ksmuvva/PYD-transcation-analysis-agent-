"""
Data models for financial transaction analysis agent.

All models use Pydantic for type safety and validation.

Implements REQ-011 and REQ-012: Tool signatures and output schema definitions
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Literal

from pydantic import BaseModel, Field, field_validator


# ==============================================================================
# Exceptions (REQ-013)
# ==============================================================================

class MCTSConvergenceError(Exception):
    """
    Raised when MCTS fails to converge within iteration budget.

    Implements REQ-013: Error handling for MCTS convergence failures.

    Attributes:
        tool_name: Name of the tool that failed
        transaction_id: Transaction ID being processed
        iterations_completed: Number of iterations before failure
        final_variance: Final reward variance at failure
    """
    def __init__(
        self,
        tool_name: str,
        transaction_id: str,
        iterations_completed: int,
        final_variance: float,
        message: str = "MCTS failed to converge"
    ):
        self.tool_name = tool_name
        self.transaction_id = transaction_id
        self.iterations_completed = iterations_completed
        self.final_variance = final_variance
        super().__init__(
            f"{message}: {tool_name} on {transaction_id} "
            f"(iterations={iterations_completed}, variance={final_variance:.4f})"
        )


class Currency(str, Enum):
    """Supported currency codes."""

    GBP = "GBP"
    USD = "USD"
    EUR = "EUR"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"


class Transaction(BaseModel):
    """
    Input transaction model.

    Example:
        >>> transaction = Transaction(
        ...     transaction_id="TX001",
        ...     amount=350.00,
        ...     currency=Currency.GBP,
        ...     date=datetime(2025, 1, 15),
        ...     merchant="Amazon UK",
        ...     description="Office supplies purchase"
        ... )
    """

    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., ge=0, description="Transaction amount (must be non-negative)")
    currency: Currency = Field(..., description="Currency code")
    date: datetime = Field(..., description="Transaction date")
    merchant: str = Field(..., min_length=1, description="Merchant name")
    category: Optional[str] = Field(None, description="Optional category")
    description: str = Field(..., min_length=1, description="Transaction description")

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: str | datetime) -> datetime:
        """Parse date from multiple formats."""
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # Try common formats
            for fmt in [
                "%Y-%m-%d",
                "%Y-%m-%d %H:%M:%S",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%fZ",
            ]:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            # If pandas Timestamp
            try:
                import pandas as pd

                if isinstance(v, pd.Timestamp):
                    return v.to_pydatetime()
            except ImportError:
                pass
        raise ValueError(f"Unable to parse date: {v}")


# ==============================================================================
# MCTS Metadata (REQ-015)
# ==============================================================================

class MCTSMetadata(BaseModel):
    """
    MCTS metadata for traceability and tuning analysis.

    Implements REQ-015: MCTS metadata propagation

    Attributes:
        root_node_visits: Number of visits to root node
        best_action_path: List of actions taken in best path
        average_reward: Average reward across all iterations
        exploration_constant_used: UCB1 exploration constant used
        final_reward_variance: Variance in rewards at completion
        total_nodes_explored: Total nodes created in search tree
        max_depth_reached: Maximum depth reached during search
    """
    root_node_visits: int = Field(..., ge=0, description="Root node visit count")
    best_action_path: list[str] = Field(default_factory=list, description="Best action sequence")
    average_reward: float = Field(..., ge=0, le=1, description="Average reward")
    exploration_constant_used: float = Field(..., ge=0.1, le=2.0, description="UCB1 exploration constant")
    final_reward_variance: float = Field(..., ge=0, description="Final reward variance")
    total_nodes_explored: int = Field(..., ge=0, description="Total nodes in tree")
    max_depth_reached: int = Field(..., ge=0, description="Maximum depth reached")


# ==============================================================================
# Tool 1: Filter Transactions (REQ-006, REQ-011, REQ-012)
# ==============================================================================

class FilterResult(BaseModel):
    """
    Result from Tool 1: filter_above_250 (REQ-011, REQ-012).

    Implements REQ-006: Binary reward for GBP amount >= 250

    This replaces TransactionFilterResult with the proper signature per REQ-011.
    """
    is_above_threshold: bool = Field(..., description="True if >= 250 GBP")
    amount_gbp: float = Field(..., ge=0, description="Amount in GBP")
    conversion_path_used: str = Field(..., description="Currency conversion path (e.g., 'USD->GBP via ECB')")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in conversion accuracy")
    mcts_metadata: MCTSMetadata = Field(..., description="MCTS search metadata")


class TransactionFilterResult(BaseModel):
    """
    Batch filtering result (legacy compatibility).

    For batch processing across multiple transactions.
    """

    filtered_count: int = Field(..., ge=0, description="Number of transactions filtered")
    total_amount: float = Field(..., description="Total amount of filtered transactions")
    currency: Currency = Field(..., description="Base currency used for filtering")
    average_amount: float = Field(..., description="Average transaction amount")


# ==============================================================================
# Tool 2: Classify Transactions (REQ-007, REQ-011, REQ-012)
# ==============================================================================

class ClassificationResult(BaseModel):
    """
    Result from Tool 2: classify_transaction (REQ-011, REQ-012).

    Implements REQ-007: Multi-class classification with one-hot encoded reward.
    Categories: Business, Personal, Investment, Gambling
    Reward: 1.0 if predicted category matches ground truth, 0.0 otherwise
    """

    transaction_id: str = Field(..., description="Transaction identifier")
    category: Literal["Business", "Personal", "Investment", "Gambling"] = Field(
        ..., description="Classification category (REQ-007)"
    )
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    mcts_path: list[str] = Field(default_factory=list, description="MCTS action path taken")
    mcts_iterations: int = Field(..., gt=0, description="Number of MCTS iterations performed")
    mcts_metadata: MCTSMetadata = Field(..., description="MCTS search metadata")

    # Legacy compatibility
    primary_classification: Optional[str] = Field(None, description="Legacy: same as category")
    alternative_classifications: list[tuple[str, float]] = Field(
        default_factory=list, description="Alternative categories with scores"
    )
    reasoning_trace: str = Field("", description="Detailed reasoning explanation")

    def __init__(self, **data):
        """Initialize and sync category with primary_classification for backward compatibility."""
        if "category" in data and "primary_classification" not in data:
            data["primary_classification"] = data["category"]
        elif "primary_classification" in data and "category" not in data:
            data["category"] = data["primary_classification"]
        super().__init__(**data)


# ==============================================================================
# Tool 3: Detect Fraudulent Transactions (REQ-008, REQ-011, REQ-012)
# ==============================================================================

class FraudRiskLevel(str, Enum):
    """
    Fraud risk levels (REQ-008).

    Mapped to numeric rewards:
    - CRITICAL = 1.0
    - HIGH = 0.75
    - MEDIUM = 0.5
    - LOW = 0.0
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    def to_reward(self) -> float:
        """
        Convert risk level to numeric reward (REQ-008).

        Returns:
            Numeric reward value
        """
        rewards = {
            FraudRiskLevel.CRITICAL: 1.0,
            FraudRiskLevel.HIGH: 0.75,
            FraudRiskLevel.MEDIUM: 0.5,
            FraudRiskLevel.LOW: 0.0,
        }
        return rewards[self]


class FraudResult(BaseModel):
    """
    Result from Tool 3: detect_fraud (REQ-011, REQ-012).

    Implements REQ-008: Risk-level based reward function.
    Reward mapping: CRITICAL=1.0, HIGH=0.75, MEDIUM=0.5, LOW=0.0
    Reward is 1.0 only if predicted risk matches labeled risk exactly.
    """

    transaction_id: str = Field(..., description="Transaction identifier")
    risk_level: FraudRiskLevel = Field(..., description="Assessed fraud risk level (REQ-008)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in fraud assessment (0-1)")
    mcts_path: list[str] = Field(default_factory=list, description="MCTS action path taken")
    mcts_reward: float = Field(..., ge=0, le=1, description="Final MCTS reward value")
    fraud_indicators: list[str] = Field(
        default_factory=list, description="List of fraud indicators detected"
    )
    mcts_metadata: MCTSMetadata = Field(..., description="MCTS search metadata")

    # Legacy compatibility
    detected_indicators: Optional[list[str]] = Field(None, description="Legacy: same as fraud_indicators")
    reasoning: str = Field("", description="MCTS reasoning explanation")
    mcts_iterations: Optional[int] = Field(None, gt=0, description="Number of MCTS iterations performed")
    recommended_actions: list[str] = Field(
        default_factory=list, description="Recommended actions to take"
    )

    def __init__(self, **data):
        """Initialize and sync fraud_indicators with detected_indicators for backward compatibility."""
        if "fraud_indicators" in data and "detected_indicators" not in data:
            data["detected_indicators"] = data["fraud_indicators"]
        elif "detected_indicators" in data and "fraud_indicators" not in data:
            data["fraud_indicators"] = data["detected_indicators"]
        if "mcts_metadata" in data and "mcts_iterations" not in data:
            data["mcts_iterations"] = data["mcts_metadata"].root_node_visits
        super().__init__(**data)


class FraudDetectionResult(BaseModel):
    """
    Legacy fraud detection result (backward compatibility).

    Use FraudResult for new implementations.
    """

    transaction_id: str = Field(..., description="Transaction identifier")
    risk_level: FraudRiskLevel = Field(..., description="Assessed fraud risk level")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in fraud assessment (0-1)")
    detected_indicators: list[str] = Field(
        default_factory=list, description="List of fraud indicators detected"
    )
    reasoning: str = Field(..., description="MCTS reasoning explanation")
    mcts_iterations: int = Field(..., gt=0, description="Number of MCTS iterations performed")
    recommended_actions: list[str] = Field(
        default_factory=list, description="Recommended actions to take"
    )


# ==============================================================================
# Tool 4: Generate Enhanced CSV (REQ-009, REQ-011, REQ-012)
# ==============================================================================

class CSVResult(BaseModel):
    """
    Result from Tool 4: generate_enhanced_csv (REQ-011, REQ-012).

    Implements REQ-009: Data completeness reward function.
    Reward: +0.2 for each required column (classification, fraud risk, confidence, MCTS explanation)
    Total reward is 1.0 only if all 4 columns are present and correctly formatted.
    """

    file_path: str = Field(..., description="Path to generated enhanced CSV file")
    row_count: int = Field(..., ge=0, description="Number of rows in enhanced CSV")
    columns_included: list[str] = Field(
        ..., description="List of columns included in the CSV"
    )
    mcts_explanations: dict[str, str] = Field(
        ..., description="Dictionary mapping transaction_id to MCTS explanation"
    )

    def calculate_completeness_reward(self) -> float:
        """
        Calculate data completeness reward per REQ-009.

        Returns:
            Reward score (0.0 to 1.0)
        """
        required_columns = [
            "classification",
            "fraud_risk",
            "confidence",
            "mcts_explanation"
        ]
        reward = 0.0
        for col in required_columns:
            if col in self.columns_included:
                reward += 0.2
        return min(reward, 1.0)  # Cap at 1.0


class EnhancedTransaction(Transaction):
    """
    Enhanced transaction with analysis results.

    Extends the base Transaction model with additional analysis columns.
    """

    above_250_gbp: bool = Field(..., description="True if transaction >= 250 GBP equivalent")
    classification: str = Field(..., description="Transaction classification")
    classification_confidence: float = Field(
        ..., ge=0, le=1, description="Classification confidence score"
    )
    fraud_risk: FraudRiskLevel = Field(..., description="Fraud risk level")
    fraud_confidence: float = Field(..., ge=0, le=1, description="Fraud detection confidence")
    fraud_reasoning: str = Field(..., description="Fraud detection reasoning")
    mcts_iterations: int = Field(..., gt=0, description="Total MCTS iterations performed")


class ProcessingReport(BaseModel):
    """
    Summary report from Tool 4: generate_enhanced_csv.

    Contains statistics about the entire processing run.
    """

    total_transactions_analyzed: int = Field(..., ge=0, description="Total transactions processed")
    transactions_above_threshold: int = Field(
        ..., ge=0, description="Transactions above threshold"
    )
    high_risk_transactions: int = Field(..., ge=0, description="High-risk transactions detected")
    critical_risk_transactions: int = Field(
        ..., ge=0, description="Critical-risk transactions detected"
    )
    processing_time_seconds: float = Field(..., ge=0, description="Total processing time")
    llm_provider: str = Field(..., description="LLM provider used")
    model_used: str = Field(..., description="LLM model used")
    mcts_iterations_total: int = Field(..., ge=0, description="Total MCTS iterations across all")

    def summary_text(self) -> str:
        """Generate human-readable summary."""
        return f"""
Processing Report
=================
Total Transactions Analyzed: {self.total_transactions_analyzed}
Transactions Above Threshold: {self.transactions_above_threshold}
High Risk Transactions: {self.high_risk_transactions}
Critical Risk Transactions: {self.critical_risk_transactions}
Processing Time: {self.processing_time_seconds:.2f}s
LLM Provider: {self.llm_provider}
Model Used: {self.model_used}
Total MCTS Iterations: {self.mcts_iterations_total}
        """.strip()
