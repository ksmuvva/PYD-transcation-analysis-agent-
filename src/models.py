"""
Data models for financial transaction analysis agent.

All models use Pydantic for type safety and validation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


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
    amount: float = Field(..., gt=0, description="Transaction amount (must be positive)")
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


class TransactionFilterResult(BaseModel):
    """Result from Tool 1: filter_transactions_above_threshold."""

    filtered_count: int = Field(..., ge=0, description="Number of transactions filtered")
    total_amount: float = Field(..., description="Total amount of filtered transactions")
    currency: Currency = Field(..., description="Base currency used for filtering")
    average_amount: float = Field(..., description="Average transaction amount")


class ClassificationResult(BaseModel):
    """
    Result from Tool 2: classify_transactions_mcts.

    Contains the classification decision for a single transaction.
    """

    transaction_id: str = Field(..., description="Transaction identifier")
    primary_classification: str = Field(..., description="Main classification category")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    alternative_classifications: list[tuple[str, float]] = Field(
        default_factory=list, description="Alternative categories with scores"
    )
    mcts_iterations: int = Field(..., gt=0, description="Number of MCTS iterations performed")
    reasoning_trace: str = Field(..., description="Detailed reasoning explanation")


class FraudRiskLevel(str, Enum):
    """Fraud risk levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FraudDetectionResult(BaseModel):
    """
    Result from Tool 3: detect_fraud_mcts.

    Contains fraud detection analysis for a single transaction.
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
