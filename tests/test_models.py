"""
Basic tests for data models.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models import (
    ClassificationResult,
    Currency,
    FraudDetectionResult,
    FraudRiskLevel,
    Transaction,
    TransactionFilterResult,
)


def test_transaction_valid():
    """Test valid transaction creation."""
    transaction = Transaction(
        transaction_id="TX001",
        amount=450.00,
        currency=Currency.GBP,
        date=datetime(2025, 1, 15),
        merchant="Amazon UK",
        description="Office supplies",
    )

    assert transaction.transaction_id == "TX001"
    assert transaction.amount == 450.00
    assert transaction.currency == Currency.GBP


def test_transaction_invalid_amount():
    """Test that negative amounts are rejected."""
    with pytest.raises(ValidationError):
        Transaction(
            transaction_id="TX001",
            amount=-100.00,  # Invalid: negative
            currency=Currency.GBP,
            date=datetime(2025, 1, 15),
            merchant="Test",
            description="Test",
        )


def test_transaction_date_parsing():
    """Test date parsing from string."""
    transaction = Transaction(
        transaction_id="TX001",
        amount=450.00,
        currency=Currency.GBP,
        date="2025-01-15",  # String date
        merchant="Amazon UK",
        description="Office supplies",
    )

    assert isinstance(transaction.date, datetime)
    assert transaction.date.year == 2025
    assert transaction.date.month == 1
    assert transaction.date.day == 15


def test_classification_result():
    """Test classification result model."""
    result = ClassificationResult(
        transaction_id="TX001",
        primary_classification="Business Expense - Office Supplies",
        confidence=0.92,
        mcts_iterations=100,
        reasoning_trace="High confidence classification",
    )

    assert result.confidence == 0.92
    assert 0.0 <= result.confidence <= 1.0


def test_fraud_detection_result():
    """Test fraud detection result model."""
    result = FraudDetectionResult(
        transaction_id="TX001",
        risk_level=FraudRiskLevel.HIGH,
        confidence=0.85,
        detected_indicators=["Large amount", "Unknown merchant"],
        reasoning="Multiple fraud indicators detected",
        mcts_iterations=100,
        recommended_actions=["Review immediately"],
    )

    assert result.risk_level == FraudRiskLevel.HIGH
    assert len(result.detected_indicators) == 2
    assert len(result.recommended_actions) == 1


def test_transaction_filter_result():
    """Test transaction filter result."""
    result = TransactionFilterResult(
        filtered_count=50,
        total_amount=125000.00,
        currency=Currency.GBP,
        average_amount=2500.00,
    )

    assert result.filtered_count == 50
    assert result.average_amount == 2500.00
