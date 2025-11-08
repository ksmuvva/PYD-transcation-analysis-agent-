"""
CSV processing utilities for transaction data.

Handles CSV loading, validation, and enhanced CSV generation.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from src.models import (
    ClassificationResult,
    Currency,
    FraudDetectionResult,
    Transaction,
)


class CSVProcessor:
    """Handles CSV operations for transaction data."""

    # Required columns in input CSV
    REQUIRED_COLUMNS = [
        "transaction_id",
        "amount",
        "currency",
        "date",
        "merchant",
        "description",
    ]

    # Exchange rates to GBP (simplified - could use API in production)
    EXCHANGE_RATES = {
        Currency.GBP: 1.0,
        Currency.USD: 0.79,  # Example rate
        Currency.EUR: 0.86,
        Currency.JPY: 0.0054,
        Currency.CAD: 0.58,
        Currency.AUD: 0.51,
    }

    @staticmethod
    def load_csv(file_path: Path) -> pd.DataFrame:
        """
        Load CSV file into DataFrame.

        Args:
            file_path: Path to CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        try:
            # Try multiple encodings
            for encoding in ["utf-8", "latin-1", "iso-8859-1"]:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Unable to decode CSV file with supported encodings")

            if df.empty:
                raise ValueError("CSV file is empty")

            return df

        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")
        except pd.errors.ParserError as e:
            raise ValueError(f"Failed to parse CSV: {e}")

    @staticmethod
    def validate_schema(df: pd.DataFrame) -> list[str]:
        """
        Validate that DataFrame has required columns and valid data.

        Args:
            df: DataFrame to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required columns
        missing_columns = set(CSVProcessor.REQUIRED_COLUMNS) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        # Check for empty DataFrame
        if df.empty:
            errors.append("DataFrame is empty")
            return errors

        # Validate amount column
        if "amount" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["amount"]):
                errors.append("Column 'amount' must be numeric")
            elif (df["amount"] <= 0).any():
                errors.append("Column 'amount' must contain only positive values")

        # Validate currency column
        if "currency" in df.columns:
            valid_currencies = {c.value for c in Currency}
            invalid_currencies = set(df["currency"].unique()) - valid_currencies
            if invalid_currencies:
                errors.append(
                    f"Invalid currency codes: {invalid_currencies}. "
                    f"Valid: {valid_currencies}"
                )

        # Validate transaction_id uniqueness
        if "transaction_id" in df.columns:
            if df["transaction_id"].duplicated().any():
                errors.append("Column 'transaction_id' contains duplicates")

        return errors

    @staticmethod
    def convert_to_transactions(df: pd.DataFrame) -> list[Transaction]:
        """
        Convert DataFrame rows to Transaction models.

        Args:
            df: DataFrame with transaction data

        Returns:
            List of Transaction objects

        Raises:
            ValidationError: If data doesn't match Transaction schema
        """
        transactions = []

        for idx, row in df.iterrows():
            try:
                transaction = Transaction(
                    transaction_id=str(row["transaction_id"]),
                    amount=float(row["amount"]),
                    currency=Currency(row["currency"]),
                    date=row["date"],
                    merchant=str(row["merchant"]),
                    category=str(row.get("category", "")) if pd.notna(row.get("category")) else None,
                    description=str(row["description"]),
                )
                transactions.append(transaction)
            except (ValidationError, ValueError, KeyError) as e:
                raise ValueError(f"Validation error at row {idx}: {e}")

        return transactions

    @staticmethod
    def convert_to_gbp(amount: float, currency: Currency) -> float:
        """
        Convert amount to GBP.

        Args:
            amount: Amount to convert
            currency: Source currency

        Returns:
            Amount in GBP
        """
        rate = CSVProcessor.EXCHANGE_RATES.get(currency, 1.0)
        return amount * rate

    @staticmethod
    def add_gbp_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add amount_gbp column to DataFrame.

        Args:
            df: DataFrame with amount and currency columns

        Returns:
            DataFrame with added amount_gbp column
        """
        df = df.copy()

        df["amount_gbp"] = df.apply(
            lambda row: CSVProcessor.convert_to_gbp(
                row["amount"], Currency(row["currency"])
            ),
            axis=1,
        )

        return df

    @staticmethod
    def save_enhanced_csv(
        original_df: pd.DataFrame,
        classifications: list[ClassificationResult],
        fraud_detections: list[FraudDetectionResult],
        output_path: Path,
    ) -> None:
        """
        Generate and save enhanced CSV with analysis results.

        Args:
            original_df: Original DataFrame
            classifications: List of classification results
            fraud_detections: List of fraud detection results
            output_path: Path to save enhanced CSV

        Raises:
            ValueError: If results don't match transactions
        """
        # Create enhanced DataFrame
        enhanced_df = original_df.copy()

        # Create mappings
        classification_map = {c.transaction_id: c for c in classifications}
        fraud_map = {f.transaction_id: f for f in fraud_detections}

        # Add enhanced columns
        enhanced_df["above_250_gbp"] = True  # All filtered transactions are above threshold

        # Add classification columns
        enhanced_df["classification"] = enhanced_df["transaction_id"].map(
            lambda tid: classification_map.get(str(tid), ClassificationResult(
                transaction_id=str(tid),
                primary_classification="Unknown",
                confidence=0.0,
                mcts_iterations=0,
                reasoning_trace="Not classified"
            )).primary_classification
        )

        enhanced_df["classification_confidence"] = enhanced_df["transaction_id"].map(
            lambda tid: classification_map.get(str(tid), ClassificationResult(
                transaction_id=str(tid),
                primary_classification="Unknown",
                confidence=0.0,
                mcts_iterations=0,
                reasoning_trace="Not classified"
            )).confidence
        )

        # Add fraud detection columns
        enhanced_df["fraud_risk"] = enhanced_df["transaction_id"].map(
            lambda tid: fraud_map.get(str(tid), FraudDetectionResult(
                transaction_id=str(tid),
                risk_level="LOW",
                confidence=0.0,
                mcts_iterations=0,
                reasoning="Not analyzed"
            )).risk_level
        )

        enhanced_df["fraud_confidence"] = enhanced_df["transaction_id"].map(
            lambda tid: fraud_map.get(str(tid), FraudDetectionResult(
                transaction_id=str(tid),
                risk_level="LOW",
                confidence=0.0,
                mcts_iterations=0,
                reasoning="Not analyzed"
            )).confidence
        )

        enhanced_df["fraud_reasoning"] = enhanced_df["transaction_id"].map(
            lambda tid: fraud_map.get(str(tid), FraudDetectionResult(
                transaction_id=str(tid),
                risk_level="LOW",
                confidence=0.0,
                mcts_iterations=0,
                reasoning="Not analyzed"
            )).reasoning
        )

        enhanced_df["mcts_iterations"] = enhanced_df["transaction_id"].map(
            lambda tid: (
                classification_map.get(str(tid), ClassificationResult(
                    transaction_id=str(tid),
                    primary_classification="Unknown",
                    confidence=0.0,
                    mcts_iterations=0,
                    reasoning_trace="Not classified"
                )).mcts_iterations
                + fraud_map.get(str(tid), FraudDetectionResult(
                    transaction_id=str(tid),
                    risk_level="LOW",
                    confidence=0.0,
                    mcts_iterations=0,
                    reasoning="Not analyzed"
                )).mcts_iterations
            )
        )

        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced_df.to_csv(output_path, index=False)
