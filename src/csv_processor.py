"""
CSV processing utilities for transaction data.

Handles CSV loading, validation, and enhanced CSV generation.
"""

from pathlib import Path

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
    def load_csv(file_path: Path | str) -> pd.DataFrame:
        """
        Load CSV file into DataFrame.

        Args:
            file_path: Path to CSV file (Path object or string)

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV is invalid
        """
        # Convert string to Path if needed
        if isinstance(file_path, str):
            file_path = Path(file_path)

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
                # Validate transaction_id is not None/NaN
                if pd.isna(row["transaction_id"]) or row["transaction_id"] is None:
                    raise ValueError("transaction_id cannot be None or NaN")

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

        # Handle empty DataFrame
        if len(df) == 0:
            df["amount_gbp"] = pd.Series([], dtype=float)
            return df

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
    ) -> "CSVResult":
        """
        Generate and save enhanced CSV with analysis results (REQ-009).

        Implements REQ-009: Data completeness reward based on required columns.
        Required columns: classification, fraud_risk, confidence, mcts_explanation
        Reward: +0.2 per column, 1.0 only if all 4 present and correctly formatted.

        Args:
            original_df: Original DataFrame
            classifications: List of classification results
            fraud_detections: List of fraud detection results
            output_path: Path to save enhanced CSV

        Returns:
            CSVResult with completeness reward calculation

        Raises:
            ValueError: If results don't match transactions
        """
        from src.models import CSVResult

        # Create enhanced DataFrame
        enhanced_df = original_df.copy()

        # Create mappings
        classification_map = {c.transaction_id: c for c in classifications}
        fraud_map = {f.transaction_id: f for f in fraud_detections}

        # Add enhanced columns
        enhanced_df["above_250_gbp"] = True  # All filtered transactions are above threshold

        # REQ-009: Add classification column (required column 1)
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

        # REQ-009: Add fraud_risk column (required column 2)
        enhanced_df["fraud_risk"] = enhanced_df["transaction_id"].map(
            lambda tid: fraud_map.get(str(tid), FraudDetectionResult(
                transaction_id=str(tid),
                risk_level="LOW",
                confidence=0.0,
                mcts_iterations=0,
                reasoning="Not analyzed"
            )).risk_level
        )

        # REQ-009: Add confidence column (required column 3) - combined confidence score
        enhanced_df["confidence"] = enhanced_df["transaction_id"].map(
            lambda tid: (
                classification_map.get(str(tid), ClassificationResult(
                    transaction_id=str(tid),
                    primary_classification="Unknown",
                    confidence=0.0,
                    mcts_iterations=0,
                    reasoning_trace="Not classified"
                )).confidence * 0.5 +
                fraud_map.get(str(tid), FraudDetectionResult(
                    transaction_id=str(tid),
                    risk_level="LOW",
                    confidence=0.0,
                    mcts_iterations=0,
                    reasoning="Not analyzed"
                )).confidence * 0.5
            )
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

        # REQ-009: Add mcts_explanation column (required column 4)
        mcts_explanations = {}
        enhanced_df["mcts_explanation"] = enhanced_df["transaction_id"].map(
            lambda tid: _generate_mcts_explanation(
                tid, classification_map.get(str(tid)), fraud_map.get(str(tid)), mcts_explanations
            )
        )

        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced_df.to_csv(output_path, index=False)

        # REQ-009: Create CSVResult with completeness reward
        result = CSVResult(
            file_path=str(output_path),
            row_count=len(enhanced_df),
            columns_included=list(enhanced_df.columns),
            mcts_explanations=mcts_explanations,
        )

        # Calculate and log completeness reward
        completeness_reward = result.calculate_completeness_reward()
        print(f"✅ CSV completeness reward: {completeness_reward:.2f} (REQ-009)")

        return result


def _generate_mcts_explanation(
    transaction_id: str,
    classification: ClassificationResult | None,
    fraud: FraudDetectionResult | None,
    mcts_explanations_dict: dict[str, str],
) -> str:
    """
    Generate MCTS explanation for a transaction (REQ-009).

    Args:
        transaction_id: Transaction ID
        classification: Classification result
        fraud: Fraud detection result
        mcts_explanations_dict: Dictionary to store explanations

    Returns:
        MCTS explanation string
    """
    if classification is None or fraud is None:
        explanation = "No MCTS analysis performed"
    else:
        # Build comprehensive MCTS explanation
        parts = []

        # Classification explanation
        if hasattr(classification, 'mcts_metadata'):
            meta = classification.mcts_metadata
            parts.append(
                f"Classification: {classification.category} "
                f"(confidence: {classification.confidence:.2f}, "
                f"visits: {meta.root_node_visits}, "
                f"path: {' → '.join(meta.best_action_path[:3]) if meta.best_action_path else 'N/A'})"
            )
        else:
            parts.append(
                f"Classification: {classification.primary_classification} "
                f"(confidence: {classification.confidence:.2f}, "
                f"iterations: {classification.mcts_iterations})"
            )

        # Fraud detection explanation
        if hasattr(fraud, 'mcts_metadata'):
            meta = fraud.mcts_metadata
            parts.append(
                f"Fraud: {fraud.risk_level} "
                f"(confidence: {fraud.confidence:.2f}, "
                f"visits: {meta.root_node_visits}, "
                f"path: {' → '.join(meta.best_action_path[:3]) if meta.best_action_path else 'N/A'})"
            )
        else:
            parts.append(
                f"Fraud: {fraud.risk_level} "
                f"(confidence: {fraud.confidence:.2f}, "
                f"iterations: {fraud.mcts_iterations})"
            )

        explanation = " | ".join(parts)

    # Store in dictionary for CSVResult
    mcts_explanations_dict[str(transaction_id)] = explanation

    return explanation
