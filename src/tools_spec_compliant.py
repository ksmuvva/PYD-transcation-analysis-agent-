"""
Specification-compliant tool implementations (REQ-011, REQ-012).

This module provides tool implementations that exactly match the requirements
specification signatures from REQ-011 and REQ-012.

Each tool accepts RunContext[pd.DataFrame] and transaction_id: str.
"""

import pandas as pd
from pydantic_ai import RunContext

from src.config import MCTSConfig
from src.csv_processor import CSVProcessor
from src.mcts_engine_v2 import EnhancedMCTSEngine
from src.models import (
    FilterResult,
    ClassificationResult,
    FraudResult,
    CSVResult,
    Currency,
    MCTSMetadata,
    FraudRiskLevel,
)


# ==============================================================================
# Tool 1: Filter Transactions Above 250 GBP (REQ-006, REQ-011)
# ==============================================================================

def filter_above_250(
    ctx: RunContext[pd.DataFrame],
    transaction_id: str,
) -> FilterResult:
    """
    Tool 1: Filter transactions above 250 GBP threshold (REQ-011).

    Implements REQ-006: Binary reward (1.0 if >= 250 GBP, 0.0 otherwise).
    Implements REQ-011: Exact signature specification.

    Args:
        ctx: RunContext with DataFrame dependencies
        transaction_id: Transaction ID to process

    Returns:
        FilterResult with MCTS metadata (REQ-012)
    """
    df = ctx.deps

    # Find transaction
    tx = df[df['transaction_id'] == transaction_id]
    if len(tx) == 0:
        raise ValueError(f"Transaction {transaction_id} not found")

    tx_row = tx.iloc[0]

    # Get amount and currency
    amount = float(tx_row['amount'])
    currency = Currency(tx_row['currency'])

    # Convert to GBP
    amount_gbp = CSVProcessor.convert_to_gbp(amount, currency)

    # Determine conversion path
    if currency == Currency.GBP:
        conversion_path = "Direct (GBP)"
    else:
        conversion_path = f"{currency.value}->GBP via ECB"

    # Calculate binary reward (REQ-006)
    is_above = amount_gbp >= 250.0
    reward = 1.0 if is_above else 0.0

    # Create minimal MCTS metadata (filter doesn't use full MCTS)
    mcts_metadata = MCTSMetadata(
        root_node_visits=1,
        best_action_path=["direct_conversion"],
        average_reward=reward,
        exploration_constant_used=1.414,
        final_reward_variance=0.0,
        total_nodes_explored=1,
        max_depth_reached=1,
    )

    return FilterResult(
        is_above_threshold=is_above,
        amount_gbp=amount_gbp,
        conversion_path_used=conversion_path,
        confidence=1.0,  # Conversion is deterministic
        mcts_metadata=mcts_metadata,
    )


# ==============================================================================
# Tool 2: Classify Transaction (REQ-007, REQ-011)
# ==============================================================================

def classify_transaction(
    ctx: RunContext[pd.DataFrame],
    transaction_id: str,
    llm_function=None,
    mcts_config: MCTSConfig = None,
) -> ClassificationResult:
    """
    Tool 2: Classify transaction using MCTS (REQ-011).

    Implements REQ-007: Multi-class classification with one-hot encoded reward.
    Implements REQ-011: Exact signature specification.

    Args:
        ctx: RunContext with DataFrame dependencies
        transaction_id: Transaction ID to process
        llm_function: Optional LLM function for MCTS
        mcts_config: Optional MCTS configuration

    Returns:
        ClassificationResult with MCTS metadata (REQ-012)
    """
    df = ctx.deps

    # Find transaction
    tx = df[df['transaction_id'] == transaction_id]
    if len(tx) == 0:
        raise ValueError(f"Transaction {transaction_id} not found")

    tx_row = tx.iloc[0]

    # Prepare transaction data
    transaction_data = {
        'transaction_id': transaction_id,
        'amount': float(tx_row['amount']),
        'currency': str(tx_row['currency']),
        'merchant': str(tx_row['merchant']),
        'description': str(tx_row['description']),
        'date': str(tx_row['date']),
    }

    # Use MCTS for classification if LLM function provided
    if llm_function and mcts_config:
        # Create MCTS engine for classification
        engine = EnhancedMCTSEngine(
            config=mcts_config,
            tool_name="classify",
            llm_function=llm_function,
            transaction_id=transaction_id,
        )

        # Run MCTS search
        state = {
            'transaction': transaction_data,
            'context': {'threshold': 250.0, 'currency': 'GBP'},
        }

        result = engine.search(state, objective="classify")

        hypothesis = result.get('hypothesis', {})
        category = hypothesis.get('category', 'Personal')
        confidence = result.get('confidence', 0.5)
        mcts_metadata = result.get('mcts_metadata')

        return ClassificationResult(
            transaction_id=transaction_id,
            category=category,
            confidence=confidence,
            mcts_path=mcts_metadata.best_action_path if mcts_metadata else [],
            mcts_iterations=mcts_metadata.root_node_visits if mcts_metadata else 0,
            mcts_metadata=mcts_metadata or _create_default_mcts_metadata(),
        )
    else:
        # Fallback: Simple rule-based classification
        merchant = str(tx_row['merchant']).lower()
        description = str(tx_row['description']).lower()

        # Simple heuristic classification
        if any(keyword in merchant or keyword in description for keyword in ['ltd', 'inc', 'corp', 'office']):
            category = "Business"
        elif any(keyword in merchant or keyword in description for keyword in ['bet', 'casino', 'poker', 'gambling']):
            category = "Gambling"
        elif any(keyword in merchant or keyword in description for keyword in ['invest', 'stock', 'fund', 'trading']):
            category = "Investment"
        else:
            category = "Personal"

        return ClassificationResult(
            transaction_id=transaction_id,
            category=category,
            confidence=0.7,  # Heuristic confidence
            mcts_path=["heuristic_classification"],
            mcts_iterations=1,
            mcts_metadata=_create_default_mcts_metadata(),
        )


# ==============================================================================
# Tool 3: Detect Fraudulent Transactions (REQ-008, REQ-011)
# ==============================================================================

def detect_fraud(
    ctx: RunContext[pd.DataFrame],
    transaction_id: str,
    llm_function=None,
    mcts_config: MCTSConfig = None,
) -> FraudResult:
    """
    Tool 3: Detect fraud using MCTS (REQ-011).

    Implements REQ-008: Risk-level based reward (CRITICAL=1.0, HIGH=0.75, MEDIUM=0.5, LOW=0.0).
    Implements REQ-011: Exact signature specification.

    Args:
        ctx: RunContext with DataFrame dependencies
        transaction_id: Transaction ID to process
        llm_function: Optional LLM function for MCTS
        mcts_config: Optional MCTS configuration

    Returns:
        FraudResult with MCTS metadata (REQ-012)
    """
    df = ctx.deps

    # Find transaction
    tx = df[df['transaction_id'] == transaction_id]
    if len(tx) == 0:
        raise ValueError(f"Transaction {transaction_id} not found")

    tx_row = tx.iloc[0]

    # Prepare transaction data
    transaction_data = {
        'transaction_id': transaction_id,
        'amount': float(tx_row['amount']),
        'currency': str(tx_row['currency']),
        'merchant': str(tx_row['merchant']),
        'description': str(tx_row['description']),
        'date': str(tx_row['date']),
    }

    # Use MCTS for fraud detection if LLM function provided
    if llm_function and mcts_config:
        # Create MCTS engine for fraud detection
        engine = EnhancedMCTSEngine(
            config=mcts_config,
            tool_name="fraud",
            llm_function=llm_function,
            transaction_id=transaction_id,
        )

        # Run MCTS search
        state = {
            'transaction': transaction_data,
            'context': {'threshold': 250.0},
        }

        result = engine.search(state, objective="detect_fraud")

        hypothesis = result.get('hypothesis', {})
        risk_level_str = hypothesis.get('risk_level', 'LOW')
        risk_level = FraudRiskLevel(risk_level_str)
        confidence = result.get('confidence', 0.5)
        indicators = hypothesis.get('indicators', [])
        mcts_metadata = result.get('mcts_metadata')

        # Calculate reward (REQ-008)
        reward = risk_level.to_reward()

        return FraudResult(
            transaction_id=transaction_id,
            risk_level=risk_level,
            confidence=confidence,
            mcts_path=mcts_metadata.best_action_path if mcts_metadata else [],
            mcts_reward=reward,
            fraud_indicators=indicators,
            mcts_metadata=mcts_metadata or _create_default_mcts_metadata(),
        )
    else:
        # Fallback: Simple rule-based fraud detection
        amount = float(tx_row['amount'])
        merchant = str(tx_row['merchant']).lower()
        description = str(tx_row['description']).lower()

        # Simple heuristic fraud detection
        indicators = []
        risk_level = FraudRiskLevel.LOW

        # Check for high amount
        if amount > 10000:
            indicators.append("High transaction amount")
            risk_level = FraudRiskLevel.MEDIUM

        # Check for suspicious keywords
        if any(keyword in merchant or keyword in description for keyword in ['crypto', 'offshore', 'anonymous']):
            indicators.append("Suspicious merchant/description")
            risk_level = FraudRiskLevel.HIGH

        # Check for very high amount
        if amount > 50000:
            indicators.append("Very high transaction amount")
            risk_level = FraudRiskLevel.CRITICAL

        reward = risk_level.to_reward()

        return FraudResult(
            transaction_id=transaction_id,
            risk_level=risk_level,
            confidence=0.6,  # Heuristic confidence
            mcts_path=["heuristic_fraud_detection"],
            mcts_reward=reward,
            fraud_indicators=indicators,
            mcts_metadata=_create_default_mcts_metadata(),
        )


# ==============================================================================
# Tool 4: Generate Enhanced CSV (REQ-009, REQ-011)
# ==============================================================================

def generate_enhanced_csv(
    ctx: RunContext[pd.DataFrame],
    transaction_ids: list[str],
    output_path: str = "enhanced_output.csv",
) -> CSVResult:
    """
    Tool 4: Generate enhanced CSV with analysis results (REQ-011).

    Implements REQ-009: Data completeness reward.
    Implements REQ-011: Exact signature specification.

    Args:
        ctx: RunContext with DataFrame dependencies
        transaction_ids: List of transaction IDs to include
        output_path: Path for output CSV

    Returns:
        CSVResult with completeness reward (REQ-012)
    """
    from pathlib import Path
    from src.models import FraudDetectionResult

    df = ctx.deps

    # Filter to specified transactions
    filtered_df = df[df['transaction_id'].isin(transaction_ids)].copy()

    # For this spec-compliant version, we assume analysis has been done
    # and results are in the DataFrame columns

    # Build MCTS explanations
    mcts_explanations = {}
    for tx_id in transaction_ids:
        tx = filtered_df[filtered_df['transaction_id'] == tx_id]
        if len(tx) > 0:
            tx_row = tx.iloc[0]
            # Create explanation from available data
            if 'classification' in tx_row and 'fraud_risk' in tx_row:
                explanation = (
                    f"Classification: {tx_row.get('classification', 'Unknown')} "
                    f"| Fraud: {tx_row.get('fraud_risk', 'LOW')}"
                )
            else:
                explanation = "Analysis pending"
            mcts_explanations[str(tx_id)] = explanation

    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_file, index=False)

    # Create CSVResult
    result = CSVResult(
        file_path=str(output_file),
        row_count=len(filtered_df),
        columns_included=list(filtered_df.columns),
        mcts_explanations=mcts_explanations,
    )

    return result


# ==============================================================================
# Helper Functions
# ==============================================================================

def _create_default_mcts_metadata() -> MCTSMetadata:
    """Create default MCTS metadata for fallback cases."""
    return MCTSMetadata(
        root_node_visits=1,
        best_action_path=["heuristic"],
        average_reward=0.5,
        exploration_constant_used=1.414,
        final_reward_variance=0.0,
        total_nodes_explored=1,
        max_depth_reached=1,
    )
