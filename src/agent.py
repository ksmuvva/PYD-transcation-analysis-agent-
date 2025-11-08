"""
Pydantic AI agent with transaction analysis tools.

Implements 4 tools:
1. filter_transactions_above_threshold
2. classify_transactions_mcts
3. detect_fraud_mcts
4. generate_enhanced_csv
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from src.config import AgentConfig, ConfigManager
from src.csv_processor import CSVProcessor
from src.mcts_engine import MCTSEngine
from src.models import (
    ClassificationResult,
    FraudDetectionResult,
    FraudRiskLevel,
    ProcessingReport,
    TransactionFilterResult,
)


@dataclass
class AgentDependencies:
    """
    Shared state across all agent tools.

    Attributes:
        df: Original DataFrame
        config: Agent configuration
        mcts_engine: MCTS reasoning engine
        llm_client: LLM model client
        results: Storage for intermediate results
        start_time: Processing start time
    """

    df: pd.DataFrame
    config: AgentConfig
    mcts_engine: MCTSEngine
    llm_client: Model
    results: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)


# Create the agent
financial_agent = Agent(
    "openai:o1-mini",  # Default model, will be overridden
    deps_type=AgentDependencies,
    system_prompt="""You are a financial transaction analysis expert.
    You use Monte Carlo Tree Search (MCTS) reasoning to analyze transactions,
    classify them accurately, and detect potential fraud.

    Always provide detailed reasoning for your conclusions.
    Consider multiple hypotheses before making final decisions.
    Use confidence scores to reflect uncertainty.

    When generating hypotheses or evaluations, always respond with valid JSON.
    """,
)


@financial_agent.tool
def filter_transactions_above_threshold(
    ctx: RunContext[AgentDependencies],
    threshold: float | None = None,
    currency: str | None = None,
) -> TransactionFilterResult:
    """
    Filter transactions above specified threshold in base currency.

    This is Tool 1. It filters transactions and prepares them for analysis.

    Args:
        ctx: Agent context with dependencies
        threshold: Amount threshold (default from config)
        currency: Base currency (default from config)

    Returns:
        TransactionFilterResult with filtered statistics
    """
    threshold = threshold or ctx.deps.config.threshold_amount
    currency = currency or ctx.deps.config.base_currency.value

    df = ctx.deps.df

    # Add GBP conversion column
    df = CSVProcessor.add_gbp_column(df)

    # Filter transactions above threshold
    filtered_df = df[df["amount_gbp"] >= threshold].copy()

    # Calculate statistics
    result = TransactionFilterResult(
        filtered_count=len(filtered_df),
        total_amount=float(filtered_df["amount_gbp"].sum()),
        currency=ctx.deps.config.base_currency,
        average_amount=float(filtered_df["amount_gbp"].mean()) if len(filtered_df) > 0 else 0.0,
    )

    # Store filtered DataFrame for next tools
    ctx.deps.results["filtered_df"] = filtered_df

    return result


@financial_agent.tool
def classify_single_transaction_mcts(
    ctx: RunContext[AgentDependencies],
    transaction_data: dict[str, Any],
) -> ClassificationResult:
    """
    Classify a single transaction using MCTS reasoning.

    This is Tool 2 (helper for batch processing).

    Args:
        ctx: Agent context with dependencies
        transaction_data: Transaction dictionary

    Returns:
        ClassificationResult for the transaction
    """
    mcts = ctx.deps.mcts_engine

    # Prepare state for MCTS
    state = {
        "transaction": transaction_data,
        "context": {
            "threshold": ctx.deps.config.threshold_amount,
            "currency": ctx.deps.config.base_currency.value,
        },
    }

    # Run MCTS search
    mcts_result = mcts.search(state, objective="classify")

    # Extract results
    hypothesis = mcts_result.get("hypothesis", {})
    category = hypothesis.get("category", "Uncategorized")
    confidence = mcts_result.get("confidence", 0.5)
    reasoning = mcts_result.get("reasoning", "Classification completed")

    # Build alternative classifications if available
    alternatives = []
    # (In a more sophisticated implementation, we'd track multiple hypotheses)

    result = ClassificationResult(
        transaction_id=str(transaction_data.get("transaction_id", "")),
        primary_classification=category,
        confidence=confidence,
        alternative_classifications=alternatives,
        mcts_iterations=ctx.deps.config.mcts.iterations,
        reasoning_trace=reasoning,
    )

    return result


@financial_agent.tool
def detect_fraud_single_transaction_mcts(
    ctx: RunContext[AgentDependencies],
    transaction_data: dict[str, Any],
) -> FraudDetectionResult:
    """
    Detect fraud for a single transaction using MCTS reasoning.

    This is Tool 3 (helper for batch processing).

    Args:
        ctx: Agent context with dependencies
        transaction_data: Transaction dictionary

    Returns:
        FraudDetectionResult for the transaction
    """
    mcts = ctx.deps.mcts_engine

    # Prepare state for MCTS
    state = {
        "transaction": transaction_data,
        "context": {
            "threshold": ctx.deps.config.threshold_amount,
        },
    }

    # Run MCTS search
    mcts_result = mcts.search(state, objective="detect_fraud")

    # Extract results
    hypothesis = mcts_result.get("hypothesis", {})
    risk_level_str = hypothesis.get("risk_level", "LOW")

    # Map to FraudRiskLevel enum
    try:
        risk_level = FraudRiskLevel(risk_level_str)
    except ValueError:
        risk_level = FraudRiskLevel.LOW

    confidence = mcts_result.get("confidence", 0.5)
    reasoning = mcts_result.get("reasoning", "Fraud detection completed")
    indicators = hypothesis.get("indicators", [])
    actions = mcts_result.get("actions", [])

    result = FraudDetectionResult(
        transaction_id=str(transaction_data.get("transaction_id", "")),
        risk_level=risk_level,
        confidence=confidence,
        detected_indicators=indicators,
        reasoning=reasoning,
        mcts_iterations=ctx.deps.config.mcts.iterations,
        recommended_actions=actions,
    )

    return result


def run_analysis(
    df: pd.DataFrame,
    config: AgentConfig,
    output_path: Path,
    progress_callback: Any = None,
) -> ProcessingReport:
    """
    Run complete transaction analysis pipeline.

    Orchestrates all 4 tools sequentially:
    1. Filter transactions
    2. Classify each transaction
    3. Detect fraud for each transaction
    4. Generate enhanced CSV

    Args:
        df: Input DataFrame
        config: Agent configuration
        output_path: Path for enhanced CSV output
        progress_callback: Optional callback for progress updates

    Returns:
        ProcessingReport with summary statistics
    """
    start_time = time.time()

    # Create LLM client
    llm_client = ConfigManager.create_llm_client(config.llm)

    # Create MCTS engine with LLM function
    def llm_function(prompt: str) -> str:
        """Wrapper to call LLM synchronously."""
        # For Pydantic AI, we'll use the model's run method
        # This is a simplified synchronous wrapper
        try:
            # Create a simple agent just for LLM calls
            simple_agent = Agent(llm_client)
            result = simple_agent.run_sync(prompt)
            return result.data
        except Exception as e:
            return f"Error: {str(e)}"

    mcts_engine = MCTSEngine(config.mcts, llm_function)

    # Create dependencies
    deps = AgentDependencies(
        df=df,
        config=config,
        mcts_engine=mcts_engine,
        llm_client=llm_client,
        start_time=start_time,
    )

    # Step 1: Filter transactions
    if progress_callback:
        progress_callback("Filtering transactions...")

    filter_result = filter_transactions_above_threshold(
        RunContext(deps=deps, retry=0, tool_name="filter_transactions_above_threshold")
    )

    filtered_df = deps.results["filtered_df"]

    if len(filtered_df) == 0:
        raise ValueError(
            f"No transactions above threshold {config.threshold_amount} {config.base_currency.value}"
        )

    # Step 2: Classify transactions
    if progress_callback:
        progress_callback(f"Classifying {len(filtered_df)} transactions...")

    classifications = []
    for idx, row in filtered_df.iterrows():
        transaction_data = row.to_dict()

        result = classify_single_transaction_mcts(
            RunContext(deps=deps, retry=0, tool_name="classify_single_transaction_mcts"),
            transaction_data=transaction_data,
        )
        classifications.append(result)

        if progress_callback:
            progress_callback(
                f"Classified {len(classifications)}/{len(filtered_df)} transactions"
            )

    # Step 3: Detect fraud
    if progress_callback:
        progress_callback(f"Detecting fraud for {len(filtered_df)} transactions...")

    fraud_detections = []
    for idx, row in filtered_df.iterrows():
        transaction_data = row.to_dict()

        result = detect_fraud_single_transaction_mcts(
            RunContext(deps=deps, retry=0, tool_name="detect_fraud_single_transaction_mcts"),
            transaction_data=transaction_data,
        )
        fraud_detections.append(result)

        if progress_callback:
            progress_callback(
                f"Analyzed fraud for {len(fraud_detections)}/{len(filtered_df)} transactions"
            )

    # Step 4: Generate enhanced CSV
    if progress_callback:
        progress_callback("Generating enhanced CSV...")

    CSVProcessor.save_enhanced_csv(
        filtered_df,
        classifications,
        fraud_detections,
        output_path,
    )

    # Create processing report
    high_risk_count = sum(
        1 for f in fraud_detections if f.risk_level in [FraudRiskLevel.HIGH, FraudRiskLevel.CRITICAL]
    )
    critical_risk_count = sum(
        1 for f in fraud_detections if f.risk_level == FraudRiskLevel.CRITICAL
    )

    processing_time = time.time() - start_time

    report = ProcessingReport(
        total_transactions_analyzed=len(filtered_df),
        transactions_above_threshold=len(filtered_df),
        high_risk_transactions=high_risk_count,
        critical_risk_transactions=critical_risk_count,
        processing_time_seconds=processing_time,
        llm_provider=config.llm.provider,
        model_used=config.llm.model,
        mcts_iterations_total=len(filtered_df) * config.mcts.iterations * 2,  # classify + fraud
    )

    return report
