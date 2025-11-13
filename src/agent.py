"""
Pydantic AI agent with transaction analysis tools.

Implements 4 tools:
1. filter_transactions_above_threshold
2. classify_transactions_mcts
3. detect_fraud_mcts
4. generate_enhanced_csv
"""

import os
import time
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
from src.session_context import SessionContext, create_session_context
from src.telemetry import get_telemetry

# Backward compatibility alias for tests
AgentDependencies = SessionContext

# Create the agent with conditional model initialization
# Use a test model if no API key is available (for testing)
def _create_agent() -> Agent:
    """
    Create the financial agent with SessionContext dependencies (REQ-SM-002).

    Uses Pydantic AI's deps_type to carry session state across tool calls.
    """
    # Check if we're in a test environment (no API keys set)
    has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if not has_openai_key and not has_anthropic_key:
        # Use test model for testing (doesn't require API key)
        from pydantic_ai.models.test import TestModel
        model = TestModel()
    else:
        # Use default OpenAI model
        model = "openai:o1-mini"

    return Agent(
        model,
        deps_type=SessionContext,  # REQ-SM-002: Use SessionContext for session memory
        system_prompt="""You are a financial transaction analysis expert.
        You use Monte Carlo Tree Search (MCTS) reasoning to analyze transactions,
        classify them accurately, and detect potential fraud.

        Always provide detailed reasoning for your conclusions.
        Consider multiple hypotheses before making final decisions.
        Use confidence scores to reflect uncertainty.

        When generating hypotheses or evaluations, always respond with valid JSON.
        """,
    )


financial_agent = _create_agent()


@financial_agent.tool
def filter_transactions_above_threshold(
    ctx: RunContext[SessionContext],
    threshold: float | None = None,
    currency: str | None = None,
) -> TransactionFilterResult:
    """
    Filter transactions above specified threshold in base currency (REQ-SM-005).

    This is Tool 1. It filters transactions and prepares them for analysis.
    Writes results to SessionContext.filtered_transactions for Tool 2/3 consumption.

    Args:
        ctx: Agent context with SessionContext dependencies
        threshold: Amount threshold (default from config)
        currency: Base currency (default from config)

    Returns:
        TransactionFilterResult with filtered statistics
    """
    tool_start_time = time.time()
    telemetry = get_telemetry()

    threshold = threshold if threshold is not None else ctx.deps.config.threshold_amount
    currency = currency if currency is not None else ctx.deps.config.base_currency.value

    df = ctx.deps.raw_csv_data

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

    # REQ-SM-005: Write to session memory (immutable update pattern REQ-SM-022)
    ctx.deps.filtered_transactions = filtered_df

    # Also store in legacy results dict for backward compatibility
    ctx.deps.results["filtered_df"] = filtered_df

    # Log execution (REQ-SM-003)
    tool_duration = time.time() - tool_start_time
    ctx.deps.log_tool_execution(
        tool_name="filter_transactions_above_threshold",
        status="success",
        duration_seconds=tool_duration,
        metadata={
            "filtered_count": len(filtered_df),
            "threshold": threshold,
            "session_id": ctx.deps.session_id,
        }
    )

    # Log to Logfire with session_id (REQ-SM-017)
    telemetry.log_info(
        "Tool 1: Filter completed",
        session_id=ctx.deps.session_id,
        filtered_count=len(filtered_df),
        threshold=threshold
    )

    return result


@financial_agent.tool
def classify_single_transaction_mcts(
    ctx: RunContext[SessionContext],
    transaction_data: dict[str, Any],
) -> ClassificationResult:
    """
    Classify a single transaction using MCTS reasoning (REQ-SM-006, REQ-SM-013).

    This is Tool 2 (helper for batch processing).
    - Reads from SessionContext.filtered_transactions (REQ-SM-006)
    - Checks MCTS cache before running search (REQ-SM-013)
    - Stores result in SessionContext.classification_results

    Args:
        ctx: Agent context with SessionContext dependencies
        transaction_data: Transaction dictionary

    Returns:
        ClassificationResult for the transaction
    """
    tool_start_time = time.time()
    telemetry = get_telemetry()
    transaction_id = str(transaction_data.get("transaction_id", ""))

    # REQ-SM-013: Check MCTS cache first
    cache_key = ctx.deps.get_cache_key(transaction_data)
    cached_result = ctx.deps.get_from_cache(cache_key, tool_type="classification")

    if cached_result is not None:
        # Cache hit!
        ctx.deps.mcts_cache_hits += 1
        tool_duration = time.time() - tool_start_time

        ctx.deps.log_tool_execution(
            tool_name="classify_single_transaction_mcts",
            status="cached",
            duration_seconds=tool_duration,
            transaction_id=transaction_id,
            metadata={"cache_key": cache_key, "session_id": ctx.deps.session_id}
        )

        telemetry.log_info(
            "Tool 2: Classification (cache hit)",
            session_id=ctx.deps.session_id,
            transaction_id=transaction_id,
            cache_key=cache_key
        )

        # Store in session memory
        ctx.deps.classification_results[transaction_id] = cached_result

        return cached_result

    # Cache miss - run MCTS
    ctx.deps.mcts_cache_misses += 1
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
        transaction_id=transaction_id,
        primary_classification=category,
        confidence=confidence,
        alternative_classifications=alternatives,
        mcts_iterations=ctx.deps.config.mcts.iterations,
        reasoning_trace=reasoning,
    )

    # REQ-SM-013: Add to cache
    ctx.deps.add_to_cache(cache_key, transaction_id, classification=result)

    # Store in session memory (REQ-SM-006)
    ctx.deps.classification_results[transaction_id] = result

    # Log execution
    tool_duration = time.time() - tool_start_time
    ctx.deps.log_tool_execution(
        tool_name="classify_single_transaction_mcts",
        status="success",
        duration_seconds=tool_duration,
        transaction_id=transaction_id,
        metadata={
            "cache_key": cache_key,
            "category": category,
            "session_id": ctx.deps.session_id
        }
    )

    telemetry.log_info(
        "Tool 2: Classification completed",
        session_id=ctx.deps.session_id,
        transaction_id=transaction_id,
        category=category,
        confidence=confidence
    )

    return result


@financial_agent.tool
def detect_fraud_single_transaction_mcts(
    ctx: RunContext[SessionContext],
    transaction_data: dict[str, Any],
) -> FraudDetectionResult:
    """
    Detect fraud for a single transaction using MCTS reasoning (REQ-SM-007, REQ-SM-013).

    This is Tool 3 (helper for batch processing).
    - Reads from SessionContext.filtered_transactions and classification_results (REQ-SM-007)
    - Checks MCTS cache before running search (REQ-SM-013)
    - Stores result in SessionContext.fraud_results

    Args:
        ctx: Agent context with SessionContext dependencies
        transaction_data: Transaction dictionary

    Returns:
        FraudDetectionResult for the transaction
    """
    tool_start_time = time.time()
    telemetry = get_telemetry()
    transaction_id = str(transaction_data.get("transaction_id", ""))

    # REQ-SM-013: Check MCTS cache first
    cache_key = ctx.deps.get_cache_key(transaction_data)
    cached_result = ctx.deps.get_from_cache(cache_key, tool_type="fraud")

    if cached_result is not None:
        # Cache hit!
        ctx.deps.mcts_cache_hits += 1
        tool_duration = time.time() - tool_start_time

        ctx.deps.log_tool_execution(
            tool_name="detect_fraud_single_transaction_mcts",
            status="cached",
            duration_seconds=tool_duration,
            transaction_id=transaction_id,
            metadata={"cache_key": cache_key, "session_id": ctx.deps.session_id}
        )

        telemetry.log_info(
            "Tool 3: Fraud detection (cache hit)",
            session_id=ctx.deps.session_id,
            transaction_id=transaction_id,
            cache_key=cache_key
        )

        # Store in session memory
        ctx.deps.fraud_results[transaction_id] = cached_result

        return cached_result

    # Cache miss - run MCTS
    ctx.deps.mcts_cache_misses += 1
    mcts = ctx.deps.mcts_engine

    # REQ-SM-007: Use classification to inform fraud detection
    # Get classification if available (Gambling category gets higher risk baseline)
    classification = ctx.deps.classification_results.get(transaction_id)
    category_hint = classification.primary_classification if classification else None

    # Prepare state for MCTS
    state = {
        "transaction": transaction_data,
        "context": {
            "threshold": ctx.deps.config.threshold_amount,
            "classification": category_hint,  # REQ-SM-007: Use classification result
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
        transaction_id=transaction_id,
        risk_level=risk_level,
        confidence=confidence,
        detected_indicators=indicators,
        reasoning=reasoning,
        mcts_iterations=ctx.deps.config.mcts.iterations,
        recommended_actions=actions,
    )

    # REQ-SM-013: Add to cache
    ctx.deps.add_to_cache(cache_key, transaction_id, fraud_result=result)

    # Store in session memory (REQ-SM-007)
    ctx.deps.fraud_results[transaction_id] = result

    # Log execution
    tool_duration = time.time() - tool_start_time
    ctx.deps.log_tool_execution(
        tool_name="detect_fraud_single_transaction_mcts",
        status="success",
        duration_seconds=tool_duration,
        transaction_id=transaction_id,
        metadata={
            "cache_key": cache_key,
            "risk_level": risk_level.value,
            "classification_used": category_hint,
            "session_id": ctx.deps.session_id
        }
    )

    telemetry.log_info(
        "Tool 3: Fraud detection completed",
        session_id=ctx.deps.session_id,
        transaction_id=transaction_id,
        risk_level=risk_level.value,
        confidence=confidence
    )

    return result


def run_analysis(
    df: pd.DataFrame,
    config: AgentConfig,
    output_path: Path,
    csv_file_name: str = "transactions.csv",
    progress_callback: Any = None,
) -> ProcessingReport:
    """
    Run complete transaction analysis pipeline with SessionContext (REQ-SM-001, REQ-SM-016).

    Orchestrates all 4 tools sequentially:
    1. Filter transactions
    2. Classify each transaction
    3. Detect fraud for each transaction
    4. Generate enhanced CSV

    Each invocation creates a new session with unique session_id (REQ-SM-009).

    Args:
        df: Input DataFrame
        config: Agent configuration
        output_path: Path for enhanced CSV output
        csv_file_name: Original CSV filename for session tracking
        progress_callback: Optional callback for progress updates

    Returns:
        ProcessingReport with summary statistics
    """
    telemetry = get_telemetry()
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

    # REQ-SM-009: Create SessionContext with unique session_id
    session_ctx = create_session_context(
        csv_data=df,
        csv_file_name=csv_file_name,
        config=config,
        mcts_engine=mcts_engine,
        llm_client=llm_client
    )

    # REQ-SM-016: Create top-level span with session_id
    with telemetry.span(
        "transaction_analysis_pipeline",
        session_id=session_ctx.session_id,  # REQ-SM-016: Add session_id to root span
        csv_file_name=csv_file_name,
        total_transactions=len(df),
        threshold=config.threshold_amount,
        currency=config.base_currency.value,
        llm_provider=config.llm.provider,
        llm_model=config.llm.model,
        mcts_iterations=config.mcts.iterations,
    ):

        # Step 1: Filter transactions (REQ-SM-005)
        with telemetry.span(
            "filter_transactions",
            session_id=session_ctx.session_id,  # REQ-SM-017: Propagate session_id
            threshold=config.threshold_amount
        ):
            if progress_callback:
                progress_callback("Filtering transactions...")

            # Filter transactions above threshold
            filter_transactions_above_threshold(
                RunContext(deps=session_ctx, retry=0, tool_name="filter_transactions_above_threshold")
            )

            # REQ-SM-006: Tool 2 will read from session_ctx.filtered_transactions
            filtered_df = session_ctx.filtered_transactions

            if filtered_df is None or len(filtered_df) == 0:
                raise ValueError(
                    f"No transactions above threshold {config.threshold_amount} {config.base_currency.value}"
                )

        # Step 2: Classify transactions (REQ-SM-006, REQ-SM-013)
        with telemetry.span(
            "classify_all_transactions",
            session_id=session_ctx.session_id,  # REQ-SM-017
            count=len(filtered_df)
        ):
            if progress_callback:
                progress_callback(f"Classifying {len(filtered_df)} transactions...")

            classifications = []
            for idx, row in filtered_df.iterrows():
                transaction_data = row.to_dict()

                # Create span for individual transaction classification
                with telemetry.span(
                    "classify_transaction",
                    session_id=session_ctx.session_id,  # REQ-SM-017
                    transaction_id=str(idx),
                    amount=float(transaction_data.get("amount", 0)),
                    currency=str(transaction_data.get("currency", "")),
                ):
                    result = classify_single_transaction_mcts(
                        RunContext(deps=session_ctx, retry=0, tool_name="classify_single_transaction_mcts"),
                        transaction_data=transaction_data,
                    )
                    classifications.append(result)

                    # Record classification result
                    telemetry.record_transaction_analysis(
                        transaction_id=str(idx),
                        amount=float(transaction_data.get("amount", 0)),
                        currency=str(transaction_data.get("currency", "")),
                        classification=result.primary_classification,
                        confidence=result.confidence,
                    )

                if progress_callback:
                    progress_callback(
                        f"Classified {len(classifications)}/{len(filtered_df)} transactions"
                    )

        # Step 3: Detect fraud (REQ-SM-007, REQ-SM-013)
        with telemetry.span(
            "detect_fraud_all_transactions",
            session_id=session_ctx.session_id,  # REQ-SM-017
            count=len(filtered_df)
        ):
            if progress_callback:
                progress_callback(f"Detecting fraud for {len(filtered_df)} transactions...")

            fraud_detections = []
            for idx, row in filtered_df.iterrows():
                transaction_data = row.to_dict()

                # Create span for individual fraud detection
                with telemetry.span(
                    "detect_fraud_transaction",
                    session_id=session_ctx.session_id,  # REQ-SM-017
                    transaction_id=str(idx),
                    amount=float(transaction_data.get("amount", 0)),
                    currency=str(transaction_data.get("currency", "")),
                ):
                    result = detect_fraud_single_transaction_mcts(
                        RunContext(deps=session_ctx, retry=0, tool_name="detect_fraud_single_transaction_mcts"),
                        transaction_data=transaction_data,
                    )
                    fraud_detections.append(result)

                    # Record fraud detection result
                    telemetry.record_transaction_analysis(
                        transaction_id=str(idx),
                        amount=float(transaction_data.get("amount", 0)),
                        currency=str(transaction_data.get("currency", "")),
                        fraud_risk=result.risk_level.value,
                        confidence=result.confidence,
                    )

                if progress_callback:
                    progress_callback(
                        f"Analyzed fraud for {len(fraud_detections)}/{len(filtered_df)} transactions"
                    )

        # Step 4: Generate enhanced CSV (REQ-SM-008, REQ-009)
        with telemetry.span(
            "generate_enhanced_csv",
            session_id=session_ctx.session_id  # REQ-SM-017
        ):
            if progress_callback:
                progress_callback("Generating enhanced CSV...")

            csv_result = CSVProcessor.save_enhanced_csv(
                filtered_df,
                classifications,
                fraud_detections,
                output_path,
            )

            # Store output path in session (REQ-SM-008)
            session_ctx.output_csv_path = csv_result.file_path

            # Log CSV completeness reward (REQ-009)
            completeness_reward = csv_result.calculate_completeness_reward()
            telemetry.log_info(
                "CSV generation completed",
                session_id=session_ctx.session_id,
                file_path=csv_result.file_path,
                row_count=csv_result.row_count,
                columns_included=len(csv_result.columns_included),
                completeness_reward=completeness_reward,
            )

        # REQ-SM-018: Finalize session metrics
        final_metrics = session_ctx.finalize_metrics()

        # Calculate risk counts
        high_risk_count = sum(
            1 for f in fraud_detections if f.risk_level in [FraudRiskLevel.HIGH, FraudRiskLevel.CRITICAL]
        )
        critical_risk_count = sum(
            1 for f in fraud_detections if f.risk_level == FraudRiskLevel.CRITICAL
        )

        processing_time = time.time() - start_time

        # REQ-SM-019: Log session_completed event
        with telemetry.span(
            "session_completed",
            session_id=session_ctx.session_id,
            final_status="success",
            output_csv_path=session_ctx.output_csv_path,
            transactions_processed=len(filtered_df),
            peak_memory_mb=final_metrics.peak_memory_mb,
            session_total_cost_usd=final_metrics.session_total_cost_usd,
            session_execution_time_seconds=final_metrics.session_execution_time_seconds,
            mcts_cache_hit_rate=final_metrics.session_mcts_cache_hit_rate,
            mcts_cache_hits=session_ctx.mcts_cache_hits,
            mcts_cache_misses=session_ctx.mcts_cache_misses,
        ):
            telemetry.log_info(
                "Session completed successfully",
                session_id=session_ctx.session_id,
                cache_hit_rate=final_metrics.session_mcts_cache_hit_rate,
                transactions_processed=len(filtered_df)
            )

        # REQ-SM-010: Memory cleanup
        with telemetry.span("session_cleanup", session_id=session_ctx.session_id):
            memory_freed_mb = session_ctx.cleanup_memory()
            telemetry.log_info(
                "Session memory cleaned up",
                session_id=session_ctx.session_id,
                memory_freed_mb=memory_freed_mb
            )

        # Create processing report
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

        # Record final pipeline metrics
        telemetry.record_pipeline_metrics(
            total_transactions=len(df),
            transactions_analyzed=len(filtered_df),
            high_risk_count=high_risk_count,
            critical_risk_count=critical_risk_count,
            processing_time_seconds=processing_time,
            model_used=config.llm.model,
        )

        return report
