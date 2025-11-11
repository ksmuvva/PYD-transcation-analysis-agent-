"""
Session Context Management for Transaction Analysis Agent.

Implements REQ-SM-001 through REQ-SM-023 for session memory and context management.

A session is defined as the complete lifecycle from CSV upload (Tool 1 input)
to final CSV generation (Tool 4 output). Each CSV file processed represents
one isolated session.
"""

import gc
import time
import uuid
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.models import Model

from src.config import AgentConfig
from src.mcts_engine import MCTSEngine
from src.models import ClassificationResult, FraudDetectionResult


class MCTSCacheEntry(BaseModel):
    """
    Cached MCTS result for reuse within a session (REQ-SM-013).

    Attributes:
        cache_key: Unique key for this cache entry (merchant_category + amount_bucket)
        classification: Cached classification result (if available)
        fraud_result: Cached fraud detection result (if available)
        created_at: Timestamp when cache entry was created
        reuse_count: Number of times this cache entry was reused
        original_transaction_id: ID of the transaction that generated this cache entry
    """

    cache_key: str = Field(..., description="Cache key: merchant_category + amount_bucket")
    classification: Optional[ClassificationResult] = Field(None, description="Cached classification")
    fraud_result: Optional[FraudDetectionResult] = Field(None, description="Cached fraud detection")
    created_at: float = Field(default_factory=time.time, description="Creation timestamp")
    reuse_count: int = Field(0, ge=0, description="Number of cache reuses")
    original_transaction_id: str = Field(..., description="Original transaction ID")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SessionMetrics(BaseModel):
    """
    Aggregated session-level metrics (REQ-SM-018).

    Attributes:
        session_total_cost_usd: Sum of all tool costs in USD
        session_execution_time_seconds: Duration from Tool 1 start to Tool 4 completion
        session_mcts_cache_hit_rate: Cache hit rate across all transactions
        session_tool_failure_count: Count of MCTSConvergenceError in session
        peak_memory_mb: Peak memory usage during session
        transactions_processed: Total number of transactions processed
    """

    session_total_cost_usd: float = Field(0.0, ge=0.0, description="Total LLM cost")
    session_execution_time_seconds: float = Field(0.0, ge=0.0, description="Total execution time")
    session_mcts_cache_hit_rate: float = Field(0.0, ge=0.0, le=1.0, description="Cache hit rate")
    session_tool_failure_count: int = Field(0, ge=0, description="Tool failure count")
    peak_memory_mb: float = Field(0.0, ge=0.0, description="Peak memory usage")
    transactions_processed: int = Field(0, ge=0, description="Transactions processed")


class ExecutionLogEntry(BaseModel):
    """
    Single execution log entry for audit trail (REQ-SM-003).

    Attributes:
        timestamp: When the tool was called
        tool_name: Name of the tool executed
        transaction_id: Transaction ID being processed (if applicable)
        status: Execution status (success, failed, cached)
        duration_seconds: How long the tool took to execute
        metadata: Additional metadata (e.g., MCTS iterations, cache hit)
    """

    timestamp: float = Field(default_factory=time.time, description="Execution timestamp")
    tool_name: str = Field(..., description="Tool name")
    transaction_id: Optional[str] = Field(None, description="Transaction ID")
    status: str = Field(..., description="Execution status")
    duration_seconds: float = Field(0.0, ge=0.0, description="Execution duration")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SessionContext(BaseModel):
    """
    Pydantic AI session context for transaction analysis (REQ-SM-003).

    This replaces the AgentDependencies dataclass with a proper Pydantic model
    that supports immutable updates, validation, and comprehensive session tracking.

    A session represents the complete pipeline from CSV upload to enhanced CSV output.
    Each CSV file processed gets a unique session_id (REQ-SM-001).

    Attributes:
        session_id: Unique UUID per CSV processing session (REQ-SM-009)
        csv_file_name: Original CSV filename
        session_start_time: Session start timestamp

        raw_csv_data: Original uploaded DataFrame
        filtered_transactions: Tool 1 output (transactions >= 250 GBP)
        classification_results: Tool 2 outputs (tx_id → ClassificationResult)
        fraud_results: Tool 3 outputs (tx_id → FraudDetectionResult)

        mcts_cache: Reusable MCTS results for similar transactions (REQ-SM-013)
        mcts_cache_hits: Cache hit counter
        mcts_cache_misses: Cache miss counter

        execution_log: Sequential record of tool calls and results
        session_metrics: Aggregated session-level metrics (REQ-SM-018)

        output_csv_path: Path to final enhanced CSV (Tool 4 output)

        config: Agent configuration (immutable per session)
        mcts_engine: MCTS reasoning engine
        llm_client: Pydantic AI Model client

        results: Legacy compatibility dict for intermediate results
    """

    # ========== Session Identity (REQ-SM-001, REQ-SM-009) ==========
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique session UUID per CSV upload"
    )
    csv_file_name: str = Field(..., description="Original CSV filename")
    session_start_time: float = Field(
        default_factory=time.time,
        description="Session start timestamp"
    )

    # ========== Pipeline Data (REQ-SM-003) ==========
    raw_csv_data: pd.DataFrame = Field(..., description="Original uploaded DataFrame")
    filtered_transactions: Optional[pd.DataFrame] = Field(
        None,
        description="Tool 1 output: transactions >= threshold"
    )
    classification_results: dict[str, ClassificationResult] = Field(
        default_factory=dict,
        description="Tool 2 outputs: tx_id → ClassificationResult"
    )
    fraud_results: dict[str, FraudDetectionResult] = Field(
        default_factory=dict,
        description="Tool 3 outputs: tx_id → FraudDetectionResult"
    )

    # ========== MCTS Optimization (REQ-SM-013, REQ-SM-014, REQ-SM-015) ==========
    mcts_cache: dict[str, MCTSCacheEntry] = Field(
        default_factory=dict,
        description="Cache of MCTS results for similar transactions"
    )
    mcts_cache_hits: int = Field(0, ge=0, description="Number of cache hits")
    mcts_cache_misses: int = Field(0, ge=0, description="Number of cache misses")
    mcts_cache_enabled: bool = Field(True, description="Enable/disable caching (for debug)")

    # ========== Execution Tracking (REQ-SM-003) ==========
    execution_log: list[ExecutionLogEntry] = Field(
        default_factory=list,
        description="Sequential record of tool calls"
    )
    session_metrics: SessionMetrics = Field(
        default_factory=SessionMetrics,
        description="Aggregated session-level metrics"
    )

    # ========== Final Output (REQ-SM-008) ==========
    output_csv_path: Optional[str] = Field(
        None,
        description="Path to final enhanced CSV (Tool 4 output)"
    )

    # ========== Agent Components (Immutable per session) ==========
    config: AgentConfig = Field(..., description="Agent configuration")
    mcts_engine: MCTSEngine = Field(..., description="MCTS reasoning engine")
    llm_client: Model = Field(..., description="Pydantic AI Model client")

    # ========== Legacy Compatibility ==========
    results: dict[str, Any] = Field(
        default_factory=dict,
        description="Legacy intermediate results storage"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow DataFrame, MCTSEngine, Model

    # ========== Cache Management (REQ-SM-013, REQ-SM-014) ==========

    def get_cache_key(self, transaction: dict[str, Any]) -> str:
        """
        Compute cache key for a transaction (REQ-SM-013).

        Cache key is based on:
        - merchant_category (exact match)
        - amount_gbp (bucketed to nearest £5 for fuzzy matching)

        Args:
            transaction: Transaction dict with keys like merchant_category, amount_gbp

        Returns:
            Cache key string (e.g., "Retail_250" or "Gambling_1000")
        """
        merchant_category = transaction.get('merchant_category', 'unknown')
        merchant_category = merchant_category.replace(' ', '_')

        # Get amount in GBP (convert if needed)
        amount_gbp = transaction.get('amount_gbp')
        if amount_gbp is None:
            # Fallback: check for 'amount' and 'currency'
            amount = transaction.get('amount', 0.0)
            # Simplified: assume GBP if not specified
            amount_gbp = amount

        # Bucket to nearest £5 for fuzzy matching (±5% tolerance)
        amount_bucket = round(amount_gbp / 5) * 5

        return f"{merchant_category}_{int(amount_bucket)}"

    def get_from_cache(self, cache_key: str, tool_type: str) -> Optional[ClassificationResult | FraudDetectionResult]:
        """
        Retrieve cached result if available (REQ-SM-013).

        Args:
            cache_key: Cache key for the transaction
            tool_type: "classification" or "fraud"

        Returns:
            Cached result if available, else None
        """
        if not self.mcts_cache_enabled:
            return None

        cache_entry = self.mcts_cache.get(cache_key)
        if cache_entry is None:
            return None

        # Update reuse count
        cache_entry.reuse_count += 1

        # Return the appropriate result based on tool type
        result = None
        if tool_type == "classification":
            result = cache_entry.classification
        elif tool_type == "fraud":
            result = cache_entry.fraud_result

        # Only increment cache hits if we actually have a result to return
        if result is not None:
            self.mcts_cache_hits += 1

        return result

    def add_to_cache(
        self,
        cache_key: str,
        transaction_id: str,
        classification: Optional[ClassificationResult] = None,
        fraud_result: Optional[FraudDetectionResult] = None
    ) -> None:
        """
        Add result to cache (REQ-SM-013).

        Args:
            cache_key: Cache key for the transaction
            transaction_id: Original transaction ID
            classification: Classification result to cache
            fraud_result: Fraud detection result to cache
        """
        if not self.mcts_cache_enabled:
            return

        # Check cache size limit (REQ-SM-014)
        if self.get_cache_size_mb() > 50.0:
            self.invalidate_cache()

        if cache_key not in self.mcts_cache:
            self.mcts_cache[cache_key] = MCTSCacheEntry(
                cache_key=cache_key,
                original_transaction_id=transaction_id,
                classification=classification,
                fraud_result=fraud_result
            )
        else:
            # Update existing entry
            entry = self.mcts_cache[cache_key]
            if classification is not None:
                entry.classification = classification
            if fraud_result is not None:
                entry.fraud_result = fraud_result

    def get_cache_size_mb(self) -> float:
        """
        Estimate cache size in MB (REQ-SM-014).

        Returns:
            Approximate cache size in megabytes
        """
        import sys

        total_size = 0
        for cache_entry in self.mcts_cache.values():
            # Rough estimate: each entry is ~1-5 KB depending on MCTS metadata
            total_size += sys.getsizeof(cache_entry.model_dump_json())

        return total_size / (1024 * 1024)  # Convert to MB

    def invalidate_cache(self) -> None:
        """
        Clear MCTS cache (REQ-SM-014, REQ-SM-015).

        Called when:
        - Cache size exceeds 50MB
        - Session starts (automatic via default_factory)
        - Debug mode disables cache
        """
        self.mcts_cache.clear()
        self.mcts_cache_hits = 0
        self.mcts_cache_misses = 0

    def get_cache_hit_rate(self) -> float:
        """
        Calculate cache hit rate (REQ-SM-018).

        Returns:
            Cache hit rate (0.0 to 1.0), or 0.0 if no cache accesses
        """
        total_accesses = self.mcts_cache_hits + self.mcts_cache_misses
        if total_accesses == 0:
            return 0.0
        return self.mcts_cache_hits / total_accesses

    # ========== Execution Logging (REQ-SM-003) ==========

    def log_tool_execution(
        self,
        tool_name: str,
        status: str,
        duration_seconds: float,
        transaction_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Log tool execution to audit trail (REQ-SM-003).

        Args:
            tool_name: Name of the tool executed
            status: Execution status (success, failed, cached)
            duration_seconds: How long the tool took
            transaction_id: Transaction ID being processed
            metadata: Additional metadata
        """
        entry = ExecutionLogEntry(
            tool_name=tool_name,
            transaction_id=transaction_id,
            status=status,
            duration_seconds=duration_seconds,
            metadata=metadata or {}
        )
        self.execution_log.append(entry)

    # ========== Memory Cleanup (REQ-SM-010) ==========

    def cleanup_memory(self) -> float:
        """
        Cleanup session memory after Tool 4 completion (REQ-SM-010).

        Deletes:
        - Intermediate DataFrames (filtered_transactions, raw_csv_data)
        - MCTS cache
        - Execution log (optional, can be archived)

        Returns:
            Estimated memory freed in MB
        """
        import sys

        memory_before = 0

        # Estimate memory usage before cleanup
        if self.raw_csv_data is not None:
            memory_before += self.raw_csv_data.memory_usage(deep=True).sum()
        if self.filtered_transactions is not None:
            memory_before += self.filtered_transactions.memory_usage(deep=True).sum()
        memory_before += sys.getsizeof(str(self.mcts_cache))

        # Delete intermediate DataFrames
        self.raw_csv_data = None  # type: ignore
        self.filtered_transactions = None

        # Clear MCTS cache
        self.mcts_cache.clear()

        # Optionally clear execution log (keep for now for debugging)
        # self.execution_log.clear()

        # Force garbage collection
        gc.collect()

        memory_freed_mb = memory_before / (1024 * 1024)
        return memory_freed_mb

    # ========== Session Metrics (REQ-SM-018) ==========

    def update_metrics(
        self,
        cost_usd: float = 0.0,
        execution_time: float = 0.0,
        failure_count: int = 0
    ) -> None:
        """
        Update session-level metrics (REQ-SM-018).

        Args:
            cost_usd: Cost to add to total
            execution_time: Time to add to total
            failure_count: Failures to add to count
        """
        self.session_metrics.session_total_cost_usd += cost_usd
        self.session_metrics.session_execution_time_seconds += execution_time
        self.session_metrics.session_tool_failure_count += failure_count
        self.session_metrics.session_mcts_cache_hit_rate = self.get_cache_hit_rate()

    def finalize_metrics(self) -> SessionMetrics:
        """
        Finalize session metrics at completion (REQ-SM-019).

        Returns:
            Final SessionMetrics object
        """
        self.session_metrics.session_execution_time_seconds = time.time() - self.session_start_time
        self.session_metrics.session_mcts_cache_hit_rate = self.get_cache_hit_rate()
        self.session_metrics.transactions_processed = len(self.classification_results)

        # Estimate peak memory (rough approximation)
        if self.raw_csv_data is not None:
            self.session_metrics.peak_memory_mb = (
                self.raw_csv_data.memory_usage(deep=True).sum() / (1024 * 1024)
            )

        return self.session_metrics


# ========== Utility Functions ==========

def create_session_context(
    csv_data: pd.DataFrame,
    csv_file_name: str,
    config: AgentConfig,
    mcts_engine: MCTSEngine,
    llm_client: Model
) -> SessionContext:
    """
    Factory function to create a new SessionContext (REQ-SM-009).

    Args:
        csv_data: Original CSV DataFrame
        csv_file_name: Original CSV filename
        config: Agent configuration
        mcts_engine: MCTS engine instance
        llm_client: Pydantic AI Model client

    Returns:
        New SessionContext with unique session_id
    """
    return SessionContext(
        csv_file_name=csv_file_name,
        raw_csv_data=csv_data,
        config=config,
        mcts_engine=mcts_engine,
        llm_client=llm_client
    )
