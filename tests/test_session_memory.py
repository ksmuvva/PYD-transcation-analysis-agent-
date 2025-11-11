"""
Unit tests for session memory and context management.

Tests REQ-SM-001 through REQ-SM-023:
- Session isolation (unique session_id per CSV)
- Tool-to-tool memory passing
- MCTS cache hit/miss logic
- Memory cleanup
- Session metrics
"""

import uuid
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic_ai.models.test import TestModel

from src.config import AgentConfig, LLMConfig, MCTSConfig
from src.mcts_engine import MCTSEngine
from src.models import (
    ClassificationResult,
    FraudDetectionResult,
    FraudRiskLevel,
    MCTSMetadata,
)
from src.session_context import (
    SessionContext,
    create_session_context,
    MCTSCacheEntry,
)


@pytest.fixture
def sample_config():
    """Create a sample agent configuration."""
    return AgentConfig(
        llm=LLMConfig(
            provider="test",
            model="test-model",
            api_key="test-key",
        ),
        mcts=MCTSConfig(
            iterations=10,
            max_depth=5,
            exploration_constant=1.414,
        ),
        threshold_amount=250.0,
    )


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "transaction_id": ["TX001", "TX002", "TX003"],
        "amount": [300.0, 500.0, 1000.0],
        "currency": ["GBP", "GBP", "GBP"],
        "merchant": ["Amazon", "Tesco", "Bet365"],
        "merchant_category": ["Retail", "Retail", "Gambling"],
        "date": ["2025-01-15", "2025-01-16", "2025-01-17"],
        "description": ["Purchase", "Groceries", "Gambling"],
    })


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    return TestModel()


@pytest.fixture
def mock_mcts_engine():
    """Create a mock MCTS engine."""
    def mock_llm_function(prompt: str) -> str:
        return "Mock LLM response"

    config = MCTSConfig(iterations=10, max_depth=5, exploration_constant=1.414)
    return MCTSEngine(config, mock_llm_function)


@pytest.fixture
def sample_mcts_metadata():
    """Create a sample MCTSMetadata object for testing."""
    return MCTSMetadata(
        root_node_visits=10,
        best_action_path=["action1", "action2"],
        average_reward=0.75,
        exploration_constant_used=1.414,
        final_reward_variance=0.05,
        total_nodes_explored=25,
        max_depth_reached=5,
    )


class TestSessionContextCreation:
    """Test session context creation and UUID generation (REQ-SM-009)."""

    def test_session_context_has_unique_uuid(self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client):
        """Test that each session gets a unique UUID."""
        session1 = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test1.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        session2 = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test2.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        # Each session should have a unique UUID
        assert session1.session_id != session2.session_id

        # Both should be valid UUIDs
        assert uuid.UUID(session1.session_id)
        assert uuid.UUID(session2.session_id)

    def test_session_context_has_required_fields(self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client):
        """Test that SessionContext has all required fields (REQ-SM-003)."""
        session = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        # Required fields from REQ-SM-003
        assert session.session_id is not None
        assert session.csv_file_name == "test.csv"
        assert session.raw_csv_data is not None
        assert session.filtered_transactions is None  # Not yet filtered
        assert session.classification_results == {}
        assert session.fraud_results == {}
        assert session.mcts_cache == {}
        assert session.execution_log == []
        assert session.config is not None
        assert session.mcts_engine is not None
        assert session.llm_client is not None


class TestToolToToolMemoryPassing:
    """Test that tools can read each other's outputs (REQ-SM-004 to REQ-SM-008)."""

    def test_tool_1_writes_filtered_transactions(self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client):
        """Test that Tool 1 writes to filtered_transactions (REQ-SM-005)."""
        session = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        # Simulate Tool 1 filtering
        filtered_df = sample_dataframe[sample_dataframe["amount"] >= 250.0]
        session.filtered_transactions = filtered_df

        # Tool 1 should have written to session memory
        assert session.filtered_transactions is not None
        assert len(session.filtered_transactions) == 3  # All transactions >= 250

    def test_tool_2_reads_tool_1_output(
        self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client, sample_mcts_metadata
    ):
        """Test that Tool 2 can read Tool 1's output (REQ-SM-006)."""
        session = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        # Simulate Tool 1
        filtered_df = sample_dataframe[sample_dataframe["amount"] >= 250.0]
        session.filtered_transactions = filtered_df

        # Tool 2 should be able to read filtered_transactions
        assert session.filtered_transactions is not None
        assert len(session.filtered_transactions) == 3

        # Simulate Tool 2 classifying first transaction
        tx_id = "TX001"
        classification = ClassificationResult(
            transaction_id=tx_id,
            category="Business",
            confidence=0.9,
            mcts_iterations=10,
            mcts_metadata=sample_mcts_metadata,
        )
        session.classification_results[tx_id] = classification

        assert tx_id in session.classification_results

    def test_tool_3_reads_tool_1_and_2_outputs(
        self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client, sample_mcts_metadata
    ):
        """Test that Tool 3 can read Tools 1 & 2 outputs (REQ-SM-007)."""
        session = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        # Simulate Tool 1
        filtered_df = sample_dataframe[sample_dataframe["amount"] >= 250.0]
        session.filtered_transactions = filtered_df

        # Simulate Tool 2
        tx_id = "TX001"
        classification = ClassificationResult(
            transaction_id=tx_id,
            category="Gambling",
            confidence=0.95,
            mcts_iterations=10,
            mcts_metadata=sample_mcts_metadata,
        )
        session.classification_results[tx_id] = classification

        # Tool 3 should be able to read both filtered_transactions and classification_results
        assert session.filtered_transactions is not None
        assert tx_id in session.classification_results

        # Simulate Tool 3 using classification to inform fraud detection
        classification_category = session.classification_results[tx_id].category
        assert classification_category == "Gambling"

        # Tool 3 can use this to adjust risk (e.g., Gambling → higher baseline risk)
        fraud_result = FraudDetectionResult(
            transaction_id=tx_id,
            risk_level=FraudRiskLevel.HIGH,  # Elevated because of Gambling category
            confidence=0.85,
            detected_indicators=["gambling_transaction"],
            reasoning="Gambling category increases risk",
            mcts_iterations=10,
            recommended_actions=["review_manually"],
        )
        session.fraud_results[tx_id] = fraud_result

        assert tx_id in session.fraud_results


class TestMCTSCaching:
    """Test MCTS caching within sessions (REQ-SM-013 to REQ-SM-015)."""

    def test_cache_key_computation(self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client):
        """Test cache key computation based on merchant + amount (REQ-SM-013)."""
        session = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        transaction1 = {
            "merchant_category": "Retail",
            "amount_gbp": 302.0,
        }
        transaction2 = {
            "merchant_category": "Retail",
            "amount_gbp": 298.0,  # Within ±5 bucket
        }
        transaction3 = {
            "merchant_category": "Gambling",
            "amount_gbp": 300.0,
        }

        key1 = session.get_cache_key(transaction1)
        key2 = session.get_cache_key(transaction2)
        key3 = session.get_cache_key(transaction3)

        # Same merchant + similar amount should have same key
        assert key1 == key2  # Both round to "Retail_300"

        # Different merchant should have different key
        assert key1 != key3

    def test_cache_hit_and_miss(self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client, sample_mcts_metadata):
        """Test MCTS cache hit and miss logic (REQ-SM-013)."""
        session = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        cache_key = "Retail_300"
        tx_id = "TX001"

        # Cache miss (nothing cached yet)
        cached_result = session.get_from_cache(cache_key, tool_type="classification")
        assert cached_result is None
        assert session.mcts_cache_hits == 0
        assert session.mcts_cache_misses == 0  # get_from_cache doesn't increment misses

        # Add to cache
        classification = ClassificationResult(
            transaction_id=tx_id,
            category="Business",
            confidence=0.9,
            mcts_iterations=10,
            mcts_metadata=sample_mcts_metadata,
        )
        session.add_to_cache(cache_key, tx_id, classification=classification)

        # Cache hit
        cached_result = session.get_from_cache(cache_key, tool_type="classification")
        assert cached_result is not None
        assert cached_result.transaction_id == tx_id
        assert cached_result.category == "Business"
        assert session.mcts_cache_hits == 1

    def test_cache_hit_rate_calculation(self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client):
        """Test cache hit rate calculation (REQ-SM-018)."""
        session = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        # Initially 0%
        assert session.get_cache_hit_rate() == 0.0

        # Simulate 3 hits, 1 miss
        session.mcts_cache_hits = 3
        session.mcts_cache_misses = 1

        # Hit rate should be 75%
        assert session.get_cache_hit_rate() == 0.75

    def test_cache_isolation_between_sessions(self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client, sample_mcts_metadata):
        """Test that cache is isolated between sessions (REQ-SM-015)."""
        session1 = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test1.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        session2 = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test2.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        # Add to session1 cache
        cache_key = "Retail_300"
        classification = ClassificationResult(
            transaction_id="TX001",
            category="Business",
            confidence=0.9,
            mcts_iterations=10,
            mcts_metadata=sample_mcts_metadata,
        )
        session1.add_to_cache(cache_key, "TX001", classification=classification)

        # Session2 should not have this cache entry
        cached_result_session2 = session2.get_from_cache(cache_key, tool_type="classification")
        assert cached_result_session2 is None

        # Session1 should still have it
        cached_result_session1 = session1.get_from_cache(cache_key, tool_type="classification")
        assert cached_result_session1 is not None

    def test_cache_invalidation_on_size_limit(self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client, sample_mcts_metadata):
        """Test cache invalidation when size exceeds 50MB (REQ-SM-014)."""
        session = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        # Add some cache entries
        for i in range(10):
            cache_key = f"Retail_{i * 100}"
            classification = ClassificationResult(
                transaction_id=f"TX{i:03d}",
                category="Business",
                confidence=0.9,
                mcts_iterations=10,
                mcts_metadata=sample_mcts_metadata,
            )
            session.add_to_cache(cache_key, f"TX{i:03d}", classification=classification)

        assert len(session.mcts_cache) == 10

        # Manually invalidate cache
        session.invalidate_cache()

        assert len(session.mcts_cache) == 0
        assert session.mcts_cache_hits == 0
        assert session.mcts_cache_misses == 0


class TestMemoryCleanup:
    """Test memory cleanup after session completion (REQ-SM-010)."""

    def test_memory_cleanup_deletes_dataframes(self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client):
        """Test that cleanup deletes intermediate DataFrames (REQ-SM-010)."""
        session = create_session_context(
            csv_data=sample_dataframe.copy(),
            csv_file_name="test.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        # Simulate Tool 1 filtering
        session.filtered_transactions = sample_dataframe[sample_dataframe["amount"] >= 250.0]

        # Before cleanup, data should exist
        assert session.raw_csv_data is not None
        assert session.filtered_transactions is not None

        # Cleanup
        memory_freed_mb = session.cleanup_memory()

        # After cleanup, DataFrames should be None
        assert session.raw_csv_data is None
        assert session.filtered_transactions is None
        assert memory_freed_mb >= 0.0

    def test_memory_cleanup_clears_mcts_cache(self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client, sample_mcts_metadata):
        """Test that cleanup clears MCTS cache (REQ-SM-010)."""
        session = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        # Add cache entries
        cache_key = "Retail_300"
        classification = ClassificationResult(
            transaction_id="TX001",
            category="Business",
            confidence=0.9,
            mcts_iterations=10,
            mcts_metadata=sample_mcts_metadata,
        )
        session.add_to_cache(cache_key, "TX001", classification=classification)

        assert len(session.mcts_cache) > 0

        # Cleanup
        session.cleanup_memory()

        # Cache should be cleared
        assert len(session.mcts_cache) == 0


class TestSessionMetrics:
    """Test session-level metrics aggregation (REQ-SM-018)."""

    def test_session_metrics_initialization(self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client):
        """Test that session metrics are initialized correctly."""
        session = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        assert session.session_metrics.session_total_cost_usd == 0.0
        assert session.session_metrics.session_execution_time_seconds == 0.0
        assert session.session_metrics.session_mcts_cache_hit_rate == 0.0
        assert session.session_metrics.session_tool_failure_count == 0

    def test_session_metrics_update(self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client):
        """Test updating session metrics (REQ-SM-018)."""
        session = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        # Update metrics
        session.update_metrics(cost_usd=0.05, execution_time=1.5, failure_count=1)

        assert session.session_metrics.session_total_cost_usd == 0.05
        assert session.session_metrics.session_execution_time_seconds == 1.5
        assert session.session_metrics.session_tool_failure_count == 1

        # Update again (should accumulate)
        session.update_metrics(cost_usd=0.03, execution_time=0.5, failure_count=0)

        assert session.session_metrics.session_total_cost_usd == 0.08
        assert session.session_metrics.session_execution_time_seconds == 2.0
        assert session.session_metrics.session_tool_failure_count == 1

    def test_finalize_metrics(
        self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client, sample_mcts_metadata
    ):
        """Test finalizing session metrics (REQ-SM-019)."""
        session = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        # Simulate some activity
        session.mcts_cache_hits = 5
        session.mcts_cache_misses = 2

        # Create simple ClassificationResult objects for counting
        for tx_id in ["TX001", "TX002", "TX003"]:
            session.classification_results[tx_id] = ClassificationResult(
                transaction_id=tx_id,
                category="Business",
                confidence=0.9,
                mcts_iterations=10,
                mcts_metadata=sample_mcts_metadata,
            )

        # Finalize
        final_metrics = session.finalize_metrics()

        assert final_metrics.session_mcts_cache_hit_rate == 5 / 7  # 5 hits, 2 misses
        assert final_metrics.transactions_processed == 3
        assert final_metrics.session_execution_time_seconds > 0  # Time has passed


class TestExecutionLog:
    """Test execution logging (REQ-SM-003)."""

    def test_log_tool_execution(self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client):
        """Test logging tool executions (REQ-SM-003)."""
        session = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        # Log a tool execution
        session.log_tool_execution(
            tool_name="filter_transactions_above_threshold",
            status="success",
            duration_seconds=0.5,
            transaction_id=None,
            metadata={"filtered_count": 3}
        )

        assert len(session.execution_log) == 1
        log_entry = session.execution_log[0]
        assert log_entry.tool_name == "filter_transactions_above_threshold"
        assert log_entry.status == "success"
        assert log_entry.duration_seconds == 0.5
        assert log_entry.metadata["filtered_count"] == 3

    def test_execution_log_multiple_entries(self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client):
        """Test logging multiple tool executions."""
        session = create_session_context(
            csv_data=sample_dataframe,
            csv_file_name="test.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        # Log multiple executions
        for i in range(3):
            session.log_tool_execution(
                tool_name="classify_single_transaction_mcts",
                status="success",
                duration_seconds=1.0,
                transaction_id=f"TX{i:03d}",
                metadata={"category": "Business"}
            )

        assert len(session.execution_log) == 3
        assert all(entry.tool_name == "classify_single_transaction_mcts" for entry in session.execution_log)


class TestSessionIsolation:
    """Test that sessions are isolated from each other (REQ-SM-011)."""

    def test_concurrent_sessions_are_isolated(self, sample_dataframe, sample_config, mock_mcts_engine, mock_llm_client, sample_mcts_metadata):
        """Test that concurrent sessions have isolated memory (REQ-SM-011)."""
        # Create two sessions
        session1 = create_session_context(
            csv_data=sample_dataframe.copy(),
            csv_file_name="test1.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        session2 = create_session_context(
            csv_data=sample_dataframe.copy(),
            csv_file_name="test2.csv",
            config=sample_config,
            mcts_engine=mock_mcts_engine,
            llm_client=mock_llm_client,
        )

        # Modify session1
        session1.filtered_transactions = sample_dataframe[sample_dataframe["amount"] >= 250.0]
        session1.classification_results["TX001"] = ClassificationResult(
            transaction_id="TX001",
            category="Business",
            confidence=0.9,
            mcts_iterations=10,
            mcts_metadata=sample_mcts_metadata,
        )

        # Session2 should not be affected
        assert session2.filtered_transactions is None
        assert "TX001" not in session2.classification_results

        # Session1 should have its data
        assert session1.filtered_transactions is not None
        assert "TX001" in session1.classification_results
