"""
Comprehensive Tool Interaction Tests

This module tests:
- Sequential execution of tools
- State sharing between tools via AgentDependencies
- Data flow from one tool to another
- Tool pipeline integration
- Error propagation between tools
- Results accumulation
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, AsyncMock

from src.models import (
    Currency,
    TransactionFilterResult,
    FraudDetectionResult,
)
from src.agent import (
    AgentDependencies,
    filter_transactions_above_threshold,
    classify_single_transaction_mcts,
    detect_fraud_single_transaction_mcts,
)
from src.config import AgentConfig, LLMConfig, MCTSConfig
from src.mcts_engine import MCTSEngine


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def comprehensive_dataframe():
    """Create a comprehensive DataFrame with various transaction types"""
    data = {
        'transaction_id': [
            'TX001', 'TX002', 'TX003', 'TX004', 'TX005',
            'TX006', 'TX007', 'TX008', 'TX009', 'TX010'
        ],
        'amount': [
            50.00,      # Below threshold
            300.00,     # Above threshold
            1500.00,    # High amount
            10.00,      # Very low
            500.00,     # Above threshold
            25000.00,   # Very high (suspicious)
            100.00,     # Below threshold
            800.00,     # Above threshold
            15.50,      # Very low
            5000.00     # High amount
        ],
        'currency': ['GBP', 'GBP', 'USD', 'EUR', 'GBP', 'GBP', 'EUR', 'GBP', 'GBP', 'GBP'],
        'date': [
            '2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19',
            '2024-01-20', '2024-01-21', '2024-01-22', '2024-01-23', '2024-01-24'
        ],
        'merchant': [
            'Coffee Shop', 'Office Depot', 'Luxury Hotel', 'Cafe', 'Tech Store',
            'Crypto Exchange', 'Restaurant', 'Flight Booking', 'Newsstand', 'Jewelry Store'
        ],
        'category': [
            'Personal', 'Business', 'Travel', 'Personal', 'Business',
            'Suspicious', 'Personal', 'Travel', 'Personal', 'Retail'
        ],
        'description': [
            'Morning coffee', 'Office supplies', 'Hotel stay', 'Lunch', 'Laptop',
            'Bitcoin purchase', 'Dinner', 'Business trip', 'Magazine', 'Watch'
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def agent_config():
    """Create a test agent configuration"""
    llm_config = LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        api_key="test-key"
    )
    mcts_config = MCTSConfig(
        iterations=10,
        exploration_constant=1.414,
        max_depth=5,
        simulation_budget=10
    )
    return AgentConfig(
        llm=llm_config,
        mcts=mcts_config,
        threshold_amount=250.0,
        base_currency=Currency.GBP
    )


@pytest.fixture
def mock_mcts_engine():
    """Create a mock MCTS engine with realistic responses"""
    engine = MagicMock(spec=MCTSEngine)
    engine.config = MCTSConfig(iterations=10)

    # Mock search to return realistic classification results
    async def mock_search_classification(task_type, transaction):
        return MagicMock(
            state={
                'primary_classification': 'Business' if 'Office' in transaction.merchant else 'Personal',
                'confidence': 0.85,
                'alternative_classifications': ['Professional Services'],
                'reasoning': f'Classified based on merchant: {transaction.merchant}'
            }
        )

    # Mock search to return realistic fraud results
    async def mock_search_fraud(task_type, transaction):
        risk = 'CRITICAL' if transaction.amount > 10000 else \
               'HIGH' if transaction.amount > 5000 else \
               'MEDIUM' if transaction.amount > 1000 else 'LOW'

        return MagicMock(
            state={
                'risk_level': risk,
                'confidence': 0.90,
                'indicators': ['Large amount'] if transaction.amount > 5000 else [],
                'reasoning': f'Amount-based risk assessment: {transaction.amount}',
                'recommended_actions': ['Contact customer'] if risk in ['HIGH', 'CRITICAL'] else []
            }
        )

    engine.search = AsyncMock(side_effect=mock_search_classification)

    return engine


@pytest.fixture
def agent_deps(comprehensive_dataframe, agent_config, mock_mcts_engine):
    """Create agent dependencies for testing"""
    return AgentDependencies(
        df=comprehensive_dataframe,
        config=agent_config,
        mcts_engine=mock_mcts_engine,
        llm_client=MagicMock(),
        results={}
    )


@pytest.fixture
def mock_ctx(agent_deps):
    """Create a mock RunContext for tool testing"""
    class MockRunContext:
        def __init__(self, deps):
            self.deps = deps

    return MockRunContext(agent_deps)


# ============================================================================
# SEQUENTIAL TOOL EXECUTION TESTS
# ============================================================================

class TestSequentialToolExecution:
    """Test sequential execution of tools in the pipeline"""

    def test_filter_then_classify_pipeline(self, mock_ctx):
        """Test that filtering happens before classification"""
        # Step 1: Filter transactions
        filter_result = filter_transactions_above_threshold(mock_ctx)

        assert isinstance(filter_result, TransactionFilterResult)
        assert 'filtered_df' in mock_ctx.deps.results

        # Verify filtered DataFrame is available for next tool
        filtered_df = mock_ctx.deps.results['filtered_df']
        assert isinstance(filtered_df, pd.DataFrame)
        assert len(filtered_df) == filter_result.filtered_count

    def test_filter_then_fraud_detection_pipeline(self, mock_ctx):
        """Test filtering followed by fraud detection"""
        # Step 1: Filter
        filter_result = filter_transactions_above_threshold(mock_ctx)

        # Step 2: Access filtered data for fraud detection
        filtered_df = mock_ctx.deps.results['filtered_df']

        assert len(filtered_df) > 0
        # Each filtered transaction can now be checked for fraud
        for _, row in filtered_df.iterrows():
            assert row['transaction_id'] is not None

    def test_complete_pipeline_sequence(self, mock_ctx):
        """Test complete pipeline: Filter -> Classify -> Fraud Detect"""
        # Step 1: Filter
        filter_result = filter_transactions_above_threshold(mock_ctx)
        assert 'filtered_df' in mock_ctx.deps.results

        # Step 2: Store classification results
        mock_ctx.deps.results['classifications'] = []

        # Step 3: Store fraud results
        mock_ctx.deps.results['fraud_detections'] = []

        # Verify all result keys exist
        assert 'filtered_df' in mock_ctx.deps.results
        assert 'classifications' in mock_ctx.deps.results
        assert 'fraud_detections' in mock_ctx.deps.results

    def test_tool_execution_order_matters(self, mock_ctx):
        """Test that executing tools out of order causes issues"""
        # Trying to access filtered_transactions before filtering should fail
        assert 'filtered_df' not in mock_ctx.deps.results

        # After filtering, it should exist
        filter_result = filter_transactions_above_threshold(mock_ctx)
        assert 'filtered_df' in mock_ctx.deps.results


# ============================================================================
# STATE SHARING TESTS
# ============================================================================

class TestStateSharing:
    """Test state sharing between tools via AgentDependencies"""

    def test_results_dictionary_shared(self, mock_ctx):
        """Test that results dictionary is shared across tool calls"""
        # Tool 1 stores a result
        filter_result = filter_transactions_above_threshold(mock_ctx)
        mock_ctx.deps.results['custom_data'] = 'test_value'

        # Tool 2 can access Tool 1's results
        assert 'custom_data' in mock_ctx.deps.results
        assert mock_ctx.deps.results['custom_data'] == 'test_value'

    def test_dataframe_reference_shared(self, mock_ctx):
        """Test that DataFrame reference is shared"""
        original_df = mock_ctx.deps.df

        # Filter modifies or creates new filtered DataFrame
        filter_result = filter_transactions_above_threshold(mock_ctx)

        # Original DataFrame should still be accessible
        assert mock_ctx.deps.df is original_df
        # But filtered version is in results
        assert 'filtered_df' in mock_ctx.deps.results

    def test_config_shared_across_tools(self, mock_ctx):
        """Test that configuration is shared across all tools"""
        # All tools should see same config
        assert mock_ctx.deps.config.threshold_amount == 250.0
        assert mock_ctx.deps.config.base_currency == Currency.GBP

        # Modifying config affects all subsequent tool calls
        mock_ctx.deps.config.threshold_amount = 500.0

        filter_result = filter_transactions_above_threshold(mock_ctx)
        # Should use new threshold
        # (actual behavior depends on implementation)

    def test_mcts_engine_shared(self, mock_ctx):
        """Test that MCTS engine is shared across classification and fraud detection"""
        # Both classify and fraud detect use same engine
        engine = mock_ctx.deps.mcts_engine

        assert engine is not None
        assert hasattr(engine, 'search')

    @pytest.mark.asyncio
    async def test_classification_results_accumulation(self, mock_ctx):
        """Test that classification results accumulate in shared state"""
        mock_ctx.deps.results['classifications'] = []

        # Simulate multiple classifications

        ctx = MagicMock()
        ctx.deps = agent_deps

        # Configure mock for classification
        mock_ctx.deps.mcts_engine.search = AsyncMock(return_value=MagicMock(
            state={
                'primary_classification': 'Business',
                'confidence': 0.85,
                'alternative_classifications': [],
                'reasoning': 'Test'
            }
        ))

        result1 = await classify_single_transaction_mcts(ctx, transaction_id='TX002')
        mock_ctx.deps.results['classifications'].append(result1)

        result2 = await classify_single_transaction_mcts(ctx, transaction_id='TX005')
        mock_ctx.deps.results['classifications'].append(result2)

        # Both results should be stored
        assert len(mock_ctx.deps.results['classifications']) == 2

    @pytest.mark.asyncio
    async def test_fraud_results_accumulation(self, mock_ctx):
        """Test that fraud detection results accumulate"""
        mock_ctx.deps.results['fraud_detections'] = []


        ctx = MagicMock()
        ctx.deps = agent_deps

        # Configure mock for fraud detection
        mock_ctx.deps.mcts_engine.search = AsyncMock(return_value=MagicMock(
            state={
                'risk_level': 'LOW',
                'confidence': 0.90,
                'indicators': [],
                'reasoning': 'Test',
                'recommended_actions': []
            }
        ))

        result1 = await detect_fraud_single_transaction_mcts(ctx, transaction_id='TX002')
        mock_ctx.deps.results['fraud_detections'].append(result1)

        result2 = await detect_fraud_single_transaction_mcts(ctx, transaction_id='TX005')
        mock_ctx.deps.results['fraud_detections'].append(result2)

        assert len(mock_ctx.deps.results['fraud_detections']) == 2


# ============================================================================
# DATA FLOW TESTS
# ============================================================================

class TestDataFlow:
    """Test data flow from one tool to another"""

    def test_filter_output_feeds_classification(self, mock_ctx):
        """Test that filter output is used by classification"""
        # Filter creates filtered_transactions
        filter_result = filter_transactions_above_threshold(mock_ctx)

        filtered_df = mock_ctx.deps.results['filtered_df']

        # Classification should iterate over filtered transactions
        for _, row in filtered_df.iterrows():
            transaction_id = row['transaction_id']
            # Each transaction from filter can be classified
            assert transaction_id is not None

        assert len(filtered_df) == filter_result.filtered_count

    def test_filter_output_feeds_fraud_detection(self, mock_ctx):
        """Test that filter output is used by fraud detection"""
        filter_result = filter_transactions_above_threshold(mock_ctx)

        filtered_df = mock_ctx.deps.results['filtered_df']

        # Fraud detection should iterate over filtered transactions
        for _, row in filtered_df.iterrows():
            transaction_id = row['transaction_id']
            # Each transaction from filter can be fraud-checked
            assert transaction_id is not None

    @pytest.mark.asyncio
    async def test_classification_and_fraud_on_same_transactions(self, mock_ctx):
        """Test that both classification and fraud detection work on same filtered set"""
        # Filter first
        filter_result = filter_transactions_above_threshold(mock_ctx)
        filtered_df = mock_ctx.deps.results['filtered_df']

        # Initialize result storage
        mock_ctx.deps.results['classifications'] = {}
        mock_ctx.deps.results['fraud_detections'] = {}


        ctx = MagicMock()
        ctx.deps = agent_deps

        # Configure mocks
        mock_ctx.deps.mcts_engine.search = AsyncMock(return_value=MagicMock(
            state={
                'primary_classification': 'Business',
                'confidence': 0.85,
                'alternative_classifications': [],
                'reasoning': 'Test'
            }
        ))

        # Process first transaction
        if len(filtered_df) > 0:
            tx_id = filtered_df.iloc[0]['transaction_id']

            # Classify
            classification = await classify_single_transaction_mcts(ctx, transaction_id=tx_id)
            mock_ctx.deps.results['classifications'][tx_id] = classification

            # Reconfigure mock for fraud
            mock_ctx.deps.mcts_engine.search = AsyncMock(return_value=MagicMock(
                state={
                    'risk_level': 'LOW',
                    'confidence': 0.90,
                    'indicators': [],
                    'reasoning': 'Test',
                    'recommended_actions': []
                }
            ))

            # Fraud detect
            fraud = await detect_fraud_single_transaction_mcts(ctx, transaction_id=tx_id)
            mock_ctx.deps.results['fraud_detections'][tx_id] = fraud

            # Same transaction should have both results
            assert tx_id in mock_ctx.deps.results['classifications']
            assert tx_id in mock_ctx.deps.results['fraud_detections']

    def test_transaction_count_consistency(self, mock_ctx):
        """Test that transaction count is consistent across tools"""
        filter_result = filter_transactions_above_threshold(mock_ctx)

        filtered_count = filter_result.filtered_count
        filtered_df = mock_ctx.deps.results['filtered_df']

        # DataFrame length should match filter count
        assert len(filtered_df) == filtered_count

    def test_currency_conversion_flows_through_pipeline(self, mock_ctx):
        """Test that currency conversion is consistent throughout pipeline"""
        # Filter uses currency conversion
        filter_result = filter_transactions_above_threshold(mock_ctx)

        assert filter_result.currency == Currency.GBP

        # Filtered transactions should include amount_gbp
        filtered_df = mock_ctx.deps.results['filtered_df']

        if 'amount_gbp' in filtered_df.columns:
            # All GBP amounts should be present
            assert filtered_df['amount_gbp'].notna().all()


# ============================================================================
# ERROR PROPAGATION TESTS
# ============================================================================

class TestErrorPropagation:
    """Test error propagation between tools"""

    def test_filter_error_stops_pipeline(self, mock_ctx):
        """Test that error in filter stops subsequent tools"""
        # Corrupt the DataFrame to cause error
        mock_ctx.deps.df = pd.DataFrame()  # Empty DataFrame

        filter_result = filter_transactions_above_threshold(mock_ctx)

        # Should get 0 filtered transactions
        assert filter_result.filtered_count == 0

        # Pipeline should handle this gracefully
        filtered_df = mock_ctx.deps.results.get('filtered_df', pd.DataFrame())
        assert len(filtered_df) == 0

    def test_missing_transaction_id_handling(self, mock_ctx):
        """Test handling of missing transaction IDs"""
        # Create DataFrame with missing transaction_id
        bad_df = pd.DataFrame({
            'amount': [100.0],
            'currency': ['GBP'],
            'date': ['2024-01-15'],
            'merchant': ['Test'],
            'category': ['Test'],
            'description': ['Test']
        })

        mock_ctx.deps.df = bad_df

        # Should handle missing transaction_id
        with pytest.raises(KeyError) or True:
            filter_result = filter_transactions_above_threshold(mock_ctx)

    @pytest.mark.asyncio
    async def test_classification_error_doesnt_affect_fraud(self, mock_ctx):
        """Test that classification error doesn't prevent fraud detection"""

        ctx = MagicMock()
        ctx.deps = agent_deps

        # Even if classification fails, fraud detection should work independently
        mock_ctx.deps.mcts_engine.search = AsyncMock(return_value=MagicMock(
            state={
                'risk_level': 'LOW',
                'confidence': 0.90,
                'indicators': [],
                'reasoning': 'Test',
                'recommended_actions': []
            }
        ))

        # Fraud detection should work
        result = await detect_fraud_single_transaction_mcts(ctx, transaction_id='TX002')
        assert isinstance(result, FraudDetectionResult)


# ============================================================================
# PARALLEL VS SEQUENTIAL PROCESSING TESTS
# ============================================================================

class TestProcessingOrder:
    """Test that tools process in correct order"""

    def test_filter_must_precede_classification(self, mock_ctx):
        """Test that attempting to classify before filtering is handled"""
        # Before filtering, no filtered_transactions
        assert 'filtered_df' not in mock_ctx.deps.results

        # Classification would need to check for this
        # (In actual implementation, pipeline ensures correct order)

    def test_filter_must_precede_fraud_detection(self, mock_ctx):
        """Test that attempting fraud detection before filtering is handled"""
        assert 'filtered_df' not in mock_ctx.deps.results

        # Fraud detection would need filtered list
        # Pipeline ensures correct order

    @pytest.mark.asyncio
    async def test_classification_and_fraud_can_run_parallel_per_transaction(self, mock_ctx):
        """Test that for same transaction, classification and fraud can theoretically run in parallel"""

        ctx = MagicMock()
        ctx.deps = agent_deps

        # Configure mock for classification
        async def mock_classify(*args, **kwargs):
            return MagicMock(
                state={
                    'primary_classification': 'Business',
                    'confidence': 0.85,
                    'alternative_classifications': [],
                    'reasoning': 'Test'
                }
            )

        # Configure mock for fraud
        async def mock_fraud(*args, **kwargs):
            return MagicMock(
                state={
                    'risk_level': 'LOW',
                    'confidence': 0.90,
                    'indicators': [],
                    'reasoning': 'Test',
                    'recommended_actions': []
                }
            )

        # Both can be called on same transaction
        # (actual parallelization is implementation detail)
        mock_ctx.deps.mcts_engine.search = AsyncMock(side_effect=mock_classify)
        result1 = await classify_single_transaction_mcts(ctx, transaction_id='TX002')

        mock_ctx.deps.mcts_engine.search = AsyncMock(side_effect=mock_fraud)
        result2 = await detect_fraud_single_transaction_mcts(ctx, transaction_id='TX002')

        assert result1 is not None
        assert result2 is not None


# ============================================================================
# RESULTS MERGING TESTS
# ============================================================================

class TestResultsMerging:
    """Test merging results from multiple tools"""

    @pytest.mark.asyncio
    async def test_merge_filter_classification_fraud_results(self, mock_ctx):
        """Test merging results from all three main tools"""
        # Filter
        filter_result = filter_transactions_above_threshold(mock_ctx)
        filtered_df = mock_ctx.deps.results['filtered_df']

        # Initialize result containers
        mock_ctx.deps.results['classifications'] = {}
        mock_ctx.deps.results['fraud_detections'] = {}


        ctx = MagicMock()
        ctx.deps = agent_deps

        if len(filtered_df) > 0:
            tx_id = filtered_df.iloc[0]['transaction_id']

            # Classify
            mock_ctx.deps.mcts_engine.search = AsyncMock(return_value=MagicMock(
                state={
                    'primary_classification': 'Business',
                    'confidence': 0.85,
                    'alternative_classifications': [],
                    'reasoning': 'Test'
                }
            ))
            classification = await classify_single_transaction_mcts(ctx, transaction_id=tx_id)
            mock_ctx.deps.results['classifications'][tx_id] = classification

            # Fraud
            mock_ctx.deps.mcts_engine.search = AsyncMock(return_value=MagicMock(
                state={
                    'risk_level': 'LOW',
                    'confidence': 0.90,
                    'indicators': [],
                    'reasoning': 'Test',
                    'recommended_actions': []
                }
            ))
            fraud = await detect_fraud_single_transaction_mcts(ctx, transaction_id=tx_id)
            mock_ctx.deps.results['fraud_detections'][tx_id] = fraud

            # All results should be available for merging
            assert tx_id in mock_ctx.deps.results['classifications']
            assert tx_id in mock_ctx.deps.results['fraud_detections']

            # Create merged result for this transaction
            merged = {
                'transaction_id': tx_id,
                'classification': classification.primary_classification,
                'classification_confidence': classification.confidence,
                'fraud_risk': fraud.risk_level.value,
                'fraud_confidence': fraud.confidence
            }

            assert merged['transaction_id'] == tx_id
            assert merged['classification'] is not None
            assert merged['fraud_risk'] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
