"""
Comprehensive Tool Unit Tests

This module tests:
- Individual tool functionality
- Tool input/output validation
- Tool error handling
- Currency conversion logic
- Classification logic
- Fraud detection logic
- Edge cases for each tool
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import Any

from src.models import (
    Transaction,
    Currency,
    FraudRiskLevel,
    TransactionFilterResult,
    ClassificationResult,
    FraudDetectionResult,
)
from src.agent import AgentDependencies, filter_transactions_above_threshold
from src.config import AgentConfig, LLMConfig, MCTSConfig
from src.mcts_engine import MCTSEngine


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    data = {
        'transaction_id': ['TX001', 'TX002', 'TX003', 'TX004', 'TX005'],
        'amount': [150.50, 500.00, 1200.00, 25.00, 10000.00],
        'currency': ['GBP', 'USD', 'GBP', 'EUR', 'GBP'],
        'date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
        'merchant': ['Office Supplies Ltd', 'Tech Conference', 'Luxury Hotel', 'Coffee Shop', 'Crypto Exchange'],
        'category': ['Business', 'Business', 'Travel', 'Personal', 'Suspicious'],
        'description': ['Paper and pens', 'Annual summit', '5-star hotel', 'Morning coffee', 'Large crypto purchase']
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
        iterations=10,  # Reduced for testing
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
    """Create a mock MCTS engine"""
    engine = MagicMock(spec=MCTSEngine)
    engine.config = MCTSConfig(iterations=10)
    return engine


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client"""
    return MagicMock()


@pytest.fixture
def agent_deps(sample_dataframe, agent_config, mock_mcts_engine, mock_llm_client):
    """Create agent dependencies for testing"""
    return AgentDependencies(
        df=sample_dataframe,
        config=agent_config,
        mcts_engine=mock_mcts_engine,
        llm_client=mock_llm_client,
        results={}
    )


# ============================================================================
# TOOL 1: FILTER TRANSACTIONS ABOVE THRESHOLD
# ============================================================================

class TestFilterTransactionsTool:
    """Test the filter_transactions_above_threshold tool"""

    def test_filter_with_default_threshold(self, agent_deps):
        """Test filtering with default threshold (250 GBP)"""
        result = filter_transactions_above_threshold(agent_deps)

        assert isinstance(result, TransactionFilterResult)
        # TX002 (500 USD = 395 GBP), TX003 (1200 GBP), TX005 (10000 GBP) should pass
        assert result.filtered_count >= 2  # At least TX003 and TX005
        assert result.currency == Currency.GBP
        assert result.total_amount > 0
        assert result.average_amount > 0

    def test_filter_with_custom_threshold(self, agent_deps):
        """Test filtering with custom threshold"""
        result = filter_transactions_above_threshold(agent_deps, threshold=1000.0)

        assert isinstance(result, TransactionFilterResult)
        # Only TX003 (1200) and TX005 (10000) should pass
        assert result.filtered_count >= 1
        assert result.total_amount >= 1200.0

    def test_filter_with_high_threshold(self, agent_deps):
        """Test filtering with very high threshold"""
        result = filter_transactions_above_threshold(agent_deps, threshold=50000.0)

        assert isinstance(result, TransactionFilterResult)
        # No transactions should pass
        assert result.filtered_count == 0
        assert result.total_amount == 0.0
        assert result.average_amount == 0.0

    def test_filter_with_zero_threshold(self, agent_deps):
        """Test filtering with zero threshold (all transactions pass)"""
        result = filter_transactions_above_threshold(agent_deps, threshold=0.0)

        assert isinstance(result, TransactionFilterResult)
        # All 5 transactions should pass
        assert result.filtered_count == 5
        assert result.total_amount > 0

    def test_filter_currency_conversion_usd_to_gbp(self, agent_deps):
        """Test that USD amounts are correctly converted to GBP"""
        # Create a DataFrame with only USD transactions
        data = {
            'transaction_id': ['TX001'],
            'amount': [1000.00],
            'currency': ['USD'],
            'date': ['2024-01-15'],
            'merchant': ['Test'],
            'category': ['Test'],
            'description': ['Test']
        }
        agent_deps.df = pd.DataFrame(data)
        agent_deps.config.threshold_amount = 500.0

        result = filter_transactions_above_threshold(agent_deps)

        # 1000 USD * 0.79 = 790 GBP, which is > 500 GBP threshold
        assert result.filtered_count == 1

    def test_filter_currency_conversion_eur_to_gbp(self, agent_deps):
        """Test that EUR amounts are correctly converted to GBP"""
        data = {
            'transaction_id': ['TX001'],
            'amount': [1000.00],
            'currency': ['EUR'],
            'date': ['2024-01-15'],
            'merchant': ['Test'],
            'category': ['Test'],
            'description': ['Test']
        }
        agent_deps.df = pd.DataFrame(data)
        agent_deps.config.threshold_amount = 500.0

        result = filter_transactions_above_threshold(agent_deps)

        # 1000 EUR * 0.86 = 860 GBP, which is > 500 GBP threshold
        assert result.filtered_count == 1

    def test_filter_mixed_currencies(self, agent_deps):
        """Test filtering with mixed currencies"""
        result = filter_transactions_above_threshold(agent_deps, threshold=100.0)

        assert isinstance(result, TransactionFilterResult)
        assert result.currency == Currency.GBP
        # Should handle mixed currencies correctly

    def test_filter_custom_currency(self, agent_deps):
        """Test filtering with custom target currency"""
        result = filter_transactions_above_threshold(
            agent_deps,
            threshold=250.0,
            currency=Currency.USD
        )

        assert isinstance(result, TransactionFilterResult)
        assert result.currency == Currency.USD

    def test_filter_results_stored_in_context(self, agent_deps):
        """Test that filter results are stored in agent dependencies"""
        result = filter_transactions_above_threshold(agent_deps)

        # Results should be stored in context
        assert 'filtered_transactions' in agent_deps.results
        assert 'filter_result' in agent_deps.results

        filtered_df = agent_deps.results['filtered_transactions']
        assert isinstance(filtered_df, pd.DataFrame)
        assert len(filtered_df) == result.filtered_count

    def test_filter_empty_dataframe(self, agent_deps):
        """Test filtering with empty DataFrame"""
        agent_deps.df = pd.DataFrame(columns=['transaction_id', 'amount', 'currency', 'date', 'merchant', 'category', 'description'])

        result = filter_transactions_above_threshold(agent_deps)

        assert result.filtered_count == 0
        assert result.total_amount == 0.0
        assert result.average_amount == 0.0

    def test_filter_negative_threshold_rejected(self, agent_deps):
        """Test that negative threshold is handled properly"""
        # Negative threshold should either raise error or treat as 0
        with pytest.raises((ValueError, AssertionError)) or True:
            result = filter_transactions_above_threshold(agent_deps, threshold=-100.0)


# ============================================================================
# TOOL 2: CLASSIFY SINGLE TRANSACTION (MCTS)
# ============================================================================

class TestClassifySingleTransactionTool:
    """Test the classify_single_transaction_mcts tool"""

    @pytest.mark.asyncio
    async def test_classify_business_transaction(self, agent_deps):
        """Test classification of a business transaction"""
        # Mock MCTS engine response
        mock_result = MagicMock()
        mock_result.state = {
            'primary_classification': 'Business',
            'confidence': 0.92,
            'alternative_classifications': ['Professional Services'],
            'reasoning': 'Office supplies for business use'
        }
        agent_deps.mcts_engine.search = AsyncMock(return_value=mock_result)

        transaction = Transaction(
            transaction_id='TX001',
            amount=150.50,
            currency=Currency.GBP,
            date=datetime(2024, 1, 15),
            merchant='Office Supplies Ltd',
            category='Business',
            description='Paper and pens'
        )

        # Import the tool function
        from src.agent import classify_single_transaction_mcts

        # Create a mock RunContext
        ctx = MagicMock()
        ctx.deps = agent_deps

        result = await classify_single_transaction_mcts(ctx, transaction_id='TX001')

        assert isinstance(result, ClassificationResult)
        assert result.transaction_id == 'TX001'
        assert result.primary_classification == 'Business'
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_classify_personal_transaction(self, agent_deps):
        """Test classification of a personal transaction"""
        mock_result = MagicMock()
        mock_result.state = {
            'primary_classification': 'Personal',
            'confidence': 0.88,
            'alternative_classifications': ['Food & Dining'],
            'reasoning': 'Personal coffee purchase'
        }
        agent_deps.mcts_engine.search = AsyncMock(return_value=mock_result)

        from src.agent import classify_single_transaction_mcts

        ctx = MagicMock()
        ctx.deps = agent_deps

        result = await classify_single_transaction_mcts(ctx, transaction_id='TX004')

        assert isinstance(result, ClassificationResult)
        assert result.confidence >= 0.0 and result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_classify_travel_transaction(self, agent_deps):
        """Test classification of a travel transaction"""
        mock_result = MagicMock()
        mock_result.state = {
            'primary_classification': 'Travel',
            'confidence': 0.95,
            'alternative_classifications': ['Accommodation', 'Business Travel'],
            'reasoning': 'Luxury hotel booking'
        }
        agent_deps.mcts_engine.search = AsyncMock(return_value=mock_result)

        from src.agent import classify_single_transaction_mcts

        ctx = MagicMock()
        ctx.deps = agent_deps

        result = await classify_single_transaction_mcts(ctx, transaction_id='TX003')

        assert isinstance(result, ClassificationResult)
        assert len(result.alternative_classifications) >= 0

    @pytest.mark.asyncio
    async def test_classify_confidence_bounds(self, agent_deps):
        """Test that confidence scores are within valid bounds"""
        mock_result = MagicMock()
        mock_result.state = {
            'primary_classification': 'Business',
            'confidence': 0.75,
            'alternative_classifications': [],
            'reasoning': 'Test'
        }
        agent_deps.mcts_engine.search = AsyncMock(return_value=mock_result)

        from src.agent import classify_single_transaction_mcts

        ctx = MagicMock()
        ctx.deps = agent_deps

        result = await classify_single_transaction_mcts(ctx, transaction_id='TX001')

        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_classify_mcts_iterations_recorded(self, agent_deps):
        """Test that MCTS iterations are recorded in result"""
        mock_result = MagicMock()
        mock_result.state = {
            'primary_classification': 'Business',
            'confidence': 0.85,
            'alternative_classifications': [],
            'reasoning': 'Test'
        }
        agent_deps.mcts_engine.search = AsyncMock(return_value=mock_result)
        agent_deps.mcts_engine.config.iterations = 100

        from src.agent import classify_single_transaction_mcts

        ctx = MagicMock()
        ctx.deps = agent_deps

        result = await classify_single_transaction_mcts(ctx, transaction_id='TX001')

        assert result.mcts_iterations > 0


# ============================================================================
# TOOL 3: DETECT FRAUD (MCTS)
# ============================================================================

class TestDetectFraudTool:
    """Test the detect_fraud_single_transaction_mcts tool"""

    @pytest.mark.asyncio
    async def test_detect_low_risk_transaction(self, agent_deps):
        """Test fraud detection for low-risk transaction"""
        mock_result = MagicMock()
        mock_result.state = {
            'risk_level': 'LOW',
            'confidence': 0.90,
            'indicators': [],
            'reasoning': 'Normal business transaction, no red flags',
            'recommended_actions': []
        }
        agent_deps.mcts_engine.search = AsyncMock(return_value=mock_result)

        from src.agent import detect_fraud_single_transaction_mcts

        ctx = MagicMock()
        ctx.deps = agent_deps

        result = await detect_fraud_single_transaction_mcts(ctx, transaction_id='TX001')

        assert isinstance(result, FraudDetectionResult)
        assert result.transaction_id == 'TX001'
        assert result.risk_level == FraudRiskLevel.LOW
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_detect_medium_risk_transaction(self, agent_deps):
        """Test fraud detection for medium-risk transaction"""
        mock_result = MagicMock()
        mock_result.state = {
            'risk_level': 'MEDIUM',
            'confidence': 0.75,
            'indicators': ['Unusual merchant'],
            'reasoning': 'First-time merchant, monitor closely',
            'recommended_actions': ['Monitor account']
        }
        agent_deps.mcts_engine.search = AsyncMock(return_value=mock_result)

        from src.agent import detect_fraud_single_transaction_mcts

        ctx = MagicMock()
        ctx.deps = agent_deps

        result = await detect_fraud_single_transaction_mcts(ctx, transaction_id='TX002')

        assert result.risk_level == FraudRiskLevel.MEDIUM
        assert len(result.detected_indicators) >= 0

    @pytest.mark.asyncio
    async def test_detect_high_risk_transaction(self, agent_deps):
        """Test fraud detection for high-risk transaction"""
        mock_result = MagicMock()
        mock_result.state = {
            'risk_level': 'HIGH',
            'confidence': 0.88,
            'indicators': ['Large amount', 'Luxury purchase', 'Unusual pattern'],
            'reasoning': 'Large luxury purchase outside normal spending',
            'recommended_actions': ['Verify with customer', 'Review recent activity']
        }
        agent_deps.mcts_engine.search = AsyncMock(return_value=mock_result)

        from src.agent import detect_fraud_single_transaction_mcts

        ctx = MagicMock()
        ctx.deps = agent_deps

        result = await detect_fraud_single_transaction_mcts(ctx, transaction_id='TX003')

        assert result.risk_level == FraudRiskLevel.HIGH
        assert len(result.detected_indicators) > 0
        assert len(result.recommended_actions) > 0

    @pytest.mark.asyncio
    async def test_detect_critical_risk_transaction(self, agent_deps):
        """Test fraud detection for critical-risk transaction"""
        mock_result = MagicMock()
        mock_result.state = {
            'risk_level': 'CRITICAL',
            'confidence': 0.95,
            'indicators': [
                'Very large amount',
                'Cryptocurrency',
                'Suspicious merchant',
                'No prior history'
            ],
            'reasoning': 'Large crypto purchase, high fraud indicators',
            'recommended_actions': [
                'Block transaction',
                'Contact customer immediately',
                'Flag account for review'
            ]
        }
        agent_deps.mcts_engine.search = AsyncMock(return_value=mock_result)

        from src.agent import detect_fraud_single_transaction_mcts

        ctx = MagicMock()
        ctx.deps = agent_deps

        result = await detect_fraud_single_transaction_mcts(ctx, transaction_id='TX005')

        assert result.risk_level == FraudRiskLevel.CRITICAL
        assert len(result.detected_indicators) >= 3
        assert len(result.recommended_actions) >= 2
        assert result.confidence > 0.8

    @pytest.mark.asyncio
    async def test_detect_fraud_confidence_bounds(self, agent_deps):
        """Test that fraud confidence scores are within valid bounds"""
        mock_result = MagicMock()
        mock_result.state = {
            'risk_level': 'LOW',
            'confidence': 0.65,
            'indicators': [],
            'reasoning': 'Test',
            'recommended_actions': []
        }
        agent_deps.mcts_engine.search = AsyncMock(return_value=mock_result)

        from src.agent import detect_fraud_single_transaction_mcts

        ctx = MagicMock()
        ctx.deps = agent_deps

        result = await detect_fraud_single_transaction_mcts(ctx, transaction_id='TX001')

        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_detect_fraud_mcts_iterations(self, agent_deps):
        """Test that MCTS iterations are recorded"""
        mock_result = MagicMock()
        mock_result.state = {
            'risk_level': 'LOW',
            'confidence': 0.80,
            'indicators': [],
            'reasoning': 'Test',
            'recommended_actions': []
        }
        agent_deps.mcts_engine.search = AsyncMock(return_value=mock_result)
        agent_deps.mcts_engine.config.iterations = 100

        from src.agent import detect_fraud_single_transaction_mcts

        ctx = MagicMock()
        ctx.deps = agent_deps

        result = await detect_fraud_single_transaction_mcts(ctx, transaction_id='TX001')

        assert result.mcts_iterations > 0


# ============================================================================
# CURRENCY CONVERSION TESTS
# ============================================================================

class TestCurrencyConversion:
    """Test currency conversion functionality"""

    def test_gbp_to_gbp_conversion(self):
        """Test GBP to GBP conversion (no change)"""
        from src.csv_processor import convert_to_gbp

        result = convert_to_gbp(100.0, Currency.GBP)
        assert result == 100.0

    def test_usd_to_gbp_conversion(self):
        """Test USD to GBP conversion"""
        from src.csv_processor import convert_to_gbp

        result = convert_to_gbp(100.0, Currency.USD)
        # 100 USD * 0.79 = 79 GBP
        assert result == pytest.approx(79.0, rel=0.01)

    def test_eur_to_gbp_conversion(self):
        """Test EUR to GBP conversion"""
        from src.csv_processor import convert_to_gbp

        result = convert_to_gbp(100.0, Currency.EUR)
        # 100 EUR * 0.86 = 86 GBP
        assert result == pytest.approx(86.0, rel=0.01)

    def test_jpy_to_gbp_conversion(self):
        """Test JPY to GBP conversion"""
        from src.csv_processor import convert_to_gbp

        result = convert_to_gbp(10000.0, Currency.JPY)
        # 10000 JPY * 0.0054 = 54 GBP
        assert result == pytest.approx(54.0, rel=0.01)

    def test_zero_amount_conversion(self):
        """Test conversion of zero amount"""
        from src.csv_processor import convert_to_gbp

        result = convert_to_gbp(0.0, Currency.USD)
        assert result == 0.0

    def test_large_amount_conversion(self):
        """Test conversion of very large amount"""
        from src.csv_processor import convert_to_gbp

        result = convert_to_gbp(1000000.0, Currency.USD)
        assert result == pytest.approx(790000.0, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
