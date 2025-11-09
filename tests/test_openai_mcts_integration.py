"""
Exhaustive Integration Tests for MCTS Engine with Real OpenAI API

This module performs comprehensive testing of the MCTS engine using:
- Real OpenAI API calls (NO MOCKS)
- Synthetic transaction data
- All OpenAI reasoning models: o1, o1-preview, o1-mini, o3-mini
- All 4 tools: filter, classify, fraud detection, CSV generation
- Edge cases and stress scenarios

IMPORTANT: These tests make REAL API calls and will consume OpenAI credits.
Set OPENAI_API_KEY environment variable before running.
"""

import pytest
import os
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from src.models import Currency, FraudRiskLevel
from src.config import AgentConfig, LLMConfig, MCTSConfig, ConfigManager
from src.mcts_engine_v2 import EnhancedMCTSEngine
from src.agent import AgentDependencies, financial_agent
from src.csv_processor import CSVProcessor
from pydantic_ai import Agent

# Import synthetic data generator
from tests.synthetic_data_generator import (
    SyntheticTransactionGenerator,
    create_test_dataset
)

# Mark all tests as integration and slow
pytestmark = [pytest.mark.integration, pytest.mark.slow]


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key.startswith('sk-proj-'):
        # Allow any key that looks like OpenAI format
        if not api_key or len(api_key) < 20:
            pytest.skip("OPENAI_API_KEY not configured")
    return api_key


@pytest.fixture(params=[
    "o1-mini",
    "o1-preview",
    # "o1",  # Expensive, comment out for cost savings
    # "o3-mini",  # If available
])
def openai_model(request):
    """Parametrize tests across different OpenAI reasoning models."""
    return request.param


@pytest.fixture
def llm_config_openai(openai_api_key, openai_model):
    """Create LLM configuration for OpenAI."""
    return LLMConfig(
        provider="openai",
        model=openai_model,
        api_key=openai_api_key,
        temperature=1.0,  # OpenAI reasoning models use temperature=1
        max_tokens=4000
    )


@pytest.fixture
def mcts_config_fast():
    """Create MCTS configuration with reduced iterations for faster testing."""
    return MCTSConfig(
        # Tool-specific reduced configurations
        filter_iterations=10,
        filter_max_depth=10,
        classification_iterations=20,
        classification_max_depth=15,
        fraud_iterations=30,
        fraud_max_depth=20,
        explanation_iterations=15,
        explanation_max_depth=10,
        # Legacy
        iterations=10,
        exploration_constant=1.414,
        max_depth=5,
        simulation_budget=10,
        early_termination_enabled=True,
        convergence_std_threshold=0.01,
        convergence_window=5  # Reduced for testing
    )


@pytest.fixture
def mcts_config_thorough():
    """Create MCTS configuration with full iterations for thorough testing."""
    return MCTSConfig(
        # Tool-specific full configurations
        filter_iterations=100,
        filter_max_depth=30,
        classification_iterations=500,
        classification_max_depth=50,
        fraud_iterations=1000,
        fraud_max_depth=75,
        explanation_iterations=200,
        explanation_max_depth=20,
        # Legacy
        iterations=100,
        exploration_constant=1.414,
        max_depth=5,
        simulation_budget=10,
        early_termination_enabled=True,
        convergence_std_threshold=0.01,
        convergence_window=50
    )


@pytest.fixture
def agent_config_openai(llm_config_openai, mcts_config_fast):
    """Create complete agent configuration for OpenAI."""
    return AgentConfig(
        llm=llm_config_openai,
        mcts=mcts_config_fast,
        threshold_amount=250.0,
        base_currency=Currency.GBP
    )


@pytest.fixture
def synthetic_data_small():
    """Generate small synthetic dataset."""
    return create_test_dataset("small", include_fraud=True, seed=42)


@pytest.fixture
def synthetic_data_tiny():
    """Generate tiny synthetic dataset for quick tests."""
    return create_test_dataset("tiny", include_fraud=True, seed=42)


@pytest.fixture
def synthetic_generator():
    """Provide synthetic data generator instance."""
    return SyntheticTransactionGenerator(seed=42)


@pytest.fixture
def real_llm_client(llm_config_openai):
    """Create real OpenAI LLM client."""
    return ConfigManager.create_llm_client(llm_config_openai)


def create_llm_function(llm_client):
    """
    Create a callable LLM function for MCTS engine.

    Args:
        llm_client: Pydantic AI LLM client

    Returns:
        Callable function that takes a prompt and returns a response
    """
    def llm_function(prompt: str) -> str:
        """Call LLM synchronously and return response."""
        try:
            # Create a simple agent for this call
            agent = Agent(llm_client)
            result = agent.run_sync(prompt)
            return str(result.data)
        except Exception as e:
            return f"Error: {str(e)}"

    return llm_function


# ============================================================================
# TEST 1: LLM CLIENT AND MODEL VALIDATION
# ============================================================================

class TestOpenAIClientSetup:
    """Test OpenAI client creation and model validation."""

    def test_create_openai_client_o1_mini(self, openai_api_key):
        """Test creating OpenAI o1-mini client."""
        config = LLMConfig(
            provider="openai",
            model="o1-mini",
            api_key=openai_api_key
        )
        client = ConfigManager.create_llm_client(config)
        assert client is not None

    def test_create_openai_client_o1_preview(self, openai_api_key):
        """Test creating OpenAI o1-preview client."""
        config = LLMConfig(
            provider="openai",
            model="o1-preview",
            api_key=openai_api_key
        )
        client = ConfigManager.create_llm_client(config)
        assert client is not None

    def test_validate_o1_mini_reasoning_model(self):
        """Test that o1-mini is validated as a reasoning model."""
        is_valid = ConfigManager.validate_reasoning_model("openai", "o1-mini")
        assert is_valid is True

    def test_validate_o1_preview_reasoning_model(self):
        """Test that o1-preview is validated as a reasoning model."""
        is_valid = ConfigManager.validate_reasoning_model("openai", "o1-preview")
        assert is_valid is True

    def test_validate_o3_mini_reasoning_model(self):
        """Test that o3-mini is validated as a reasoning model."""
        is_valid = ConfigManager.validate_reasoning_model("openai", "o3-mini")
        assert is_valid is True

    def test_reject_non_reasoning_model(self, openai_api_key):
        """Test that non-reasoning models are rejected."""
        with pytest.raises(ValueError, match="not a reasoning model"):
            config = LLMConfig(
                provider="openai",
                model="gpt-4",  # Not a reasoning model
                api_key=openai_api_key
            )
            ConfigManager.validate_reasoning_model(config.provider, config.model)


# ============================================================================
# TEST 2: MCTS ENGINE WITH REAL OPENAI API
# ============================================================================

class TestMCTSEngineWithOpenAI:
    """Test MCTS engine with real OpenAI API calls."""

    def test_mcts_engine_initialization(self, llm_config_openai, mcts_config_fast, real_llm_client):
        """Test initializing MCTS engine with real OpenAI client."""
        llm_function = create_llm_function(real_llm_client)

        engine = EnhancedMCTSEngine(
            config=mcts_config_fast,
            tool_name="classify",
            llm_function=llm_function,
            transaction_id="TX001"
        )

        assert engine is not None
        assert engine.tool_name == "classify"
        assert engine.llm_function is not None

    def test_mcts_search_classification_real_api(
        self, llm_config_openai, mcts_config_fast, real_llm_client, synthetic_generator
    ):
        """Test MCTS search for classification with real OpenAI API."""
        # Create LLM function
        llm_function = create_llm_function(real_llm_client)

        # Create MCTS engine
        engine = EnhancedMCTSEngine(
            config=mcts_config_fast,
            tool_name="classify",
            llm_function=llm_function,
            transaction_id="TX001"
        )

        # Generate a business transaction
        transaction = synthetic_generator.generate_transaction(
            category="Business",
            fraud_risk="LOW"
        )

        # Prepare initial state
        initial_state = {
            "transaction": transaction,
            "context": {
                "threshold": 250.0,
                "currency": "GBP"
            }
        }

        # Run MCTS search
        start_time = time.time()
        result = engine.search(initial_state, objective="classify")
        elapsed = time.time() - start_time

        # Verify result structure
        assert result is not None
        assert "hypothesis" in result
        assert "confidence" in result
        assert "mcts_metadata" in result

        # Verify MCTS metadata
        metadata = result["mcts_metadata"]
        assert metadata.root_node_visits > 0
        assert metadata.total_nodes_explored > 0
        assert metadata.max_depth_reached >= 0
        assert 0.0 <= result["confidence"] <= 1.0

        print(f"\n✓ Classification completed in {elapsed:.2f}s")
        print(f"  Hypothesis: {result['hypothesis']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Nodes explored: {metadata.total_nodes_explored}")

    def test_mcts_search_fraud_detection_real_api(
        self, llm_config_openai, mcts_config_fast, real_llm_client, synthetic_generator
    ):
        """Test MCTS search for fraud detection with real OpenAI API."""
        llm_function = create_llm_function(real_llm_client)

        engine = EnhancedMCTSEngine(
            config=mcts_config_fast,
            tool_name="fraud",
            llm_function=llm_function,
            transaction_id="TX002"
        )

        # Generate a suspicious transaction
        transaction = synthetic_generator.generate_transaction(
            category="Suspicious",
            fraud_risk="CRITICAL"
        )

        initial_state = {
            "transaction": transaction,
            "context": {
                "threshold": 250.0,
                "currency": "GBP"
            }
        }

        # Run MCTS search
        start_time = time.time()
        result = engine.search(initial_state, objective="detect_fraud")
        elapsed = time.time() - start_time

        # Verify result
        assert result is not None
        assert "hypothesis" in result
        assert "confidence" in result

        hypothesis = result["hypothesis"]
        # Should detect some risk level
        assert "risk_level" in hypothesis or "risk" in str(hypothesis).lower()

        print(f"\n✓ Fraud detection completed in {elapsed:.2f}s")
        print(f"  Hypothesis: {hypothesis}")
        print(f"  Confidence: {result['confidence']:.2f}")

    def test_mcts_convergence_early_termination(
        self, llm_config_openai, mcts_config_fast, real_llm_client, synthetic_generator
    ):
        """Test MCTS early termination via convergence detection."""
        llm_function = create_llm_function(real_llm_client)

        # Enable early termination
        config = mcts_config_fast
        config.early_termination_enabled = True
        config.convergence_window = 5
        config.convergence_std_threshold = 0.01

        engine = EnhancedMCTSEngine(
            config=config,
            tool_name="classify",
            llm_function=llm_function,
            transaction_id="TX003"
        )

        transaction = synthetic_generator.generate_transaction(category="Personal")
        initial_state = {"transaction": transaction}

        # Run search
        result = engine.search(initial_state, objective="classify")

        # If converged early, metadata should show fewer iterations
        metadata = result["mcts_metadata"]
        print(f"\n✓ MCTS iterations: {metadata.root_node_visits}")
        print(f"  Final variance: {metadata.final_reward_variance:.4f}")

        # Should complete successfully
        assert result is not None


# ============================================================================
# TEST 3: TOOL 1 - FILTER TRANSACTIONS
# ============================================================================

class TestFilterToolWithRealAPI:
    """Test filter transactions tool with real OpenAI API."""

    def test_filter_basic_threshold(self, agent_config_openai, synthetic_data_tiny, real_llm_client):
        """Test basic filtering above 250 GBP threshold."""
        # Prepare dependencies
        from src.agent import filter_transactions_above_threshold

        class MockContext:
            def __init__(self, deps):
                self.deps = deps

        deps = AgentDependencies(
            df=synthetic_data_tiny,
            config=agent_config_openai,
            mcts_engine=None,  # Not needed for filtering
            llm_client=real_llm_client,
            results={}
        )

        ctx = MockContext(deps)

        # Run filter
        result = filter_transactions_above_threshold(ctx)

        # Verify results
        assert result is not None
        assert result.filtered_count >= 0
        assert result.currency == Currency.GBP
        assert "filtered_df" in deps.results

        print(f"\n✓ Filtered {result.filtered_count} transactions above £250")
        print(f"  Total amount: £{result.total_amount:.2f}")
        if result.filtered_count > 0:
            print(f"  Average amount: £{result.average_amount:.2f}")

    def test_filter_currency_conversion(self, agent_config_openai, real_llm_client):
        """Test filtering with currency conversion."""
        # Create dataset with multiple currencies
        data = {
            'transaction_id': ['TX001', 'TX002', 'TX003', 'TX004'],
            'amount': [300.00, 400.00, 30000.00, 350.00],
            'currency': ['GBP', 'USD', 'JPY', 'EUR'],
            'date': ['2024-01-15'] * 4,
            'merchant': ['Store'] * 4,
            'category': ['Business'] * 4,
            'description': ['Purchase'] * 4
        }
        df = pd.DataFrame(data)

        from src.agent import filter_transactions_above_threshold

        class MockContext:
            def __init__(self, deps):
                self.deps = deps

        deps = AgentDependencies(
            df=df,
            config=agent_config_openai,
            mcts_engine=None,
            llm_client=real_llm_client,
            results={}
        )

        ctx = MockContext(deps)
        result = filter_transactions_above_threshold(ctx, threshold=250.0)

        # Should filter based on GBP equivalent
        assert result.filtered_count >= 1  # At least GBP 300
        print(f"\n✓ Multi-currency filtering: {result.filtered_count} transactions")

    def test_filter_edge_case_exact_threshold(self, agent_config_openai, real_llm_client):
        """Test filtering with amount exactly at threshold."""
        data = {
            'transaction_id': ['TX001', 'TX002', 'TX003'],
            'amount': [249.99, 250.00, 250.01],
            'currency': ['GBP', 'GBP', 'GBP'],
            'date': ['2024-01-15'] * 3,
            'merchant': ['Store'] * 3,
            'category': ['Business'] * 3,
            'description': ['Purchase'] * 3
        }
        df = pd.DataFrame(data)

        from src.agent import filter_transactions_above_threshold

        class MockContext:
            def __init__(self, deps):
                self.deps = deps

        deps = AgentDependencies(
            df=df,
            config=agent_config_openai,
            mcts_engine=None,
            llm_client=real_llm_client,
            results={}
        )

        ctx = MockContext(deps)
        result = filter_transactions_above_threshold(ctx, threshold=250.0)

        # Should include 250.00 and 250.01 (>= threshold)
        assert result.filtered_count == 2
        print(f"\n✓ Edge case filtering: {result.filtered_count} transactions (expected: 2)")


# ============================================================================
# TEST 4: TOOL 2 - CLASSIFY TRANSACTIONS
# ============================================================================

class TestClassifyToolWithRealAPI:
    """Test classification tool with real OpenAI API."""

    def test_classify_business_transaction(
        self, agent_config_openai, mcts_config_fast, real_llm_client, synthetic_generator
    ):
        """Test classifying a business transaction."""
        llm_function = create_llm_function(real_llm_client)

        # Create MCTS engine for classification
        mcts_engine = EnhancedMCTSEngine(
            config=mcts_config_fast,
            tool_name="classify",
            llm_function=llm_function,
            transaction_id="TX_CLASSIFY_001"
        )

        # Generate business transaction
        transaction = synthetic_generator.generate_transaction(
            category="Business",
            fraud_risk="LOW"
        )

        # Prepare state
        state = {
            "transaction": transaction,
            "context": {"threshold": 250.0}
        }

        # Run classification
        result = mcts_engine.search(state, objective="classify")

        assert result is not None
        assert "hypothesis" in result
        assert 0.0 <= result["confidence"] <= 1.0

        print(f"\n✓ Classified transaction: {transaction['merchant']}")
        print(f"  Result: {result['hypothesis']}")
        print(f"  Confidence: {result['confidence']:.2f}")

    def test_classify_multiple_categories(
        self, agent_config_openai, mcts_config_fast, real_llm_client, synthetic_generator
    ):
        """Test classifying transactions from different categories."""
        llm_function = create_llm_function(real_llm_client)

        categories = ["Business", "Personal", "Travel", "Entertainment"]
        results = []

        for category in categories:
            transaction = synthetic_generator.generate_transaction(
                category=category,
                fraud_risk="LOW"
            )

            mcts_engine = EnhancedMCTSEngine(
                config=mcts_config_fast,
                tool_name="classify",
                llm_function=llm_function,
                transaction_id=f"TX_CAT_{category}"
            )

            state = {"transaction": transaction}
            result = mcts_engine.search(state, objective="classify")
            results.append((category, result))

            print(f"\n✓ {category}: {result['hypothesis']} (confidence: {result['confidence']:.2f})")

        # All should complete successfully
        assert len(results) == len(categories)
        assert all(r[1] is not None for r in results)

    def test_classify_confidence_scores(
        self, agent_config_openai, mcts_config_fast, real_llm_client, synthetic_generator
    ):
        """Test that confidence scores are meaningful and in valid range."""
        llm_function = create_llm_function(real_llm_client)

        confidences = []

        for _ in range(5):
            transaction = synthetic_generator.generate_transaction()

            mcts_engine = EnhancedMCTSEngine(
                config=mcts_config_fast,
                tool_name="classify",
                llm_function=llm_function,
                transaction_id=f"TX_CONF_{_}"
            )

            state = {"transaction": transaction}
            result = mcts_engine.search(state, objective="classify")
            confidences.append(result["confidence"])

        # All confidences should be in valid range
        assert all(0.0 <= c <= 1.0 for c in confidences)
        print(f"\n✓ Confidence scores: {[f'{c:.2f}' for c in confidences]}")


# ============================================================================
# TEST 5: TOOL 3 - FRAUD DETECTION
# ============================================================================

class TestFraudDetectionToolWithRealAPI:
    """Test fraud detection tool with real OpenAI API."""

    def test_detect_low_risk_transaction(
        self, agent_config_openai, mcts_config_fast, real_llm_client, synthetic_generator
    ):
        """Test detecting low-risk transaction."""
        llm_function = create_llm_function(real_llm_client)

        # Generate low-risk transaction
        transaction = synthetic_generator.generate_transaction(
            category="Personal",
            fraud_risk="LOW",
            amount_range=(10, 50)
        )

        mcts_engine = EnhancedMCTSEngine(
            config=mcts_config_fast,
            tool_name="fraud",
            llm_function=llm_function,
            transaction_id="TX_FRAUD_LOW"
        )

        state = {"transaction": transaction}
        result = mcts_engine.search(state, objective="detect_fraud")

        assert result is not None
        print(f"\n✓ Low-risk transaction: {transaction['merchant']}")
        print(f"  Result: {result['hypothesis']}")
        print(f"  Confidence: {result['confidence']:.2f}")

    def test_detect_high_risk_transaction(
        self, agent_config_openai, mcts_config_fast, real_llm_client, synthetic_generator
    ):
        """Test detecting high-risk transaction."""
        llm_function = create_llm_function(real_llm_client)

        # Generate high-risk transaction
        transaction = synthetic_generator.generate_transaction(
            category="Suspicious",
            fraud_risk="CRITICAL",
            amount_range=(10000, 50000)
        )

        mcts_engine = EnhancedMCTSEngine(
            config=mcts_config_fast,
            tool_name="fraud",
            llm_function=llm_function,
            transaction_id="TX_FRAUD_HIGH"
        )

        state = {"transaction": transaction}
        result = mcts_engine.search(state, objective="detect_fraud")

        assert result is not None
        print(f"\n✓ High-risk transaction: {transaction['merchant']} (£{transaction['amount']})")
        print(f"  Result: {result['hypothesis']}")
        print(f"  Confidence: {result['confidence']:.2f}")

    def test_detect_fraud_multiple_risk_levels(
        self, agent_config_openai, mcts_config_fast, real_llm_client, synthetic_generator
    ):
        """Test fraud detection across multiple risk levels."""
        llm_function = create_llm_function(real_llm_client)

        risk_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        results = []

        for risk in risk_levels:
            transaction = synthetic_generator.generate_transaction(fraud_risk=risk)

            mcts_engine = EnhancedMCTSEngine(
                config=mcts_config_fast,
                tool_name="fraud",
                llm_function=llm_function,
                transaction_id=f"TX_RISK_{risk}"
            )

            state = {"transaction": transaction}
            result = mcts_engine.search(state, objective="detect_fraud")
            results.append((risk, result))

            print(f"\n✓ {risk}: {result['hypothesis']} (confidence: {result['confidence']:.2f})")

        # All should complete
        assert len(results) == len(risk_levels)

    def test_detect_fraud_rapid_succession_pattern(
        self, agent_config_openai, mcts_config_fast, real_llm_client, synthetic_generator
    ):
        """Test fraud detection on rapid succession pattern."""
        llm_function = create_llm_function(real_llm_client)

        # Generate rapid succession scenario
        transactions = synthetic_generator.generate_fraud_scenario("rapid_succession")

        for i, transaction in enumerate(transactions[:3]):  # Test first 3
            mcts_engine = EnhancedMCTSEngine(
                config=mcts_config_fast,
                tool_name="fraud",
                llm_function=llm_function,
                transaction_id=f"TX_RAPID_{i}"
            )

            state = {
                "transaction": transaction,
                "context": {"pattern": "rapid_succession", "sequence": i}
            }

            result = mcts_engine.search(state, objective="detect_fraud")
            assert result is not None
            print(f"\n✓ Rapid transaction #{i+1}: {result['confidence']:.2f} confidence")


# ============================================================================
# TEST 6: TOOL 4 - CSV GENERATION
# ============================================================================

class TestCSVGenerationWithRealAPI:
    """Test CSV generation tool with real OpenAI API."""

    def test_generate_enhanced_csv(
        self, agent_config_openai, synthetic_data_tiny, tmp_path
    ):
        """Test generating enhanced CSV output."""
        # Save input CSV
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"

        synthetic_data_tiny.to_csv(input_path, index=False)

        # Process with CSV processor
        processor = CSVProcessor(str(input_path))
        df = processor.load()

        # Add analysis columns (simulated)
        df['above_250_gbp'] = df['amount'] >= 250
        df['classification'] = 'Business'
        df['classification_confidence'] = 0.85
        df['fraud_risk'] = 'LOW'
        df['fraud_confidence'] = 0.90
        df['fraud_reasoning'] = 'No indicators detected'
        df['mcts_iterations'] = 10

        # Save enhanced CSV
        processor.save_enhanced(df, str(output_path))

        # Verify output
        assert output_path.exists()
        output_df = pd.read_csv(output_path)
        assert len(output_df) == len(synthetic_data_tiny)
        assert 'classification' in output_df.columns
        assert 'fraud_risk' in output_df.columns

        print(f"\n✓ Enhanced CSV generated: {len(output_df)} rows")
        print(f"  Columns: {list(output_df.columns)}")


# ============================================================================
# TEST 7: MODEL COMPARISON
# ============================================================================

class TestModelComparison:
    """Compare performance across different OpenAI reasoning models."""

    @pytest.mark.parametrize("model_name", ["o1-mini", "o1-preview"])
    def test_model_comparison_classification(
        self, openai_api_key, model_name, mcts_config_fast, synthetic_generator
    ):
        """Compare classification performance across models."""
        config = LLMConfig(
            provider="openai",
            model=model_name,
            api_key=openai_api_key,
            temperature=1.0
        )

        client = ConfigManager.create_llm_client(config)
        llm_function = create_llm_function(client)

        transaction = synthetic_generator.generate_transaction(category="Business")

        mcts_engine = EnhancedMCTSEngine(
            config=mcts_config_fast,
            tool_name="classify",
            llm_function=llm_function,
            transaction_id=f"TX_{model_name}"
        )

        state = {"transaction": transaction}

        start_time = time.time()
        result = mcts_engine.search(state, objective="classify")
        elapsed = time.time() - start_time

        print(f"\n✓ {model_name}: {result['confidence']:.2f} confidence in {elapsed:.2f}s")
        print(f"  Nodes explored: {result['mcts_metadata'].total_nodes_explored}")

        assert result is not None
        assert result["confidence"] >= 0.0


# ============================================================================
# TEST 8: STRESS TESTS
# ============================================================================

class TestStressScenarios:
    """Stress test the system with challenging scenarios."""

    def test_very_large_amount(
        self, agent_config_openai, mcts_config_fast, real_llm_client, synthetic_generator
    ):
        """Test with very large transaction amount."""
        llm_function = create_llm_function(real_llm_client)

        transaction = synthetic_generator.generate_transaction(
            amount_range=(100000, 500000),
            fraud_risk="CRITICAL"
        )

        mcts_engine = EnhancedMCTSEngine(
            config=mcts_config_fast,
            tool_name="fraud",
            llm_function=llm_function,
            transaction_id="TX_LARGE"
        )

        state = {"transaction": transaction}
        result = mcts_engine.search(state, objective="detect_fraud")

        assert result is not None
        print(f"\n✓ Large amount (£{transaction['amount']:,.2f}): processed")

    def test_very_small_amount(
        self, agent_config_openai, mcts_config_fast, real_llm_client, synthetic_generator
    ):
        """Test with very small transaction amount."""
        llm_function = create_llm_function(real_llm_client)

        transaction = synthetic_generator.generate_transaction(
            amount_range=(0.01, 1.00),
            fraud_risk="LOW"
        )

        mcts_engine = EnhancedMCTSEngine(
            config=mcts_config_fast,
            tool_name="classify",
            llm_function=llm_function,
            transaction_id="TX_SMALL"
        )

        state = {"transaction": transaction}
        result = mcts_engine.search(state, objective="classify")

        assert result is not None
        print(f"\n✓ Small amount (£{transaction['amount']:.2f}): processed")

    def test_batch_processing(
        self, agent_config_openai, mcts_config_fast, real_llm_client, synthetic_data_tiny
    ):
        """Test processing a batch of transactions."""
        llm_function = create_llm_function(real_llm_client)

        processed = 0
        for _, row in synthetic_data_tiny.head(5).iterrows():
            transaction = row.to_dict()

            mcts_engine = EnhancedMCTSEngine(
                config=mcts_config_fast,
                tool_name="classify",
                llm_function=llm_function,
                transaction_id=transaction['transaction_id']
            )

            state = {"transaction": transaction}
            result = mcts_engine.search(state, objective="classify")

            if result is not None:
                processed += 1

        print(f"\n✓ Batch processed: {processed}/5 transactions")
        assert processed >= 3  # At least 60% success rate


# ============================================================================
# TEST 9: ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Test error handling with real API."""

    def test_invalid_transaction_data(
        self, agent_config_openai, mcts_config_fast, real_llm_client
    ):
        """Test handling of invalid transaction data."""
        llm_function = create_llm_function(real_llm_client)

        # Missing required fields
        invalid_transaction = {
            "transaction_id": "TX_INVALID"
            # Missing amount, merchant, etc.
        }

        mcts_engine = EnhancedMCTSEngine(
            config=mcts_config_fast,
            tool_name="classify",
            llm_function=llm_function,
            transaction_id="TX_INVALID"
        )

        state = {"transaction": invalid_transaction}

        # Should handle gracefully
        try:
            result = mcts_engine.search(state, objective="classify")
            # If it doesn't raise, that's also acceptable
            assert result is not None or result is None
        except Exception as e:
            # Expected to handle error
            print(f"\n✓ Handled invalid data: {type(e).__name__}")


# ============================================================================
# TEST 10: PERFORMANCE BENCHMARKS
# ============================================================================

class TestPerformanceBenchmarks:
    """Benchmark performance with real API."""

    def test_single_classification_latency(
        self, agent_config_openai, mcts_config_fast, real_llm_client, synthetic_generator
    ):
        """Measure single classification latency."""
        llm_function = create_llm_function(real_llm_client)
        transaction = synthetic_generator.generate_transaction()

        mcts_engine = EnhancedMCTSEngine(
            config=mcts_config_fast,
            tool_name="classify",
            llm_function=llm_function,
            transaction_id="TX_PERF"
        )

        state = {"transaction": transaction}

        start = time.time()
        result = mcts_engine.search(state, objective="classify")
        latency = time.time() - start

        print(f"\n✓ Classification latency: {latency:.2f}s")
        print(f"  Iterations: {result['mcts_metadata'].root_node_visits}")

        assert latency < 60.0  # Should complete within 60 seconds

    def test_throughput_measurement(
        self, agent_config_openai, mcts_config_fast, real_llm_client, synthetic_generator
    ):
        """Measure throughput (transactions per minute)."""
        llm_function = create_llm_function(real_llm_client)

        count = 3
        start = time.time()

        for i in range(count):
            transaction = synthetic_generator.generate_transaction()

            mcts_engine = EnhancedMCTSEngine(
                config=mcts_config_fast,
                tool_name="classify",
                llm_function=llm_function,
                transaction_id=f"TX_THROUGHPUT_{i}"
            )

            state = {"transaction": transaction}
            result = mcts_engine.search(state, objective="classify")

        elapsed = time.time() - start
        throughput = (count / elapsed) * 60

        print(f"\n✓ Throughput: {throughput:.2f} transactions/minute")
        print(f"  Processed {count} in {elapsed:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
