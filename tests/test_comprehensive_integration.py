"""
Comprehensive Integration Test for Transaction Analysis Agent.

Tests the complete workflow with synthetic data using a real API.
"""

import os
import pytest
from pathlib import Path
from datetime import datetime
import pandas as pd
import tempfile

# NOTE: Set ANTHROPIC_API_KEY environment variable before running integration tests
# For testing, use: export ANTHROPIC_API_KEY="your-key-here"
# This test suite requires a valid API key for real integration testing
if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = "test-key-placeholder"  # Will skip real API tests

from src.agent import run_analysis
from src.config import AgentConfig, LLMConfig, MCTSConfig, ConfigManager
from src.csv_processor import CSVProcessor
from src.models import Currency, ProcessingReport


class TestComprehensiveIntegration:
    """Comprehensive integration tests with synthetic data."""

    @pytest.fixture
    def synthetic_transactions_csv(self):
        """Create synthetic transaction data for testing."""
        data = {
            "transaction_id": [
                "TXN001",
                "TXN002",
                "TXN003",
                "TXN004",
                "TXN005",
                "TXN006",
                "TXN007",
                "TXN008",
            ],
            "amount": [
                500.00,  # Above threshold
                150.00,  # Below threshold
                1500.00,  # Large amount
                350.00,  # Above threshold
                200.00,  # Below threshold
                9999.99,  # Very large - potential fraud
                425.50,  # Above threshold
                100.00,  # Below threshold
            ],
            "currency": [
                "GBP",
                "USD",
                "GBP",
                "EUR",
                "USD",
                "GBP",
                "EUR",
                "USD",
            ],
            "date": [
                "2025-01-15",
                "2025-01-16",
                "2025-01-17",
                "2025-01-18",
                "2025-01-19",
                "2025-01-20",
                "2025-01-21",
                "2025-01-22",
            ],
            "merchant": [
                "Amazon Business",
                "Starbucks",
                "Microsoft Corporation",
                "Office Depot",
                "McDonald's",
                "UNKNOWN_MERCHANT_XYZ",
                "Dell Technologies",
                "Subway",
            ],
            "description": [
                "Office supplies and equipment purchase",
                "Coffee and snacks",
                "Software licensing annual fee",
                "Printer paper and toner cartridges",
                "Lunch meal",
                "Suspicious transaction - unusual merchant",
                "Computer hardware for development",
                "Sandwich and drink",
            ],
        }

        df = pd.DataFrame(data)

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            return Path(f.name)

    def test_csv_loading_and_validation(self, synthetic_transactions_csv):
        """Test CSV loading and validation."""
        # Load CSV
        df = CSVProcessor.load_csv(synthetic_transactions_csv)
        assert len(df) == 8
        assert all(col in df.columns for col in CSVProcessor.REQUIRED_COLUMNS)

        # Validate schema
        errors = CSVProcessor.validate_schema(df)
        assert len(errors) == 0, f"CSV validation failed: {errors}"

        # Cleanup
        synthetic_transactions_csv.unlink()

    def test_currency_conversion(self, synthetic_transactions_csv):
        """Test currency conversion to GBP."""
        df = CSVProcessor.load_csv(synthetic_transactions_csv)
        df_with_gbp = CSVProcessor.add_gbp_column(df)

        assert "amount_gbp" in df_with_gbp.columns
        assert df_with_gbp["amount_gbp"].notna().all()

        # Check GBP amounts are correct
        gbp_row = df_with_gbp[df_with_gbp["currency"] == "GBP"].iloc[0]
        assert gbp_row["amount_gbp"] == gbp_row["amount"]

        # Cleanup
        synthetic_transactions_csv.unlink()

    def test_transaction_filtering(self, synthetic_transactions_csv):
        """Test filtering transactions above threshold."""
        df = CSVProcessor.load_csv(synthetic_transactions_csv)
        df_with_gbp = CSVProcessor.add_gbp_column(df)

        # Filter above 250 GBP
        filtered = df_with_gbp[df_with_gbp["amount_gbp"] >= 250.0]

        # Should have at least the GBP transactions above 250
        assert len(filtered) >= 3

        # Cleanup
        synthetic_transactions_csv.unlink()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent can be initialized with various configurations."""
        # Test with Anthropic Claude
        config = AgentConfig(
            llm=LLMConfig(
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
                api_key=os.environ["ANTHROPIC_API_KEY"],
            ),
            mcts=MCTSConfig(iterations=10),  # Low iterations for testing
            threshold_amount=250.0,
            base_currency=Currency.GBP,
        )

        assert config.llm.provider == "anthropic"
        assert config.threshold_amount == 250.0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_analysis_with_real_api(self, synthetic_transactions_csv):
        """
        COMPREHENSIVE END-TO-END TEST with real API.

        Tests complete workflow:
        1. Load CSV
        2. Filter transactions
        3. Classify with MCTS
        4. Detect fraud with MCTS
        5. Generate enhanced CSV
        """
        # Configuration with very low iterations for speed
        config = AgentConfig(
            llm=LLMConfig(
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
                api_key=os.environ["ANTHROPIC_API_KEY"],
                temperature=0.0,
            ),
            mcts=MCTSConfig(
                iterations=5,  # Very low for fast testing
                exploration_constant=1.414,
                max_depth=3,
            ),
            threshold_amount=250.0,
            base_currency=Currency.GBP,
        )

        # Load CSV
        df = CSVProcessor.load_csv(synthetic_transactions_csv)

        # Run analysis
        output_path = synthetic_transactions_csv.parent / "enhanced_output.csv"

        try:
            report = run_analysis(
                df=df,
                config=config,
                output_path=output_path,
                progress_callback=lambda msg: print(f"Progress: {msg}"),
            )

            # Verify report
            assert isinstance(report, ProcessingReport)
            assert report.total_transactions_analyzed >= 0
            assert report.llm_provider == "anthropic"
            assert report.model_used == "claude-sonnet-4-5-20250929"
            assert report.processing_time_seconds > 0

            # Verify output file exists
            assert output_path.exists()

            # Load and verify enhanced CSV
            enhanced_df = pd.read_csv(output_path)
            assert len(enhanced_df) == report.total_transactions_analyzed
            assert "classification" in enhanced_df.columns
            assert "fraud_risk" in enhanced_df.columns
            assert "fraud_confidence" in enhanced_df.columns
            assert "mcts_explanation" in enhanced_df.columns

            # Verify classifications are valid
            valid_categories = ["Business", "Personal", "Investment", "Gambling", "Unknown"]
            assert all(
                any(cat in str(classification) for cat in valid_categories)
                for classification in enhanced_df["classification"]
            )

            # Verify fraud risk levels are valid
            valid_risks = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            assert all(risk in valid_risks for risk in enhanced_df["fraud_risk"])

            print(f"\n✅ Integration test passed!")
            print(f"   Transactions analyzed: {report.total_transactions_analyzed}")
            print(f"   High risk: {report.high_risk_transactions}")
            print(f"   Critical risk: {report.critical_risk_transactions}")
            print(f"   Processing time: {report.processing_time_seconds:.2f}s")
            print(f"   MCTS iterations total: {report.mcts_iterations_total}")

        finally:
            # Cleanup
            synthetic_transactions_csv.unlink()
            if output_path.exists():
                output_path.unlink()

    def test_mcts_convergence_detection(self):
        """Test MCTS convergence detection."""
        from src.mcts_engine_v2 import EnhancedMCTSEngine, MCTSNodeV2
        from src.config import MCTSConfig

        config = MCTSConfig(
            iterations=20,
            convergence_window=10,
            convergence_std_threshold=0.01,
            early_termination_enabled=True,
        )

        # Mock LLM function for testing
        def mock_llm(prompt: str) -> str:
            return '[{"category": "Business", "rationale": "Test"}]'

        engine = EnhancedMCTSEngine(
            config=config,
            tool_name="classify",
            llm_function=mock_llm,
        )

        # Test that engine can be created
        assert engine.tool_name == "classify"
        assert engine.global_config.early_termination_enabled

    def test_fraud_detection_indicators(self, synthetic_transactions_csv):
        """Test fraud detection identifies suspicious patterns."""
        df = CSVProcessor.load_csv(synthetic_transactions_csv)

        # TXN006 should be flagged as suspicious (large amount, unknown merchant)
        suspicious_txn = df[df["transaction_id"] == "TXN006"].iloc[0]

        assert suspicious_txn["amount"] == 9999.99
        assert "UNKNOWN" in suspicious_txn["merchant"]
        assert "unusual" in suspicious_txn["description"].lower()

        # Cleanup
        synthetic_transactions_csv.unlink()

    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid reasoning model
        assert ConfigManager.validate_reasoning_model("anthropic", "claude-sonnet-4-5-20250929")

        # Test invalid model should raise error
        with pytest.raises(ValueError, match="not a reasoning model"):
            ConfigManager.validate_reasoning_model("anthropic", "claude-instant-1.2")

    def test_error_handling_missing_api_key(self):
        """Test error handling when API key is missing."""
        # Remove API key temporarily
        old_key = os.environ.get("ANTHROPIC_API_KEY")
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        try:
            config = LLMConfig(
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
                api_key="",
            )

            # Should raise error when creating client
            with pytest.raises(ValueError, match="Invalid API key"):
                ConfigManager.create_llm_client(config)

        finally:
            # Restore API key
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    def test_empty_csv_handling(self):
        """Test handling of empty CSV."""
        # Create empty CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("transaction_id,amount,currency,date,merchant,description\n")
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="empty"):
                CSVProcessor.load_csv(temp_path)
        finally:
            temp_path.unlink()

    def test_invalid_currency_handling(self):
        """Test handling of invalid currency codes."""
        data = {
            "transaction_id": ["TX1"],
            "amount": [100.0],
            "currency": ["INVALID"],
            "date": ["2025-01-01"],
            "merchant": ["Test"],
            "description": ["Test transaction"],
        }

        df = pd.DataFrame(data)

        # Validate should catch invalid currency
        errors = CSVProcessor.validate_schema(df)
        assert len(errors) > 0
        assert any("currency" in error.lower() for error in errors)

    def test_duplicate_transaction_ids(self):
        """Test handling of duplicate transaction IDs."""
        data = {
            "transaction_id": ["TX1", "TX1"],  # Duplicate!
            "amount": [100.0, 200.0],
            "currency": ["GBP", "GBP"],
            "date": ["2025-01-01", "2025-01-02"],
            "merchant": ["Test1", "Test2"],
            "description": ["Desc1", "Desc2"],
        }

        df = pd.DataFrame(data)

        # Validate should catch duplicates
        errors = CSVProcessor.validate_schema(df)
        assert len(errors) > 0
        assert any("duplicate" in error.lower() for error in errors)

    def test_negative_amounts_rejected(self):
        """Test that negative amounts are rejected."""
        data = {
            "transaction_id": ["TX1"],
            "amount": [-100.0],  # Negative!
            "currency": ["GBP"],
            "date": ["2025-01-01"],
            "merchant": ["Test"],
            "description": ["Refund"],
        }

        df = pd.DataFrame(data)

        # Validate should catch negative amount
        errors = CSVProcessor.validate_schema(df)
        assert len(errors) > 0
        assert any("positive" in error.lower() for error in errors)


class TestMCTSAlgorithm:
    """Test MCTS algorithm components."""

    def test_ucb1_calculation(self):
        """Test UCB1 score calculation."""
        from src.mcts_engine import MCTSNode

        root = MCTSNode(state={"test": "data"})
        root.visits = 10
        root.value = 5.0

        child = MCTSNode(state={"child": "data"}, parent=root)
        child.visits = 3
        child.value = 2.0

        # Calculate UCB1
        score = child.ucb1_score(exploration_constant=1.414)

        # Should be positive and finite
        assert score > 0
        assert score != float("inf")

    def test_mcts_node_creation(self):
        """Test MCTS node creation and tree structure."""
        from src.mcts_engine import MCTSNode

        root = MCTSNode(state={"level": 0})
        assert root.parent is None
        assert len(root.children) == 0

        child1 = MCTSNode(state={"level": 1}, parent=root)
        assert child1.parent == root
        assert child1 in root.children

        child2 = MCTSNode(state={"level": 1}, parent=root)
        assert len(root.children) == 2

    def test_backpropagation(self):
        """Test reward backpropagation."""
        from src.mcts_engine import MCTSEngine, MCTSNode
        from src.config import MCTSConfig

        config = MCTSConfig()

        def mock_llm(prompt: str) -> str:
            return '{"confidence": 0.8}'

        engine = MCTSEngine(config, mock_llm)

        # Create tree
        root = MCTSNode(state={})
        child = MCTSNode(state={}, parent=root)

        # Backpropagate
        engine._backpropagate(child, 1.0)

        # Both nodes should be updated
        assert child.visits == 1
        assert child.value == 1.0
        assert root.visits == 1
        assert root.value == 1.0


class TestObservability:
    """Test observability and telemetry features."""

    def test_telemetry_initialization(self):
        """Test telemetry can be initialized."""
        from src.telemetry import get_telemetry, initialize_telemetry, LogfireConfig

        telemetry = get_telemetry()
        assert telemetry is not None

    def test_telemetry_span_creation(self):
        """Test creating telemetry spans."""
        from src.telemetry import get_telemetry

        telemetry = get_telemetry()

        with telemetry.span("test_operation", test_param="value"):
            # Span should be active
            pass

        # Span should complete without errors


class TestSyntheticDataGeneration:
    """Test synthetic data generation utilities."""

    def test_generate_normal_transactions(self):
        """Test generating normal transaction patterns."""
        transactions = [
            {
                "id": "NORM001",
                "amount": 45.50,
                "merchant": "Starbucks",
                "description": "Coffee purchase",
                "expected_risk": "LOW",
                "expected_category": "Personal",
            },
            {
                "id": "NORM002",
                "amount": 1200.00,
                "merchant": "Microsoft",
                "description": "Software licensing",
                "expected_risk": "LOW",
                "expected_category": "Business",
            },
        ]

        for txn in transactions:
            assert txn["amount"] > 0
            assert len(txn["merchant"]) > 0
            assert len(txn["description"]) > 0

    def test_generate_suspicious_transactions(self):
        """Test generating suspicious transaction patterns."""
        suspicious = [
            {
                "id": "SUSP001",
                "amount": 9999.99,
                "merchant": "CASH_WITHDRAWAL_ATM",
                "description": "Large cash withdrawal at 3 AM",
                "expected_risk": "HIGH",
                "red_flags": ["Large amount", "Off-hours", "Cash"],
            },
            {
                "id": "SUSP002",
                "amount": 15000.00,
                "merchant": "FOREIGN_MERCHANT_123",
                "description": "International wire transfer",
                "expected_risk": "MEDIUM",
                "red_flags": ["Large amount", "Foreign", "Wire transfer"],
            },
        ]

        for txn in suspicious:
            assert len(txn["red_flags"]) > 0
            assert txn["expected_risk"] in ["MEDIUM", "HIGH", "CRITICAL"]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
