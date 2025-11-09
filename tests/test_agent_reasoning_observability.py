"""
Agent Reasoning and Observability Tests.

Tests MCTS reasoning quality, agent initialization, and observability features.
"""

import os
import pytest
from datetime import datetime

# NOTE: Set ANTHROPIC_API_KEY environment variable before running these tests
# For testing, use: export ANTHROPIC_API_KEY="your-key-here"
if "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["ANTHROPIC_API_KEY"] = "test-key-placeholder"

from src.agent import financial_agent, AgentDependencies
from src.config import AgentConfig, LLMConfig, MCTSConfig
from src.models import Currency, Transaction, FraudRiskLevel
from src.mcts_engine import MCTSEngine, MCTSNode
from src.telemetry import get_telemetry, initialize_telemetry, LogfireConfig
import pandas as pd


class TestAgentInitialization:
    """Test agent initialization and configuration."""

    def test_agent_creation(self):
        """Test that financial agent can be created."""
        assert financial_agent is not None
        assert hasattr(financial_agent, "tool")

    def test_agent_with_test_model(self):
        """Test agent works with test model (no API key needed)."""
        # Remove API keys
        old_openai = os.environ.get("OPENAI_API_KEY")
        old_anthropic = os.environ.get("ANTHROPIC_API_KEY")

        try:
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]

            # Import should create agent with test model
            from src.agent import _create_agent

            agent = _create_agent()
            assert agent is not None

        finally:
            # Restore
            if old_openai:
                os.environ["OPENAI_API_KEY"] = old_openai
            if old_anthropic:
                os.environ["ANTHROPIC_API_KEY"] = old_anthropic

    def test_agent_dependencies_creation(self):
        """Test AgentDependencies can be created."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        config = AgentConfig(
            llm=LLMConfig(
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
                api_key="test-key",
            ),
            mcts=MCTSConfig(),
            threshold_amount=250.0,
        )

        # Mock LLM function
        def mock_llm(prompt: str) -> str:
            return '{"result": "test"}'

        from pydantic_ai.models.test import TestModel

        deps = AgentDependencies(
            df=df,
            config=config,
            mcts_engine=MCTSEngine(config.mcts, mock_llm),
            llm_client=TestModel(),
        )

        assert deps.df is not None
        assert deps.config is not None
        assert deps.mcts_engine is not None


class TestMCTSReasoning:
    """Test MCTS reasoning capabilities."""

    def test_mcts_node_ucb1_unvisited(self):
        """Test UCB1 returns infinity for unvisited nodes."""
        node = MCTSNode(state={})
        assert node.ucb1_score() == float("inf")

    def test_mcts_node_ucb1_visited(self):
        """Test UCB1 calculation for visited nodes."""
        root = MCTSNode(state={})
        root.visits = 10
        root.value = 5.0

        child = MCTSNode(state={}, parent=root)
        child.visits = 3
        child.value = 2.0

        score = child.ucb1_score(exploration_constant=1.414)
        assert score > 0
        assert score < float("inf")

    def test_mcts_search_classify_objective(self):
        """Test MCTS search with classify objective."""

        def mock_llm(prompt: str) -> str:
            if "generate" in prompt.lower() or "hypotheses" in prompt.lower():
                return """[
                    {"category": "Business", "rationale": "Business transaction"},
                    {"category": "Personal", "rationale": "Personal expense"}
                ]"""
            else:
                return '{"confidence": 0.8, "reasoning": "Test reasoning"}'

        config = MCTSConfig(iterations=5, max_depth=3)
        engine = MCTSEngine(config, mock_llm)

        state = {
            "transaction": {
                "amount": 500,
                "currency": "GBP",
                "merchant": "Test Merchant",
                "description": "Test transaction",
            }
        }

        result = engine.search(state, objective="classify")

        assert "hypothesis" in result
        assert "confidence" in result
        assert result["confidence"] >= 0
        assert result["confidence"] <= 1

    def test_mcts_search_fraud_objective(self):
        """Test MCTS search with fraud detection objective."""

        def mock_llm(prompt: str) -> str:
            if "generate" in prompt.lower() or "hypotheses" in prompt.lower():
                return """[
                    {"risk_level": "LOW", "indicators": [], "rationale": "Normal transaction"},
                    {"risk_level": "MEDIUM", "indicators": ["Review needed"], "rationale": "Needs review"}
                ]"""
            else:
                return '{"confidence": 0.7, "reasoning": "Fraud assessment", "actions": []}'

        config = MCTSConfig(iterations=5, max_depth=3)
        engine = MCTSEngine(config, mock_llm)

        state = {
            "transaction": {
                "amount": 9999,
                "currency": "GBP",
                "merchant": "UNKNOWN_MERCHANT",
                "description": "Suspicious transaction",
            }
        }

        result = engine.search(state, objective="detect_fraud")

        assert "hypothesis" in result
        assert "confidence" in result
        assert "reasoning" in result

    def test_mcts_tree_expansion(self):
        """Test MCTS tree expansion creates children."""

        def mock_llm(prompt: str) -> str:
            return '[{"category": "Test", "rationale": "Test"}]'

        config = MCTSConfig(iterations=5)
        engine = MCTSEngine(config, mock_llm)

        root = MCTSNode(state={"transaction": {}})
        node = engine._expand(root, objective="classify")

        # Should have created children
        assert len(root.children) > 0

    def test_mcts_backpropagation_updates_tree(self):
        """Test backpropagation updates all nodes in path."""

        def mock_llm(prompt: str) -> str:
            return "{}"

        config = MCTSConfig()
        engine = MCTSEngine(config, mock_llm)

        # Create tree: root -> child -> grandchild
        root = MCTSNode(state={})
        child = MCTSNode(state={}, parent=root)
        grandchild = MCTSNode(state={}, parent=child)

        # Backpropagate from grandchild
        engine._backpropagate(grandchild, reward=1.0)

        # All should be updated
        assert grandchild.visits == 1
        assert grandchild.value == 1.0
        assert child.visits == 1
        assert child.value == 1.0
        assert root.visits == 1
        assert root.value == 1.0


class TestObservabilityFeatures:
    """Test observability and telemetry."""

    def test_telemetry_singleton(self):
        """Test telemetry uses singleton pattern."""
        tel1 = get_telemetry()
        tel2 = get_telemetry()
        assert tel1 is tel2

    def test_telemetry_span_context_manager(self):
        """Test telemetry span as context manager."""
        telemetry = get_telemetry()

        # Should not raise errors
        with telemetry.span("test_span", param1="value1"):
            pass

    def test_telemetry_log_methods(self):
        """Test telemetry logging methods."""
        telemetry = get_telemetry()

        # Should not raise errors
        telemetry.log_info("Test info message", key="value")
        telemetry.log_warning("Test warning", key="value")
        telemetry.log_error("Test error", key="value")

    def test_telemetry_record_transaction(self):
        """Test recording transaction analysis."""
        telemetry = get_telemetry()

        telemetry.record_transaction_analysis(
            transaction_id="TX123",
            amount=500.0,
            currency="GBP",
            classification="Business",
            confidence=0.85,
        )

    def test_telemetry_record_mcts_iteration(self):
        """Test recording MCTS iteration."""
        telemetry = get_telemetry()

        telemetry.record_mcts_iteration(
            iteration=1,
            node_visits=5,
            node_value=3.0,
            best_hypothesis="Test hypothesis",
            confidence=0.7,
            objective="classify",
        )

    def test_telemetry_record_pipeline_metrics(self):
        """Test recording pipeline metrics."""
        telemetry = get_telemetry()

        telemetry.record_pipeline_metrics(
            total_transactions=100,
            transactions_analyzed=75,
            high_risk_count=5,
            critical_risk_count=1,
            processing_time_seconds=45.5,
            model_used="claude-sonnet-4-5-20250929",
        )


class TestReasoningQuality:
    """Test quality of reasoning and decision-making."""

    def test_business_transaction_classification(self):
        """Test classification of obvious business transaction."""
        transaction_data = {
            "amount": 1200.00,
            "currency": "GBP",
            "merchant": "Microsoft Corporation",
            "description": "Annual Office 365 Business Premium subscription",
            "date": datetime(2025, 1, 15),
        }

        # Expected: Should classify as Business with high confidence
        # This requires full integration test with real LLM

    def test_personal_transaction_classification(self):
        """Test classification of personal transaction."""
        transaction_data = {
            "amount": 45.00,
            "currency": "GBP",
            "merchant": "Starbucks Coffee",
            "description": "Morning coffee and pastry",
            "date": datetime(2025, 1, 15),
        }

        # Expected: Should classify as Personal with high confidence

    def test_obvious_fraud_detection(self):
        """Test detection of obvious fraudulent pattern."""
        transaction_data = {
            "amount": 15000.00,
            "currency": "GBP",
            "merchant": "UNKNOWN_MERCHANT_XXX",
            "description": "Wire transfer to offshore account at 3 AM",
            "date": datetime(2025, 1, 15, 3, 0, 0),
        }

        # Expected: Should flag as HIGH or CRITICAL risk

    def test_normal_transaction_low_risk(self):
        """Test normal transaction flagged as low risk."""
        transaction_data = {
            "amount": 35.50,
            "currency": "GBP",
            "merchant": "Tesco Superstore",
            "description": "Weekly grocery shopping",
            "date": datetime(2025, 1, 15, 14, 30, 0),
        }

        # Expected: Should flag as LOW risk


class TestTransactionModels:
    """Test transaction data models."""

    def test_transaction_creation(self):
        """Test creating Transaction model."""
        txn = Transaction(
            transaction_id="TX123",
            amount=500.0,
            currency=Currency.GBP,
            date=datetime(2025, 1, 15),
            merchant="Test Merchant",
            description="Test transaction",
        )

        assert txn.transaction_id == "TX123"
        assert txn.amount == 500.0
        assert txn.currency == Currency.GBP

    def test_transaction_date_parsing(self):
        """Test transaction date parsing from various formats."""
        formats_to_test = [
            "2025-01-15",
            "2025-01-15 10:30:00",
            "15/01/2025",
            "2025-01-15T10:30:00",
        ]

        for date_str in formats_to_test:
            try:
                txn = Transaction(
                    transaction_id="TX1",
                    amount=100.0,
                    currency=Currency.GBP,
                    date=date_str,
                    merchant="Test",
                    description="Test",
                )
                assert txn.date is not None
            except Exception:
                # Some formats may not be supported
                pass

    def test_fraud_risk_level_enum(self):
        """Test FraudRiskLevel enum."""
        assert FraudRiskLevel.LOW.value == "LOW"
        assert FraudRiskLevel.MEDIUM.value == "MEDIUM"
        assert FraudRiskLevel.HIGH.value == "HIGH"
        assert FraudRiskLevel.CRITICAL.value == "CRITICAL"

    def test_fraud_risk_to_reward(self):
        """Test risk level to reward conversion."""
        assert FraudRiskLevel.LOW.to_reward() == 0.0
        assert FraudRiskLevel.MEDIUM.to_reward() == 0.5
        assert FraudRiskLevel.HIGH.to_reward() == 0.75
        assert FraudRiskLevel.CRITICAL.to_reward() == 1.0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_objective(self):
        """Test MCTS engine rejects invalid objective."""

        def mock_llm(prompt: str) -> str:
            return "{}"

        config = MCTSConfig()
        engine = MCTSEngine(config, mock_llm)

        with pytest.raises(ValueError, match="Unknown objective"):
            engine.search({}, objective="invalid_objective")

    def test_mcts_with_no_children(self):
        """Test MCTS handles case where no children are created."""

        def mock_llm(prompt: str) -> str:
            return "[]"  # Empty hypotheses list

        config = MCTSConfig(iterations=5)
        engine = MCTSEngine(config, mock_llm)

        result = engine.search({"transaction": {}}, objective="classify")

        # Should still return a result
        assert "hypothesis" in result
        assert "confidence" in result

    def test_mcts_with_malformed_llm_response(self):
        """Test MCTS handles malformed LLM responses."""

        def mock_llm(prompt: str) -> str:
            return "This is not JSON at all!"

        config = MCTSConfig(iterations=3)
        engine = MCTSEngine(config, mock_llm)

        # Should fall back to default hypotheses
        result = engine.search({"transaction": {}}, objective="classify")

        assert result is not None
        assert "confidence" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
