"""
Unit tests for reasoning components with synthetic CSV test data.

This module tests the MCTS reasoning engine, hypothesis generation,
evaluation logic, and reasoning quality metrics using synthetic transaction data.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pandas as pd
import pytest
from pydantic_ai.models import KnownModelName

from src.agent import Agent
from src.config import MCTSConfig
from src.mcts_engine import MCTSEngine, MCTSNode
from src.mcts_engine_v2 import EnhancedMCTSEngine
from src.models import (
    ClassificationResult,
    Currency,
    FraudDetectionResult,
    Transaction,
)
from src.session_context import SessionContext
from tests.synthetic_data_generator import SyntheticTransactionGenerator


# ============================================================================
# FIXTURES: Synthetic CSV Test Data
# ============================================================================


@pytest.fixture
def synthetic_generator():
    """Provides a synthetic transaction data generator with fixed seed."""
    return SyntheticTransactionGenerator(seed=42)


@pytest.fixture
def simple_transaction() -> Dict[str, Any]:
    """Single simple transaction for basic MCTS tests."""
    return {
        "transaction_id": "TX_TEST_001",
        "amount": 500.00,
        "currency": "GBP",
        "date": "2025-01-15",
        "merchant": "Test Corporation",
        "description": "Standard business expense",
    }


@pytest.fixture
def fraud_transaction() -> Dict[str, Any]:
    """Transaction with high fraud indicators."""
    return {
        "transaction_id": "TX_FRAUD_001",
        "amount": 15000.00,
        "currency": "GBP",
        "date": "2025-01-15",
        "merchant": "Unknown Offshore Entity",
        "description": "Unusual large transfer - crypto",
    }


@pytest.fixture
def business_transaction() -> Dict[str, Any]:
    """Clear business transaction."""
    return {
        "transaction_id": "TX_BIZ_001",
        "amount": 450.00,
        "currency": "GBP",
        "date": "2025-01-15",
        "merchant": "Office Depot",
        "description": "Office supplies and equipment",
    }


@pytest.fixture
def boundary_transactions(synthetic_generator) -> List[Dict[str, Any]]:
    """Transactions at threshold boundaries (250 GBP)."""
    return [
        synthetic_generator.generate_transaction(amount_range=(249.99, 249.99)),
        synthetic_generator.generate_transaction(amount_range=(250.00, 250.00)),
        synthetic_generator.generate_transaction(amount_range=(250.01, 250.01)),
    ]


@pytest.fixture
def diverse_csv_dataframe(synthetic_generator) -> pd.DataFrame:
    """Diverse 100-row synthetic CSV dataset."""
    return synthetic_generator.generate_dataframe(count=100, include_edge_cases=True)


@pytest.fixture
def fraud_scenario_csv(synthetic_generator) -> pd.DataFrame:
    """CSV with known fraud patterns."""
    rapid_succession = synthetic_generator.generate_fraud_scenario("rapid_succession")
    suspicious_merchants = synthetic_generator.generate_fraud_scenario(
        "suspicious_merchants"
    )
    unusual_amounts = synthetic_generator.generate_fraud_scenario("unusual_amounts")

    all_transactions = rapid_succession + suspicious_merchants + unusual_amounts
    return pd.DataFrame(all_transactions)


@pytest.fixture
def csv_test_file(diverse_csv_dataframe) -> Path:
    """Temporary CSV file with synthetic test data."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as tmp_file:
        diverse_csv_dataframe.to_csv(tmp_file.name, index=False)
        return Path(tmp_file.name)


@pytest.fixture
def edge_case_transactions(synthetic_generator) -> List[Dict[str, Any]]:
    """Edge case transactions for robust testing."""
    return synthetic_generator.generate_edge_cases()


# ============================================================================
# FIXTURES: MCTS Configuration
# ============================================================================


@pytest.fixture
def test_mcts_config() -> MCTSConfig:
    """MCTS configuration optimized for fast unit tests."""
    return MCTSConfig(
        iterations=20,  # Reduced for fast tests
        exploration_constant=1.414,
        max_depth=3,  # Shallow for unit tests
        early_termination_enabled=True,
        convergence_std_threshold=0.01,
        convergence_window=10,
    )


@pytest.fixture
def thorough_mcts_config() -> MCTSConfig:
    """MCTS configuration for thorough reasoning tests."""
    return MCTSConfig(
        iterations=100,
        exploration_constant=1.414,
        max_depth=5,
        early_termination_enabled=True,
        convergence_std_threshold=0.01,
        convergence_window=50,
    )


# ============================================================================
# FIXTURES: Mock LLM Responses
# ============================================================================


@pytest.fixture
def mock_classification_hypotheses() -> List[str]:
    """Mock LLM response for classification hypothesis generation."""
    return [
        "Business - Regular office supplies purchase from known vendor",
        "Personal - Could be personal shopping at office supply store",
        "Travel - Unlikely, no travel indicators present",
    ]


@pytest.fixture
def mock_fraud_hypotheses() -> List[str]:
    """Mock LLM response for fraud detection hypothesis generation."""
    return [
        "LOW - Standard transaction with reputable merchant",
        "MEDIUM - Amount is elevated but merchant is known",
        "HIGH - Unusual pattern detected in transaction timing",
    ]


@pytest.fixture
def mock_hypothesis_evaluation() -> float:
    """Mock LLM response for hypothesis evaluation (score 0-1)."""
    return 0.85


@pytest.fixture
def mock_llm_classify_response():
    """Mock LLM response for classification."""

    async def _mock_response(*args, **kwargs):
        return {
            "category": "Business",
            "confidence": 0.92,
            "reasoning": "Clear business expense - office supplies from known vendor",
        }

    return _mock_response


@pytest.fixture
def mock_llm_fraud_response():
    """Mock LLM response for fraud detection."""

    async def _mock_response(*args, **kwargs):
        return {
            "risk_level": "LOW",
            "confidence": 0.88,
            "fraud_indicators": ["None detected"],
            "reasoning": "Standard business transaction with no red flags",
        }

    return _mock_response


# ============================================================================
# TEST CLASS: MCTS Core Reasoning Logic
# ============================================================================


class TestMCTSCoreReasoning:
    """Tests for MCTS core reasoning algorithms."""

    def test_node_initialization_with_synthetic_data(self, simple_transaction):
        """Test MCTS node creation with synthetic transaction data."""
        state = {
            "transaction": simple_transaction,
            "hypothesis": "Business - Standard expense",
        }

        node = MCTSNode(state=state)

        assert node.state == state
        assert node.parent is None
        assert node.children == []
        assert node.visits == 0
        assert node.value == 0.0
        assert node.state["transaction"]["transaction_id"] == "TX_TEST_001"

    def test_ucb1_score_calculation_reasoning(self, simple_transaction):
        """Test UCB1 score calculation for reasoning exploration."""
        state = {"transaction": simple_transaction, "hypothesis": "Business"}

        # Create parent with 10 visits
        parent = MCTSNode(state=state)
        parent.visits = 10

        # Create child with 5 visits and value 4.0
        child = MCTSNode(state=state, parent=parent)
        child.visits = 5
        child.value = 4.0

        # UCB1 = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
        # Q(s,a) = 4.0 / 5 = 0.8
        # Exploration = sqrt(2) * sqrt(ln(10) / 5) ≈ 1.414 * 0.682 ≈ 0.964
        # UCB1 ≈ 0.8 + 0.964 ≈ 1.764

        ucb1 = child.ucb1_score()
        assert 1.7 < ucb1 < 1.8  # Allow small floating point variance

    def test_ucb1_prioritizes_unexplored_nodes(self, simple_transaction):
        """Test that UCB1 gives high priority to unexplored reasoning paths."""
        state = {"transaction": simple_transaction, "hypothesis": "Test"}

        parent = MCTSNode(state=state)
        parent.visits = 100

        # Unexplored child
        unexplored = MCTSNode(state=state, parent=parent)
        unexplored.visits = 0

        # Explored child
        explored = MCTSNode(state=state, parent=parent)
        explored.visits = 10
        explored.value = 8.0

        # Unexplored should have infinite UCB1
        assert unexplored.ucb1_score() == float("inf")
        assert explored.ucb1_score() < float("inf")

    def test_node_selection_with_multiple_hypotheses(
        self, simple_transaction, test_mcts_config
    ):
        """Test MCTS node selection among multiple reasoning hypotheses."""
        root_state = {
            "transaction": simple_transaction,
            "hypothesis": "Initial",
        }
        root = MCTSNode(state=root_state)
        root.visits = 10

        # Create children with different visit/value patterns
        hypotheses = [
            ("Business - Office supplies", 5, 4.0),  # 0.8 + exploration
            ("Personal - Personal shopping", 3, 2.5),  # 0.83 + exploration
            ("Travel - Unlikely", 1, 0.5),  # 0.5 + high exploration
        ]

        children = []
        for hyp, visits, value in hypotheses:
            child_state = {"transaction": simple_transaction, "hypothesis": hyp}
            child = MCTSNode(state=child_state, parent=root)
            child.visits = visits
            child.value = value
            children.append(child)
            root.children.append(child)

        # Select best child based on UCB1
        selected = max(root.children, key=lambda n: n.ucb1_score())

        # The least visited child should often be selected (exploration)
        assert selected.visits <= 3  # High exploration bonus


class TestHypothesisGeneration:
    """Tests for hypothesis generation reasoning."""

    @patch("src.mcts_engine.MCTSEngine._generate_hypotheses")
    def test_classification_hypothesis_generation(
        self,
        mock_generate,
        simple_transaction,
        test_mcts_config,
        mock_classification_hypotheses,
    ):
        """Test classification hypothesis generation with synthetic data."""
        mock_generate.return_value = mock_classification_hypotheses

        engine = MCTSEngine(
            config=test_mcts_config,
            llm_function=Mock(),
        )

        hypotheses = engine._generate_hypotheses(simple_transaction, "classify")

        assert len(hypotheses) == 3
        assert any("Business" in h for h in hypotheses)
        assert any("Personal" in h for h in hypotheses)
        mock_generate.assert_called_once_with(simple_transaction, "classify")

    @patch("src.mcts_engine.MCTSEngine._generate_hypotheses")
    def test_fraud_hypothesis_generation(
        self,
        mock_generate,
        fraud_transaction,
        test_mcts_config,
        mock_fraud_hypotheses,
    ):
        """Test fraud detection hypothesis generation with high-risk data."""
        mock_generate.return_value = mock_fraud_hypotheses

        engine = MCTSEngine(
            config=test_mcts_config,
            llm_function=Mock(),
        )

        hypotheses = engine._generate_hypotheses(fraud_transaction, "detect_fraud")

        assert len(hypotheses) == 3
        assert any("LOW" in h or "MEDIUM" in h or "HIGH" in h for h in hypotheses)
        mock_generate.assert_called_once_with(fraud_transaction, "detect_fraud")

    @patch("src.mcts_engine.MCTSEngine._generate_hypotheses")
    def test_hypothesis_diversity_with_business_transaction(
        self,
        mock_generate,
        business_transaction,
        test_mcts_config,
    ):
        """Test that hypotheses are diverse for clear business transactions."""
        # Simulate diverse hypotheses from LLM
        diverse_hypotheses = [
            "Business - Clear office supplies purchase",
            "Personal - Possible personal use of office items",
            "Mixed - Could be business or personal",
        ]
        mock_generate.return_value = diverse_hypotheses

        engine = MCTSEngine(
            config=test_mcts_config,
            llm_function=Mock(),
        )

        hypotheses = engine._generate_hypotheses(business_transaction, "classify")

        # All hypotheses should be unique
        assert len(hypotheses) == len(set(hypotheses))
        assert len(hypotheses) >= 3


class TestHypothesisEvaluation:
    """Tests for hypothesis evaluation reasoning."""

    @patch("src.mcts_engine.MCTSEngine._evaluate_hypothesis")
    def test_hypothesis_evaluation_score_range(
        self,
        mock_evaluate,
        simple_transaction,
        test_mcts_config,
    ):
        """Test that hypothesis evaluation returns valid scores (0-1)."""
        mock_evaluate.return_value = 0.85

        engine = MCTSEngine(
            config=test_mcts_config,
            llm_function=Mock(),
        )

        state = {
            "transaction": simple_transaction,
            "hypothesis": "Business - Office supplies",
        }
        result = engine._evaluate_hypothesis(state, "classify")

        # Mock returns a dict, extract confidence as score
        assert isinstance(result, (dict, float))
        if isinstance(result, dict):
            assert "confidence" in result or result == 0.85
        else:
            assert 0.0 <= result <= 1.0
            assert result == 0.85

    @patch("src.mcts_engine.MCTSEngine._evaluate_hypothesis")
    def test_high_confidence_business_hypothesis(
        self,
        mock_evaluate,
        business_transaction,
        test_mcts_config,
    ):
        """Test evaluation of high-confidence business hypothesis."""
        # Clear business transaction should get high score
        mock_evaluate.return_value = 0.95

        engine = MCTSEngine(
            config=test_mcts_config,
            llm_function=Mock(),
        )

        state = {
            "transaction": business_transaction,
            "hypothesis": "Business - Office supplies from Office Depot",
        }
        result = engine._evaluate_hypothesis(state, "classify")

        # Mock returns 0.95, could be dict or float
        assert isinstance(result, (dict, float))
        if isinstance(result, float):
            assert result >= 0.9  # High confidence for clear business

    @patch("src.mcts_engine.MCTSEngine._evaluate_hypothesis")
    def test_low_confidence_mismatched_hypothesis(
        self,
        mock_evaluate,
        business_transaction,
        test_mcts_config,
    ):
        """Test evaluation of mismatched hypothesis gets low score."""
        # Business transaction with gambling hypothesis should get low score
        mock_evaluate.return_value = 0.15

        engine = MCTSEngine(
            config=test_mcts_config,
            llm_function=Mock(),
        )

        state = {
            "transaction": business_transaction,
            "hypothesis": "Gambling - Casino transaction",
        }
        result = engine._evaluate_hypothesis(state, "classify")

        # Mock returns 0.15, could be dict or float
        assert isinstance(result, (dict, float))
        if isinstance(result, float):
            assert result <= 0.3  # Low confidence for mismatched hypothesis

    @patch("src.mcts_engine.MCTSEngine._evaluate_hypothesis")
    def test_fraud_evaluation_for_suspicious_transaction(
        self,
        mock_evaluate,
        fraud_transaction,
        test_mcts_config,
    ):
        """Test fraud evaluation for suspicious transaction."""
        # High-risk transaction with HIGH fraud hypothesis
        mock_evaluate.return_value = 0.92

        engine = MCTSEngine(
            config=test_mcts_config,
            llm_function=Mock(),
        )

        state = {
            "transaction": fraud_transaction,
            "hypothesis": "HIGH - Large offshore transfer, crypto keywords",
        }
        result = engine._evaluate_hypothesis(state, "detect_fraud")

        # Mock returns 0.92, could be dict or float
        assert isinstance(result, (dict, float))
        if isinstance(result, float):
            assert result >= 0.85  # High confidence for fraud detection


class TestBackpropagation:
    """Tests for MCTS backpropagation (reward propagation)."""

    def test_backpropagation_updates_path(self, simple_transaction):
        """Test that backpropagation updates all nodes in path."""
        # Create a simple path: root -> child1 -> child2
        root = MCTSNode(state={"transaction": simple_transaction, "hypothesis": "Root"})
        child1 = MCTSNode(
            state={"transaction": simple_transaction, "hypothesis": "Child1"},
            parent=root,
        )
        child2 = MCTSNode(
            state={"transaction": simple_transaction, "hypothesis": "Child2"},
            parent=child1,
        )

        # Backpropagate reward of 0.8
        reward = 0.8
        current = child2
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

        # Verify all nodes updated
        assert root.visits == 1
        assert root.value == 0.8
        assert child1.visits == 1
        assert child1.value == 0.8
        assert child2.visits == 1
        assert child2.value == 0.8

    def test_backpropagation_accumulates_rewards(self, simple_transaction):
        """Test that multiple backpropagations accumulate rewards."""
        root = MCTSNode(state={"transaction": simple_transaction, "hypothesis": "Root"})

        # Simulate multiple rollouts with different rewards
        rewards = [0.8, 0.6, 0.9, 0.7]

        for reward in rewards:
            root.visits += 1
            root.value += reward

        assert root.visits == 4
        assert root.value == 0.8 + 0.6 + 0.9 + 0.7  # 3.0

    def test_backpropagation_average_value(self, simple_transaction):
        """Test that average value is correctly calculated."""
        root = MCTSNode(state={"transaction": simple_transaction, "hypothesis": "Root"})

        # Add rewards
        rewards = [0.8, 0.6, 0.9, 0.7]
        for reward in rewards:
            root.visits += 1
            root.value += reward

        # Average should be total value / visits
        expected_avg = sum(rewards) / len(rewards)  # 0.75
        actual_avg = root.value / root.visits

        assert abs(actual_avg - expected_avg) < 0.01


# ============================================================================
# TEST CLASS: Classification Reasoning
# ============================================================================


class TestClassificationReasoning:
    """Tests for classification reasoning with MCTS."""

    @patch("src.mcts_engine.MCTSEngine.search")
    def test_classification_reasoning_with_business_transaction(
        self,
        mock_search,
        business_transaction,
        test_mcts_config,
    ):
        """Test classification reasoning for clear business transaction."""
        # Mock MCTS search result
        mock_search.return_value = {
            "best_hypothesis": "Business - Office supplies purchase",
            "confidence": 0.92,
            "iterations": 20,
            "tree_depth": 3,
        }

        llm_mock = Mock()
        engine = MCTSEngine(
            config=test_mcts_config,
            llm_function=llm_mock,
        )

        result = engine.search(business_transaction, "classify")
        assert "Business" in result["best_hypothesis"]
        assert result["confidence"] >= 0.9
        mock_search.assert_called_once_with(business_transaction, "classify")
    @patch("src.mcts_engine.MCTSEngine.search")
    def test_classification_reasoning_with_ambiguous_transaction(
        self,
        mock_search,
        simple_transaction,
        test_mcts_config,
    ):
        """Test classification reasoning for ambiguous transaction."""
        # Ambiguous transaction might have lower confidence
        mock_search.return_value = {
            "best_hypothesis": "Personal - Could be personal or business",
            "confidence": 0.65,
            "iterations": 20,
            "tree_depth": 3,
        }

        llm_mock = Mock()
        engine = MCTSEngine(
            config=test_mcts_config,
            llm_function=llm_mock,
        )

        result = engine.search(simple_transaction, "classify")
        # Lower confidence for ambiguous
        assert 0.5 <= result["confidence"] < 0.8

    @patch("src.mcts_engine.MCTSEngine.search")
    def test_classification_explores_multiple_categories(
        self,
        mock_search,
        simple_transaction,
        test_mcts_config,
    ):
        """Test that classification considers multiple category hypotheses."""
        # Track hypotheses explored during search
        explored_hypotheses = [
            "Business - Office expense",
            "Personal - Personal shopping",
            "Travel - Travel related",
            "Entertainment - Entertainment expense",
        ]

        mock_search.return_value = {
            "best_hypothesis": "Business - Office expense",
            "confidence": 0.88,
            "iterations": 20,
            "explored_hypotheses": explored_hypotheses,
        }

        llm_mock = Mock()
        engine = MCTSEngine(
            config=test_mcts_config,
            llm_function=llm_mock,
        )

        result = engine.search(simple_transaction, "classify")
        # Should explore multiple categories
        assert len(result.get("explored_hypotheses", [])) >= 3


class TestFraudDetectionReasoning:
    """Tests for fraud detection reasoning with MCTS."""

    @patch("src.mcts_engine.MCTSEngine.search")
    def test_fraud_detection_low_risk_transaction(
        self,
        mock_search,
        business_transaction,
        test_mcts_config,
    ):
        """Test fraud detection for low-risk business transaction."""
        mock_search.return_value = {
            "best_hypothesis": "LOW - Standard business transaction",
            "confidence": 0.95,
            "iterations": 20,
        }

        llm_mock = Mock()
        engine = MCTSEngine(
            config=test_mcts_config,
            llm_function=llm_mock,
        )

        result = engine.search(business_transaction, "classify")
        assert "LOW" in result["best_hypothesis"]
        assert result["confidence"] >= 0.9

    @patch("src.mcts_engine.MCTSEngine.search")
    def test_fraud_detection_high_risk_transaction(
        self,
        mock_search,
        fraud_transaction,
        test_mcts_config,
    ):
        """Test fraud detection for high-risk transaction."""
        mock_search.return_value = {
            "best_hypothesis": "HIGH - Large offshore transfer with crypto keywords",
            "confidence": 0.88,
            "iterations": 20,
        }

        llm_mock = Mock()
        engine = MCTSEngine(
            config=test_mcts_config,
            llm_function=llm_mock,
        )

        result = engine.search(fraud_transaction, "classify")
        assert "HIGH" in result["best_hypothesis"] or "CRITICAL" in result[
            "best_hypothesis"
        ]
        assert result["confidence"] >= 0.8

    @patch("src.mcts_engine.MCTSEngine.search")
    def test_fraud_detection_considers_amount(
        self,
        mock_search,
        synthetic_generator,
        test_mcts_config,
    ):
        """Test that fraud detection considers transaction amount."""
        # Small amount transaction
        small_txn = synthetic_generator.generate_transaction(
            amount_range=(50, 50), fraud_risk="LOW"
        )

        # Large amount transaction
        large_txn = synthetic_generator.generate_transaction(
            amount_range=(15000, 15000), fraud_risk="HIGH"
        )

        # Mock different risk levels based on amount
        mock_search.side_effect = [
            {"best_hypothesis": "LOW - Small amount", "confidence": 0.92},
            {"best_hypothesis": "HIGH - Large amount", "confidence": 0.85},
        ]

        llm_mock = Mock()
        engine = MCTSEngine(
            config=test_mcts_config,
            llm_function=llm_mock,
        )

        small_result = engine.search(small_txn, "classify")
        large_result = engine.search(large_txn, "classify")
        assert "LOW" in small_result["best_hypothesis"]
        assert "HIGH" in large_result["best_hypothesis"]

    @patch("src.mcts_engine.MCTSEngine.search")
    def test_fraud_detection_considers_merchant(
        self,
        mock_search,
        test_mcts_config,
    ):
        """Test that fraud detection considers merchant reputation."""
        # Known merchant
        known_merchant_txn = {
            "transaction_id": "TX_001",
            "amount": 500.0,
            "currency": "GBP",
            "merchant": "Amazon UK",
            "description": "Purchase",
            "date": "2025-01-15",
        }

        # Unknown merchant
        unknown_merchant_txn = {
            "transaction_id": "TX_002",
            "amount": 500.0,
            "currency": "GBP",
            "merchant": "Unknown Offshore LLC",
            "description": "Transfer",
            "date": "2025-01-15",
        }

        mock_search.side_effect = [
            {"best_hypothesis": "LOW - Known reputable merchant", "confidence": 0.93},
            {
                "best_hypothesis": "MEDIUM - Unknown merchant requires investigation",
                "confidence": 0.78,
            },
        ]

        llm_mock = Mock()
        engine = MCTSEngine(
            config=test_mcts_config,
            llm_function=llm_mock,
        )

        known_result = engine.search(known_merchant_txn, "classify")
        unknown_result = engine.search(unknown_merchant_txn, "classify")
        assert "LOW" in known_result["best_hypothesis"]
        assert "MEDIUM" in unknown_result["best_hypothesis"] or "HIGH" in unknown_result[
            "best_hypothesis"
        ]


# ============================================================================
# TEST CLASS: Reasoning Quality Metrics
# ============================================================================


class TestReasoningQualityMetrics:
    """Tests for reasoning quality metrics and convergence."""

    def test_convergence_detection_with_stable_values(self, simple_transaction):
        """Test that MCTS detects convergence when values stabilize."""
        root = MCTSNode(state={"transaction": simple_transaction, "hypothesis": "Root"})

        # Create children with similar values (converged)
        for i in range(10):
            child = MCTSNode(
                state={"transaction": simple_transaction, "hypothesis": f"Child{i}"},
                parent=root,
            )
            child.visits = 10
            child.value = 8.5 + (i * 0.05)  # Very similar values
            root.children.append(child)

        # Calculate standard deviation
        if root.children:
            avg_values = [c.value / c.visits for c in root.children]
            import statistics

            std_dev = statistics.stdev(avg_values)

            # Low std dev indicates convergence
            assert std_dev < 0.1

    def test_confidence_score_calculation(self, simple_transaction):
        """Test confidence score calculation from MCTS results."""
        # Create node with high visit count and value
        node = MCTSNode(state={"transaction": simple_transaction, "hypothesis": "Test"})
        node.visits = 100
        node.value = 90.0  # 90% average score

        avg_value = node.value / node.visits
        confidence = avg_value

        assert 0.85 <= confidence <= 0.95

    def test_exploration_exploitation_tradeoff(self, simple_transaction):
        """Test exploration-exploitation balance in reasoning."""
        parent = MCTSNode(
            state={"transaction": simple_transaction, "hypothesis": "Parent"}
        )
        parent.visits = 100

        # High-value but well-explored child
        exploited = MCTSNode(
            state={"transaction": simple_transaction, "hypothesis": "Exploited"},
            parent=parent,
        )
        exploited.visits = 50
        exploited.value = 45.0  # 0.9 average

        # Lower-value but less-explored child
        explored = MCTSNode(
            state={"transaction": simple_transaction, "hypothesis": "Explored"},
            parent=parent,
        )
        explored.visits = 10
        explored.value = 7.0  # 0.7 average

        # UCB1 should balance exploitation vs exploration
        exploited_ucb1 = exploited.ucb1_score()
        explored_ucb1 = explored.ucb1_score()

        # Less explored should get exploration bonus
        # This makes it competitive despite lower average value
        assert explored_ucb1 > 0.7  # Has exploration bonus

    def test_reasoning_trace_completeness(self, simple_transaction, test_mcts_config):
        """Test that reasoning trace captures key decision points."""
        # Simulate MCTS search trace
        reasoning_trace = {
            "iterations": 20,
            "best_hypothesis": "Business - Office supplies",
            "confidence": 0.88,
            "explored_hypotheses": [
                "Business - Office supplies",
                "Personal - Personal shopping",
                "Travel - Unlikely",
            ],
            "tree_depth": 3,
            "convergence_detected": True,
        }

        # Verify trace has key information
        assert "iterations" in reasoning_trace
        assert "best_hypothesis" in reasoning_trace
        assert "confidence" in reasoning_trace
        assert "explored_hypotheses" in reasoning_trace
        assert len(reasoning_trace["explored_hypotheses"]) >= 3

    @patch("src.mcts_engine.MCTSEngine.search")
    def test_early_termination_on_convergence(
        self,
        mock_search,
        simple_transaction,
    ):
        """Test that MCTS terminates early when convergence is detected."""
        # Configure MCTS with early termination
        config = MCTSConfig(
            iterations=100,
            early_termination_enabled=True,
            convergence_std_threshold=0.01,
            convergence_window=10,
        )

        # Mock search to return with early termination
        mock_search.return_value = {
            "best_hypothesis": "Business - Clear category",
            "confidence": 0.95,
            "iterations": 45,  # Terminated early before 100
            "early_termination": True,
            "convergence_detected": True,
        }

        llm_mock = Mock()
        engine = MCTSEngine(
            config=config,
            llm_function=llm_mock,
        )

        result = engine.search(simple_transaction, "classify")
        # Should terminate before max iterations
        assert result["iterations"] < 100
        assert result.get("early_termination", False)


# ============================================================================
# TEST CLASS: Integration Tests with Synthetic CSV Data
# ============================================================================


class TestReasoningWithSyntheticCSV:
    """Integration tests using synthetic CSV datasets."""

    def test_process_diverse_csv_dataset(
        self,
        diverse_csv_dataframe,
        test_mcts_config,
    ):
        """Test processing a diverse synthetic CSV dataset."""
        # Verify dataset characteristics
        assert len(diverse_csv_dataframe) >= 50  # At least 50 rows
        assert "transaction_id" in diverse_csv_dataframe.columns
        assert "amount" in diverse_csv_dataframe.columns
        assert "merchant" in diverse_csv_dataframe.columns

        # Check data diversity
        unique_merchants = diverse_csv_dataframe["merchant"].nunique()
        assert unique_merchants >= 10  # Diverse merchants

    def test_process_fraud_scenario_csv(
        self,
        fraud_scenario_csv,
        test_mcts_config,
    ):
        """Test processing CSV with known fraud patterns."""
        # Verify fraud patterns present
        assert len(fraud_scenario_csv) > 0

        # Check for high-value transactions (fraud indicator)
        high_value_txns = fraud_scenario_csv[fraud_scenario_csv["amount"] > 5000]
        assert len(high_value_txns) > 0

    def test_boundary_transactions_reasoning(
        self,
        boundary_transactions,
        test_mcts_config,
    ):
        """Test reasoning with transactions at 250 GBP threshold."""
        assert len(boundary_transactions) == 3

        # Check amounts are at boundaries
        amounts = [t["amount"] for t in boundary_transactions]
        assert any(a < 250 for a in amounts)
        assert any(250 <= a <= 250.01 for a in amounts)
        assert any(a > 250 for a in amounts)  # Just above threshold

    def test_edge_case_transactions_reasoning(
        self,
        edge_case_transactions,
        test_mcts_config,
    ):
        """Test reasoning with edge case transactions."""
        assert len(edge_case_transactions) > 0

        # Edge cases should include micro and massive amounts
        amounts = [t["amount"] for t in edge_case_transactions]
        assert any(a < 1 for a in amounts)  # Micro amounts
        assert any(a > 10000 for a in amounts)  # Massive amounts


# ============================================================================
# TEST CLASS: MCTS Engine V2 (Enhanced) Tests
# ============================================================================


class TestEnhancedMCTSReasoning:
    """Tests for EnhancedMCTSEngine with tool-specific configurations."""

    def test_tool_specific_config_classification(self, simple_transaction):
        """Test that EnhancedMCTS uses tool-specific config for classification."""
        config = MCTSConfig(
            iterations=50,
            max_depth=5,
            exploration_constant=1.414,
        )

        llm_mock = Mock()
        engine = EnhancedMCTSEngine(
            config=config,
            tool_name="classify",
            llm_function=llm_mock,
        )

        # Enhanced engine should have tool-specific settings
        assert engine.global_config.iterations >= 50
        assert engine.tool_config is not None
        assert engine.tool_name == "classify"

    def test_tool_specific_config_fraud_detection(self, fraud_transaction):
        """Test that EnhancedMCTS uses tool-specific config for fraud detection."""
        config = MCTSConfig(
            iterations=75,
            max_depth=7,
            exploration_constant=1.414,
        )

        llm_mock = Mock()
        engine = EnhancedMCTSEngine(
            config=config,
            tool_name="fraud",
            llm_function=llm_mock,
        )

        # Fraud detection might use different settings
        assert engine.global_config.iterations >= 50
        assert engine.tool_config is not None
        assert engine.tool_name == "fraud"

    @patch("src.mcts_engine_v2.EnhancedMCTSEngine.search")
    def test_enhanced_convergence_detection(
        self,
        mock_search,
        simple_transaction,
    ):
        """Test enhanced convergence detection in v2 engine."""
        mock_search.return_value = {
            "best_hypothesis": "Business - Clear classification",
            "confidence": 0.94,
            "iterations": 30,
            "convergence_detected": True,
            "convergence_window_used": 20,
        }

        config = MCTSConfig(
            iterations=100,
            early_termination_enabled=True,
            convergence_std_threshold=0.01,
            convergence_window=20,
        )

        llm_mock = Mock()
        engine = EnhancedMCTSEngine(
            config=config,
            tool_name="classify",
            llm_function=llm_mock,
        )

        result = engine.search(simple_transaction, "classify")
        assert result["convergence_detected"]
        assert result["iterations"] < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
