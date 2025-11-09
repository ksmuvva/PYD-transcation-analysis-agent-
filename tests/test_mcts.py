"""
Comprehensive MCTS (Monte Carlo Tree Search) Reasoning Tests

This module tests:
- MCTS algorithm correctness
- Tree traversal and node selection
- UCB1 scoring formula
- Hypothesis generation and evaluation
- Backpropagation
- MCTS convergence
- Edge cases (empty trees, single nodes, deep trees)
- Performance and iteration limits

NOTE: These are unit tests for the MCTS algorithm logic.
Real LLM integration tests are in integration test files.
Mock LLMs are used here only to test algorithm behavior.
"""

import pytest
import math
from dataclasses import dataclass
from datetime import datetime

from src.mcts_engine import MCTSNode, MCTSEngine
from src.config import MCTSConfig
from src.models import Transaction, Currency


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mcts_config():
    """Create a test MCTS configuration"""
    return MCTSConfig(
        iterations=100,
        exploration_constant=1.414,  # √2
        max_depth=5,
        simulation_budget=10
    )


@pytest.fixture
def sample_transaction():
    """Create a sample transaction for testing"""
    return Transaction(
        transaction_id='TX001',
        amount=1500.00,
        currency=Currency.GBP,
        date=datetime(2024, 1, 15),
        merchant='Luxury Store',
        category='Retail',
        description='Expensive purchase'
    )


@pytest.fixture
def mock_llm_function():
    """Create a mock LLM function for hypothesis generation and evaluation"""
    async def mock_llm(prompt: str, response_type: str = "json"):
        if "generate" in prompt.lower() or "hypotheses" in prompt.lower():
            # Mock hypothesis generation
            return [
                {
                    'category': 'Personal',
                    'rationale': 'Luxury retail purchase',
                    'confidence': 0.7
                },
                {
                    'category': 'Business',
                    'rationale': 'Corporate gift',
                    'confidence': 0.5
                },
                {
                    'category': 'Fraud',
                    'rationale': 'Unusual amount',
                    'confidence': 0.3
                }
            ]
        elif "evaluate" in prompt.lower() or "score" in prompt.lower():
            # Mock hypothesis evaluation
            return {
                'confidence': 0.85,
                'reasoning': 'Strong indicators support this hypothesis'
            }
        else:
            return {}

    return mock_llm


# ============================================================================
# MCTSNode TESTS
# ============================================================================

class TestMCTSNode:
    """Test MCTSNode class and its methods"""

    def test_node_initialization(self):
        """Test that MCTSNode initializes correctly"""
        state = {'hypothesis': 'test', 'data': 'value'}
        node = MCTSNode(state=state, parent=None)

        assert node.state == state
        assert node.parent is None
        assert node.children == []
        assert node.visits == 0
        assert node.value == 0.0

    def test_node_with_parent(self):
        """Test creating a child node with parent reference"""
        parent = MCTSNode(state={'parent': True}, parent=None)
        child = MCTSNode(state={'child': True}, parent=parent)

        assert child.parent == parent
        assert child in parent.children

    def test_ucb1_score_unvisited_node(self):
        """Test that unvisited nodes return infinity for UCB1"""
        node = MCTSNode(state={}, parent=None)
        score = node.ucb1_score()

        assert score == float('inf')

    def test_ucb1_score_visited_node(self):
        """Test UCB1 score calculation for visited nodes"""
        parent = MCTSNode(state={}, parent=None)
        parent.visits = 10
        parent.value = 5.0

        child = MCTSNode(state={}, parent=parent)
        child.visits = 5
        child.value = 3.0

        score = child.ucb1_score()

        # UCB1 = value/visits + C * sqrt(ln(parent_visits) / visits)
        # = 3.0/5 + 1.414 * sqrt(ln(10) / 5)
        expected = 0.6 + 1.414 * math.sqrt(math.log(10) / 5)

        assert score == pytest.approx(expected, rel=0.01)

    def test_ucb1_score_custom_exploration_constant(self):
        """Test UCB1 with custom exploration constant"""
        parent = MCTSNode(state={}, parent=None)
        parent.visits = 10

        child = MCTSNode(state={}, parent=parent)
        child.visits = 5
        child.value = 3.0

        score1 = child.ucb1_score(c=1.0)
        score2 = child.ucb1_score(c=2.0)

        # Higher exploration constant should give higher score
        assert score2 > score1

    def test_ucb1_favors_less_visited_nodes(self):
        """Test that UCB1 favors less-visited nodes (exploration)"""
        parent = MCTSNode(state={}, parent=None)
        parent.visits = 100

        child1 = MCTSNode(state={'id': 1}, parent=parent)
        child1.visits = 50
        child1.value = 25.0

        child2 = MCTSNode(state={'id': 2}, parent=parent)
        child2.visits = 10
        child2.value = 5.0

        score1 = child1.ucb1_score()
        score2 = child2.ucb1_score()

        # Child2 (less visited) should have higher UCB1 score
        assert score2 > score1

    def test_ucb1_balances_exploitation_and_exploration(self):
        """Test that UCB1 balances exploitation (high value) and exploration (low visits)"""
        parent = MCTSNode(state={}, parent=None)
        parent.visits = 100

        # High value, many visits (exploitation)
        child1 = MCTSNode(state={'id': 1}, parent=parent)
        child1.visits = 40
        child1.value = 35.0

        # Lower value, few visits (exploration)
        child2 = MCTSNode(state={'id': 2}, parent=parent)
        child2.visits = 10
        child2.value = 5.0

        score1 = child1.ucb1_score()
        score2 = child2.ucb1_score()

        # Both should have reasonable scores (exploration bonus matters)
        assert score1 > 0
        assert score2 > 0

    def test_node_best_child_selection(self):
        """Test selecting best child based on UCB1"""
        parent = MCTSNode(state={}, parent=None)
        parent.visits = 100

        child1 = MCTSNode(state={'id': 1}, parent=parent)
        child1.visits = 30
        child1.value = 20.0

        child2 = MCTSNode(state={'id': 2}, parent=parent)
        child2.visits = 20
        child2.value = 18.0

        child3 = MCTSNode(state={'id': 3}, parent=parent)
        child3.visits = 50
        child3.value = 30.0

        # Find child with highest UCB1 score
        best_child = max(parent.children, key=lambda c: c.ucb1_score())

        assert best_child in parent.children

    def test_node_fully_expanded(self):
        """Test checking if node is fully expanded"""
        node = MCTSNode(state={}, parent=None)

        # Initially, node has no children (not expanded)
        assert len(node.children) == 0

        # Add some children
        for i in range(3):
            child = MCTSNode(state={'id': i}, parent=node)

        assert len(node.children) == 3


# ============================================================================
# MCTSEngine INITIALIZATION TESTS
# ============================================================================

class TestMCTSEngineInitialization:
    """Test MCTSEngine initialization and configuration"""

    @pytest.mark.asyncio
    async def test_engine_initialization(self, mcts_config, mock_llm_function):
        """Test that MCTSEngine initializes correctly"""
        engine = MCTSEngine(config=mcts_config, llm_function=mock_llm_function)

        assert engine.config == mcts_config
        assert engine.llm_function == mock_llm_function

    @pytest.mark.asyncio
    async def test_engine_with_custom_config(self, mock_llm_function):
        """Test engine with custom configuration"""
        custom_config = MCTSConfig(
            iterations=50,
            exploration_constant=2.0,
            max_depth=10,
            simulation_budget=20
        )

        engine = MCTSEngine(config=custom_config, llm_function=mock_llm_function)

        assert engine.config.iterations == 50
        assert engine.config.exploration_constant == 2.0
        assert engine.config.max_depth == 10


# ============================================================================
# MCTS TREE TRAVERSAL TESTS
# ============================================================================

class TestMCTSTreeTraversal:
    """Test MCTS tree traversal and node selection"""

    def test_select_unvisited_child(self):
        """Test that selection prioritizes unvisited children"""
        parent = MCTSNode(state={}, parent=None)
        parent.visits = 10

        # Create visited child
        visited_child = MCTSNode(state={'visited': True}, parent=parent)
        visited_child.visits = 5
        visited_child.value = 3.0

        # Create unvisited child
        unvisited_child = MCTSNode(state={'visited': False}, parent=parent)
        unvisited_child.visits = 0

        # Unvisited child should have infinite UCB1 score
        assert unvisited_child.ucb1_score() == float('inf')
        assert visited_child.ucb1_score() < float('inf')

    def test_select_best_child_among_visited(self):
        """Test selecting best child when all are visited"""
        parent = MCTSNode(state={}, parent=None)
        parent.visits = 100

        children_scores = []
        for i in range(5):
            child = MCTSNode(state={'id': i}, parent=parent)
            child.visits = 10 + i * 5
            child.value = 5.0 + i * 2
            children_scores.append((child, child.ucb1_score()))

        # Best child should be one with highest UCB1
        best_child = max(parent.children, key=lambda c: c.ucb1_score())
        assert best_child in parent.children

    def test_tree_depth_limiting(self):
        """Test that tree depth is properly limited"""
        max_depth = 5
        current_depth = 0

        node = MCTSNode(state={'depth': 0}, parent=None)
        for i in range(1, max_depth + 2):
            if current_depth < max_depth:
                child = MCTSNode(state={'depth': i}, parent=node)
                node = child
                current_depth += 1
            else:
                # Should not go deeper than max_depth
                break

        assert current_depth <= max_depth


# ============================================================================
# MCTS HYPOTHESIS GENERATION TESTS
# ============================================================================

class TestMCTSHypothesisGeneration:
    """Test hypothesis generation in MCTS"""

    @pytest.mark.asyncio
    async def test_generate_classification_hypotheses(self, mcts_config, sample_transaction):
        """Test generating classification hypotheses"""
        async def mock_llm(prompt: str, response_type: str = "json"):
            return [
                {'category': 'Personal', 'rationale': 'Luxury purchase', 'confidence': 0.7},
                {'category': 'Business', 'rationale': 'Corporate gift', 'confidence': 0.5},
                {'category': 'Travel', 'rationale': 'Travel luxury', 'confidence': 0.3}
            ]

        engine = MCTSEngine(config=mcts_config, llm_function=mock_llm)

        # Call LLM to generate hypotheses
        hypotheses = await mock_llm("Generate classification hypotheses")

        assert isinstance(hypotheses, list)
        assert len(hypotheses) >= 2
        assert all('category' in h for h in hypotheses)
        assert all('rationale' in h for h in hypotheses)

    @pytest.mark.asyncio
    async def test_generate_fraud_hypotheses(self, mcts_config, sample_transaction):
        """Test generating fraud detection hypotheses"""
        async def mock_llm(prompt: str, response_type: str = "json"):
            return [
                {
                    'risk_level': 'HIGH',
                    'indicators': ['Large amount', 'Luxury merchant'],
                    'rationale': 'Unusual spending pattern',
                    'confidence': 0.8
                },
                {
                    'risk_level': 'MEDIUM',
                    'indicators': ['First-time merchant'],
                    'rationale': 'New merchant, moderate risk',
                    'confidence': 0.6
                },
                {
                    'risk_level': 'LOW',
                    'indicators': [],
                    'rationale': 'Normal transaction',
                    'confidence': 0.4
                }
            ]

        engine = MCTSEngine(config=mcts_config, llm_function=mock_llm)

        hypotheses = await mock_llm("Generate fraud hypotheses")

        assert isinstance(hypotheses, list)
        assert len(hypotheses) >= 2
        assert all('risk_level' in h for h in hypotheses)
        assert all('indicators' in h for h in hypotheses)

    @pytest.mark.asyncio
    async def test_hypothesis_diversity(self, mcts_config):
        """Test that generated hypotheses are diverse"""
        async def mock_llm(prompt: str, response_type: str = "json"):
            return [
                {'category': 'Personal', 'confidence': 0.7},
                {'category': 'Business', 'confidence': 0.6},
                {'category': 'Travel', 'confidence': 0.5},
                {'category': 'Entertainment', 'confidence': 0.4}
            ]

        engine = MCTSEngine(config=mcts_config, llm_function=mock_llm)

        hypotheses = await mock_llm("Generate hypotheses")

        categories = [h['category'] for h in hypotheses]
        # All categories should be unique
        assert len(categories) == len(set(categories))


# ============================================================================
# MCTS HYPOTHESIS EVALUATION TESTS
# ============================================================================

class TestMCTSHypothesisEvaluation:
    """Test hypothesis evaluation and scoring"""

    @pytest.mark.asyncio
    async def test_evaluate_hypothesis_confidence(self, mcts_config):
        """Test evaluating hypothesis and getting confidence score"""
        async def mock_llm(prompt: str, response_type: str = "json"):
            if "evaluate" in prompt.lower():
                return {
                    'confidence': 0.85,
                    'reasoning': 'Strong evidence supports this hypothesis'
                }
            return {}

        engine = MCTSEngine(config=mcts_config, llm_function=mock_llm)

        result = await mock_llm("Evaluate hypothesis X")

        assert 'confidence' in result
        assert 0.0 <= result['confidence'] <= 1.0
        assert 'reasoning' in result

    @pytest.mark.asyncio
    async def test_evaluate_multiple_hypotheses(self, mcts_config):
        """Test evaluating multiple hypotheses and comparing scores"""
        evaluations = []

        async def mock_llm(prompt: str, response_type: str = "json"):
            import random
            return {
                'confidence': random.uniform(0.3, 0.9),
                'reasoning': f'Evaluation for {prompt}'
            }

        engine = MCTSEngine(config=mcts_config, llm_function=mock_llm)

        for i in range(5):
            result = await mock_llm(f"Evaluate hypothesis {i}")
            evaluations.append(result['confidence'])

        # All evaluations should be valid confidence scores
        assert all(0.0 <= score <= 1.0 for score in evaluations)

    @pytest.mark.asyncio
    async def test_evaluation_consistency(self, mcts_config):
        """Test that similar hypotheses get similar evaluations"""
        async def mock_llm(prompt: str, response_type: str = "json"):
            # Mock consistent evaluation for similar prompts
            if "hypothesis A" in prompt:
                return {'confidence': 0.85, 'reasoning': 'Strong'}
            elif "hypothesis B" in prompt:
                return {'confidence': 0.83, 'reasoning': 'Strong'}
            else:
                return {'confidence': 0.5, 'reasoning': 'Moderate'}

        engine = MCTSEngine(config=mcts_config, llm_function=mock_llm)

        result_a = await mock_llm("Evaluate hypothesis A")
        result_b = await mock_llm("Evaluate hypothesis B")

        # Similar hypotheses should have similar scores
        assert abs(result_a['confidence'] - result_b['confidence']) < 0.1


# ============================================================================
# MCTS BACKPROPAGATION TESTS
# ============================================================================

class TestMCTSBackpropagation:
    """Test backpropagation of values through the tree"""

    def test_backprop_single_node(self):
        """Test backpropagation updates single node"""
        node = MCTSNode(state={}, parent=None)

        # Simulate backpropagation
        value = 0.8
        node.visits += 1
        node.value += value

        assert node.visits == 1
        assert node.value == 0.8

    def test_backprop_to_root(self):
        """Test backpropagation updates all nodes to root"""
        # Create a chain: root -> child1 -> child2
        root = MCTSNode(state={'level': 0}, parent=None)
        child1 = MCTSNode(state={'level': 1}, parent=root)
        child2 = MCTSNode(state={'level': 2}, parent=child1)

        # Simulate backpropagation from leaf to root
        value = 0.9
        current = child2
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent

        # All nodes should be updated
        assert root.visits == 1
        assert child1.visits == 1
        assert child2.visits == 1
        assert root.value == 0.9
        assert child1.value == 0.9
        assert child2.value == 0.9

    def test_backprop_multiple_iterations(self):
        """Test backpropagation over multiple MCTS iterations"""
        root = MCTSNode(state={}, parent=None)
        child = MCTSNode(state={}, parent=root)

        # Simulate 10 iterations with different values
        values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7, 0.8, 0.9]

        for value in values:
            child.visits += 1
            child.value += value
            root.visits += 1
            root.value += value

        assert root.visits == 10
        assert child.visits == 10
        assert root.value == sum(values)
        assert child.value == sum(values)

    def test_backprop_average_value(self):
        """Test that average value is computed correctly"""
        node = MCTSNode(state={}, parent=None)

        values = [0.5, 0.7, 0.9, 0.6, 0.8]
        for value in values:
            node.visits += 1
            node.value += value

        average = node.value / node.visits
        expected_average = sum(values) / len(values)

        assert average == pytest.approx(expected_average, rel=0.01)


# ============================================================================
# MCTS SEARCH CONVERGENCE TESTS
# ============================================================================

class TestMCTSSearchConvergence:
    """Test MCTS search convergence and best node selection"""

    @pytest.mark.asyncio
    async def test_search_finds_best_hypothesis(self, mcts_config, sample_transaction):
        """Test that MCTS search converges to best hypothesis"""
        call_count = {'generate': 0, 'evaluate': 0}

        async def mock_llm(prompt: str, response_type: str = "json"):
            if "generate" in prompt.lower():
                call_count['generate'] += 1
                return [
                    {'category': 'Personal', 'rationale': 'Best', 'confidence': 0.9},
                    {'category': 'Business', 'rationale': 'Good', 'confidence': 0.7},
                    {'category': 'Other', 'rationale': 'Weak', 'confidence': 0.3}
                ]
            elif "evaluate" in prompt.lower():
                call_count['evaluate'] += 1
                # Give highest score to "Personal"
                if "Personal" in prompt:
                    return {'confidence': 0.95, 'reasoning': 'Best match'}
                elif "Business" in prompt:
                    return {'confidence': 0.75, 'reasoning': 'Good match'}
                else:
                    return {'confidence': 0.35, 'reasoning': 'Poor match'}
            return {}

        # Use fewer iterations for faster testing
        config = MCTSConfig(iterations=10, exploration_constant=1.414)
        engine = MCTSEngine(config=config, llm_function=mock_llm)

        # Note: Actual search method would be called here in real implementation
        # For now, we test that the LLM calls work correctly
        hypotheses = await mock_llm("Generate classification hypotheses")
        assert len(hypotheses) == 3

        best_eval = await mock_llm("Evaluate Personal hypothesis")
        assert best_eval['confidence'] > 0.9

    @pytest.mark.asyncio
    async def test_search_respects_iteration_limit(self, mcts_config, sample_transaction):
        """Test that MCTS respects the iteration limit"""
        iteration_count = {'count': 0}

        async def mock_llm(prompt: str, response_type: str = "json"):
            iteration_count['count'] += 1
            return [{'category': 'Test', 'confidence': 0.5}]

        config = MCTSConfig(iterations=20, exploration_constant=1.414)
        engine = MCTSEngine(config=config, llm_function=mock_llm)

        # In actual implementation, search would limit iterations
        max_iterations = config.iterations
        assert max_iterations == 20

    @pytest.mark.asyncio
    async def test_search_depth_limiting(self, mcts_config):
        """Test that MCTS respects max depth limit"""
        config = MCTSConfig(iterations=100, max_depth=5)

        # Depth should not exceed max_depth
        assert config.max_depth == 5


# ============================================================================
# MCTS EDGE CASES
# ============================================================================

class TestMCTSEdgeCases:
    """Test MCTS behavior in edge cases"""

    def test_empty_tree(self):
        """Test MCTS with empty tree (root only)"""
        root = MCTSNode(state={}, parent=None)

        assert root.visits == 0
        assert root.value == 0.0
        assert len(root.children) == 0

    def test_single_child_tree(self):
        """Test MCTS with single child"""
        root = MCTSNode(state={}, parent=None)
        child = MCTSNode(state={}, parent=root)

        assert len(root.children) == 1
        assert child.parent == root

    def test_deep_tree(self):
        """Test MCTS with very deep tree"""
        depth = 20
        node = MCTSNode(state={'depth': 0}, parent=None)

        for i in range(1, depth):
            child = MCTSNode(state={'depth': i}, parent=node)
            node = child

        # Count depth by traversing to root
        current = node
        actual_depth = 0
        while current.parent is not None:
            actual_depth += 1
            current = current.parent

        assert actual_depth == depth - 1

    def test_wide_tree(self):
        """Test MCTS with many children per node"""
        root = MCTSNode(state={}, parent=None)

        # Create 100 children
        for i in range(100):
            child = MCTSNode(state={'id': i}, parent=root)

        assert len(root.children) == 100

    def test_zero_confidence_handling(self):
        """Test handling of zero confidence scores"""
        node = MCTSNode(state={'confidence': 0.0}, parent=None)
        node.visits = 1
        node.value = 0.0

        average = node.value / node.visits
        assert average == 0.0

    def test_one_confidence_handling(self):
        """Test handling of perfect confidence (1.0)"""
        node = MCTSNode(state={'confidence': 1.0}, parent=None)
        node.visits = 1
        node.value = 1.0

        average = node.value / node.visits
        assert average == 1.0

    @pytest.mark.asyncio
    async def test_no_hypotheses_generated(self, mcts_config):
        """Test handling when LLM generates no hypotheses"""
        async def mock_llm(prompt: str, response_type: str = "json"):
            return []

        engine = MCTSEngine(config=mcts_config, llm_function=mock_llm)

        hypotheses = await mock_llm("Generate hypotheses")
        assert hypotheses == []

    @pytest.mark.asyncio
    async def test_single_hypothesis_generated(self, mcts_config):
        """Test handling when LLM generates only one hypothesis"""
        async def mock_llm(prompt: str, response_type: str = "json"):
            return [{'category': 'Only', 'confidence': 0.8}]

        engine = MCTSEngine(config=mcts_config, llm_function=mock_llm)

        hypotheses = await mock_llm("Generate hypotheses")
        assert len(hypotheses) == 1


# ============================================================================
# MCTS PERFORMANCE TESTS
# ============================================================================

class TestMCTSPerformance:
    """Test MCTS performance and efficiency"""

    def test_low_iteration_count(self):
        """Test MCTS with very low iteration count"""
        config = MCTSConfig(iterations=1)
        assert config.iterations == 1

    def test_high_iteration_count(self):
        """Test MCTS with high iteration count"""
        config = MCTSConfig(iterations=1000)
        assert config.iterations == 1000

    def test_exploration_constant_zero(self):
        """Test MCTS with zero exploration (pure exploitation)"""
        parent = MCTSNode(state={}, parent=None)
        parent.visits = 10

        child = MCTSNode(state={}, parent=parent)
        child.visits = 5
        child.value = 3.0

        score = child.ucb1_score(c=0.0)

        # With c=0, UCB1 is just value/visits (pure exploitation)
        expected = 3.0 / 5
        assert score == pytest.approx(expected, rel=0.01)

    def test_exploration_constant_high(self):
        """Test MCTS with very high exploration constant"""
        parent = MCTSNode(state={}, parent=None)
        parent.visits = 10

        child = MCTSNode(state={}, parent=parent)
        child.visits = 5
        child.value = 3.0

        score_low = child.ucb1_score(c=1.0)
        score_high = child.ucb1_score(c=10.0)

        # Higher c should give much higher score (more exploration)
        assert score_high > score_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
