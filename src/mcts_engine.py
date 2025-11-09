"""
Monte Carlo Tree Search (MCTS) reasoning engine with LLM integration.

Implements MCTS algorithm for transaction classification and fraud detection.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from src.config import MCTSConfig
from src.telemetry import get_telemetry


@dataclass
class MCTSNode:
    """
    Represents a node in the MCTS tree.

    Attributes:
        state: Current state (transaction data + hypothesis)
        parent: Parent node (None for root)
        children: List of child nodes
        visits: Number of times this node has been visited
        value: Accumulated value from simulations
    """

    state: dict[str, Any]
    parent: "MCTSNode | None" = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0

    def __post_init__(self):
        """Add this node to parent's children if parent is provided."""
        if self.parent is not None:
            self.parent.children.append(self)

    def ucb1_score(self, exploration_constant: float = 1.414, c: float | None = None) -> float:
        """
        Calculate Upper Confidence Bound (UCB1) score.

        UCB1 balances exploitation (high value) and exploration (low visits).

        Formula: value/visits + C * sqrt(ln(parent_visits) / visits)

        Args:
            exploration_constant: Exploration parameter (default √2 ≈ 1.414)
            c: Alias for exploration_constant (for convenience)

        Returns:
            UCB1 score (infinity for unvisited nodes)
        """
        # Allow 'c' as an alias for exploration_constant
        if c is not None:
            exploration_constant = c

        if self.visits == 0:
            return float("inf")

        if self.parent is None:
            return self.value / self.visits

        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration


class MCTSEngine:
    """
    MCTS reasoning engine with LLM integration.

    Uses Monte Carlo Tree Search to explore hypothesis space guided by LLM.
    """

    def __init__(self, config: MCTSConfig, llm_function: Callable[[str], str]):
        """
        Initialize MCTS engine.

        Args:
            config: MCTS configuration parameters
            llm_function: Function to call LLM (takes prompt, returns response)
        """
        self.config = config
        self.llm_function = llm_function
        self.root: MCTSNode | None = None

    def search(self, initial_state: dict[str, Any], objective: str) -> dict[str, Any]:
        """
        Perform MCTS search to find best hypothesis.

        Args:
            initial_state: Starting state (transaction + context)
            objective: "classify" or "detect_fraud"

        Returns:
            Best result with hypothesis, confidence, and reasoning
        """
        telemetry = get_telemetry()

        with telemetry.span(
            f"mcts_search_{objective}",
            objective=objective,
            max_iterations=self.config.iterations,
            exploration_constant=self.config.exploration_constant,
        ):
            self.root = MCTSNode(state=initial_state)

            # Run MCTS iterations
            for iteration in range(self.config.iterations):
                # 1. Selection: Traverse tree using UCB1
                node = self._select(self.root)

                # 2. Expansion: Generate new hypotheses if needed
                if node.visits > 0 and not node.children:
                    node = self._expand(node, objective)

                # 3. Simulation: Evaluate hypothesis using LLM
                reward = self._simulate(node, objective)

                # 4. Backpropagation: Update values up the tree
                self._backpropagate(node, reward)

                # Record iteration metrics
                best_hypothesis = node.state.get("hypothesis", "unknown")
                telemetry.record_mcts_iteration(
                    iteration=iteration,
                    node_visits=node.visits,
                    node_value=node.value,
                    best_hypothesis=best_hypothesis,
                    confidence=reward,
                    objective=objective,
                )

            # Return best result
            if not self.root.children:
                # No expansion happened, use root
                result = self._extract_result(self.root, objective)
            else:
                best_child = max(self.root.children, key=lambda n: n.visits)
                result = self._extract_result(best_child, objective)

            # Log final search metrics
            telemetry.log_info(
                "MCTS search completed",
                objective=objective,
                total_nodes=self._count_nodes(self.root),
                root_visits=self.root.visits,
                best_confidence=result.get("confidence", 0.0),
                classification=result.get("classification", "unknown") if objective == "classify" else None,
                risk_level=result.get("risk_level", "unknown") if objective == "detect_fraud" else None,
            )

            return result

    def _count_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in the tree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select most promising leaf node using UCB1.

        Args:
            node: Starting node (typically root)

        Returns:
            Selected leaf node
        """
        while node.children:
            node = max(node.children, key=lambda n: n.ucb1_score(self.config.exploration_constant))
        return node

    def _expand(self, node: MCTSNode, objective: str) -> MCTSNode:
        """
        Expand node by generating hypotheses using LLM.

        Args:
            node: Node to expand
            objective: "classify" or "detect_fraud"

        Returns:
            First child node (or original node if expansion failed)
        """
        hypotheses = self._generate_hypotheses(node.state, objective)

        # Create child nodes for each hypothesis
        # Note: __post_init__ automatically adds children to parent.children
        for hypothesis in hypotheses:
            child_state = {**node.state, "hypothesis": hypothesis}
            MCTSNode(state=child_state, parent=node)

        return node.children[0] if node.children else node

    def _simulate(self, node: MCTSNode, objective: str) -> float:
        """
        Simulate (evaluate) a hypothesis using LLM.

        Args:
            node: Node with hypothesis to evaluate
            objective: "classify" or "detect_fraud"

        Returns:
            Confidence score (0-1)
        """
        result = self._evaluate_hypothesis(node.state, objective)
        # Store reasoning in node state for later extraction
        node.state["evaluation"] = result
        return result.get("confidence", 0.5)

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagate reward up the tree.

        Args:
            node: Starting node
            reward: Reward value to propagate
        """
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    def _generate_hypotheses(self, state: dict[str, Any], objective: str) -> list[dict[str, Any]]:
        """
        Generate hypotheses using LLM.

        Args:
            state: Current state (transaction + context)
            objective: "classify" or "detect_fraud"

        Returns:
            List of hypothesis dictionaries
        """
        transaction = state.get("transaction", {})

        if objective == "classify":
            prompt = self._build_classification_hypothesis_prompt(transaction)
        elif objective == "detect_fraud":
            prompt = self._build_fraud_hypothesis_prompt(transaction)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Call LLM
        response = self.llm_function(prompt)

        # Parse response
        return self._parse_hypotheses(response, objective)

    def _evaluate_hypothesis(self, state: dict[str, Any], objective: str) -> dict[str, Any]:
        """
        Evaluate a hypothesis using LLM.

        Args:
            state: State with hypothesis
            objective: "classify" or "detect_fraud"

        Returns:
            Evaluation result with confidence and reasoning
        """
        transaction = state.get("transaction", {})
        hypothesis = state.get("hypothesis", {})

        if objective == "classify":
            prompt = self._build_classification_evaluation_prompt(transaction, hypothesis)
        elif objective == "detect_fraud":
            prompt = self._build_fraud_evaluation_prompt(transaction, hypothesis)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Call LLM
        response = self.llm_function(prompt)

        # Parse response
        return self._parse_evaluation(response, hypothesis)

    def _build_classification_hypothesis_prompt(self, transaction: dict[str, Any]) -> str:
        """Build prompt for classification hypothesis generation."""
        return f"""Given this financial transaction, generate 3-5 possible classification hypotheses.

Transaction:
- Amount: {transaction.get('amount')} {transaction.get('currency')}
- Merchant: {transaction.get('merchant')}
- Description: {transaction.get('description')}
- Date: {transaction.get('date')}

For each hypothesis, provide:
1. Category name (e.g., "Business Expense - Office Supplies", "Personal - Entertainment", etc.)
2. Brief rationale (1 sentence)

Format your response as a JSON array of objects with "category" and "rationale" fields.
Example:
[
  {{"category": "Business Expense - Office Supplies", "rationale": "Purchase from business supplier"}},
  {{"category": "Personal - Electronics", "rationale": "Consumer electronics purchase"}}
]
"""

    def _build_fraud_hypothesis_prompt(self, transaction: dict[str, Any]) -> str:
        """Build prompt for fraud detection hypothesis generation."""
        return f"""Given this financial transaction, generate 3-5 possible fraud risk hypotheses.

Transaction:
- Amount: {transaction.get('amount')} {transaction.get('currency')}
- Merchant: {transaction.get('merchant')}
- Description: {transaction.get('description')}
- Date: {transaction.get('date')}

For each hypothesis, provide:
1. Risk level (LOW, MEDIUM, HIGH, CRITICAL)
2. Fraud indicators (list of potential red flags)
3. Brief rationale

Format your response as a JSON array of objects with "risk_level", "indicators", and "rationale" fields.
Example:
[
  {{"risk_level": "LOW", "indicators": ["Normal merchant", "Reasonable amount"], "rationale": "Standard transaction"}},
  {{"risk_level": "HIGH", "indicators": ["Unusual amount", "Off-hours"], "rationale": "Suspicious timing and amount"}}
]
"""

    def _build_classification_evaluation_prompt(
        self, transaction: dict[str, Any], hypothesis: dict[str, Any]
    ) -> str:
        """Build prompt for classification hypothesis evaluation."""
        return f"""Evaluate how well this classification hypothesis fits the transaction.

Transaction:
- Amount: {transaction.get('amount')} {transaction.get('currency')}
- Merchant: {transaction.get('merchant')}
- Description: {transaction.get('description')}
- Date: {transaction.get('date')}

Hypothesis:
- Category: {hypothesis.get('category')}
- Rationale: {hypothesis.get('rationale')}

Provide:
1. Confidence score (0.0 to 1.0) for this classification
2. Detailed reasoning (2-3 sentences)

Format as JSON:
{{"confidence": 0.85, "reasoning": "Your detailed reasoning here"}}
"""

    def _build_fraud_evaluation_prompt(
        self, transaction: dict[str, Any], hypothesis: dict[str, Any]
    ) -> str:
        """Build prompt for fraud hypothesis evaluation."""
        return f"""Evaluate how likely this fraud hypothesis is for the transaction.

Transaction:
- Amount: {transaction.get('amount')} {transaction.get('currency')}
- Merchant: {transaction.get('merchant')}
- Description: {transaction.get('description')}
- Date: {transaction.get('date')}

Hypothesis:
- Risk Level: {hypothesis.get('risk_level')}
- Indicators: {hypothesis.get('indicators')}
- Rationale: {hypothesis.get('rationale')}

Provide:
1. Confidence score (0.0 to 1.0) for this fraud assessment
2. Detailed reasoning (2-3 sentences)
3. Recommended actions (if risk is MEDIUM or higher)

Format as JSON:
{{"confidence": 0.75, "reasoning": "Your detailed reasoning", "actions": ["Action 1", "Action 2"]}}
"""

    def _parse_hypotheses(self, response: str, objective: str) -> list[dict[str, Any]]:
        """
        Parse LLM response into hypotheses.

        Args:
            response: LLM response text
            objective: "classify" or "detect_fraud"

        Returns:
            List of hypothesis dictionaries
        """
        try:
            # Try to extract JSON from response
            # Look for JSON array in the response
            start = response.find("[")
            end = response.rfind("]") + 1

            if start != -1 and end > start:
                json_str = response[start:end]
                hypotheses = json.loads(json_str)
                return hypotheses[:5]  # Limit to 5 hypotheses
        except json.JSONDecodeError:
            pass

        # Fallback: create default hypotheses
        if objective == "classify":
            return [
                {"category": "Business Expense", "rationale": "Default classification"},
                {"category": "Personal Expense", "rationale": "Alternative classification"},
            ]
        else:  # detect_fraud
            return [
                {"risk_level": "LOW", "indicators": [], "rationale": "No obvious fraud indicators"},
                {
                    "risk_level": "MEDIUM",
                    "indicators": ["Requires review"],
                    "rationale": "Standard review needed",
                },
            ]

    def _parse_evaluation(
        self, response: str, hypothesis: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Parse LLM evaluation response.

        Args:
            response: LLM response text
            hypothesis: Original hypothesis

        Returns:
            Evaluation result with confidence and reasoning
        """
        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1

            if start != -1 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                return {
                    "hypothesis": hypothesis,
                    "confidence": float(result.get("confidence", 0.5)),
                    "reasoning": result.get("reasoning", ""),
                    "actions": result.get("actions", []),
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback
        return {
            "hypothesis": hypothesis,
            "confidence": 0.5,
            "reasoning": "Unable to parse LLM response",
            "actions": [],
        }

    def _extract_result(self, node: MCTSNode, objective: str) -> dict[str, Any]:
        """
        Extract final result from best node.

        Args:
            node: Best node from search
            objective: "classify" or "detect_fraud"

        Returns:
            Structured result dictionary
        """
        hypothesis = node.state.get("hypothesis", {})
        evaluation = node.state.get("evaluation", {})

        confidence = node.value / node.visits if node.visits > 0 else 0.0

        return {
            "hypothesis": hypothesis,
            "confidence": confidence,
            "visits": node.visits,
            "reasoning": evaluation.get("reasoning", "MCTS search completed"),
            "actions": evaluation.get("actions", []),
        }
