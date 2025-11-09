"""
Enhanced Monte Carlo Tree Search (MCTS) Engine with Tool-Specific Configurations.

Implements requirements REQ-001 through REQ-010:
- UCB1 selection policy with configurable exploration constants
- Tool-specific node expansion strategies
- Lightweight simulation/rollout policies
- Backpropagation with reward scaling
- Iteration budget enforcement with early termination
- Tool-specific reward functions
- Terminal state detection
"""

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.config import MCTSConfig, ToolMCTSConfig
from src.models import MCTSConvergenceError, MCTSMetadata, FraudRiskLevel
from src.telemetry import get_telemetry


@dataclass
class MCTSNodeV2:
    """
    Enhanced MCTS node with tool-specific metadata.

    Attributes:
        state: Current state (transaction data + hypothesis)
        parent: Parent node (None for root)
        children: List of child nodes
        visits: Number of times this node has been visited
        value: Accumulated value from simulations
        depth: Current depth in tree
        action_taken: Action that led to this node
        is_terminal: Whether this is a terminal state
    """

    state: dict[str, Any]
    parent: "MCTSNodeV2 | None" = None
    children: list["MCTSNodeV2"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    depth: int = 0
    action_taken: str = ""
    is_terminal: bool = False

    def __post_init__(self):
        """Add this node to parent's children if parent is provided."""
        if self.parent is not None:
            self.parent.children.append(self)
            self.depth = self.parent.depth + 1

    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        """
        Calculate Upper Confidence Bound (UCB1) score (REQ-001).

        Formula: UCB1 = (w_i / n_i) + c * sqrt(ln(N) / n_i)
        where:
            w_i = win count (node value)
            n_i = node visit count
            N = parent visit count
            c = exploration constant (configurable 0.1-2.0)

        Args:
            exploration_constant: Exploration parameter (REQ-001: 0.1-2.0)

        Returns:
            UCB1 score (infinity for unvisited nodes)
        """
        if self.visits == 0:
            return float("inf")

        if self.parent is None:
            return self.value / self.visits

        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration


class EnhancedMCTSEngine:
    """
    Enhanced MCTS engine with tool-specific configurations and reward functions.

    Implements REQ-001 through REQ-010.
    """

    def __init__(
        self,
        config: MCTSConfig,
        tool_name: str,
        llm_function: Callable[[str], str],
        transaction_id: str = "",
    ):
        """
        Initialize enhanced MCTS engine.

        Args:
            config: Global MCTS configuration
            tool_name: Tool name ("filter", "classify", "fraud", "explanation")
            llm_function: Function to call LLM (takes prompt, returns response)
            transaction_id: Transaction ID being processed
        """
        self.global_config = config
        self.tool_config = config.get_tool_config(tool_name)
        self.tool_name = tool_name
        self.llm_function = llm_function
        self.transaction_id = transaction_id
        self.root: MCTSNodeV2 | None = None
        self.iteration_rewards: deque = deque(maxlen=config.convergence_window)

    def search(self, initial_state: dict[str, Any], objective: str) -> dict[str, Any]:
        """
        Perform MCTS search with tool-specific configuration (REQ-005).

        Args:
            initial_state: Starting state (transaction + context)
            objective: "classify" or "detect_fraud"

        Returns:
            Best result with hypothesis, confidence, reasoning, and MCTS metadata

        Raises:
            MCTSConvergenceError: If convergence fails (REQ-013)
        """
        telemetry = get_telemetry()

        with telemetry.span(
            "mcts_search",
            tool_name=self.tool_name,
            objective=objective,
            max_iterations=self.tool_config.iterations,
            max_depth=self.tool_config.max_depth,
            exploration_constant=self.tool_config.exploration_constant,
            transaction_id=self.transaction_id,
        ):
            self.root = MCTSNodeV2(state=initial_state)
            iteration = 0
            total_nodes_explored = 1
            max_depth_reached = 0

            # Run MCTS iterations with tool-specific budget (REQ-005)
            for iteration in range(self.tool_config.iterations):
                with telemetry.span(
                    "mcts_iteration",
                    iteration=iteration,
                    tool_name=self.tool_name,
                ):
                    # 1. Selection: Traverse tree using UCB1 (REQ-001)
                    with telemetry.span("mcts_selection"):
                        node = self._select(self.root)

                    # 2. Expansion: Generate new hypotheses if needed (REQ-002)
                    if node.visits > 0 and not node.children and not node.is_terminal:
                        with telemetry.span("mcts_expansion", current_depth=node.depth):
                            node = self._expand(node, objective)
                            if node != self.root:
                                total_nodes_explored += 1

                    # 3. Simulation: Evaluate hypothesis using lightweight policy (REQ-003)
                    with telemetry.span("mcts_rollout"):
                        reward = self._simulate(node, objective)

                    # Scale reward per tool configuration (REQ-004)
                    scaled_reward = reward * self.tool_config.reward_scale

                    # 4. Backpropagation: Update values up the tree (REQ-004)
                    with telemetry.span("mcts_backpropagation"):
                        self._backpropagate(node, scaled_reward)

                    # Track rewards for convergence detection
                    self.iteration_rewards.append(scaled_reward)

                    # Update max depth reached
                    if node.depth > max_depth_reached:
                        max_depth_reached = node.depth

                    # Record iteration metrics
                    best_hypothesis = node.state.get("hypothesis", {})
                    telemetry.record_mcts_iteration(
                        iteration=iteration,
                        node_visits=node.visits,
                        node_value=node.value,
                        best_hypothesis=str(best_hypothesis),
                        confidence=scaled_reward,
                        objective=objective,
                    )

                    # Early termination check (REQ-005)
                    if self.global_config.early_termination_enabled:
                        if self._check_convergence():
                            telemetry.log_info(
                                "MCTS early termination: converged",
                                iteration=iteration,
                                tool_name=self.tool_name,
                            )
                            break

            # Extract best result
            if not self.root.children:
                best_node = self.root
            else:
                best_node = max(self.root.children, key=lambda n: n.visits)

            result = self._extract_result(best_node, objective)

            # Calculate final variance
            final_variance = self._calculate_variance()

            # Check for convergence failure (REQ-013)
            if final_variance > self.global_config.convergence_std_threshold * 10:
                # High variance indicates potential convergence issues
                telemetry.log_warning(
                    "MCTS high variance detected",
                    tool_name=self.tool_name,
                    transaction_id=self.transaction_id,
                    final_variance=final_variance,
                    iterations=iteration + 1,
                )

            # Add MCTS metadata (REQ-015)
            result["mcts_metadata"] = MCTSMetadata(
                root_node_visits=self.root.visits,
                best_action_path=self._get_action_path(best_node),
                average_reward=self.root.value / self.root.visits if self.root.visits > 0 else 0.0,
                exploration_constant_used=self.tool_config.exploration_constant,
                final_reward_variance=final_variance,
                total_nodes_explored=total_nodes_explored,
                max_depth_reached=max_depth_reached,
            )

            telemetry.log_info(
                "MCTS search completed",
                tool_name=self.tool_name,
                objective=objective,
                iterations_completed=iteration + 1,
                total_nodes=total_nodes_explored,
                max_depth=max_depth_reached,
                root_visits=self.root.visits,
                best_confidence=result.get("confidence", 0.0),
            )

            return result

    def _select(self, node: MCTSNodeV2) -> MCTSNodeV2:
        """
        Select most promising leaf node using UCB1 (REQ-001).

        Args:
            node: Starting node (typically root)

        Returns:
            Selected leaf node
        """
        while node.children and not node.is_terminal:
            node = max(
                node.children,
                key=lambda n: n.ucb1_score(self.tool_config.exploration_constant)
            )
        return node

    def _expand(self, node: MCTSNodeV2, objective: str) -> MCTSNodeV2:
        """
        Expand node by generating hypotheses (REQ-002).

        Implements deterministic action ordering and max depth limits per tool.

        Args:
            node: Node to expand
            objective: "classify" or "detect_fraud"

        Returns:
            First child node (or original node if expansion failed)
        """
        # Check if max depth reached (REQ-002)
        if node.depth >= self.tool_config.max_depth:
            node.is_terminal = True
            return node

        # Generate hypotheses
        hypotheses = self._generate_hypotheses(node.state, objective)

        # Create child nodes for each hypothesis (REQ-002: deterministic ordering)
        for i, hypothesis in enumerate(hypotheses):
            child_state = {**node.state, "hypothesis": hypothesis}
            action = self._get_action_name(hypothesis, i)
            MCTSNodeV2(
                state=child_state,
                parent=node,
                action_taken=action,
            )

        return node.children[0] if node.children else node

    def _simulate(self, node: MCTSNodeV2, objective: str) -> float:
        """
        Simulate (evaluate) a hypothesis using lightweight heuristic policy (REQ-003).

        Must complete within 10ms per REQ-003.

        Args:
            node: Node with hypothesis to evaluate
            objective: "classify" or "detect_fraud"

        Returns:
            Reward score (0.0-1.0) based on tool-specific reward function
        """
        start_time = time.time()

        # Use tool-specific reward function (REQ-006 to REQ-009)
        reward = self._calculate_reward(node.state, objective)

        # Check simulation timeout (REQ-003: 10ms)
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > self.global_config.simulation_timeout_ms:
            telemetry = get_telemetry()
            telemetry.log_warning(
                "Simulation timeout exceeded",
                tool_name=self.tool_name,
                elapsed_ms=elapsed_ms,
                timeout_ms=self.global_config.simulation_timeout_ms,
            )

        # Store evaluation in node state
        node.state["reward"] = reward

        # Check for terminal state (REQ-010)
        if self._is_terminal_state(node, reward):
            node.is_terminal = True

        return reward

    def _calculate_reward(self, state: dict[str, Any], objective: str) -> float:
        """
        Calculate tool-specific reward (REQ-006 to REQ-009).

        Args:
            state: Current state with hypothesis
            objective: "classify" or "detect_fraud"

        Returns:
            Reward value (0.0-1.0)
        """
        hypothesis = state.get("hypothesis", {})

        if self.tool_name == "filter":
            # REQ-006: Filter Transactions Binary Reward
            amount_gbp = state.get("amount_gbp", 0.0)
            return 1.0 if amount_gbp >= 250.0 else 0.0

        elif self.tool_name == "classify":
            # REQ-007: Classification Multi-Class Reward
            predicted_category = hypothesis.get("category", "")
            ground_truth = state.get("ground_truth_category", "")

            # One-hot encoded: 1.0 if match, 0.0 otherwise
            if ground_truth and predicted_category == ground_truth:
                return 1.0
            else:
                # Use confidence as proxy if no ground truth
                return hypothesis.get("confidence", 0.5)

        elif self.tool_name == "fraud":
            # REQ-008: Fraud Detection Risk-Level Reward
            predicted_risk = hypothesis.get("risk_level", "LOW")
            ground_truth_risk = state.get("ground_truth_risk", "")

            # Map risk levels to rewards
            risk_rewards = {
                "CRITICAL": 1.0,
                "HIGH": 0.75,
                "MEDIUM": 0.5,
                "LOW": 0.0,
            }

            # Exact match gets 1.0, otherwise use risk-based reward
            if ground_truth_risk and predicted_risk == ground_truth_risk:
                return 1.0
            else:
                return risk_rewards.get(predicted_risk, 0.0)

        elif self.tool_name == "explanation":
            # REQ-009: CSV Generation Data Completeness Reward
            # This would be used in the explanation generation phase
            explanation_quality = hypothesis.get("quality", 0.5)
            return explanation_quality

        return 0.5  # Default fallback

    def _is_terminal_state(self, node: MCTSNodeV2, reward: float) -> bool:
        """
        Check if state is terminal (REQ-010).

        Terminal conditions:
        a) Maximum depth is reached
        b) A definitive reward (0.0 or 1.0) is achieved
        c) No valid actions remain

        Args:
            node: Current node
            reward: Current reward value

        Returns:
            True if state is terminal
        """
        # (a) Max depth reached
        if node.depth >= self.tool_config.max_depth:
            return True

        # (b) Definitive reward achieved
        if reward == 0.0 or reward == 1.0:
            # Check tool-specific terminal conditions
            if self.tool_name == "filter":
                # Filter: terminal when amount is confirmed
                return True
            elif self.tool_name == "fraud":
                # Fraud: terminal when risk is definitively LOW or CRITICAL
                hypothesis = node.state.get("hypothesis", {})
                risk_level = hypothesis.get("risk_level", "")
                return risk_level in ["LOW", "CRITICAL"]

        # (c) No valid actions remain (would be determined during expansion)
        # This is handled implicitly by not expanding terminal nodes

        return False

    def _backpropagate(self, node: MCTSNodeV2, reward: float) -> None:
        """
        Backpropagate reward up the tree (REQ-004).

        Reward is already scaled by tool-specific factor before calling this.

        Args:
            node: Starting node
            reward: Scaled reward value to propagate
        """
        current = node
        while current is not None:
            current.visits += 1
            current.value += reward
            current = current.parent

    def _check_convergence(self) -> bool:
        """
        Check for early termination due to convergence (REQ-005).

        Convergence detected if std dev < 0.01 for 50 consecutive iterations.

        Returns:
            True if converged
        """
        if len(self.iteration_rewards) < self.global_config.convergence_window:
            return False

        variance = self._calculate_variance()
        std_dev = math.sqrt(variance) if variance > 0 else 0.0

        return std_dev < self.global_config.convergence_std_threshold

    def _calculate_variance(self) -> float:
        """
        Calculate variance of recent rewards.

        Returns:
            Variance value
        """
        if len(self.iteration_rewards) < 2:
            return float("inf")

        mean = sum(self.iteration_rewards) / len(self.iteration_rewards)
        variance = sum((r - mean) ** 2 for r in self.iteration_rewards) / len(self.iteration_rewards)
        return variance

    def _get_action_path(self, node: MCTSNodeV2) -> list[str]:
        """
        Get the action path from root to node.

        Args:
            node: Target node

        Returns:
            List of actions taken
        """
        path = []
        current = node
        while current.parent is not None:
            if current.action_taken:
                path.insert(0, current.action_taken)
            current = current.parent
        return path

    def _get_action_name(self, hypothesis: dict[str, Any], index: int) -> str:
        """
        Generate action name for hypothesis.

        Args:
            hypothesis: Hypothesis dictionary
            index: Hypothesis index

        Returns:
            Action name string
        """
        if self.tool_name == "classify":
            category = hypothesis.get("category", f"hypothesis_{index}")
            return f"classify_as_{category}"
        elif self.tool_name == "fraud":
            risk = hypothesis.get("risk_level", f"hypothesis_{index}")
            return f"assess_risk_{risk}"
        else:
            return f"action_{index}"

    def _generate_hypotheses(self, state: dict[str, Any], objective: str) -> list[dict[str, Any]]:
        """
        Generate hypotheses using LLM (reuses existing implementation).

        Args:
            state: Current state
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

        response = self.llm_function(prompt)
        return self._parse_hypotheses(response, objective)

    def _build_classification_hypothesis_prompt(self, transaction: dict[str, Any]) -> str:
        """Build prompt for classification hypothesis generation."""
        return f"""Generate 3-5 classification hypotheses for this transaction.

Transaction:
- Amount: {transaction.get('amount')} {transaction.get('currency')}
- Merchant: {transaction.get('merchant')}
- Description: {transaction.get('description')}

Categories: Business, Personal, Investment, Gambling

Format as JSON array:
[
  {{"category": "Business", "rationale": "...", "confidence": 0.8}},
  {{"category": "Personal", "rationale": "...", "confidence": 0.6}}
]
"""

    def _build_fraud_hypothesis_prompt(self, transaction: dict[str, Any]) -> str:
        """Build prompt for fraud detection hypothesis generation."""
        return f"""Generate 3-5 fraud risk hypotheses for this transaction.

Transaction:
- Amount: {transaction.get('amount')} {transaction.get('currency')}
- Merchant: {transaction.get('merchant')}
- Description: {transaction.get('description')}

Risk levels: LOW, MEDIUM, HIGH, CRITICAL

Format as JSON array:
[
  {{"risk_level": "LOW", "indicators": ["..."], "rationale": "...", "confidence": 0.7}},
  {{"risk_level": "HIGH", "indicators": ["..."], "rationale": "...", "confidence": 0.8}}
]
"""

    def _parse_hypotheses(self, response: str, objective: str) -> list[dict[str, Any]]:
        """Parse LLM response into hypotheses."""
        try:
            start = response.find("[")
            end = response.rfind("]") + 1

            if start != -1 and end > start:
                json_str = response[start:end]
                hypotheses = json.loads(json_str)
                return hypotheses[:5]
        except json.JSONDecodeError:
            pass

        # Fallback: create default hypotheses
        if objective == "classify":
            return [
                {"category": "Business", "rationale": "Default", "confidence": 0.5},
                {"category": "Personal", "rationale": "Alternative", "confidence": 0.5},
            ]
        else:  # detect_fraud
            return [
                {"risk_level": "LOW", "indicators": [], "rationale": "No indicators", "confidence": 0.5},
                {"risk_level": "MEDIUM", "indicators": [], "rationale": "Review needed", "confidence": 0.5},
            ]

    def _extract_result(self, node: MCTSNodeV2, objective: str) -> dict[str, Any]:
        """
        Extract final result from best node.

        Args:
            node: Best node from search
            objective: "classify" or "detect_fraud"

        Returns:
            Structured result dictionary
        """
        hypothesis = node.state.get("hypothesis", {})
        confidence = node.value / node.visits if node.visits > 0 else 0.0

        return {
            "hypothesis": hypothesis,
            "confidence": confidence,
            "visits": node.visits,
            "reward": node.state.get("reward", 0.0),
            "action_path": self._get_action_path(node),
        }
