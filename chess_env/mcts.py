"""
Monte Carlo Tree Search (MCTS) for AlphaZero-style training.

MCTS improves upon raw policy network outputs by performing lookahead search.
Each move is selected based on visit counts from many simulations, providing
a much stronger training signal than direct policy sampling.

Key formulas:
- UCB selection: a = argmax(Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)))
- Q(s,a) = W(s,a) / N(s,a)  (mean value of action)
- After search, policy target = N(s,a) / sum(N(s,:))  (visit count distribution)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import chess
import numpy as np
import torch

from chess_env.encoding import ActionEncoder, ObservationEncoder


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    num_simulations: int = 100  # Simulations per move (AlphaZero used 800)
    c_puct: float = 1.5  # Exploration constant
    temperature: float = 1.0  # Temperature for action selection (1.0 = proportional to visits)
    dirichlet_alpha: float = 0.3  # Dirichlet noise alpha for root exploration
    dirichlet_epsilon: float = 0.25  # Weight of Dirichlet noise at root


class MCTSNode:
    """
    A node in the MCTS tree.

    Each node corresponds to a board state and stores statistics for each action.
    """

    def __init__(
        self,
        board: chess.Board,
        parent: Optional[MCTSNode] = None,
        parent_action: Optional[int] = None,
        prior: float = 0.0,
    ):
        self.board = board.copy()
        self.parent = parent
        self.parent_action = parent_action
        self.prior = prior  # P(s,a) from parent's perspective

        # Statistics
        self.visit_count = 0  # N(s)
        self.value_sum = 0.0  # W(s) - total value from this node's perspective

        # Children: action -> MCTSNode
        self.children: dict[int, MCTSNode] = {}

        # Whether this node has been expanded (evaluated by network)
        self.is_expanded = False

        # Cache legal actions
        self._legal_actions: Optional[list[int]] = None
        self._action_encoder = ActionEncoder()

    @property
    def legal_actions(self) -> list[int]:
        """Get list of legal action indices."""
        if self._legal_actions is None:
            self._legal_actions = [
                self._action_encoder.encode(self.board, move)
                for move in self.board.legal_moves
            ]
        return self._legal_actions

    @property
    def q_value(self) -> float:
        """Mean value Q(s) = W(s) / N(s)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self.board.is_game_over()

    def terminal_value(self) -> float:
        """Get value of terminal state from current player's perspective."""
        result = self.board.result()
        if result == "1-0":
            return 1.0 if self.board.turn == chess.WHITE else -1.0
        elif result == "0-1":
            return 1.0 if self.board.turn == chess.BLACK else -1.0
        else:
            return 0.0  # Draw

    def ucb_score(self, child_action: int, c_puct: float) -> float:
        """
        Compute UCB score for selecting a child action.

        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        child = self.children.get(child_action)

        if child is None:
            # Unexpanded child - use prior only
            q = 0.0
            n = 0
            prior = 0.0  # Will be set during expansion
        else:
            # Negate Q because child's value is from opponent's perspective
            q = -child.q_value
            n = child.visit_count
            prior = child.prior

        exploration = c_puct * prior * math.sqrt(self.visit_count) / (1 + n)
        return q + exploration

    def select_child(self, c_puct: float) -> int:
        """Select best child action using UCB."""
        best_score = float('-inf')
        best_action = None

        for action in self.legal_actions:
            score = self.ucb_score(action, c_puct)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def expand(self, action_priors: np.ndarray) -> None:
        """
        Expand this node by creating children with prior probabilities.

        Args:
            action_priors: Policy network output (probability for each action)
        """
        self.is_expanded = True

        # Mask to legal actions and renormalize
        legal_mask = np.zeros(len(action_priors), dtype=bool)
        for action in self.legal_actions:
            legal_mask[action] = True

        masked_priors = action_priors * legal_mask
        prior_sum = masked_priors.sum()
        if prior_sum > 0:
            masked_priors /= prior_sum
        else:
            # Uniform over legal actions if network gives zero
            masked_priors[legal_mask] = 1.0 / len(self.legal_actions)

        # Create child nodes (lazily - only store priors for now)
        for action in self.legal_actions:
            if action not in self.children:
                # Create child board state
                child_board = self.board.copy()
                move = self._action_encoder.decode(self.board, action)
                child_board.push(move)

                self.children[action] = MCTSNode(
                    board=child_board,
                    parent=self,
                    parent_action=action,
                    prior=masked_priors[action],
                )

    def backup(self, value: float) -> None:
        """
        Backpropagate value up the tree.

        Args:
            value: Value from the perspective of the node being backed up FROM
        """
        node = self
        while node is not None:
            node.visit_count += 1
            # Value alternates sign as we go up (opponent's loss is our gain)
            node.value_sum += value
            value = -value
            node = node.parent

    def get_policy_target(self, temperature: float = 1.0) -> np.ndarray:
        """
        Get policy target from visit counts.

        Args:
            temperature: Controls exploration (0 = greedy, 1 = proportional to visits)

        Returns:
            Probability distribution over actions based on visit counts
        """
        policy = np.zeros(4672, dtype=np.float32)

        if temperature == 0:
            # Greedy - pick most visited
            best_action = max(self.children.keys(), key=lambda a: self.children[a].visit_count)
            policy[best_action] = 1.0
        else:
            # Proportional to visit counts raised to 1/temperature
            visits = np.array([
                self.children[a].visit_count if a in self.children else 0
                for a in range(4672)
            ], dtype=np.float32)

            if temperature != 1.0:
                visits = visits ** (1.0 / temperature)

            visit_sum = visits.sum()
            if visit_sum > 0:
                policy = visits / visit_sum

        return policy

    def select_action(self, temperature: float = 1.0) -> int:
        """Select action based on visit counts."""
        policy = self.get_policy_target(temperature)

        if temperature == 0:
            return int(np.argmax(policy))
        else:
            return int(np.random.choice(len(policy), p=policy))


class MCTS:
    """
    Monte Carlo Tree Search algorithm.

    Uses a neural network for policy priors and value estimation.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        config: MCTSConfig = None,
        device: str = "cpu",
    ):
        self.network = network
        self.config = config or MCTSConfig()
        self.device = torch.device(device)
        self.obs_encoder = ObservationEncoder()
        self.action_encoder = ActionEncoder()

    def _evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        """
        Evaluate a position using the neural network.

        Returns:
            policy: Action probabilities (4672,)
            value: Position value from current player's perspective
        """
        self.network.eval()

        # Encode observation
        obs = self.obs_encoder.encode(board)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            # Get policy and value from network
            policy_logits, value = self.network(obs_tensor)
            policy = torch.softmax(policy_logits, dim=-1).cpu().numpy().flatten()
            value = value.cpu().item()

        return policy, value

    def _add_dirichlet_noise(self, node: MCTSNode, action_priors: np.ndarray) -> np.ndarray:
        """Add Dirichlet noise to root node priors for exploration."""
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(node.legal_actions))

        noisy_priors = action_priors.copy()
        for i, action in enumerate(node.legal_actions):
            noisy_priors[action] = (
                (1 - self.config.dirichlet_epsilon) * action_priors[action]
                + self.config.dirichlet_epsilon * noise[i]
            )

        return noisy_priors

    def search(self, board: chess.Board) -> MCTSNode:
        """
        Run MCTS from a given position.

        Args:
            board: Starting position

        Returns:
            Root node with search statistics
        """
        # Create root node
        root = MCTSNode(board)

        # Evaluate and expand root
        policy, value = self._evaluate(board)

        # Add exploration noise at root
        policy = self._add_dirichlet_noise(root, policy)
        root.expand(policy)

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root

            # Selection: traverse tree using UCB
            while node.is_expanded and not node.is_terminal():
                action = node.select_child(self.config.c_puct)
                if action in node.children:
                    node = node.children[action]
                else:
                    break

            # Get value for backup
            if node.is_terminal():
                value = node.terminal_value()
            else:
                # Expansion: evaluate with network and expand
                policy, value = self._evaluate(node.board)
                if not node.is_terminal():
                    node.expand(policy)

            # Backup
            node.backup(value)

        return root

    def get_action_and_policy(
        self,
        board: chess.Board,
        temperature: float = None,
    ) -> tuple[int, np.ndarray, float]:
        """
        Run MCTS and return selected action, policy target, and root value.

        Args:
            board: Current position
            temperature: Override config temperature (None = use config)

        Returns:
            action: Selected action
            policy: Policy target (visit count distribution)
            value: Root node value estimate
        """
        if temperature is None:
            temperature = self.config.temperature

        root = self.search(board)

        action = root.select_action(temperature)
        policy = root.get_policy_target(temperature)
        value = root.q_value

        return action, policy, value


@dataclass
class MCTSTrajectory:
    """Training data from one game of MCTS self-play."""
    observations: list[np.ndarray] = field(default_factory=list)
    policy_targets: list[np.ndarray] = field(default_factory=list)
    values: list[float] = field(default_factory=list)  # Filled in at game end

    def add(self, obs: np.ndarray, policy: np.ndarray) -> None:
        """Add a position to the trajectory."""
        self.observations.append(obs)
        self.policy_targets.append(policy)

    def set_outcome(self, result: str) -> None:
        """
        Set final values based on game outcome.

        Args:
            result: Game result ("1-0", "0-1", or "1/2-1/2")
        """
        n = len(self.observations)

        if result == "1-0":
            # White wins - odd indices (black to move) get -1, even get +1
            self.values = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
        elif result == "0-1":
            # Black wins
            self.values = [-1.0 if i % 2 == 0 else 1.0 for i in range(n)]
        else:
            # Draw
            self.values = [0.0] * n


def mcts_self_play(
    mcts: MCTS,
    max_moves: int = 200,
    temperature_threshold: int = 30,
) -> MCTSTrajectory:
    """
    Play one game of MCTS self-play.

    Args:
        mcts: MCTS instance with network
        max_moves: Maximum moves before declaring draw
        temperature_threshold: Use temperature=1 for first N moves, then 0

    Returns:
        Trajectory with observations, policy targets, and values
    """
    trajectory = MCTSTrajectory()
    board = chess.Board()
    obs_encoder = ObservationEncoder()

    for move_num in range(max_moves):
        if board.is_game_over():
            break

        # Temperature annealing: explore early, exploit late
        temp = 1.0 if move_num < temperature_threshold else 0.0

        # Run MCTS
        action, policy, _ = mcts.get_action_and_policy(board, temperature=temp)

        # Store training data
        obs = obs_encoder.encode(board)
        trajectory.add(obs, policy)

        # Apply move
        move = mcts.action_encoder.decode(board, action)
        board.push(move)
        obs_encoder.push(board)

    # Set values based on game outcome
    if board.is_game_over():
        result = board.result()
    else:
        result = "1/2-1/2"  # Truncation = draw

    trajectory.set_outcome(result)

    return trajectory
