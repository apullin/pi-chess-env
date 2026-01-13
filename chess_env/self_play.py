"""
Self-Play Wrapper for Chess RL Training.

This module provides infrastructure for training chess agents through self-play,
where the same model plays both White and Black (or against a frozen past version).

=== THE SELF-PLAY PARADIGM ===

Traditional RL has an agent interacting with an environment. But chess is a two-player
game - there's no fixed "environment response." Instead, the environment IS the opponent.

Self-play solves this by having the agent play against itself:
1. Model plays as White, selects action
2. Same model (playing as Black) sees the resulting position and responds
3. Repeat until game ends
4. Both "players" learn from the same game

Why this works:
- Creates an adversarial curriculum: as the agent improves, so does its opponent
- No need for external chess engines or labeled games
- Scales to superhuman play (see AlphaZero, AlphaGo)

=== ROLLOUT STRUCTURE ===

A single game generates TWO trajectories (one per color):

    Game: e4 e5 Nf3 Nc6 Bb5 a6 ...
                    │
          ┌────────┴────────┐
          │                 │
    White trajectory:    Black trajectory:
    s0 -> a0 -> r0       s1 -> a1 -> r1
    s2 -> a2 -> r2       s3 -> a3 -> r3
    ...                  ...

Each trajectory contains:
- observations: from that color's perspective (board flipped for Black)
- actions: that color's moves
- rewards: 0 during game, +1/-1/0 at end
- values: critic's estimate of position value (for GAE)

=== OPPONENT STRATEGIES ===

SelfPlayWrapper supports several opponent modes:

1. SELF (default): Current policy plays both sides
   - Best for initial training
   - Creates balanced games

2. FROZEN: Play against a frozen snapshot of the policy
   - Prevents "forgetting" by maintaining some opponent diversity
   - Typical approach: freeze every N training steps

3. POOL: Play against a pool of past versions
   - Maximum diversity
   - Prevents cyclical policies
   - Used by AlphaZero

4. EXTERNAL: Play against an external engine (Stockfish, etc.)
   - Useful for evaluation
   - Can be used for curriculum (gradually increase engine strength)

=== VALUE FUNCTION AND GAE ===

The value function V(s) estimates the expected game outcome from position s:
- V(s) ≈ +1 if position is winning
- V(s) ≈ 0 if position is drawn
- V(s) ≈ -1 if position is losing

Generalized Advantage Estimation (GAE) computes advantages:
    A_t = Σ (γλ)^l * δ_{t+l}
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)

This propagates the terminal reward signal back through the game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol

import numpy as np
import torch

from chess_env.env import ChessTensorEnv


class OpponentMode(Enum):
    """How the opponent is determined in self-play."""
    SELF = "self"           # Current policy plays both sides
    FROZEN = "frozen"       # Frozen copy of current policy
    POOL = "pool"           # Random sample from pool of past policies
    EXTERNAL = "external"   # External engine or function


class Policy(Protocol):
    """Protocol for a policy that can select actions."""

    def __call__(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """
        Select an action given observation and legal action mask.

        Args:
            observation: Board state tensor (8, 8, 119)
            action_mask: Boolean array (4672,) of legal actions
            deterministic: If True, select argmax; else sample

        Returns:
            action: Selected action index
            log_prob: Log probability of the action
            value: Critic's value estimate for this state
        """
        ...


@dataclass
class Trajectory:
    """
    A single trajectory from one player's perspective.

    Contains all data needed for PPO training:
    - observations, actions, rewards for policy gradient
    - values, log_probs for advantage estimation
    - masks for handling variable-length sequences
    """
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    # Metadata
    color: str = "white"  # "white" or "black"
    game_result: Optional[str] = None  # "1-0", "0-1", "1/2-1/2"
    num_moves: int = 0
    termination_reason: Optional[str] = None

    def __len__(self) -> int:
        return len(self.observations)

    def compute_returns_and_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        final_value: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and GAE advantages for this trajectory.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            final_value: Value estimate for state after last action (usually 0 for terminal)

        Returns:
            returns: Discounted returns for each timestep
            advantages: GAE advantages for each timestep
        """
        n = len(self.rewards)
        if n == 0:
            return np.array([]), np.array([])

        returns = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)

        # Work backwards from end of trajectory
        last_gae = 0.0
        last_value = final_value

        for t in reversed(range(n)):
            # If this is a terminal state, next value is 0
            next_value = 0.0 if self.dones[t] else last_value

            # Temporal difference error
            delta = self.rewards[t] + gamma * next_value - self.values[t]

            # GAE
            advantages[t] = delta + gamma * gae_lambda * (0.0 if self.dones[t] else last_gae)
            last_gae = advantages[t]
            last_value = self.values[t]

            # Return = advantage + value
            returns[t] = advantages[t] + self.values[t]

        return returns, advantages

    def to_tensors(self, device: str = "cpu") -> dict[str, torch.Tensor]:
        """Convert trajectory to PyTorch tensors for training."""
        returns, advantages = self.compute_returns_and_advantages()

        return {
            "observations": torch.tensor(
                np.array(self.observations), dtype=torch.float32, device=device
            ),
            "actions": torch.tensor(self.actions, dtype=torch.long, device=device),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32, device=device),
            "values": torch.tensor(self.values, dtype=torch.float32, device=device),
            "returns": torch.tensor(returns, dtype=torch.float32, device=device),
            "advantages": torch.tensor(advantages, dtype=torch.float32, device=device),
        }


@dataclass
class GameResult:
    """Result of a complete self-play game."""
    white_trajectory: Trajectory
    black_trajectory: Trajectory
    game_pgn: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    num_moves: int
    termination_reason: str

    def all_trajectories(self) -> list[Trajectory]:
        """Get both trajectories as a list."""
        return [self.white_trajectory, self.black_trajectory]


class SelfPlayWrapper:
    """
    Wrapper that runs self-play games and collects trajectories.

    This handles the complexity of:
    - Running games where both players use the same (or different) policies
    - Collecting separate trajectories for White and Black
    - Assigning rewards at game end
    - Building training batches from game results

    Usage:
        wrapper = SelfPlayWrapper(env)

        # Run a single game
        result = wrapper.play_game(policy)

        # Run multiple games in parallel (when you have vectorized envs)
        results = wrapper.play_games(policy, num_games=100)

        # Collect trajectories for training
        trajectories = wrapper.collect_trajectories(policy, num_games=100)
    """

    def __init__(
        self,
        env: ChessTensorEnv,
        opponent_mode: OpponentMode = OpponentMode.SELF,
        opponent_policy: Optional[Policy] = None,
    ):
        self.env = env
        self.opponent_mode = opponent_mode
        self.opponent_policy = opponent_policy

    def play_game(
        self,
        policy: Policy,
        deterministic: bool = False,
        render: bool = False,
    ) -> GameResult:
        """
        Play a complete game using self-play.

        Args:
            policy: The policy to use for the current player
            deterministic: If True, always select best action (for eval)
            render: If True, print board after each move

        Returns:
            GameResult containing trajectories for both players
        """
        # Initialize trajectories
        white_traj = Trajectory(color="white")
        black_traj = Trajectory(color="black")

        # Start game
        obs, info = self.env.reset()
        move_history = []

        while True:
            # Determine which policy to use
            if self.env.board.turn:  # White to move
                current_traj = white_traj
                current_policy = policy
            else:  # Black to move
                current_traj = black_traj
                current_policy = self._get_opponent_policy(policy)

            # Get action from policy
            action_mask = info["action_mask"]
            action, log_prob, value = current_policy(obs, action_mask, deterministic)

            # Store experience
            current_traj.observations.append(obs)
            current_traj.actions.append(action)
            current_traj.values.append(value)
            current_traj.log_probs.append(log_prob)

            # Take action
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Record move
            if "last_move_san" in info:
                move_history.append(info.get("last_move_san", "??"))

            if render:
                self.env.render()

            # Store reward and done flag
            # Reward goes to the player who just moved
            current_traj.rewards.append(reward)
            current_traj.dones.append(terminated or truncated)

            if terminated or truncated:
                break

        # Fill in metadata
        result = info.get("game_result", "1/2-1/2")
        termination = info.get("termination_reason", "unknown")

        white_traj.game_result = result
        white_traj.num_moves = len(white_traj.actions)
        white_traj.termination_reason = termination

        black_traj.game_result = result
        black_traj.num_moves = len(black_traj.actions)
        black_traj.termination_reason = termination

        # Build simple PGN
        pgn = self._build_pgn(move_history, result)

        return GameResult(
            white_trajectory=white_traj,
            black_trajectory=black_traj,
            game_pgn=pgn,
            result=result,
            num_moves=len(move_history),
            termination_reason=termination,
        )

    def collect_trajectories(
        self,
        policy: Policy,
        num_games: int,
        deterministic: bool = False,
    ) -> list[Trajectory]:
        """
        Collect trajectories from multiple self-play games.

        Args:
            policy: Policy to use
            num_games: Number of games to play
            deterministic: If True, play deterministically

        Returns:
            List of all trajectories (2 per game, one for each color)
        """
        trajectories = []

        for _ in range(num_games):
            result = self.play_game(policy, deterministic=deterministic)
            trajectories.extend(result.all_trajectories())

        return trajectories

    def trajectories_to_batch(
        self,
        trajectories: list[Trajectory],
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """
        Combine multiple trajectories into a single training batch.

        Args:
            trajectories: List of trajectories to combine
            device: Device to put tensors on

        Returns:
            Dict of batched tensors ready for PPO training
        """
        # Convert each trajectory to tensors
        all_tensors = [t.to_tensors(device) for t in trajectories]

        # Concatenate along batch dimension
        batch = {}
        for key in all_tensors[0].keys():
            batch[key] = torch.cat([t[key] for t in all_tensors], dim=0)

        return batch

    def _get_opponent_policy(self, current_policy: Policy) -> Policy:
        """Get the opponent policy based on mode."""
        if self.opponent_mode == OpponentMode.SELF:
            return current_policy
        elif self.opponent_mode == OpponentMode.FROZEN:
            if self.opponent_policy is None:
                return current_policy  # Fall back to self
            return self.opponent_policy
        elif self.opponent_mode == OpponentMode.EXTERNAL:
            if self.opponent_policy is None:
                raise ValueError("External opponent mode requires opponent_policy")
            return self.opponent_policy
        else:
            # Pool mode - would need additional logic
            return current_policy

    def _build_pgn(self, moves: list[str], result: str) -> str:
        """Build a simple PGN string from move list."""
        lines = []
        for i in range(0, len(moves), 2):
            move_num = i // 2 + 1
            white_move = moves[i] if i < len(moves) else ""
            black_move = moves[i + 1] if i + 1 < len(moves) else ""
            lines.append(f"{move_num}. {white_move} {black_move}")
        lines.append(result)
        return " ".join(lines)

    def set_frozen_opponent(self, policy: Policy) -> None:
        """Update the frozen opponent policy."""
        self.opponent_policy = policy
        self.opponent_mode = OpponentMode.FROZEN


def create_random_policy() -> Policy:
    """
    Create a random policy for testing.

    Selects uniformly from legal moves, returns dummy value/log_prob.
    """
    def random_policy(
        observation: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        legal_actions = np.where(action_mask)[0]
        if len(legal_actions) == 0:
            # No legal moves - shouldn't happen in practice
            return 0, 0.0, 0.0

        action = np.random.choice(legal_actions)
        log_prob = -np.log(len(legal_actions))  # Uniform distribution
        value = 0.0  # Random policy has no value estimate

        return int(action), float(log_prob), float(value)

    return random_policy
