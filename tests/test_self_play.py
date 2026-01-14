"""
Tests for the self-play wrapper.

These tests verify:
1. Games run to completion
2. Trajectories are collected correctly
3. Rewards are assigned correctly at game end
4. GAE computation works
"""

import numpy as np
import pytest

from chess_env.env import ChessTensorEnv
from chess_env.self_play import (
    GameResult,
    SelfPlayWrapper,
    Trajectory,
    create_random_policy,
)


class TestTrajectory:
    """Tests for Trajectory data structure."""

    def test_empty_trajectory(self):
        """Empty trajectory should have length 0."""
        traj = Trajectory()
        assert len(traj) == 0

    def test_trajectory_length(self):
        """Length should match number of observations."""
        traj = Trajectory()
        traj.observations = [np.zeros((8, 8, 119)) for _ in range(5)]
        assert len(traj) == 5

    def test_compute_returns_empty(self):
        """Empty trajectory should return empty arrays."""
        traj = Trajectory()
        returns, advantages = traj.compute_returns_and_advantages()

        assert len(returns) == 0
        assert len(advantages) == 0

    def test_compute_returns_single_step(self):
        """Single step trajectory should compute correctly."""
        traj = Trajectory()
        traj.observations = [np.zeros((8, 8, 119))]
        traj.actions = [0]
        traj.rewards = [1.0]  # Win
        traj.values = [0.5]
        traj.log_probs = [-1.0]
        traj.dones = [True]

        returns, advantages = traj.compute_returns_and_advantages()

        assert len(returns) == 1
        assert len(advantages) == 1
        # Return should equal reward since done=True
        assert returns[0] == pytest.approx(1.0)
        # Advantage = reward - value
        assert advantages[0] == pytest.approx(0.5)

    def test_compute_returns_multi_step(self):
        """Multi-step trajectory should propagate rewards."""
        traj = Trajectory()
        # 3 step game, reward only at end
        traj.observations = [np.zeros((8, 8, 119)) for _ in range(3)]
        traj.actions = [0, 1, 2]
        traj.rewards = [0.0, 0.0, 1.0]  # Win at end
        traj.values = [0.3, 0.5, 0.7]
        traj.log_probs = [-1.0, -1.0, -1.0]
        traj.dones = [False, False, True]

        returns, advantages = traj.compute_returns_and_advantages(
            gamma=0.99, gae_lambda=0.95
        )

        assert len(returns) == 3
        # Returns should be discounted backwards
        assert returns[2] == pytest.approx(1.0)  # Final reward
        assert returns[1] > returns[0]  # Closer to reward

    def test_to_tensors(self):
        """to_tensors should produce valid PyTorch tensors."""
        pytest.importorskip("torch")
        import torch

        traj = Trajectory()
        traj.observations = [np.zeros((8, 8, 119)) for _ in range(3)]
        traj.actions = [0, 1, 2]
        traj.rewards = [0.0, 0.0, 1.0]
        traj.values = [0.3, 0.5, 0.7]
        traj.log_probs = [-1.0, -1.0, -1.0]
        traj.dones = [False, False, True]

        tensors = traj.to_tensors()

        assert "observations" in tensors
        assert "actions" in tensors
        assert "returns" in tensors
        assert "advantages" in tensors

        assert tensors["observations"].shape == (3, 8, 8, 119)
        assert tensors["actions"].shape == (3,)


class TestRandomPolicy:
    """Tests for the random policy."""

    def test_random_policy_returns_legal_action(self):
        """Random policy should only return legal actions."""
        policy = create_random_policy()
        env = ChessTensorEnv()
        obs, info = env.reset()

        for _ in range(100):  # Multiple samples
            action, log_prob, value = policy(obs, info["action_mask"])

            assert info["action_mask"][action], "Random policy returned illegal action"

    def test_random_policy_output_types(self):
        """Random policy should return correct types."""
        policy = create_random_policy()
        env = ChessTensorEnv()
        obs, info = env.reset()

        action, log_prob, value = policy(obs, info["action_mask"])

        assert isinstance(action, int)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)


class TestSelfPlayWrapper:
    """Tests for the SelfPlayWrapper."""

    def test_play_game_completes(self):
        """play_game should run to completion."""
        env = ChessTensorEnv(max_moves=200)
        wrapper = SelfPlayWrapper(env)
        policy = create_random_policy()

        result = wrapper.play_game(policy)

        assert isinstance(result, GameResult)
        assert result.white_trajectory is not None
        assert result.black_trajectory is not None
        assert result.result in ["1-0", "0-1", "1/2-1/2"]

    def test_trajectories_have_moves(self):
        """Both trajectories should have moves recorded."""
        env = ChessTensorEnv(max_moves=200)
        wrapper = SelfPlayWrapper(env)
        policy = create_random_policy()

        result = wrapper.play_game(policy)

        # Both sides should have made at least one move (unless immediate checkmate)
        total_moves = len(result.white_trajectory) + len(result.black_trajectory)
        assert total_moves > 0
        assert total_moves == result.num_moves

    def test_rewards_sum_to_zero_or_are_terminal(self):
        """In zero-sum game, rewards should balance or be terminal."""
        env = ChessTensorEnv(max_moves=100, reward_shaping=False)
        wrapper = SelfPlayWrapper(env)
        policy = create_random_policy()

        result = wrapper.play_game(policy)

        # Sum all rewards
        white_total = sum(result.white_trajectory.rewards)
        black_total = sum(result.black_trajectory.rewards)

        # In a complete game:
        # - Draw: both get 0 total
        # - Win: winner gets +1, loser gets -1 (but loser's last reward)
        # During game, all intermediate rewards should be 0
        intermediate_white = result.white_trajectory.rewards[:-1] if result.white_trajectory.rewards else []
        intermediate_black = result.black_trajectory.rewards[:-1] if result.black_trajectory.rewards else []

        for r in intermediate_white:
            assert r == 0.0, "Non-zero intermediate reward for White"
        for r in intermediate_black:
            assert r == 0.0, "Non-zero intermediate reward for Black"

    def test_collect_trajectories(self):
        """collect_trajectories should return multiple trajectories."""
        env = ChessTensorEnv(max_moves=50)
        wrapper = SelfPlayWrapper(env)
        policy = create_random_policy()

        trajectories = wrapper.collect_trajectories(policy, num_games=3)

        # 2 trajectories per game
        assert len(trajectories) == 6

    def test_trajectories_to_batch(self):
        """trajectories_to_batch should combine into single batch."""
        pytest.importorskip("torch")

        env = ChessTensorEnv(max_moves=50)
        wrapper = SelfPlayWrapper(env)
        policy = create_random_policy()

        trajectories = wrapper.collect_trajectories(policy, num_games=2)
        batch = wrapper.trajectories_to_batch(trajectories)

        assert "observations" in batch
        assert "actions" in batch
        assert "returns" in batch
        assert "advantages" in batch

        # Batch size should be sum of all trajectory lengths
        expected_size = sum(len(t) for t in trajectories)
        assert batch["observations"].shape[0] == expected_size

    def test_pgn_is_valid(self):
        """Generated PGN should be valid format."""
        env = ChessTensorEnv(max_moves=20)
        wrapper = SelfPlayWrapper(env)
        policy = create_random_policy()

        result = wrapper.play_game(policy)

        pgn = result.game_pgn
        assert isinstance(pgn, str)
        assert len(pgn) > 0
        # Should end with result
        assert pgn.endswith(result.result)

    def test_metadata_is_populated(self):
        """Trajectory metadata should be populated."""
        env = ChessTensorEnv(max_moves=50)
        wrapper = SelfPlayWrapper(env)
        policy = create_random_policy()

        result = wrapper.play_game(policy)

        assert result.white_trajectory.color == "white"
        assert result.black_trajectory.color == "black"
        assert result.white_trajectory.game_result is not None
        assert result.termination_reason is not None


class TestSelfPlayEdgeCases:
    """Edge case tests for self-play."""

    def test_very_short_game(self):
        """Should handle very short games (e.g., fool's mate)."""
        # Start from a position one move before checkmate
        # This is after f3, e5, g4 - Black can play Qh4#
        fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
        env = ChessTensorEnv(start_fen=fen)
        wrapper = SelfPlayWrapper(env)

        # Policy that plays Qh4 (checkmate) then random
        def checkmate_policy(obs, mask, det=False):
            # Try to find Qh4
            for action in np.where(mask)[0]:
                move = env.action_encoder.decode(env.board, action)
                if move:
                    try:
                        san = env.board.san(move)
                        if san == "Qh4#":
                            return int(action), 0.0, 0.0
                    except Exception:
                        pass
            # Fallback to random
            action = np.random.choice(np.where(mask)[0])
            return int(action), 0.0, 0.0

        result = wrapper.play_game(checkmate_policy)

        assert result.result == "0-1", f"Expected 0-1, got {result.result}"
        assert result.termination_reason == "checkmate"
        assert result.num_moves == 1  # Just the one checkmate move

    def test_game_with_truncation(self):
        """Should handle truncated games correctly."""
        env = ChessTensorEnv(max_moves=4)
        wrapper = SelfPlayWrapper(env)
        policy = create_random_policy()

        result = wrapper.play_game(policy)

        assert result.num_moves == 4
        # Truncation should be treated as draw
        assert result.result == "1/2-1/2"
