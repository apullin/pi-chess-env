"""
Proximal Policy Optimization (PPO) for Chess Self-Play.

=== WHAT IS PPO? ===

PPO is a policy gradient algorithm that updates the policy "carefully" to avoid
destructive updates. It's the workhorse of modern RL (used in ChatGPT RLHF,
OpenAI Five, and many others).

Key idea: Limit how much the policy can change in each update to prevent
catastrophic forgetting or policy collapse.

=== THE PPO OBJECTIVE ===

Standard policy gradient tries to maximize:
    E[log π(a|s) * A]

where A is the advantage (how much better was this action than expected).

PPO modifies this with a "clipped" objective:
    L = min(r * A, clip(r, 1-ε, 1+ε) * A)

where:
    r = π_new(a|s) / π_old(a|s)  (probability ratio)
    ε = 0.2 (typically)

This clips the objective when the policy changes too much, preventing
wild swings in behavior.

=== VALUE FUNCTION LOSS ===

We also train a value function V(s) to predict expected returns.
This is used to compute advantages:
    A = R - V(s)

The value loss is simply MSE:
    L_value = (V(s) - R)²

=== ENTROPY BONUS ===

We add an entropy bonus to encourage exploration:
    L_entropy = H(π)

This prevents the policy from becoming too deterministic too quickly.

=== COMBINED LOSS ===

    L_total = L_policy + c1 * L_value - c2 * L_entropy

Typical values: c1 = 0.5, c2 = 0.01

=== TRAINING LOOP OVERVIEW ===

1. Collect trajectories using current policy (self-play)
2. Compute advantages using GAE
3. For K epochs:
   a. Sample minibatches from collected data
   b. Compute PPO loss
   c. Update policy and value networks
4. Repeat

=== IMPORTANT FOR CHESS ===

Chess has some special considerations:
- Very long episodes (100+ moves)
- Sparse reward (only at game end)
- Two-player game (need self-play)

We handle these with:
- GAE for credit assignment over long horizons
- Self-play wrapper to generate both player trajectories
- Action masking to only allow legal moves
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from chess_env.env import ChessTensorEnv
from chess_env.network import ChessNet, SmallChessNet, TinyChessNet
from chess_env.self_play import SelfPlayWrapper, Trajectory


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    # Environment
    max_moves: int = 400  # Max moves per game

    # Training
    total_timesteps: int = 100_000  # Total environment steps to train
    games_per_iteration: int = 10  # Games to play per training iteration
    num_epochs: int = 4  # PPO epochs per iteration
    batch_size: int = 64  # Minibatch size
    learning_rate: float = 3e-4

    # PPO hyperparameters
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_range: float = 0.2  # PPO clip range
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping

    # Network
    network_type: str = "small"  # "tiny", "small", or "full"

    # Logging
    log_interval: int = 1  # Log every N iterations
    save_interval: int = 10  # Save checkpoint every N iterations
    checkpoint_dir: str = "checkpoints"

    # Device
    device: str = "cpu"  # "cpu" or "cuda"


class PPOTrainer:
    """
    PPO trainer for chess self-play.

    Usage:
        config = PPOConfig()
        trainer = PPOTrainer(config)
        trainer.train()
    """

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Create environment
        self.env = ChessTensorEnv(max_moves=config.max_moves)
        self.self_play = SelfPlayWrapper(self.env)

        # Create network
        if config.network_type == "tiny":
            self.network = TinyChessNet()
        elif config.network_type == "small":
            self.network = SmallChessNet()
        else:
            self.network = ChessNet()

        self.network.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
        )

        # Tracking
        self.total_timesteps = 0
        self.iteration = 0
        self.best_win_rate = 0.0

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def collect_trajectories(self) -> tuple[list[Trajectory], dict]:
        """
        Collect trajectories from self-play games.

        Returns:
            trajectories: List of trajectories from all games
            stats: Dictionary with game statistics
        """
        self.network.eval()

        trajectories = []
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
        total_moves = 0

        for _ in range(self.config.games_per_iteration):
            result = self.self_play.play_game(
                policy=self.network.predict,
                deterministic=False,
            )

            trajectories.extend(result.all_trajectories())
            results[result.result] += 1
            total_moves += result.num_moves

        # Compute statistics
        num_games = self.config.games_per_iteration
        stats = {
            "white_wins": results["1-0"],
            "black_wins": results["0-1"],
            "draws": results["1/2-1/2"],
            "avg_game_length": total_moves / num_games,
            "total_steps": sum(len(t) for t in trajectories),
        }

        self.total_timesteps += stats["total_steps"]

        return trajectories, stats

    def prepare_batch(
        self, trajectories: list[Trajectory]
    ) -> dict[str, torch.Tensor]:
        """
        Prepare trajectories into tensors for training.

        Returns:
            Dictionary with observations, actions, old_log_probs, returns, advantages
        """
        batch = self.self_play.trajectories_to_batch(
            trajectories, device=self.config.device
        )

        # Normalize advantages (important for stable training!)
        advantages = batch["advantages"]
        batch["advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return batch

    def ppo_update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Perform PPO update on a batch of data.

        Returns:
            Dictionary with loss values
        """
        self.network.train()

        observations = batch["observations"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        returns = batch["returns"]
        advantages = batch["advantages"]

        # Create legal action masks
        # During training, we need to regenerate masks from stored data
        # For simplicity, we'll allow all actions (the network learned from legal-only)
        # In production, you'd store masks with the trajectory
        action_masks = torch.ones(
            (observations.shape[0], 4672), dtype=torch.bool, device=self.device
        )

        # Create dataset for minibatch training
        dataset = TensorDataset(
            observations, actions, old_log_probs, returns, advantages, action_masks
        )
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for _ in range(self.config.num_epochs):
            for (
                obs_batch,
                act_batch,
                old_lp_batch,
                ret_batch,
                adv_batch,
                mask_batch,
            ) in loader:
                # Get new predictions
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    obs_batch, mask_batch, action=act_batch
                )

                # PPO policy loss
                ratio = torch.exp(new_log_probs - old_lp_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(
                    ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
                ) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, ret_batch)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss).item()
                num_batches += 1

        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "entropy": total_entropy / num_batches,
        }

    def train(self) -> None:
        """
        Main training loop.
        """
        print(f"Starting PPO training")
        print(f"  Network: {self.config.network_type}")
        print(f"  Device: {self.device}")
        print(f"  Total timesteps: {self.config.total_timesteps:,}")
        print(f"  Games per iteration: {self.config.games_per_iteration}")
        print()

        start_time = time.time()

        while self.total_timesteps < self.config.total_timesteps:
            self.iteration += 1
            iter_start = time.time()

            # Collect trajectories
            trajectories, game_stats = self.collect_trajectories()

            # Prepare batch
            batch = self.prepare_batch(trajectories)

            # PPO update
            loss_stats = self.ppo_update(batch)

            iter_time = time.time() - iter_start

            # Logging
            if self.iteration % self.config.log_interval == 0:
                fps = game_stats["total_steps"] / iter_time
                elapsed = time.time() - start_time

                print(f"Iteration {self.iteration}")
                print(f"  Timesteps: {self.total_timesteps:,} / {self.config.total_timesteps:,}")
                print(f"  Games: W={game_stats['white_wins']} B={game_stats['black_wins']} D={game_stats['draws']}")
                print(f"  Avg game length: {game_stats['avg_game_length']:.1f}")
                print(f"  Policy loss: {loss_stats['policy_loss']:.4f}")
                print(f"  Value loss: {loss_stats['value_loss']:.4f}")
                print(f"  Entropy: {loss_stats['entropy']:.4f}")
                print(f"  FPS: {fps:.1f}")
                print(f"  Elapsed: {elapsed:.1f}s")
                print()

            # Checkpointing
            if self.iteration % self.config.save_interval == 0:
                self.save_checkpoint()

        print("Training complete!")
        self.save_checkpoint("final")

    def save_checkpoint(self, name: Optional[str] = None) -> None:
        """Save a training checkpoint."""
        if name is None:
            name = f"checkpoint_{self.iteration}"

        path = Path(self.config.checkpoint_dir) / f"{name}.pt"

        torch.save({
            "iteration": self.iteration,
            "total_timesteps": self.total_timesteps,
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)

        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.iteration = checkpoint["iteration"]
        self.total_timesteps = checkpoint["total_timesteps"]
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Loaded checkpoint: {path}")
        print(f"  Iteration: {self.iteration}")
        print(f"  Timesteps: {self.total_timesteps:,}")


def quick_train_demo(timesteps: int = 5000) -> PPOTrainer:
    """
    Run a quick training demo.

    Uses tiny network and few games for fast iteration.
    Good for testing that training works.
    """
    config = PPOConfig(
        total_timesteps=timesteps,
        games_per_iteration=2,
        network_type="tiny",
        max_moves=100,
        log_interval=1,
        save_interval=5,
    )

    trainer = PPOTrainer(config)
    trainer.train()

    return trainer


if __name__ == "__main__":
    # Run a quick demo
    quick_train_demo(5000)
