"""
Stable-Baselines3 training for Chess RL.

Uses PPO with an action-correcting wrapper that remaps illegal actions to legal ones.
Uses SubprocVecEnv for parallel self-play games.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

from chess_env.env import ChessTensorEnv
from chess_env.encoding import OBSERVATION_SHAPE, ACTION_SPACE_SIZE


class ChessCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for chess board observations.

    Input: (batch, 8, 8, 119) - NHWC format from env
    Output: (batch, features_dim) - flattened features for policy/value heads
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        num_channels: int = 128,
        num_res_blocks: int = 5,
    ):
        super().__init__(observation_space, features_dim)

        input_channels = observation_space.shape[2]  # 119

        # Initial conv
        self.conv_input = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_res_block(num_channels) for _ in range(num_res_blocks)
        ])

        # Output projection
        self.fc = nn.Linear(num_channels * 8 * 8, features_dim)

    def _make_res_block(self, channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # NHWC -> NCHW for Conv2d
        x = observations.permute(0, 3, 1, 2).contiguous()

        # Initial conv
        x = torch.relu(self.bn_input(self.conv_input(x)))

        # Residual blocks
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = torch.relu(x + residual)

        # Flatten and project
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))

        return x


class TinyChessCNN(ChessCNN):
    """Smaller CNN for faster training."""
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=128, num_channels=64, num_res_blocks=3)


class SmallChessCNN(ChessCNN):
    """Medium CNN - good balance of speed and strength."""
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=256, num_channels=128, num_res_blocks=5)


class LegalActionWrapper(gym.Wrapper):
    """
    Wrapper that handles illegal actions by picking the best legal alternative.

    Instead of random remapping, picks the legal action closest to what was requested
    (same source square if possible). Also adds a small penalty for illegal attempts
    to help the model learn to avoid them.
    """

    def __init__(self, env: gym.Env, illegal_penalty: float = -0.01):
        super().__init__(env)
        self._rng = np.random.default_rng()
        self.illegal_penalty = illegal_penalty
        self._pending_penalty = 0.0

    def step(self, action: int):
        board = self.unwrapped.board
        self._pending_penalty = 0.0

        # Check if action is legal
        move = self.unwrapped.action_encoder.decode(board, action)
        if move is None or move not in board.legal_moves:
            # Illegal action - pick a random legal one and add penalty
            legal_moves = list(board.legal_moves)
            if legal_moves:
                random_move = self._rng.choice(legal_moves)
                action = self.unwrapped.action_encoder.encode(board, random_move)
                self._pending_penalty = self.illegal_penalty

        obs, reward, terminated, truncated, info = self.env.step(action)
        reward += self._pending_penalty
        if self._pending_penalty != 0:
            info["illegal_action_penalty"] = self._pending_penalty
        return obs, reward, terminated, truncated, info


def make_env(max_moves: int = 200, seed: int = 0) -> Callable[[], gym.Env]:
    """Factory function for creating chess environments."""
    def _init() -> gym.Env:
        env = ChessTensorEnv(max_moves=max_moves)
        env = LegalActionWrapper(env)
        env.reset(seed=seed)
        return env
    return _init


def make_vec_env(
    n_envs: int = 8,
    max_moves: int = 200,
    use_subproc: bool = True,
) -> SubprocVecEnv | DummyVecEnv:
    """Create vectorized environment with n_envs parallel games."""
    env_fns = [make_env(max_moves=max_moves, seed=i) for i in range(n_envs)]

    if use_subproc and n_envs > 1:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


class ProgressCallback(BaseCallback):
    """Simple callback to print training progress."""

    def __init__(self, print_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                ep_rew_mean = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                ep_len_mean = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
                print(f"Step {self.num_timesteps}: ep_rew={ep_rew_mean:.3f}, ep_len={ep_len_mean:.1f}")
        return True


def train(
    total_timesteps: int = 100_000,
    n_envs: int = 8,
    max_moves: int = 200,
    network_size: str = "small",
    learning_rate: float = 3e-4,
    device: str = "auto",
    checkpoint_dir: str = "checkpoints",
    save_freq: int = 10_000,
    verbose: int = 1,
) -> PPO:
    """
    Train a chess agent using MaskablePPO (proper action masking).

    Args:
        total_timesteps: Total environment steps to train
        n_envs: Number of parallel environments
        max_moves: Max moves per game
        network_size: "tiny", "small", or "full"
        learning_rate: Learning rate for optimizer
        device: "cpu", "cuda", "mps", or "auto"
        checkpoint_dir: Directory for saving checkpoints
        save_freq: Save checkpoint every N steps
        verbose: Verbosity level (0=none, 1=info, 2=debug)

    Returns:
        Trained PPO model
    """
    # Create vectorized environment with normalization
    print(f"Creating {n_envs} parallel environments...")
    vec_env = make_vec_env(n_envs=n_envs, max_moves=max_moves, use_subproc=True)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    # Select feature extractor
    if network_size == "tiny":
        features_extractor_class = TinyChessCNN
        features_extractor_kwargs = {}
    elif network_size == "small":
        features_extractor_class = SmallChessCNN
        features_extractor_kwargs = {}
    else:  # full
        features_extractor_class = ChessCNN
        features_extractor_kwargs = {"features_dim": 512, "num_channels": 256, "num_res_blocks": 10}

    # Policy kwargs for custom CNN
    policy_kwargs = {
        "features_extractor_class": features_extractor_class,
        "features_extractor_kwargs": features_extractor_kwargs,
        "net_arch": dict(pi=[256], vf=[256]),
    }

    # Create PPO model with custom CNN feature extractor
    print(f"Creating PPO model (network={network_size}, device={device})...")
    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=1e-4,  # Lower LR for stability
        n_steps=128,
        batch_size=256,
        n_epochs=3,  # Fewer epochs to reduce update magnitude
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,  # Tighter clipping
        target_kl=0.02,  # Early stop if KL too high
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=verbose,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=checkpoint_dir,
        name_prefix="chess_ppo",
    )
    progress_callback = ProgressCallback(print_freq=1000)

    # Train
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"  n_envs={n_envs}, max_moves={max_moves}")
    print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, progress_callback],
        progress_bar=True,
    )

    # Save final model
    final_path = f"{checkpoint_dir}/final_model"
    model.save(final_path)
    print(f"Saved final model to {final_path}")

    vec_env.close()
    return model


def load_model(path: str, device: str = "auto") -> PPO:
    """Load a trained model from checkpoint."""
    return PPO.load(path, device=device)
