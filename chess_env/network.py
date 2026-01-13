"""
Neural Network Architecture for Chess RL.

This module implements an AlphaZero-inspired neural network with:
- Shared residual backbone (feature extraction)
- Policy head (action probabilities)
- Value head (position evaluation)

=== ARCHITECTURE OVERVIEW ===

    Input: 8×8×119 tensor (board state)
           │
           ▼
    ┌─────────────────┐
    │  Convolutional  │  Initial feature extraction
    │     Block       │  (Conv2d + BatchNorm + ReLU)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Residual      │  Stack of residual blocks
    │    Blocks       │  (skip connections for gradient flow)
    │   (× N_RES)     │
    └────────┬────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌───────────┐ ┌───────────┐
│  Policy   │ │   Value   │
│   Head    │ │   Head    │
└─────┬─────┘ └─────┬─────┘
      │             │
      ▼             ▼
  4672 logits    1 scalar
  (action probs) (position value)

=== WHY RESIDUAL BLOCKS? ===

Residual connections (skip connections) solve the "vanishing gradient" problem:
- In deep networks, gradients can become tiny when backpropagating
- Skip connections provide a "highway" for gradients to flow directly
- This allows training much deeper networks effectively

    Input ───┐
      │      │
      ▼      │
    Conv     │
      │      │
      ▼      │
    Conv     │
      │      │
      ▼      │
    + ◄──────┘  (add skip connection)
      │
      ▼
   Output

=== POLICY HEAD ===

Outputs logits for each of 4672 possible actions.
During inference, illegal actions are masked (set to -inf) before softmax.

=== VALUE HEAD ===

Outputs a single scalar in [-1, 1] representing expected game outcome:
- +1: position is winning
- 0: position is drawn
- -1: position is losing

This is used for:
1. Computing advantages in PPO (A = R - V)
2. Bootstrapping in GAE for non-terminal states
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_env.encoding import ACTION_SPACE_SIZE, OBSERVATION_SHAPE


class ResidualBlock(nn.Module):
    """
    A single residual block with skip connection.

    Structure:
        input -> Conv -> BN -> ReLU -> Conv -> BN -> (+input) -> ReLU -> output
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # Skip connection
        out = F.relu(out)
        return out


class ChessNet(nn.Module):
    """
    AlphaZero-style neural network for chess.

    Args:
        num_channels: Number of channels in residual blocks (default: 128)
        num_res_blocks: Number of residual blocks (default: 10)

    The network takes board observations (8×8×119) and outputs:
        - policy_logits: (batch, 4672) raw logits for each action
        - value: (batch, 1) estimated position value in [-1, 1]

    Smaller networks (fewer channels/blocks) are faster but weaker.
    AlphaZero used 256 channels and 19 residual blocks.
    For CPU training demos, 64-128 channels and 5-10 blocks work well.
    """

    def __init__(
        self,
        num_channels: int = 128,
        num_res_blocks: int = 10,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        input_channels = OBSERVATION_SHAPE[2]  # 119

        # Initial convolution
        self.conv_input = nn.Conv2d(
            input_channels, num_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, ACTION_SPACE_SIZE)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Board observation tensor (batch, 8, 8, 119)

        Returns:
            policy_logits: (batch, 4672) action logits
            value: (batch, 1) position value
        """
        # NHWC -> NCHW: Conv2d expects (batch, channels, H, W), we store (batch, H, W, channels)
        # .contiguous() required for MPS/GPU backends after permute
        x = x.permute(0, 3, 1, 2).contiguous()

        # Initial convolution
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.reshape(policy.size(0), -1)  # Flatten
        policy_logits = self.policy_fc(policy)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.reshape(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]

        return policy_logits, value

    def predict(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """
        Predict action for a single observation (inference mode).

        This is the interface expected by SelfPlayWrapper.

        Args:
            observation: Board state (8, 8, 119)
            action_mask: Boolean mask of legal actions (4672,)
            deterministic: If True, return argmax; else sample from distribution

        Returns:
            action: Selected action index
            log_prob: Log probability of the action
            value: Estimated position value
        """
        self.eval()
        with torch.no_grad():
            # Get the device the model is on
            device = next(self.parameters()).device

            # Add batch dimension and move to device
            obs_t = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            policy_logits, value = self(obs_t)

            # Mask illegal actions (keep on same device)
            mask_t = torch.tensor(action_mask, dtype=torch.bool, device=device)
            policy_logits[0, ~mask_t] = float("-inf")

            # Compute probabilities
            probs = F.softmax(policy_logits, dim=-1)
            log_probs = F.log_softmax(policy_logits, dim=-1)

            if deterministic:
                action = policy_logits.argmax(dim=-1).item()
            else:
                # Sample from distribution
                action = torch.multinomial(probs, num_samples=1).item()

            log_prob = log_probs[0, action].item()
            value_scalar = value[0, 0].item()

        return action, log_prob, value_scalar

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log_prob, entropy, and value for a batch of observations.

        Used during PPO training.

        Args:
            obs: Batch of observations (batch, 8, 8, 119)
            action_mask: Batch of action masks (batch, 4672)
            action: Optional pre-selected actions (batch,). If None, sample new actions.

        Returns:
            action: Selected actions (batch,)
            log_prob: Log probabilities of actions (batch,)
            entropy: Policy entropy (batch,)
            value: Value estimates (batch,)
        """
        policy_logits, value = self(obs)

        # Mask illegal actions
        policy_logits = policy_logits.masked_fill(~action_mask, float("-inf"))

        # Compute distribution
        probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)

        if action is None:
            # Sample new actions
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Get log prob of selected actions
        log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Compute entropy (for exploration bonus)
        # Only consider legal actions for entropy calculation
        entropy = -(probs * log_probs).sum(dim=-1)
        # Handle inf * 0 = nan case
        entropy = torch.where(torch.isnan(entropy), torch.zeros_like(entropy), entropy)

        return action, log_prob, entropy, value.squeeze(-1)


class SmallChessNet(ChessNet):
    """
    A smaller network for faster training on CPU.

    Uses 64 channels and 5 residual blocks (vs 128/10 for default).
    Good for initial testing and demonstrations.
    """

    def __init__(self):
        super().__init__(num_channels=64, num_res_blocks=5)


class TinyChessNet(ChessNet):
    """
    Minimal network for very fast iteration.

    Uses 32 channels and 2 residual blocks.
    Not expected to play well, but trains in seconds.
    """

    def __init__(self):
        super().__init__(num_channels=32, num_res_blocks=2)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: ChessNet) -> None:
    """Print a summary of the model architecture."""
    print(f"ChessNet Summary:")
    print(f"  Channels: {model.num_channels}")
    print(f"  Residual blocks: {model.num_res_blocks}")
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"  Input shape: (batch, 8, 8, 119)")
    print(f"  Policy output: (batch, {ACTION_SPACE_SIZE})")
    print(f"  Value output: (batch, 1)")
