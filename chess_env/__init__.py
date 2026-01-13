"""
Chess Environment for RL Training

This package provides:
- ChessTensorEnv: Gymnasium environment with AlphaZero-style tensor observations
- Action/observation encoders following AlphaZero's 4672 action space
- Self-play wrapper for training
- Neural networks (policy + value heads)
- PPO training loop
- (Future) PI Verifiers integration for LLM-based chess

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────┐
    │                      ChessTensorEnv                          │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
    │  │ Observation  │  │   Action     │  │   Core Chess     │   │
    │  │   Encoder    │  │   Decoder    │  │   (python-chess) │   │
    │  │ (8x8x119)    │  │   (4672)     │  │                  │   │
    │  └──────────────┘  └──────────────┘  └──────────────────┘   │
    └─────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
            ┌───────▼───────┐       ┌───────▼───────┐
            │  Self-Play    │       │  PI Verifiers │
            │   Wrapper     │       │   Wrapper     │
            │  (PPO train)  │       │  (LLM route)  │
            └───────────────┘       └───────────────┘
"""

from chess_env.env import ChessTensorEnv
from chess_env.encoding import (
    ObservationEncoder,
    ActionEncoder,
    OBSERVATION_SHAPE,
    ACTION_SPACE_SIZE,
)
from chess_env.self_play import SelfPlayWrapper, Trajectory, create_random_policy
from chess_env.network import ChessNet, SmallChessNet, TinyChessNet
from chess_env.sb3_training import train as sb3_train, load_model, make_vec_env
from chess_env.verifiers_env import ChessTextEnv, ChessTextConfig, load_environment

__version__ = "0.1.0"
__all__ = [
    # Tensor Environment (for CNN RL)
    "ChessTensorEnv",
    # Text Environment (for LLM RL)
    "ChessTextEnv",
    "ChessTextConfig",
    "load_environment",
    # Encoding
    "ObservationEncoder",
    "ActionEncoder",
    "OBSERVATION_SHAPE",
    "ACTION_SPACE_SIZE",
    # Self-play
    "SelfPlayWrapper",
    "Trajectory",
    "create_random_policy",
    # Networks
    "ChessNet",
    "SmallChessNet",
    "TinyChessNet",
    # Training (SB3)
    "sb3_train",
    "load_model",
    "make_vec_env",
]
