#!/usr/bin/env python3
"""
Training script for chess RL using Stable-Baselines3.

Uses SB3's PPO with SubprocVecEnv for parallel self-play.

Usage:
    # Quick demo (tiny network)
    uv run python scripts/train.py --timesteps 50000 --network tiny --n-envs 4

    # Standard training (small network, 8 parallel envs)
    uv run python scripts/train.py --timesteps 100000 --network small --n-envs 8

    # Full training
    uv run python scripts/train.py --timesteps 1000000 --network full --n-envs 16
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_env.sb3_training import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train chess RL agent with SB3")

    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total timesteps to train (default: 100000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments (default: 8)",
    )
    parser.add_argument(
        "--network",
        choices=["tiny", "small", "full"],
        default="small",
        help="Network size (default: small)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Max moves per game (default: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: cpu, cuda, mps, or auto (default: auto)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        max_moves=args.max_moves,
        network_size=args.network,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
