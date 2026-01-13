#!/usr/bin/env python3
"""
Training script for AlphaZero-style MCTS self-play.

Unlike PPO training which samples directly from the policy network,
MCTS training uses tree search to generate better training targets.

Usage:
    # Quick test (few simulations, small network)
    uv run python scripts/train_mcts.py --iterations 10 --games 5 --simulations 50 --network tiny

    # Longer training
    uv run python scripts/train_mcts.py --iterations 100 --games 20 --simulations 100 --network small
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_env.network import ChessNet, SmallChessNet, TinyChessNet
from chess_env.mcts import MCTS, MCTSConfig, MCTSTrajectory, mcts_self_play


def _play_one_game(args):
    """Worker function for parallel self-play."""
    network_state, network_class, mcts_config, max_moves, device = args

    # Recreate network in worker process
    network = network_class()
    network.load_state_dict(network_state)
    network.to(device)
    network.eval()

    mcts = MCTS(network, config=mcts_config, device=device)
    return mcts_self_play(mcts, max_moves=max_moves)


def train_on_trajectories(
    network: nn.Module,
    trajectories: list[MCTSTrajectory],
    optimizer: optim.Optimizer,
    device: torch.device,
    batch_size: int = 64,
    epochs: int = 1,
) -> dict[str, float]:
    """
    Train network on collected MCTS trajectories.

    Returns:
        Dictionary with loss values and diagnostics
    """
    all_obs = []
    all_policies = []
    all_values = []

    for traj in trajectories:
        all_obs.extend(traj.observations)
        all_policies.extend(traj.policy_targets)
        all_values.extend(traj.values)

    if not all_obs:
        return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0, "policy_entropy": 0.0}

    obs_tensor = torch.tensor(np.array(all_obs), dtype=torch.float32, device=device)
    policy_tensor = torch.tensor(np.array(all_policies), dtype=torch.float32, device=device)
    value_tensor = torch.tensor(np.array(all_values), dtype=torch.float32, device=device).unsqueeze(1)

    dataset = TensorDataset(obs_tensor, policy_tensor, value_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    network.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_batches = 0

    for _ in range(epochs):
        for obs_batch, policy_batch, value_batch in loader:
            policy_logits, value_pred = network(obs_batch)

            # Policy loss: cross-entropy with MCTS policy targets
            log_probs = torch.log_softmax(policy_logits, dim=-1)
            policy_loss = -torch.sum(policy_batch * log_probs, dim=-1).mean()

            # Policy entropy (how decisive the network is)
            probs = torch.softmax(policy_logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1).mean()

            # Value loss: MSE with game outcomes
            value_loss = nn.functional.mse_loss(value_pred, value_batch)

            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            num_batches += 1

    return {
        "policy_loss": total_policy_loss / num_batches if num_batches > 0 else 0.0,
        "value_loss": total_value_loss / num_batches if num_batches > 0 else 0.0,
        "total_loss": (total_policy_loss + total_value_loss) / num_batches if num_batches > 0 else 0.0,
        "policy_entropy": total_entropy / num_batches if num_batches > 0 else 0.0,
    }


def collect_self_play_games(
    network: nn.Module,
    network_class: type,
    mcts_config: MCTSConfig,
    num_games: int,
    max_moves: int = 200,
    num_workers: int = 4,
    device: str = "cpu",
) -> tuple[list[MCTSTrajectory], dict]:
    """
    Collect training data from parallel self-play games.

    Returns:
        trajectories: List of game trajectories
        stats: Dictionary with game statistics
    """
    # Prepare args for workers
    network_state = network.state_dict()
    worker_args = [
        (network_state, network_class, mcts_config, max_moves, device)
        for _ in range(num_games)
    ]

    # Run games in parallel
    trajectories = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        trajectories = list(executor.map(_play_one_game, worker_args))

    # Compute stats
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    total_positions = 0

    for traj in trajectories:
        total_positions += len(traj.observations)
        if traj.values and traj.values[0] == 1.0:
            results["1-0"] += 1
        elif traj.values and traj.values[0] == -1.0:
            results["0-1"] += 1
        else:
            results["1/2-1/2"] += 1

    stats = {
        "white_wins": results["1-0"],
        "black_wins": results["0-1"],
        "draws": results["1/2-1/2"],
        "total_positions": total_positions,
        "avg_game_length": total_positions / num_games if num_games > 0 else 0,
    }

    return trajectories, stats


def main():
    parser = argparse.ArgumentParser(description="Train chess with MCTS self-play")

    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--games", type=int, default=10,
                        help="Self-play games per iteration")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS simulations per move")
    parser.add_argument("--network", choices=["tiny", "small", "full"], default="small",
                        help="Network size")
    parser.add_argument("--max-moves", type=int, default=200,
                        help="Max moves per game")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Training epochs per iteration")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda/mps)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel self-play workers")

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Create network
    print(f"Creating {args.network} network on {device}")
    if args.network == "tiny":
        network_class = TinyChessNet
    elif args.network == "small":
        network_class = SmallChessNet
    else:
        network_class = ChessNet
    network = network_class()
    network.to(device)

    # Optimizer
    optimizer = optim.Adam(network.parameters(), lr=args.lr)

    # MCTS config
    mcts_config = MCTSConfig(
        num_simulations=args.simulations,
        c_puct=1.5,
        temperature=1.0,
    )

    print(f"MCTS config: {args.simulations} simulations, c_puct=1.5")
    print(f"Training: {args.iterations} iterations, {args.games} games/iter, {args.workers} workers")
    print()

    # Training loop
    total_games = 0
    start_time = time.time()

    for iteration in range(1, args.iterations + 1):
        iter_start = time.time()

        # Collect self-play games in parallel
        print(f"Iteration {iteration}: collecting {args.games} games...", end=" ", flush=True)
        trajectories, game_stats = collect_self_play_games(
            network=network,
            network_class=network_class,
            mcts_config=mcts_config,
            num_games=args.games,
            max_moves=args.max_moves,
            num_workers=args.workers,
            device=args.device,
        )
        total_games += args.games

        # Train on collected data
        loss_stats = train_on_trajectories(
            network, trajectories, optimizer, device,
            batch_size=args.batch_size, epochs=args.epochs
        )

        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        games_per_sec = args.games / iter_time
        positions_per_sec = game_stats['total_positions'] / iter_time

        # SB3-style logging
        print("done")
        print("-" * 45)
        print(f"| {'time/':<25} | {'':<12} |")
        print(f"|    {'games_per_sec':<21} | {games_per_sec:<12.1f} |")
        print(f"|    {'positions_per_sec':<21} | {positions_per_sec:<12.1f} |")
        print(f"|    {'iteration':<21} | {iteration:<12} |")
        print(f"|    {'time_elapsed':<21} | {elapsed:<12.0f} |")
        print(f"|    {'total_games':<21} | {total_games:<12} |")
        print(f"| {'self_play/':<25} | {'':<12} |")
        print(f"|    {'white_wins':<21} | {game_stats['white_wins']:<12} |")
        print(f"|    {'black_wins':<21} | {game_stats['black_wins']:<12} |")
        print(f"|    {'draws':<21} | {game_stats['draws']:<12} |")
        print(f"|    {'avg_game_length':<21} | {game_stats['avg_game_length']:<12.1f} |")
        print(f"| {'train/':<25} | {'':<12} |")
        print(f"|    {'policy_loss':<21} | {loss_stats['policy_loss']:<12.4f} |")
        print(f"|    {'value_loss':<21} | {loss_stats['value_loss']:<12.4f} |")
        print(f"|    {'policy_entropy':<21} | {loss_stats['policy_entropy']:<12.2f} |")
        print("-" * 45)
        print()

        # Save checkpoint
        if iteration % args.save_interval == 0:
            path = Path(args.checkpoint_dir) / f"mcts_checkpoint_{iteration}.pt"
            torch.save({
                "iteration": iteration,
                "network_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "total_games": total_games,
            }, path)
            print(f"Saved checkpoint: {path}")

    # Save final model
    final_path = Path(args.checkpoint_dir) / "mcts_final.pt"
    torch.save({
        "iteration": args.iterations,
        "network_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "total_games": total_games,
    }, final_path)
    print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()
