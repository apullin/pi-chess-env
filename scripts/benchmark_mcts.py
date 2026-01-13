#!/usr/bin/env python3
"""
Benchmark an MCTS-trained chess model.

Usage:
    uv run python scripts/benchmark_mcts.py checkpoints/mcts_final.pt
    uv run python scripts/benchmark_mcts.py checkpoints/mcts_final.pt --simulations 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_env import ChessTensorEnv, create_random_policy
from chess_env.network import TinyChessNet, SmallChessNet, ChessNet
from chess_env.mcts import MCTS, MCTSConfig
import chess


MATE_IN_1_PUZZLES = [
    ("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "Re8"),
    ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "Ra8"),
    ("k7/8/1K6/8/8/8/8/7Q w - - 0 1", "Qa8"),
    ("7k/8/6K1/8/8/8/8/Q7 w - - 0 1", "Qa8"),
    ("k7/8/NK6/8/8/8/8/7Q w - - 0 1", "Qb1"),
    ("6rk/5Npp/8/8/8/8/8/4K2Q w - - 0 1", "Qxh7"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "Qxf7"),
    ("8/8/8/8/8/5k2/8/4QK2 w - - 0 1", "Qe3"),
    ("8/8/8/8/8/k7/8/1RK5 w - - 0 1", "Ra1"),
    ("7k/R7/R7/8/8/8/8/4K3 w - - 0 1", "Rh7"),
]


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load MCTS-trained model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["network_state_dict"]

    # Detect network size from state dict
    first_conv = state_dict["conv_input.weight"]
    num_channels = first_conv.shape[0]

    if num_channels <= 32:
        network = TinyChessNet()
    elif num_channels <= 64:
        network = SmallChessNet()
    else:
        network = ChessNet(num_channels=num_channels)

    network.load_state_dict(state_dict)
    network.to(device)
    network.eval()

    return network


def test_mate_in_1(mcts: MCTS, use_search: bool = True) -> dict:
    """Test on mate-in-1 puzzles."""
    env = ChessTensorEnv()
    correct = 0
    results = []

    for fen, expected_san in MATE_IN_1_PUZZLES:
        env.reset(options={"fen": fen})

        if use_search:
            action, _, _ = mcts.get_action_and_policy(env.board, temperature=0)
        else:
            # Just use network directly
            obs = env.obs_encoder.encode(env.board)
            mask = env.action_encoder.legal_action_mask(env.board)
            action, _, _ = mcts.network.predict(obs, mask, deterministic=True)

        predicted_san = env.action_to_san(action)

        is_correct = predicted_san == expected_san
        if not is_correct and predicted_san:
            test_board = chess.Board(fen)
            try:
                move = test_board.parse_san(predicted_san)
                test_board.push(move)
                if test_board.is_checkmate():
                    is_correct = True
            except:
                pass

        if is_correct:
            correct += 1

        results.append({
            "fen": fen,
            "expected": expected_san,
            "predicted": predicted_san,
            "correct": is_correct,
        })

    return {
        "accuracy": correct / len(MATE_IN_1_PUZZLES),
        "correct": correct,
        "total": len(MATE_IN_1_PUZZLES),
        "details": results,
    }


def test_vs_random(mcts: MCTS, num_games: int = 20, max_moves: int = 200) -> dict:
    """Test against random opponent."""
    env = ChessTensorEnv(max_moves=max_moves)
    random_policy = create_random_policy()

    results = {"model_wins": 0, "random_wins": 0, "draws": 0}
    game_lengths = []

    for game_num in range(num_games):
        env.reset()
        moves = 0
        model_plays_white = (game_num % 2 == 0)

        while True:
            is_model_turn = (env.board.turn == chess.WHITE) == model_plays_white

            if is_model_turn:
                action, _, _ = mcts.get_action_and_policy(env.board, temperature=0)
            else:
                obs = env.obs_encoder.encode(env.board)
                mask = env.action_encoder.legal_action_mask(env.board)
                action, _, _ = random_policy(obs, mask)

            obs, reward, terminated, truncated, info = env.step(action)
            moves += 1

            if terminated or truncated:
                game_lengths.append(moves)

                if terminated and env.board.is_checkmate():
                    winner_is_white = not env.board.turn
                    if winner_is_white == model_plays_white:
                        results["model_wins"] += 1
                    else:
                        results["random_wins"] += 1
                else:
                    results["draws"] += 1
                break

    return {
        "model_wins": results["model_wins"],
        "random_wins": results["random_wins"],
        "draws": results["draws"],
        "win_rate": results["model_wins"] / num_games,
        "avg_game_length": np.mean(game_lengths),
        "num_games": num_games,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark MCTS-trained chess model")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--simulations", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--games", type=int, default=20, help="Games per test")
    parser.add_argument("--no-search", action="store_true", help="Skip MCTS, use raw network")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")
    network = load_model(args.checkpoint, args.device)
    print(f"Model loaded on {args.device}")

    mcts_config = MCTSConfig(num_simulations=args.simulations)
    mcts = MCTS(network, config=mcts_config, device=args.device)

    search_mode = "raw network" if args.no_search else f"MCTS ({args.simulations} sims)"
    print(f"Using: {search_mode}")
    print()

    # Test 1: Mate-in-1
    print("=" * 60)
    print("TEST 1: Mate-in-1 Puzzles")
    print("=" * 60)
    mate_results = test_mate_in_1(mcts, use_search=not args.no_search)
    print(f"Accuracy: {mate_results['correct']}/{mate_results['total']} ({mate_results['accuracy']*100:.1f}%)")
    for r in mate_results["details"]:
        status = "Y" if r["correct"] else "X"
        print(f"  {status} Expected: {r['expected']:8s} Got: {r['predicted']}")
    print()

    # Test 2: vs Random
    print("=" * 60)
    print(f"TEST 2: vs Random Opponent ({args.games} games)")
    print("=" * 60)
    random_results = test_vs_random(mcts, num_games=args.games)
    print(f"Model wins: {random_results['model_wins']}")
    print(f"Random wins: {random_results['random_wins']}")
    print(f"Draws: {random_results['draws']}")
    print(f"Win rate: {random_results['win_rate']*100:.1f}%")
    print(f"Avg game length: {random_results['avg_game_length']:.1f} moves")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Mate-in-1 accuracy: {mate_results['accuracy']*100:.1f}%")
    print(f"Win rate vs random: {random_results['win_rate']*100:.1f}%")

    if mate_results['accuracy'] > 0.8 and random_results['win_rate'] > 0.9:
        print("Skill level: Beginner+ (knows basic tactics)")
    elif mate_results['accuracy'] > 0.5 or random_results['win_rate'] > 0.7:
        print("Skill level: Learning (some tactical awareness)")
    elif random_results['win_rate'] > 0.5:
        print("Skill level: Novice (better than random)")
    else:
        print("Skill level: Untrained (random-level play)")


if __name__ == "__main__":
    main()
