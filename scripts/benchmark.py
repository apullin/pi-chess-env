#!/usr/bin/env python3
"""
Benchmark a trained chess model.

Tests:
1. Mate-in-1 puzzle accuracy (can it find obvious wins?)
2. Win rate vs random opponent
3. Win rate vs Stockfish at various ELO levels

Usage:
    uv run python scripts/benchmark.py checkpoints/final_model.zip
    uv run python scripts/benchmark.py checkpoints/final_model.zip --elo 1320

Stockfish installation:
    brew install stockfish  # macOS
    apt install stockfish   # Ubuntu/Debian
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from chess_env import ChessTensorEnv, create_random_policy
import chess


# Mate-in-1 puzzles: (FEN, correct_move_san)
MATE_IN_1_PUZZLES = [
    # Back rank mates
    ("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "Re8"),
    ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "Ra8"),
    # Queen mates
    ("k7/8/1K6/8/8/8/8/7Q w - - 0 1", "Qa8"),
    ("7k/8/6K1/8/8/8/8/Q7 w - - 0 1", "Qa8"),
    # Knight + Queen
    ("k7/8/NK6/8/8/8/8/7Q w - - 0 1", "Qb1"),
    # Smothered mate setup (simplified)
    ("6rk/5Npp/8/8/8/8/8/4K2Q w - - 0 1", "Qxh7"),
    # Scholar's mate position
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "Qxf7"),
    # Simple queen mate
    ("8/8/8/8/8/5k2/8/4QK2 w - - 0 1", "Qe3"),
    # Rook + King
    ("8/8/8/8/8/k7/8/1RK5 w - - 0 1", "Ra1"),
    # Two rooks
    ("7k/R7/R7/8/8/8/8/4K3 w - - 0 1", "Rh7"),
]


def load_model(checkpoint_path: str, device: str = "auto"):
    """Load a trained SB3 model."""
    return PPO.load(checkpoint_path, device=device)


def model_predict(model, obs, action_mask):
    """Get model prediction, filtering to legal actions."""
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)
    # If action is illegal, find the legal action with highest probability
    if not action_mask[action]:
        # Get action probabilities from model
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.cpu().numpy().flatten()
        # Mask illegal actions and pick best legal one
        probs[~action_mask] = -np.inf
        action = int(np.argmax(probs))
    return action


def test_mate_in_1(model, device: str = "cpu") -> dict:
    """Test model on mate-in-1 puzzles."""
    env = ChessTensorEnv()
    correct = 0
    results = []

    for fen, expected_san in MATE_IN_1_PUZZLES:
        env.reset(options={"fen": fen})
        obs = env.obs_encoder.encode(env.board)
        mask = env.action_encoder.legal_action_mask(env.board)

        # Get model's move
        action = model_predict(model, obs, mask)
        predicted_san = env.action_to_san(action)

        # Check if it's the mating move
        is_correct = predicted_san == expected_san

        # Also accept any move that gives checkmate
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


def test_vs_random(model, num_games: int = 20, max_moves: int = 200) -> dict:
    """Test model against random opponent."""
    env = ChessTensorEnv(max_moves=max_moves)
    random_policy = create_random_policy()

    results = {"model_wins": 0, "random_wins": 0, "draws": 0}
    game_lengths = []

    for game_num in range(num_games):
        env.reset()
        obs = env.obs_encoder.encode(env.board)
        info = {"action_mask": env.action_encoder.legal_action_mask(env.board)}
        moves = 0
        model_plays_white = (game_num % 2 == 0)

        while True:
            is_model_turn = (env.board.turn == chess.WHITE) == model_plays_white

            if is_model_turn:
                action = model_predict(model, obs, info["action_mask"])
            else:
                action, _, _ = random_policy(obs, info["action_mask"])

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


def test_vs_stockfish(
    model,
    stockfish_path: str,
    elo: int = 1320,
    num_games: int = 10,
    max_moves: int = 200,
    time_limit: float = 0.1,
) -> dict:
    """Test model against Stockfish at a given ELO."""
    try:
        import chess.engine
    except ImportError:
        return {"error": "chess.engine not available"}

    # Try to find stockfish
    if not Path(stockfish_path).exists():
        for path in ["/usr/bin/stockfish", "/usr/local/bin/stockfish", "/opt/homebrew/bin/stockfish"]:
            if Path(path).exists():
                stockfish_path = path
                break
        else:
            return {"error": f"Stockfish not found at {stockfish_path}"}

    env = ChessTensorEnv(max_moves=max_moves)
    results = {"model_wins": 0, "stockfish_wins": 0, "draws": 0}
    game_lengths = []

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
    except Exception as e:
        return {"error": f"Failed to start Stockfish: {e}"}

    try:
        for game_num in range(num_games):
            env.reset()
            obs = env.obs_encoder.encode(env.board)
            info = {"action_mask": env.action_encoder.legal_action_mask(env.board)}
            moves = 0
            model_plays_white = (game_num % 2 == 0)

            while True:
                is_model_turn = (env.board.turn == chess.WHITE) == model_plays_white

                if is_model_turn:
                    action = model_predict(model, obs, info["action_mask"])
                    move = env.action_encoder.decode(env.board, action)
                else:
                    result = engine.play(env.board, chess.engine.Limit(time=time_limit))
                    move = result.move
                    action = env.action_encoder.encode(env.board, move)

                obs, reward, terminated, truncated, info = env.step(action)
                moves += 1

                if terminated or truncated:
                    game_lengths.append(moves)

                    if terminated and env.board.is_checkmate():
                        winner_is_white = not env.board.turn
                        if winner_is_white == model_plays_white:
                            results["model_wins"] += 1
                        else:
                            results["stockfish_wins"] += 1
                    else:
                        results["draws"] += 1
                    break
    finally:
        engine.quit()

    return {
        "model_wins": results["model_wins"],
        "stockfish_wins": results["stockfish_wins"],
        "draws": results["draws"],
        "win_rate": results["model_wins"] / num_games,
        "elo": elo,
        "avg_game_length": np.mean(game_lengths) if game_lengths else 0,
        "num_games": num_games,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark a trained chess model")
    parser.add_argument("checkpoint", type=str, help="Path to SB3 model (.zip)")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/mps/cuda/auto)")
    parser.add_argument("--games", type=int, default=20, help="Games per opponent")
    parser.add_argument("--stockfish", type=str, default="stockfish", help="Path to Stockfish")
    parser.add_argument("--elo", type=int, default=1320, help="Stockfish ELO (min 1320)")
    parser.add_argument("--skip-stockfish", action="store_true", help="Skip Stockfish test")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)
    print(f"Model loaded")
    print()

    # Test 1: Mate-in-1 puzzles
    print("=" * 60)
    print("TEST 1: Mate-in-1 Puzzles")
    print("=" * 60)
    mate_results = test_mate_in_1(model, args.device)
    print(f"Accuracy: {mate_results['correct']}/{mate_results['total']} ({mate_results['accuracy']*100:.1f}%)")
    print()
    for r in mate_results["details"]:
        status = "Y" if r["correct"] else "X"
        print(f"  {status} Expected: {r['expected']:8s} Got: {r['predicted']}")
    print()

    # Test 2: vs Random
    print("=" * 60)
    print(f"TEST 2: vs Random Opponent ({args.games} games)")
    print("=" * 60)
    random_results = test_vs_random(model, num_games=args.games)
    print(f"Model wins: {random_results['model_wins']}")
    print(f"Random wins: {random_results['random_wins']}")
    print(f"Draws: {random_results['draws']}")
    print(f"Win rate: {random_results['win_rate']*100:.1f}%")
    print(f"Avg game length: {random_results['avg_game_length']:.1f} moves")
    print()

    # Test 3: vs Stockfish
    stockfish_results = None
    if not args.skip_stockfish:
        print("=" * 60)
        print(f"TEST 3: vs Stockfish (ELO {args.elo}, {args.games} games)")
        print("=" * 60)
        stockfish_results = test_vs_stockfish(
            model, args.stockfish, elo=args.elo, num_games=args.games
        )
        if "error" in stockfish_results:
            print(f"Skipped: {stockfish_results['error']}")
        else:
            print(f"Model wins: {stockfish_results['model_wins']}")
            print(f"Stockfish wins: {stockfish_results['stockfish_wins']}")
            print(f"Draws: {stockfish_results['draws']}")
            print(f"Win rate: {stockfish_results['win_rate']*100:.1f}%")
            print(f"Avg game length: {stockfish_results['avg_game_length']:.1f} moves")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Mate-in-1 accuracy: {mate_results['accuracy']*100:.1f}%")
    print(f"Win rate vs random: {random_results['win_rate']*100:.1f}%")
    if stockfish_results and "error" not in stockfish_results:
        print(f"Win rate vs Stockfish {args.elo}: {stockfish_results['win_rate']*100:.1f}%")

    # Skill assessment
    if stockfish_results and "error" not in stockfish_results and stockfish_results['win_rate'] > 0.5:
        print(f"Skill level: ~{args.elo} ELO (beats Stockfish at this level)")
    elif mate_results['accuracy'] > 0.8 and random_results['win_rate'] > 0.9:
        print("Skill level: Beginner+ (knows basic tactics)")
    elif mate_results['accuracy'] > 0.5 or random_results['win_rate'] > 0.7:
        print("Skill level: Learning (some tactical awareness)")
    elif random_results['win_rate'] > 0.5:
        print("Skill level: Novice (better than random)")
    else:
        print("Skill level: Untrained (random-level play)")


if __name__ == "__main__":
    main()
