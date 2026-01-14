#!/usr/bin/env python3
"""
Benchmark LLM chess playing ability.

Tests:
1. Mate-in-1 puzzle accuracy (10 puzzles)
2. Move legality rate (50 random positions)
3. Win rate vs random opponent (10 games)

Usage:
    # Install ollama first
    brew install ollama
    ollama serve  # in another terminal
    ollama pull qwen2.5:0.5b

    # Run benchmark
    uv run python scripts/benchmark_llm.py qwen2.5:0.5b
    uv run python scripts/benchmark_llm.py qwen2.5:0.5b --games 5
    uv run python scripts/benchmark_llm.py --all  # benchmark all recommended models
"""

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_env.verifiers_env import ChessTextEnv, ChessTextConfig, create_system_prompt
from chess_env.llm_player import create_player, parse_move_from_response, RECOMMENDED_MODELS, REASONING_MODELS


ENHANCED_SYSTEM_PROMPT = """You are playing chess. On each turn, you will see the current board state and a list of legal moves.

CRITICAL: You MUST choose a move from the "Legal moves" list shown in the observation. Any move not in this list is illegal and will be rejected.

Format your response as:
<move>YOUR_MOVE</move>

The move must EXACTLY match one of the legal moves listed. For example, if the legal moves include "e4", respond with:
<move>e4</move>

Common move formats:
- Pawn moves: e4, d5, exd5 (capture)
- Piece moves: Nf3, Bb5, Qd1
- Castling: O-O (kingside), O-O-O (queenside)
- Promotion: e8=Q
- Check: Qf7+, Bb5+

Play to win. Always choose from the legal moves list."""


# Mate-in-1 puzzles: (FEN, correct_move_san, description)
MATE_IN_1_PUZZLES = [
    ("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "Re8", "Back rank mate"),
    ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "Ra8", "Back rank mate"),
    ("k7/8/1K6/8/8/8/8/7Q w - - 0 1", "Qa8", "Queen mate"),
    ("7k/8/6K1/8/8/8/8/Q7 w - - 0 1", "Qa8", "Queen mate"),
    ("k7/8/NK6/8/8/8/8/7Q w - - 0 1", "Qb1", "Queen + Knight"),
    ("6rk/5Npp/8/8/8/8/8/4K2Q w - - 0 1", "Qxh7", "Smothered setup"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "Qxf7", "Scholar's mate"),
    ("8/8/8/8/8/5k2/8/4QK2 w - - 0 1", "Qe3", "Simple queen mate"),
    ("8/8/8/8/8/k7/8/1RK5 w - - 0 1", "Ra1", "Rook mate"),
    ("7k/R7/R7/8/8/8/8/4K3 w - - 0 1", "Rh7", "Two rooks mate"),
]


def test_mate_in_1(player, system_prompt: str, verbose: bool = False) -> dict:
    """
    Test LLM on mate-in-1 puzzles.

    Returns:
        Dict with accuracy, correct count, total, and details
    """
    env = ChessTextEnv()
    correct = 0
    results = []
    total_time = 0.0

    for fen, expected_san, description in MATE_IN_1_PUZZLES:
        prompt, state = env.reset(start_fen=fen)

        start = time.time()
        response = player.generate(prompt, system_prompt)
        elapsed = time.time() - start
        total_time += elapsed

        # Parse the move
        predicted_move = parse_move_from_response(response)

        # Check if correct (either exact match or any checkmate)
        is_correct = False
        is_checkmate = False

        if predicted_move:
            try:
                board = chess.Board(fen)
                move = board.parse_san(predicted_move)
                board.push(move)
                is_checkmate = board.is_checkmate()
                is_correct = (predicted_move == expected_san) or is_checkmate
            except:
                pass

        if is_correct:
            correct += 1

        results.append({
            "fen": fen,
            "description": description,
            "expected": expected_san,
            "predicted": predicted_move,
            "correct": is_correct,
            "is_checkmate": is_checkmate,
            "time": elapsed,
        })

        if verbose:
            status = "Y" if is_correct else "X"
            print(f"  {status} {description}: expected {expected_san}, got {predicted_move}")

    return {
        "accuracy": correct / len(MATE_IN_1_PUZZLES),
        "correct": correct,
        "total": len(MATE_IN_1_PUZZLES),
        "avg_time": total_time / len(MATE_IN_1_PUZZLES),
        "details": results,
    }


def test_move_legality(player, system_prompt: str, num_positions: int = 50, verbose: bool = False) -> dict:
    """
    Test LLM's ability to produce legal moves from random positions.

    Returns:
        Dict with legality rate, format rate, and details
    """
    env = ChessTextEnv()
    legal_count = 0
    format_count = 0
    total_time = 0.0

    for i in range(num_positions):
        # Generate a random position by playing random moves
        board = chess.Board()
        num_moves = np.random.randint(5, 40)
        for _ in range(num_moves):
            if board.is_game_over():
                break
            move = np.random.choice(list(board.legal_moves))
            board.push(move)

        if board.is_game_over():
            continue

        # Get LLM's move
        prompt, state = env.reset(start_fen=board.fen())

        start = time.time()
        response = player.generate(prompt, system_prompt)
        elapsed = time.time() - start
        total_time += elapsed

        predicted_move = parse_move_from_response(response)

        if predicted_move:
            format_count += 1
            try:
                move = board.parse_san(predicted_move)
                if move in board.legal_moves:
                    legal_count += 1
            except:
                pass

        if verbose and (i + 1) % 10 == 0:
            print(f"  Tested {i + 1}/{num_positions} positions...")

    tested = num_positions
    return {
        "legality_rate": legal_count / tested if tested > 0 else 0,
        "format_rate": format_count / tested if tested > 0 else 0,
        "legal_count": legal_count,
        "format_count": format_count,
        "total": tested,
        "avg_time": total_time / tested if tested > 0 else 0,
    }


def test_vs_random_with_retries(player, system_prompt: str, num_games: int = 10, max_moves: int = 100, max_retries: int = 3, verbose: bool = False) -> dict:
    """
    Test LLM against random opponent with retries for illegal moves.

    Unlike test_vs_random, illegal moves don't count toward max_moves.
    The LLM gets max_retries attempts per turn before forfeiting.
    """
    config = ChessTextConfig(max_moves=max_moves * 10)  # High limit, we track ourselves
    env = ChessTextEnv(config)

    results = {"llm_wins": 0, "random_wins": 0, "draws": 0, "forfeits": 0}
    game_lengths = []
    retry_counts = []
    total_time = 0.0

    for game_num in range(num_games):
        prompt, state = env.reset()
        llm_plays_white = (game_num % 2 == 0)
        legal_moves = 0  # Only count successful moves
        total_retries = 0
        forfeited = False

        while not state.done and legal_moves < max_moves:
            is_llm_turn = (state.board.turn == chess.WHITE) == llm_plays_white

            if is_llm_turn:
                # LLM turn with retries
                retries = 0
                while retries < max_retries:
                    start = time.time()
                    response = player.generate(prompt, system_prompt)
                    elapsed = time.time() - start
                    total_time += elapsed

                    prompt, state, done, reward, info = env.step(response, state)

                    if not info.get("error"):
                        # Legal move made
                        legal_moves += 1
                        break
                    else:
                        retries += 1
                        total_retries += 1
                        if retries >= max_retries:
                            # Forfeit after too many retries - LLM loses
                            forfeited = True
                            results["random_wins"] += 1
                            results["forfeits"] += 1
                            break

                if forfeited:
                    break
            else:
                # Random opponent - always legal
                legal_moves_list = list(state.board.legal_moves)
                if not legal_moves_list:
                    break
                move = np.random.choice(legal_moves_list)
                response = f"<move>{state.board.san(move)}</move>"
                prompt, state, done, reward, info = env.step(response, state)
                legal_moves += 1

        game_lengths.append(legal_moves)
        retry_counts.append(total_retries)

        if not forfeited:
            # Determine winner
            if state.result == "1-0":
                if llm_plays_white:
                    results["llm_wins"] += 1
                else:
                    results["random_wins"] += 1
            elif state.result == "0-1":
                if not llm_plays_white:
                    results["llm_wins"] += 1
                else:
                    results["random_wins"] += 1
            else:
                results["draws"] += 1

        if verbose:
            color = "White" if llm_plays_white else "Black"
            status = "FORFEIT" if forfeited else state.result
            print(f"  Game {game_num + 1}: LLM ({color}) - {status} in {legal_moves} moves, {total_retries} retries")

    return {
        "win_rate": results["llm_wins"] / num_games,
        "llm_wins": results["llm_wins"],
        "random_wins": results["random_wins"],
        "draws": results["draws"],
        "forfeits": results["forfeits"],
        "num_games": num_games,
        "avg_game_length": np.mean(game_lengths),
        "avg_retries_per_game": np.mean(retry_counts),
        "avg_move_time": total_time / max(sum(game_lengths), 1),
    }


def test_vs_random(player, system_prompt: str, num_games: int = 10, max_moves: int = 200, verbose: bool = False) -> dict:
    """
    Test LLM against a random opponent.

    Returns:
        Dict with win rate, game results, and stats
    """
    config = ChessTextConfig(max_moves=max_moves)
    env = ChessTextEnv(config)

    results = {"llm_wins": 0, "random_wins": 0, "draws": 0}
    game_lengths = []
    illegal_moves = []
    total_time = 0.0

    for game_num in range(num_games):
        prompt, state = env.reset()
        llm_plays_white = (game_num % 2 == 0)
        moves = 0
        game_illegal = 0

        while not state.done:
            is_llm_turn = (state.board.turn == chess.WHITE) == llm_plays_white

            if is_llm_turn:
                start = time.time()
                response = player.generate(prompt, system_prompt)
                elapsed = time.time() - start
                total_time += elapsed
            else:
                # Random opponent
                legal_moves = list(state.board.legal_moves)
                if not legal_moves:
                    break
                move = np.random.choice(legal_moves)
                response = f"<move>{state.board.san(move)}</move>"

            prompt, state, done, reward, info = env.step(response, state)
            moves += 1

            if is_llm_turn and info.get("error"):
                game_illegal += 1

            if moves >= max_moves:
                break

        game_lengths.append(moves)
        illegal_moves.append(game_illegal)

        # Determine winner
        if state.result == "1-0":
            if llm_plays_white:
                results["llm_wins"] += 1
            else:
                results["random_wins"] += 1
        elif state.result == "0-1":
            if not llm_plays_white:
                results["llm_wins"] += 1
            else:
                results["random_wins"] += 1
        else:
            results["draws"] += 1

        if verbose:
            color = "White" if llm_plays_white else "Black"
            print(f"  Game {game_num + 1}: LLM ({color}) - {state.result} in {moves} moves, {game_illegal} illegal")

    return {
        "win_rate": results["llm_wins"] / num_games,
        "llm_wins": results["llm_wins"],
        "random_wins": results["random_wins"],
        "draws": results["draws"],
        "num_games": num_games,
        "avg_game_length": np.mean(game_lengths),
        "avg_illegal_per_game": np.mean(illegal_moves),
        "avg_move_time": total_time / sum(game_lengths) if sum(game_lengths) > 0 else 0,
    }


def run_benchmark(model: str, num_games: int = 10, num_positions: int = 50, verbose: bool = True, enhanced: bool = False, reasoning: bool = False):
    """Run full benchmark suite for a model."""
    modes = []
    if enhanced:
        modes.append("ENHANCED")
    if reasoning:
        modes.append("REASONING")
    mode = "+".join(modes) if modes else "STANDARD"
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {model} ({mode})")
    print(f"{'='*60}")

    # Create player - reasoning models need higher token limits for chain-of-thought
    max_tokens = 2048 if reasoning else 128
    try:
        player = create_player("ollama", model=model, temperature=0.3, max_tokens=max_tokens)
    except Exception as e:
        print(f"Failed to create player: {e}")
        return None

    # Check if model is available
    if not player.is_available():
        print(f"Model {model} not available. Pull it with: ollama pull {model}")
        return None

    # Use enhanced prompt if requested
    if enhanced:
        system_prompt = ENHANCED_SYSTEM_PROMPT
        print("Using ENHANCED system prompt (emphasizes legal moves list)")
    else:
        system_prompt = create_system_prompt(ChessTextConfig())

    if reasoning:
        print(f"Using REASONING mode (max_tokens={max_tokens} for chain-of-thought)")

    results = {}

    # Test 1: Mate-in-1
    print(f"\n{'='*60}")
    print("TEST 1: Mate-in-1 Puzzles")
    print("="*60)
    mate_results = test_mate_in_1(player, system_prompt, verbose=verbose)
    results["mate_in_1"] = mate_results
    print(f"\nAccuracy: {mate_results['correct']}/{mate_results['total']} ({mate_results['accuracy']*100:.1f}%)")
    print(f"Avg response time: {mate_results['avg_time']:.2f}s")

    # Test 2: Move Legality
    print(f"\n{'='*60}")
    print(f"TEST 2: Move Legality ({num_positions} positions)")
    print("="*60)
    legality_results = test_move_legality(player, system_prompt, num_positions, verbose=verbose)
    results["legality"] = legality_results
    print(f"\nFormat rate: {legality_results['format_rate']*100:.1f}%")
    print(f"Legality rate: {legality_results['legality_rate']*100:.1f}%")
    print(f"Avg response time: {legality_results['avg_time']:.2f}s")

    # Test 3: vs Random
    print(f"\n{'='*60}")
    if enhanced:
        print(f"TEST 3: vs Random Opponent ({num_games} games, with retries)")
        print("="*60)
        random_results = test_vs_random_with_retries(player, system_prompt, num_games, max_retries=3, verbose=verbose)
        results["vs_random"] = random_results
        print(f"\nLLM wins: {random_results['llm_wins']}")
        print(f"Random wins: {random_results['random_wins']}")
        print(f"Draws: {random_results['draws']}")
        print(f"Forfeits (3+ retries): {random_results['forfeits']}")
        print(f"Win rate: {random_results['win_rate']*100:.1f}%")
        print(f"Avg game length: {random_results['avg_game_length']:.1f} moves")
        print(f"Avg retries per game: {random_results['avg_retries_per_game']:.1f}")
    else:
        print(f"TEST 3: vs Random Opponent ({num_games} games)")
        print("="*60)
        random_results = test_vs_random(player, system_prompt, num_games, verbose=verbose)
        results["vs_random"] = random_results
        print(f"\nLLM wins: {random_results['llm_wins']}")
        print(f"Random wins: {random_results['random_wins']}")
        print(f"Draws: {random_results['draws']}")
        print(f"Win rate: {random_results['win_rate']*100:.1f}%")
        print(f"Avg game length: {random_results['avg_game_length']:.1f} moves")
        print(f"Avg illegal moves per game: {random_results['avg_illegal_per_game']:.1f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"Model: {model}")
    print(f"Mate-in-1 accuracy: {mate_results['accuracy']*100:.1f}%")
    print(f"Move legality rate: {legality_results['legality_rate']*100:.1f}%")
    print(f"Win rate vs random: {random_results['win_rate']*100:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM chess playing ability")
    parser.add_argument("model", nargs="?", type=str, help="Model name (e.g., qwen2.5:0.5b)")
    parser.add_argument("--all", action="store_true", help="Benchmark all recommended models")
    parser.add_argument("--games", type=int, default=10, help="Number of games vs random")
    parser.add_argument("--positions", type=int, default=50, help="Number of positions for legality test")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")
    parser.add_argument("--enhanced", "-e", action="store_true", help="Use enhanced prompt + retry logic")
    parser.add_argument("--reasoning", "-r", action="store_true", help="Reasoning model mode (higher token limit)")
    parser.add_argument("--list-models", action="store_true", help="List recommended models")
    args = parser.parse_args()

    if args.list_models:
        print("\nRecommended models for benchmarking:")
        print("-" * 50)
        for model, info in RECOMMENDED_MODELS.items():
            print(f"  {model:<20} {info['params']:>6} params, ~{info['size_gb']:.1f}GB")
            print(f"    {info['description']}")
        print("\nReasoning models (use with --reasoning flag):")
        print("-" * 50)
        for model, info in REASONING_MODELS.items():
            print(f"  {model:<20} {info['params']:>6} params, ~{info['size_gb']:.1f}GB")
            print(f"    {info['description']}")
        print("\nInstall with: ollama pull <model>")
        return

    if args.all:
        # Benchmark all recommended models (or reasoning models with --reasoning)
        models_to_test = REASONING_MODELS if args.reasoning else RECOMMENDED_MODELS
        all_results = {}
        for model in models_to_test.keys():
            try:
                results = run_benchmark(model, args.games, args.positions, not args.quiet, args.enhanced, args.reasoning)
                if results:
                    all_results[model] = results
            except Exception as e:
                print(f"Error benchmarking {model}: {e}")

        # Print comparison table
        if all_results:
            print(f"\n{'='*70}")
            print("COMPARISON TABLE")
            print("="*70)
            print(f"{'Model':<20} {'Mate-in-1':>12} {'Legality':>12} {'vs Random':>12}")
            print("-"*70)
            for model, results in all_results.items():
                mate = results["mate_in_1"]["accuracy"] * 100
                legal = results["legality"]["legality_rate"] * 100
                win = results["vs_random"]["win_rate"] * 100
                print(f"{model:<20} {mate:>11.1f}% {legal:>11.1f}% {win:>11.1f}%")

    elif args.model:
        run_benchmark(args.model, args.games, args.positions, not args.quiet, args.enhanced, args.reasoning)
    else:
        parser.print_help()
        print("\nExample: uv run python scripts/benchmark_llm.py qwen2.5:0.5b")


if __name__ == "__main__":
    main()
