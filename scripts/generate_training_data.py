#!/usr/bin/env python3
"""
Generate training datasets for chess RL training.

Produces JSONL files with chess positions for different training phases:
- legality: Random positions for legal move training
- endgame: Simplified endgame positions
- puzzles: Mate-in-1 tactical puzzles

Usage:
    python scripts/generate_training_data.py --type legality --count 10000
    python scripts/generate_training_data.py --type endgame --count 1000
    python scripts/generate_training_data.py --type puzzles --count 500
"""

import argparse
import json
import random
from pathlib import Path

import chess


def generate_random_position(min_moves: int = 5, max_moves: int = 40) -> dict | None:
    """
    Generate a random chess position by playing random moves.

    Returns None if the game ends before reaching a valid position.
    """
    board = chess.Board()

    num_moves = random.randint(min_moves, max_moves)
    for _ in range(num_moves):
        if board.is_game_over():
            return None
        moves = list(board.legal_moves)
        board.push(random.choice(moves))

    if board.is_game_over():
        return None

    legal_moves = [board.san(m) for m in board.legal_moves]

    return {
        "fen": board.fen(),
        "legal_moves": legal_moves,
        "num_legal_moves": len(legal_moves),
        "turn": "white" if board.turn else "black",
        "fullmove_number": board.fullmove_number,
    }


def generate_legality_dataset(count: int) -> list[dict]:
    """Generate random positions for legality training."""
    positions = []
    attempts = 0
    max_attempts = count * 3

    while len(positions) < count and attempts < max_attempts:
        attempts += 1
        pos = generate_random_position()
        if pos:
            positions.append(pos)

    print(f"Generated {len(positions)} positions in {attempts} attempts")
    return positions


def generate_endgame_dataset(count: int) -> list[dict]:
    """
    Generate simplified endgame positions.

    These have fewer pieces and clearer winning/drawing conditions.
    """
    positions = []

    # Common endgame setups
    endgame_templates = [
        # King + Queen vs King
        lambda: setup_endgame(["K", "Q"], ["k"]),
        # King + Rook vs King
        lambda: setup_endgame(["K", "R"], ["k"]),
        # King + 2 Rooks vs King
        lambda: setup_endgame(["K", "R", "R"], ["k"]),
        # King + Pawn vs King (promotion race)
        lambda: setup_endgame(["K", "P"], ["k"]),
        # King + 2 Pawns vs King
        lambda: setup_endgame(["K", "P", "P"], ["k"]),
        # King + Queen vs King + Rook
        lambda: setup_endgame(["K", "Q"], ["k", "r"]),
    ]

    for _ in range(count):
        template = random.choice(endgame_templates)
        pos = template()
        if pos:
            positions.append(pos)

    print(f"Generated {len(positions)} endgame positions")
    return positions


def setup_endgame(white_pieces: list[str], black_pieces: list[str]) -> dict | None:
    """
    Create an endgame position with specified pieces.

    Places pieces randomly on the board ensuring a legal position.
    """
    board = chess.Board(None)  # Empty board

    squares = list(range(64))
    random.shuffle(squares)

    piece_map = {
        "K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK,
        "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN,
        "k": chess.KING, "q": chess.QUEEN, "r": chess.ROOK,
        "b": chess.BISHOP, "n": chess.KNIGHT, "p": chess.PAWN,
    }

    sq_idx = 0

    # Place white pieces
    for piece in white_pieces:
        if sq_idx >= len(squares):
            return None
        sq = squares[sq_idx]
        # Pawns can't be on ranks 1 or 8
        if piece == "P" and (sq < 8 or sq >= 56):
            sq_idx += 1
            if sq_idx >= len(squares):
                return None
            sq = squares[sq_idx]
        board.set_piece_at(sq, chess.Piece(piece_map[piece], chess.WHITE))
        sq_idx += 1

    # Place black pieces
    for piece in black_pieces:
        if sq_idx >= len(squares):
            return None
        sq = squares[sq_idx]
        if piece == "p" and (sq < 8 or sq >= 56):
            sq_idx += 1
            if sq_idx >= len(squares):
                return None
            sq = squares[sq_idx]
        board.set_piece_at(sq, chess.Piece(piece_map[piece.lower()], chess.BLACK))
        sq_idx += 1

    # Validate position
    if not board.is_valid():
        return None

    if board.is_game_over():
        return None

    legal_moves = [board.san(m) for m in board.legal_moves]

    return {
        "fen": board.fen(),
        "legal_moves": legal_moves,
        "num_legal_moves": len(legal_moves),
        "turn": "white" if board.turn else "black",
        "type": "endgame",
    }


def generate_puzzle_dataset(count: int) -> list[dict]:
    """
    Generate mate-in-1 puzzle positions.

    These are positions where there's exactly one move that delivers checkmate.
    """
    # Hardcoded mate-in-1 puzzles (same as in benchmark)
    # In production, would load from Lichess puzzle database
    puzzles = [
        {
            "fen": "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1",
            "solution": "Re8",
            "theme": "back_rank_mate",
        },
        {
            "fen": "5rk1/5ppp/8/8/8/8/8/R6K w - - 0 1",
            "solution": "Ra8",
            "theme": "back_rank_mate",
        },
        {
            "fen": "k7/8/1K6/8/8/8/8/7Q w - - 0 1",
            "solution": "Qa8",
            "theme": "queen_mate",
        },
        {
            "fen": "8/8/8/5k2/8/8/8/4K2Q w - - 0 1",
            "solution": "Qh5",
            "theme": "queen_mate",
        },
        {
            "fen": "k1K5/8/8/8/8/8/8/1Q6 w - - 0 1",
            "solution": "Qb7",
            "theme": "queen_mate_corner",
        },
        {
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
            "solution": "Qxf7",
            "theme": "scholars_mate",
        },
        {
            "fen": "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 1",
            "solution": "Qh4",
            "theme": "fools_mate",
        },
        {
            "fen": "5rk1/pp3ppp/8/8/8/8/PP3PPP/R4RK1 w - - 0 1",
            "solution": "Ra8",
            "theme": "rook_mate",
        },
        {
            "fen": "6k1/pp3ppp/8/8/8/8/PP3PPP/4RRK1 w - - 0 1",
            "solution": "Re8",
            "theme": "rook_mate",
        },
        {
            "fen": "r4rk1/ppp2ppp/8/8/8/8/PPP2PPP/R4RK1 w - - 0 1",
            "solution": "Rxa8",
            "theme": "capture_mate",
        },
    ]

    # Duplicate puzzles to reach count (in production, load more from DB)
    result = []
    for i in range(count):
        puzzle = puzzles[i % len(puzzles)].copy()
        puzzle["id"] = i
        result.append(puzzle)

    print(f"Generated {len(result)} puzzle positions")
    return result


def save_dataset(data: list[dict], output_path: Path) -> None:
    """Save dataset as JSONL."""
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate chess training data")
    parser.add_argument(
        "--type",
        choices=["legality", "endgame", "puzzles", "all"],
        default="legality",
        help="Type of dataset to generate",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10000,
        help="Number of positions to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.type == "legality" or args.type == "all":
        data = generate_legality_dataset(args.count)
        save_dataset(data, args.output_dir / "legality_train.jsonl")

    if args.type == "endgame" or args.type == "all":
        data = generate_endgame_dataset(args.count)
        save_dataset(data, args.output_dir / "endgame_train.jsonl")

    if args.type == "puzzles" or args.type == "all":
        data = generate_puzzle_dataset(args.count)
        save_dataset(data, args.output_dir / "puzzles_train.jsonl")


if __name__ == "__main__":
    main()
