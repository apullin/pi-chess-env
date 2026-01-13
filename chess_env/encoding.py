"""
AlphaZero-style encoding for chess observations and actions.

=== OBSERVATION ENCODING (8 × 8 × 119 planes) ===

The observation tensor encodes the board state from the CURRENT PLAYER's perspective.
This means the board is always "flipped" so the current player's pieces are at the bottom.

Structure (119 planes total):
    - 112 planes: Historical piece positions (8 time steps × 14 planes each)
    - 7 planes: Auxiliary features (castling, side to move, etc.)

For each of the 8 time steps (T=0 is current, T=7 is 7 half-moves ago):
    Planes 0-5:   Current player's pieces (P, N, B, R, Q, K)
    Planes 6-11:  Opponent's pieces (P, N, B, R, Q, K)
    Plane 12:     Repetition count == 1 (position seen once before)
    Plane 13:     Repetition count >= 2 (position seen twice+ before)

Auxiliary planes (constant across all squares):
    Plane 112: Color (1 if White to move, 0 if Black)
    Plane 113: Total move count (normalized)
    Plane 114: White kingside castling rights
    Plane 115: White queenside castling rights
    Plane 116: Black kingside castling rights
    Plane 117: Black queenside castling rights
    Plane 118: Halfmove clock (for 50-move rule, normalized)

Why flip the board?
    - Makes the network's job easier: it always "plays from the bottom"
    - Reduces the effective state space by half
    - The network learns one perspective, not two

=== ACTION ENCODING (4672 actions = 8 × 8 × 73) ===

Each action is encoded as (from_square, move_type) where:
    - from_square: Which of the 64 squares to pick up a piece from (8×8)
    - move_type: One of 73 possible move directions/types

The 73 move types:
    Planes 0-55:  Queen-like moves (8 directions × 7 distances)
                  Directions: N, NE, E, SE, S, SW, W, NW
                  Distances: 1-7 squares
    Planes 56-63: Knight moves (8 possible L-shapes)
    Planes 64-66: Underpromotion to knight (straight, left-capture, right-capture)
    Planes 67-69: Underpromotion to bishop (straight, left-capture, right-capture)
    Planes 70-72: Underpromotion to rook (straight, left-capture, right-capture)

Note: Queen promotions use the regular queen-move planes (most common case).

Why this encoding?
    - Bijective: every legal chess move maps to exactly one action, and vice versa
    - Efficient: illegal moves are masked out before sampling
    - Proven: AlphaZero and Leela Chess Zero use this exact scheme
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import chess
import numpy as np

# Observation tensor dimensions
HISTORY_LENGTH = 8  # Number of past positions to include
PLANES_PER_POSITION = 14  # 6 + 6 pieces + 2 repetition planes
AUXILIARY_PLANES = 7  # Castling (4) + color (1) + move count (1) + halfmove (1)
TOTAL_PLANES = HISTORY_LENGTH * PLANES_PER_POSITION + AUXILIARY_PLANES  # 119

# NHWC format (8, 8, 119): intuitive "8x8 board with 119 features per square"
# Network permutes to NCHW before Conv2d (see network.py)
OBSERVATION_SHAPE = (8, 8, TOTAL_PLANES)

# Action space dimensions
QUEEN_MOVE_PLANES = 56  # 8 directions × 7 distances
KNIGHT_MOVE_PLANES = 8
UNDERPROMOTION_PLANES = 9  # 3 piece types × 3 directions
MOVE_PLANES = QUEEN_MOVE_PLANES + KNIGHT_MOVE_PLANES + UNDERPROMOTION_PLANES  # 73
ACTION_SPACE_SIZE = 8 * 8 * MOVE_PLANES  # 4672

# Direction vectors for queen-like moves (N, NE, E, SE, S, SW, W, NW)
# These are (file_delta, rank_delta) pairs
QUEEN_DIRECTIONS = [
    (0, 1),   # N
    (1, 1),   # NE
    (1, 0),   # E
    (1, -1),  # SE
    (0, -1),  # S
    (-1, -1), # SW
    (-1, 0),  # W
    (-1, 1),  # NW
]

# Knight move deltas (file_delta, rank_delta)
KNIGHT_DELTAS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]

# Piece type indices (matching python-chess order)
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


class ObservationEncoder:
    """
    Encodes chess board states into AlphaZero-style tensor observations.

    Maintains a history of board positions for temporal context.
    All observations are from the current player's perspective (board flipped for Black).

    Usage:
        encoder = ObservationEncoder()
        encoder.reset(board)  # Start new game
        obs = encoder.encode(board)  # Get 8×8×119 tensor
        encoder.push(board)  # After each move, update history
    """

    def __init__(self, history_length: int = HISTORY_LENGTH):
        self.history_length = history_length
        # Deque of (fen, repetition_count) tuples, most recent first
        self.history: deque[tuple[str, int]] = deque(maxlen=history_length)
        # Track position repetitions for threefold draw detection
        self.position_counts: dict[str, int] = {}

    def reset(self, board: Optional[chess.Board] = None) -> None:
        """Reset history for a new game."""
        self.history.clear()
        self.position_counts.clear()
        if board is not None:
            self._add_to_history(board)

    def push(self, board: chess.Board) -> None:
        """Add current position to history after a move is made."""
        self._add_to_history(board)

    def _add_to_history(self, board: chess.Board) -> None:
        """Add position to history and update repetition counts."""
        # Use board FEN without move counters for repetition tracking
        position_key = board.board_fen() + " " + ("w" if board.turn else "b")
        self.position_counts[position_key] = self.position_counts.get(position_key, 0) + 1
        rep_count = self.position_counts[position_key]
        self.history.appendleft((board.fen(), rep_count))

    def encode(self, board: chess.Board) -> np.ndarray:
        """
        Encode the current board state as an 8×8×119 tensor.

        The board is always encoded from the current player's perspective:
        - If White to move: board as-is
        - If Black to move: board flipped vertically, colors swapped

        Returns:
            numpy array of shape (8, 8, 119) with float32 values in [0, 1]
        """
        obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)

        # Determine if we need to flip perspective
        flip = not board.turn  # Flip if Black to move

        # Encode historical positions
        plane_idx = 0
        for t in range(self.history_length):
            if t < len(self.history):
                fen, rep_count = self.history[t]
                hist_board = chess.Board(fen)
            else:
                # No history available, use empty planes
                plane_idx += PLANES_PER_POSITION
                continue

            # Encode this historical position
            plane_idx = self._encode_position(obs, hist_board, plane_idx, flip, rep_count)

        # Encode auxiliary features (constant planes)
        # These describe the CURRENT position, not historical
        self._encode_auxiliary(obs, board, plane_idx)

        return obs

    def _encode_position(
        self,
        obs: np.ndarray,
        board: chess.Board,
        plane_idx: int,
        flip: bool,
        rep_count: int
    ) -> int:
        """
        Encode a single board position into the observation tensor.

        Args:
            obs: The observation tensor to fill
            board: The chess board to encode
            plane_idx: Starting plane index
            flip: Whether to flip the board (Black's perspective)
            rep_count: How many times this position has occurred

        Returns:
            Next plane index after encoding
        """
        # Determine which color is "us" vs "them"
        if flip:
            our_color = chess.BLACK
            their_color = chess.WHITE
        else:
            our_color = chess.WHITE
            their_color = chess.BLACK

        # Encode our pieces (planes 0-5)
        for piece_idx, piece_type in enumerate(PIECE_TYPES):
            for square in board.pieces(piece_type, our_color):
                rank, file = self._square_to_coords(square, flip)
                obs[rank, file, plane_idx + piece_idx] = 1.0

        # Encode their pieces (planes 6-11)
        for piece_idx, piece_type in enumerate(PIECE_TYPES):
            for square in board.pieces(piece_type, their_color):
                rank, file = self._square_to_coords(square, flip)
                obs[rank, file, plane_idx + 6 + piece_idx] = 1.0

        # Repetition planes (planes 12-13)
        if rep_count >= 1:
            obs[:, :, plane_idx + 12] = 1.0  # Position seen at least once before
        if rep_count >= 2:
            obs[:, :, plane_idx + 13] = 1.0  # Position seen at least twice before

        return plane_idx + PLANES_PER_POSITION

    def _encode_auxiliary(self, obs: np.ndarray, board: chess.Board, plane_idx: int) -> None:
        """
        Encode auxiliary features (constant planes).

        These are full planes of constant values encoding game state metadata.
        """
        # Plane 112: Color (1 if White to move)
        # Since we always flip to current player's perspective, this is always 1
        # But we encode the original color for the critic to know game phase
        obs[:, :, plane_idx] = 1.0 if board.turn else 0.0

        # Plane 113: Total move count (normalized by typical game length)
        # Normalize by 200 (typical max full moves)
        move_count = board.fullmove_number / 200.0
        obs[:, :, plane_idx + 1] = min(move_count, 1.0)

        # Planes 114-117: Castling rights
        # Note: These are from White's perspective in the original board
        obs[:, :, plane_idx + 2] = float(board.has_kingside_castling_rights(chess.WHITE))
        obs[:, :, plane_idx + 3] = float(board.has_queenside_castling_rights(chess.WHITE))
        obs[:, :, plane_idx + 4] = float(board.has_kingside_castling_rights(chess.BLACK))
        obs[:, :, plane_idx + 5] = float(board.has_queenside_castling_rights(chess.BLACK))

        # Plane 118: Halfmove clock (50-move rule), normalized
        obs[:, :, plane_idx + 6] = board.halfmove_clock / 100.0

    def _square_to_coords(self, square: int, flip: bool) -> tuple[int, int]:
        """
        Convert chess square (0-63) to tensor coordinates (rank, file).

        In python-chess:
        - Square 0 = a1, square 63 = h8
        - chess.square_file(sq) gives 0-7 (a-h)
        - chess.square_rank(sq) gives 0-7 (ranks 1-8)

        For the tensor:
        - Index [0, 0] = top-left of visual board (a8 or a1 if flipped)
        - We want [0, :] to be the back rank of the "opponent"

        Args:
            square: Chess square index (0-63)
            flip: If True, flip for Black's perspective

        Returns:
            (rank_idx, file_idx) for indexing obs[rank, file, plane]
        """
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        if flip:
            # Black's perspective: flip both rank and file
            rank = 7 - rank
            file = 7 - file
        else:
            # White's perspective: just invert rank so rank 8 is at top
            rank = 7 - rank

        return rank, file


class ActionEncoder:
    """
    Encodes/decodes between chess moves and AlphaZero action indices.

    The action space has 4672 discrete actions = 8 × 8 × 73.
    - 8 × 8: the "from" square
    - 73: the move type (direction + distance, or knight, or underpromotion)

    All encoding is from the CURRENT PLAYER's perspective (board flipped for Black).

    Usage:
        encoder = ActionEncoder()
        action = encoder.encode(board, move)  # chess.Move -> int
        move = encoder.decode(board, action)  # int -> chess.Move
        mask = encoder.legal_action_mask(board)  # bool array of shape (4672,)
    """

    def encode(self, board: chess.Board, move: chess.Move) -> int:
        """
        Encode a chess move as an action index.

        Args:
            board: Current board state (needed to determine perspective)
            move: The chess move to encode

        Returns:
            Action index in [0, 4671]
        """
        flip = not board.turn

        # Get from/to squares, potentially flipped
        from_sq = self._flip_square(move.from_square) if flip else move.from_square
        to_sq = self._flip_square(move.to_square) if flip else move.to_square

        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)
        to_file = chess.square_file(to_sq)
        to_rank = chess.square_rank(to_sq)

        delta_file = to_file - from_file
        delta_rank = to_rank - from_rank

        # Determine move plane based on move type
        if move.promotion is not None and move.promotion != chess.QUEEN:
            # Underpromotion (knight, bishop, rook)
            plane = self._encode_underpromotion(delta_file, move.promotion)
        elif abs(delta_file) == abs(delta_rank) or delta_file == 0 or delta_rank == 0:
            # Queen-like move (including queen promotions)
            plane = self._encode_queen_move(delta_file, delta_rank)
        else:
            # Knight move
            plane = self._encode_knight_move(delta_file, delta_rank)

        # Combine into single action index
        # Action = from_rank * 8 * 73 + from_file * 73 + plane
        action = from_rank * 8 * MOVE_PLANES + from_file * MOVE_PLANES + plane
        return action

    def decode(self, board: chess.Board, action: int) -> Optional[chess.Move]:
        """
        Decode an action index back to a chess move.

        Args:
            board: Current board state
            action: Action index in [0, 4671]

        Returns:
            The corresponding chess.Move, or None if invalid
        """
        flip = not board.turn

        # Extract components
        from_rank = action // (8 * MOVE_PLANES)
        from_file = (action // MOVE_PLANES) % 8
        plane = action % MOVE_PLANES

        from_sq = chess.square(from_file, from_rank)

        # Decode the move plane
        if plane < QUEEN_MOVE_PLANES:
            # Queen-like move
            delta_file, delta_rank = self._decode_queen_move(plane)
        elif plane < QUEEN_MOVE_PLANES + KNIGHT_MOVE_PLANES:
            # Knight move
            delta_file, delta_rank = self._decode_knight_move(plane - QUEEN_MOVE_PLANES)
        else:
            # Underpromotion
            delta_file, promotion = self._decode_underpromotion(plane - QUEEN_MOVE_PLANES - KNIGHT_MOVE_PLANES)
            delta_rank = 1  # Promotions always move forward one rank (from 7th to 8th)

        to_file = from_file + delta_file
        to_rank = from_rank + delta_rank

        # Check bounds
        if not (0 <= to_file < 8 and 0 <= to_rank < 8):
            return None

        to_sq = chess.square(to_file, to_rank)

        # Unflip if needed
        if flip:
            from_sq = self._flip_square(from_sq)
            to_sq = self._flip_square(to_sq)

        # Determine promotion type
        promotion = None
        if plane >= QUEEN_MOVE_PLANES + KNIGHT_MOVE_PLANES:
            # Underpromotion
            underpromo_idx = plane - QUEEN_MOVE_PLANES - KNIGHT_MOVE_PLANES
            promotion = [chess.KNIGHT, chess.BISHOP, chess.ROOK][underpromo_idx // 3]
        else:
            # Check if this is a queen promotion (pawn reaching 8th rank)
            piece = board.piece_at(from_sq)
            if piece and piece.piece_type == chess.PAWN:
                to_rank_real = chess.square_rank(to_sq)
                if (board.turn and to_rank_real == 7) or (not board.turn and to_rank_real == 0):
                    promotion = chess.QUEEN

        return chess.Move(from_sq, to_sq, promotion=promotion)

    def legal_action_mask(self, board: chess.Board) -> np.ndarray:
        """
        Generate a boolean mask over the action space for legal moves.

        Args:
            board: Current board state

        Returns:
            Boolean array of shape (4672,) where True = legal action
        """
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

        for move in board.legal_moves:
            try:
                action = self.encode(board, move)
                mask[action] = True
            except Exception:
                # Skip any moves that fail to encode (shouldn't happen)
                pass

        return mask

    def _encode_queen_move(self, delta_file: int, delta_rank: int) -> int:
        """Encode a queen-like move (straight or diagonal) as a plane index."""
        # Determine direction
        if delta_file == 0:
            direction = 0 if delta_rank > 0 else 4  # N or S
        elif delta_rank == 0:
            direction = 2 if delta_file > 0 else 6  # E or W
        elif delta_file > 0:
            direction = 1 if delta_rank > 0 else 3  # NE or SE
        else:
            direction = 7 if delta_rank > 0 else 5  # NW or SW

        # Determine distance (1-7)
        distance = max(abs(delta_file), abs(delta_rank))

        return direction * 7 + (distance - 1)

    def _decode_queen_move(self, plane: int) -> tuple[int, int]:
        """Decode a queen-move plane to (delta_file, delta_rank)."""
        direction = plane // 7
        distance = (plane % 7) + 1

        df, dr = QUEEN_DIRECTIONS[direction]
        return df * distance, dr * distance

    def _encode_knight_move(self, delta_file: int, delta_rank: int) -> int:
        """Encode a knight move as a plane index (56-63)."""
        try:
            idx = KNIGHT_DELTAS.index((delta_file, delta_rank))
            return QUEEN_MOVE_PLANES + idx
        except ValueError:
            raise ValueError(f"Invalid knight move delta: ({delta_file}, {delta_rank})")

    def _decode_knight_move(self, idx: int) -> tuple[int, int]:
        """Decode a knight move plane index to (delta_file, delta_rank)."""
        return KNIGHT_DELTAS[idx]

    def _encode_underpromotion(self, delta_file: int, piece_type: int) -> int:
        """
        Encode an underpromotion move.

        delta_file: -1 (left capture), 0 (straight), 1 (right capture)
        piece_type: chess.KNIGHT, chess.BISHOP, or chess.ROOK
        """
        piece_idx = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[piece_type]
        direction_idx = delta_file + 1  # -1 -> 0, 0 -> 1, 1 -> 2

        return QUEEN_MOVE_PLANES + KNIGHT_MOVE_PLANES + piece_idx * 3 + direction_idx

    def _decode_underpromotion(self, idx: int) -> tuple[int, int]:
        """Decode underpromotion plane to (delta_file, piece_type)."""
        piece_idx = idx // 3
        direction_idx = idx % 3

        piece_type = [chess.KNIGHT, chess.BISHOP, chess.ROOK][piece_idx]
        delta_file = direction_idx - 1  # 0 -> -1, 1 -> 0, 2 -> 1

        return delta_file, piece_type

    def _flip_square(self, square: int) -> int:
        """Flip a square for Black's perspective (rotate 180°)."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        return chess.square(7 - file, 7 - rank)
