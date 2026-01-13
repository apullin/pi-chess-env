"""
ChessTensorEnv: A Gymnasium environment for chess with AlphaZero-style encoding.

This environment implements chess as a two-player zero-sum game suitable for
self-play reinforcement learning. Key features:

1. TENSOR OBSERVATIONS (8×8×119):
   - 8 historical positions (14 planes each) + 7 auxiliary planes
   - Always from current player's perspective (board flipped for Black)
   - See encoding.py for detailed breakdown

2. DISCRETE ACTION SPACE (4672):
   - AlphaZero-style encoding: 8×8 from-squares × 73 move types
   - Illegal moves are masked - agent can only select legal actions
   - See encoding.py for move type breakdown

3. SELF-PLAY STRUCTURE:
   - Single environment, alternating players
   - Observation always shows "your" perspective
   - Reward given at game end: +1 win, 0 draw, -1 loss

4. DETERMINISM:
   - Given same initial position and move sequence, outcome is always the same
   - No stochasticity in transitions
   - Random starting positions (if enabled) use seeded RNG

Usage:
    import gymnasium as gym
    from chess_env import ChessTensorEnv

    env = ChessTensorEnv()
    obs, info = env.reset()

    while True:
        # Get legal action mask
        mask = info["action_mask"]

        # Select action (your policy goes here)
        action = your_policy(obs, mask)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

Episode Structure:
    - Each step is one half-move (ply)
    - Observation after step is from the NEXT player's perspective
    - Reward is from the perspective of the player who just moved
    - Terminal reward: +1 if mover won, -1 if mover lost, 0 if draw
"""

from __future__ import annotations

from typing import Any, Optional

import chess
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from chess_env.encoding import (
    ACTION_SPACE_SIZE,
    OBSERVATION_SHAPE,
    ActionEncoder,
    ObservationEncoder,
)


class ChessTensorEnv(gym.Env):
    """
    Gymnasium environment for chess with AlphaZero-style tensor observations.

    Attributes:
        observation_space: Box space of shape (8, 8, 119) with float32 values in [0, 1]
        action_space: Discrete space with 4672 actions
        board: The current python-chess Board
        obs_encoder: Encoder for observations
        action_encoder: Encoder/decoder for actions

    Config options:
        max_moves: Maximum half-moves before truncation (default: 400)
        illegal_move_penalty: Reward penalty for attempting illegal move (default: -1.0)
        provide_legal_moves: Include list of legal SAN moves in info (default: True)
        start_fen: Starting position FEN, or None for standard start (default: None)
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    # Piece values for reward shaping (standard chess values)
    PIECE_VALUES = {
        chess.PAWN: 0.01,
        chess.KNIGHT: 0.03,
        chess.BISHOP: 0.03,
        chess.ROOK: 0.05,
        chess.QUEEN: 0.09,
        chess.KING: 0.0,  # Can't capture king
    }

    def __init__(
        self,
        max_moves: int = 400,
        illegal_move_penalty: float = -1.0,
        provide_legal_moves: bool = True,
        start_fen: Optional[str] = None,
        render_mode: Optional[str] = None,
        reward_shaping: bool = True,
    ):
        super().__init__()
        self.reward_shaping = reward_shaping

        self.max_moves = max_moves
        self.illegal_move_penalty = illegal_move_penalty
        self.provide_legal_moves = provide_legal_moves
        self.start_fen = start_fen
        self.render_mode = render_mode

        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=OBSERVATION_SHAPE,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # Initialize encoders
        self.obs_encoder = ObservationEncoder()
        self.action_encoder = ActionEncoder()

        # Game state (initialized in reset)
        self.board: chess.Board = None  # type: ignore
        self.move_count = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to start a new game.

        Args:
            seed: Random seed (for reproducibility if using random starts)
            options: Optional dict with:
                - "fen": Override starting position for this episode
                - "moves": List of SAN moves to play from start position

        Returns:
            observation: Initial board observation (8, 8, 119)
            info: Dict with action_mask, legal_moves, fen, etc.
        """
        super().reset(seed=seed)

        # Determine starting position
        fen = None
        if options and "fen" in options:
            fen = options["fen"]
        elif self.start_fen:
            fen = self.start_fen

        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()

        # Apply any preset moves
        if options and "moves" in options:
            for san in options["moves"]:
                self.board.push_san(san)

        self.move_count = 0

        # Reset observation encoder with initial position
        self.obs_encoder.reset(self.board)

        # Generate observation and info
        obs = self.obs_encoder.encode(self.board)
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one move in the game.

        Args:
            action: Action index in [0, 4671]

        Returns:
            observation: New board state (from next player's perspective)
            reward: Reward for the player who just moved (+1 win, 0 draw, -1 loss, or penalty)
            terminated: True if game ended (checkmate, draw, or illegal move)
            truncated: True if max moves reached
            info: Dict with action_mask, legal_moves, fen, game_result, etc.
        """
        # Decode action to move
        move = self.action_encoder.decode(self.board, action)

        # Check if move is legal
        if move is None or move not in self.board.legal_moves:
            # Illegal move - game ends with penalty
            # Note: With proper action masking, this should never happen
            obs = self.obs_encoder.encode(self.board)
            info = self._get_info()
            info["illegal_move"] = True
            info["attempted_action"] = action
            return obs, self.illegal_move_penalty, True, False, info

        # Remember who moved (for reward calculation)
        mover_was_white = self.board.turn

        # Check for capture before applying move (for reward shaping)
        captured_piece = None
        if self.reward_shaping and self.board.is_capture(move):
            # Get the piece being captured
            to_square = move.to_square
            captured_piece = self.board.piece_at(to_square)
            # Handle en passant
            if captured_piece is None and self.board.is_en_passant(move):
                captured_piece = chess.Piece(chess.PAWN, not self.board.turn)

        # Apply the move
        self.board.push(move)
        self.move_count += 1

        # Update observation history
        self.obs_encoder.push(self.board)

        # Check for game end
        terminated = False
        truncated = False
        reward = 0.0

        # Reward shaping: small reward for captures
        if self.reward_shaping and captured_piece is not None:
            reward += self.PIECE_VALUES.get(captured_piece.piece_type, 0.0)

        if self.board.is_game_over():
            terminated = True
            result = self.board.result()

            if result == "1-0":
                # White wins
                reward = 1.0 if mover_was_white else -1.0
            elif result == "0-1":
                # Black wins
                reward = 1.0 if not mover_was_white else -1.0
            else:
                # Draw
                reward = 0.0

        elif self.move_count >= self.max_moves:
            truncated = True
            # Reward based on material advantage at truncation
            if self.reward_shaping:
                material = self._material_balance()
                # Positive if current player (who just moved) is ahead
                reward = material * 0.1 if mover_was_white else -material * 0.1
            else:
                reward = 0.0

        # Generate observation (from NEXT player's perspective)
        obs = self.obs_encoder.encode(self.board)
        info = self._get_info()
        info["last_move"] = move.uci()
        info["last_move_san"] = self.board.san(move) if move in self.board.legal_moves else "??"

        if terminated:
            info["game_result"] = self.board.result() if self.board.is_game_over() else "1/2-1/2"
            info["termination_reason"] = self._get_termination_reason()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_info(self) -> dict[str, Any]:
        """Generate info dict for current state."""
        info = {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn else "black",
            "fullmove_number": self.board.fullmove_number,
            "halfmove_clock": self.board.halfmove_clock,
            "is_check": self.board.is_check(),
            "action_mask": self.action_encoder.legal_action_mask(self.board),
            "num_legal_moves": len(list(self.board.legal_moves)),
        }

        if self.provide_legal_moves:
            info["legal_moves_san"] = sorted(
                self.board.san(m) for m in self.board.legal_moves
            )
            info["legal_moves_uci"] = sorted(
                m.uci() for m in self.board.legal_moves
            )

        return info

    def _material_balance(self) -> float:
        """Calculate material balance (positive = white ahead)."""
        balance = 0.0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            value = self.PIECE_VALUES[piece_type] * 100  # Scale up for readability
            balance += len(self.board.pieces(piece_type, chess.WHITE)) * value
            balance -= len(self.board.pieces(piece_type, chess.BLACK)) * value
        return balance

    def _get_termination_reason(self) -> str:
        """Get human-readable termination reason."""
        if self.board.is_checkmate():
            return "checkmate"
        elif self.board.is_stalemate():
            return "stalemate"
        elif self.board.is_insufficient_material():
            return "insufficient_material"
        elif self.board.is_fifty_moves():
            return "fifty_move_rule"
        elif self.board.is_repetition():
            return "threefold_repetition"
        else:
            return "unknown"

    def render(self) -> Optional[str]:
        """Render the current board state."""
        if self.render_mode == "ansi" or self.render_mode == "human":
            board_str = str(self.board)
            turn = "White" if self.board.turn else "Black"
            info = f"\nMove {self.board.fullmove_number}, {turn} to move"
            if self.board.is_check():
                info += " (CHECK)"
            result = board_str + info + "\n"

            if self.render_mode == "human":
                print(result)

            return result
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass

    # Utility methods for external use

    def get_legal_actions(self) -> list[int]:
        """Get list of legal action indices."""
        mask = self.action_encoder.legal_action_mask(self.board)
        return list(np.where(mask)[0])

    def action_to_san(self, action: int) -> Optional[str]:
        """Convert action index to SAN notation."""
        move = self.action_encoder.decode(self.board, action)
        if move and move in self.board.legal_moves:
            return self.board.san(move)
        return None

    def san_to_action(self, san: str) -> Optional[int]:
        """Convert SAN notation to action index."""
        try:
            move = self.board.parse_san(san)
            return self.action_encoder.encode(self.board, move)
        except ValueError:
            return None

    def clone(self) -> "ChessTensorEnv":
        """Create a deep copy of the environment."""
        new_env = ChessTensorEnv(
            max_moves=self.max_moves,
            illegal_move_penalty=self.illegal_move_penalty,
            provide_legal_moves=self.provide_legal_moves,
            start_fen=self.start_fen,
            render_mode=self.render_mode,
        )
        new_env.reset(options={"fen": self.board.fen()})
        new_env.move_count = self.move_count
        # Copy history
        new_env.obs_encoder.history = self.obs_encoder.history.copy()
        new_env.obs_encoder.position_counts = self.obs_encoder.position_counts.copy()
        return new_env
