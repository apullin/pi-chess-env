"""
Prime Intellect Verifiers-Compatible Chess Environment.

This module wraps ChessTensorEnv for use with Prime Intellect's verifiers library,
enabling LLM-based chess agents.

=== HOW THIS RELATES TO CHESSTENSORENV ===

ChessTensorEnv (tensor-based):
    - Observation: 8×8×119 tensor
    - Action: Integer 0-4671
    - For: CNN/ResNet policies, PPO training

ChessTextEnv (this file):
    - Observation: Text (ASCII board + FEN + legal moves)
    - Action: Text in <move>SAN</move> format
    - For: LLM policies, prime-rl/vf-rl training

Both use the same underlying chess logic (python-chess).
The text env is a thin adapter that:
1. Converts tensor obs → text for the LLM to read
2. Parses LLM output → SAN move → tensor action
3. Wraps rewards in a Rubric for verifiers

=== VERIFIERS CONCEPTS ===

MultiTurnEnv:
    An environment where the agent takes multiple actions before episode ends.
    Chess is multi-turn: each move is an action, game has many moves.

Rubric:
    A function that scores agent responses. For chess:
    - Format score: Did the agent output <move>...</move>?
    - Legality score: Is the move legal?
    - Outcome score: Win/draw/loss at game end

=== USAGE ===

For local testing:
    from chess_env.verifiers_env import ChessTextEnv

    env = ChessTextEnv()
    prompt, state = env.get_initial_prompt()

    # LLM responds...
    response = "<think>I'll develop my knight.</think><move>Nf3</move>"

    messages, state, done = env.step(response, state)

For Prime Intellect Hub:
    # In your chess.py file
    import verifiers as vf
    from chess_env.verifiers_env import ChessVerifiersEnv

    def load_environment(**kwargs):
        return ChessVerifiersEnv(**kwargs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import chess


@dataclass
class ChessTextConfig:
    """Configuration for text-based chess environment."""

    # Game settings
    max_moves: int = 400
    start_fen: Optional[str] = None

    # Observation format
    include_legal_moves: bool = True
    include_ascii_board: bool = True
    include_fen: bool = True
    max_legal_moves_shown: int = 60

    # Thinking
    allow_think: bool = True

    # Reward settings
    illegal_move_penalty: float = -0.2
    format_penalty: float = -0.1


@dataclass
class ChessState:
    """State for a chess game."""

    board: chess.Board = field(default_factory=chess.Board)
    move_count: int = 0
    history: list[str] = field(default_factory=list)
    done: bool = False
    result: Optional[str] = None
    termination_reason: Optional[str] = None
    last_illegal: bool = False


class ChessTextEnv:
    """
    Text-based chess environment for LLM agents.

    This is a standalone environment that can be used for testing
    without the full verifiers infrastructure.
    """

    def __init__(self, config: Optional[ChessTextConfig] = None):
        self.config = config or ChessTextConfig()

        # Compile regex for move parsing
        self.move_pattern = re.compile(r"<move>(.*?)</move>", re.IGNORECASE | re.DOTALL)
        self.think_pattern = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)

    def reset(self, start_fen: Optional[str] = None) -> tuple[str, ChessState]:
        """
        Reset the environment for a new game.

        Returns:
            prompt: Initial observation/prompt for the LLM
            state: Game state object
        """
        state = ChessState()

        if start_fen:
            state.board.set_fen(start_fen)
        elif self.config.start_fen:
            state.board.set_fen(self.config.start_fen)

        prompt = self._format_observation(state)
        return prompt, state

    def step(
        self, response: str, state: ChessState
    ) -> tuple[str, ChessState, bool, float, dict[str, Any]]:
        """
        Process an LLM response and update game state.

        Args:
            response: LLM's response text (should contain <move>...</move>)
            state: Current game state

        Returns:
            observation: Next observation text
            state: Updated state
            done: Whether game is over
            reward: Reward for this step
            info: Additional info dict
        """
        info: dict[str, Any] = {}

        # Parse the move
        move_match = self.move_pattern.search(response)
        if not move_match:
            # No move found - format error
            info["error"] = "no_move_tag"
            info["response"] = response
            state.last_illegal = True
            observation = self._format_observation(state, error="No <move>...</move> found in response")
            return observation, state, False, self.config.format_penalty, info

        san = move_match.group(1).strip()
        info["parsed_move"] = san

        # Try to apply the move
        try:
            move = state.board.parse_san(san)
            if move not in state.board.legal_moves:
                raise ValueError("Illegal move")
        except Exception as e:
            info["error"] = "illegal_move"
            info["exception"] = str(e)
            state.last_illegal = True
            observation = self._format_observation(
                state, error=f"Illegal move: {san}"
            )
            return observation, state, False, self.config.illegal_move_penalty, info

        # Move is legal - apply it
        mover_was_white = state.board.turn
        state.board.push(move)
        state.move_count += 1
        state.history.append(san)
        state.last_illegal = False
        info["move_uci"] = move.uci()

        # Check for game end
        reward = 0.0
        if state.board.is_game_over():
            state.done = True
            state.result = state.board.result()
            state.termination_reason = self._get_termination_reason(state.board)

            if state.result == "1-0":
                reward = 1.0 if mover_was_white else -1.0
            elif state.result == "0-1":
                reward = 1.0 if not mover_was_white else -1.0
            else:
                reward = 0.0

            info["game_result"] = state.result
            info["termination_reason"] = state.termination_reason

        elif state.move_count >= self.config.max_moves:
            state.done = True
            state.result = "1/2-1/2"
            state.termination_reason = "max_moves"
            reward = 0.0

        observation = self._format_observation(state)
        return observation, state, state.done, reward, info

    def _format_observation(
        self, state: ChessState, error: Optional[str] = None
    ) -> str:
        """Format the current state as a text observation."""
        lines = []

        lines.append("=== Chess ===")

        if self.config.include_ascii_board:
            lines.append("")
            lines.append(str(state.board))

        if self.config.include_fen:
            lines.append("")
            lines.append(f"FEN: {state.board.fen()}")

        lines.append("")
        lines.append(f"Side to move: {'White' if state.board.turn else 'Black'}")
        lines.append(f"Move number: {state.board.fullmove_number}")

        if state.board.is_check():
            lines.append("CHECK!")

        if state.history:
            lines.append("")
            lines.append(f"Last move: {state.history[-1]}")

        if self.config.include_legal_moves and not state.done:
            lines.append("")
            legal = sorted(state.board.san(m) for m in state.board.legal_moves)
            if len(legal) <= self.config.max_legal_moves_shown:
                lines.append(f"Legal moves ({len(legal)}): {', '.join(legal)}")
            else:
                shown = legal[: self.config.max_legal_moves_shown]
                lines.append(
                    f"Legal moves ({len(legal)}): {', '.join(shown)} ..."
                )

        if error:
            lines.append("")
            lines.append(f"ERROR: {error}")
            lines.append("Please try again with a valid move.")

        if state.done:
            lines.append("")
            lines.append(f"Game Over: {state.result}")
            lines.append(f"Reason: {state.termination_reason}")
        else:
            lines.append("")
            if self.config.allow_think:
                lines.append("Respond with your move in <move>SAN</move> format.")
                lines.append("You may optionally think in <think>...</think> first.")
            else:
                lines.append("Respond with your move in <move>SAN</move> format.")

        return "\n".join(lines)

    def _get_termination_reason(self, board: chess.Board) -> str:
        """Get human-readable termination reason."""
        if board.is_checkmate():
            return "checkmate"
        elif board.is_stalemate():
            return "stalemate"
        elif board.is_insufficient_material():
            return "insufficient_material"
        elif board.is_fifty_moves():
            return "fifty_move_rule"
        elif board.is_repetition():
            return "threefold_repetition"
        else:
            return "unknown"


# === PRIME INTELLECT VERIFIERS INTEGRATION ===
#
# The following class provides the actual vf.MultiTurnEnv implementation.
# It requires the verifiers package to be installed.
#
# Usage in your hub environment file (chess.py):
#
#     import verifiers as vf
#     from chess_env.verifiers_env import load_environment
#
#     # This is called by the Hub
#     def load_environment(**kwargs):
#         return load_environment(**kwargs)

try:
    import verifiers as vf
    from verifiers.types import Messages, State

    VERIFIERS_AVAILABLE = True
except ImportError:
    VERIFIERS_AVAILABLE = False


def create_system_prompt(config: ChessTextConfig) -> str:
    """Create the system prompt for the chess environment."""
    prompt = """You are playing chess. On each turn, you will see the current board state and must respond with a legal move.

Format your response as:
<move>YOUR_MOVE_IN_SAN_NOTATION</move>

For example:
<move>e4</move>
<move>Nf3</move>
<move>O-O</move>  (castling kingside)
<move>exd5</move>  (pawn capture)
<move>Qxf7+</move>  (queen captures with check)
<move>e8=Q</move>  (pawn promotion)
"""

    if config.allow_think:
        prompt += """
You may optionally think before your move:
<think>I should develop my pieces...</think>
<move>Nc3</move>
"""

    prompt += """
Play to win. Make legal moves only.
"""
    return prompt


def load_environment(
    include_legal_moves: bool = True,
    allow_think: bool = True,
    max_moves: int = 400,
    **kwargs,
):
    """
    Load the chess environment for Prime Intellect Hub.

    This is the entry point called by the Hub when loading your environment.

    Args:
        include_legal_moves: Show legal moves in observation
        allow_think: Allow <think>...</think> tags
        max_moves: Maximum moves per game

    Returns:
        A verifiers-compatible environment object
    """
    if not VERIFIERS_AVAILABLE:
        raise ImportError(
            "verifiers package not installed. Install with: pip install verifiers"
        )

    config = ChessTextConfig(
        include_legal_moves=include_legal_moves,
        allow_think=allow_think,
        max_moves=max_moves,
    )

    return ChessVerifiersEnv(config)


if VERIFIERS_AVAILABLE:

    class ChessVerifiersEnv(vf.MultiTurnEnv):
        """
        Verifiers-compatible multi-turn chess environment.

        This implements the vf.MultiTurnEnv interface for use with
        Prime Intellect's training and evaluation infrastructure.
        """

        def __init__(self, config: Optional[ChessTextConfig] = None):
            self.config = config or ChessTextConfig()
            self.text_env = ChessTextEnv(self.config)
            self.system_prompt = create_system_prompt(self.config)

        async def setup_state(self, **kwargs) -> State:
            """Initialize state for a new episode."""
            _, chess_state = self.text_env.reset()
            return {"chess": chess_state, "turn": 0}

        async def env_response(
            self, messages: Messages, state: State, **kwargs
        ) -> tuple[Messages, State]:
            """
            Process assistant message and return environment response.

            This is called after the model generates a response.
            We parse the move, update the board, and return the new observation.
            """
            chess_state: ChessState = state["chess"]

            # Get the last assistant message
            last_msg = messages[-1]
            if last_msg.get("role") != "assistant":
                # No assistant message yet, return initial observation
                observation = self.text_env._format_observation(chess_state)
                new_msg = {"role": "user", "content": observation}
                return messages + [new_msg], state

            response = last_msg.get("content", "")

            # Process the move
            observation, chess_state, done, reward, info = self.text_env.step(
                response, chess_state
            )

            # Update state
            state["chess"] = chess_state
            state["turn"] = state.get("turn", 0) + 1
            state["last_reward"] = reward
            state["last_info"] = info

            if done:
                state["done"] = True
                state["result"] = chess_state.result
                state["total_reward"] = reward  # Simplified; full version tracks cumulative

            # Add observation as user message
            new_msg = {"role": "user", "content": observation}
            return messages + [new_msg], state

        def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
            """Check if the episode is complete."""
            return state.get("done", False)

        def get_rubric(self) -> vf.Rubric:
            """
            Get the rubric for scoring responses.

            The rubric evaluates:
            1. Format: Did the response contain <move>...</move>?
            2. Legality: Was the move legal?
            3. Outcome: Win/draw/loss at game end
            """

            def score_response(messages: Messages, state: State) -> float:
                """Score a single response."""
                reward = state.get("last_reward", 0.0)
                return reward

            return vf.Rubric(
                name="chess_outcome",
                score_fn=score_response,
            )
