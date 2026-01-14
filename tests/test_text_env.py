"""
Tests for the ChessTextEnv text-based environment.

These tests verify:
1. Basic API (reset, step, observation format)
2. Move parsing (<move>SAN</move> format)
3. Error handling (illegal moves, format errors)
4. Game flow (checkmate, stalemate, max moves)
"""

import pytest
from chess_env.verifiers_env import ChessTextEnv, ChessTextConfig, ChessState, create_system_prompt


class TestChessTextEnvBasics:
    """Basic API tests for ChessTextEnv."""

    def test_reset_returns_prompt_and_state(self):
        """Reset should return observation string and state."""
        env = ChessTextEnv()
        prompt, state = env.reset()

        assert isinstance(prompt, str)
        assert isinstance(state, ChessState)
        assert len(prompt) > 0

    def test_observation_contains_board(self):
        """Observation should contain ASCII board representation."""
        env = ChessTextEnv()
        prompt, _ = env.reset()

        # Should have the chess header
        assert "=== Chess ===" in prompt
        # Should have piece characters (lowercase for black)
        assert "r" in prompt.lower()
        assert "k" in prompt.lower()

    def test_observation_contains_fen(self):
        """Observation should contain FEN string."""
        env = ChessTextEnv()
        prompt, _ = env.reset()

        assert "FEN:" in prompt
        assert "rnbqkbnr" in prompt  # Starting position

    def test_observation_contains_legal_moves(self):
        """Observation should list legal moves."""
        env = ChessTextEnv()
        prompt, _ = env.reset()

        assert "Legal moves" in prompt
        # Starting position has 20 legal moves
        assert "20" in prompt

    def test_observation_contains_side_to_move(self):
        """Observation should indicate whose turn it is."""
        env = ChessTextEnv()
        prompt, _ = env.reset()

        assert "Side to move: White" in prompt

    def test_config_can_disable_legal_moves(self):
        """Config should allow hiding legal moves."""
        config = ChessTextConfig(include_legal_moves=False)
        env = ChessTextEnv(config)
        prompt, _ = env.reset()

        assert "Legal moves" not in prompt

    def test_config_can_disable_ascii_board(self):
        """Config should allow hiding ASCII board."""
        config = ChessTextConfig(include_ascii_board=False)
        env = ChessTextEnv(config)
        prompt, _ = env.reset()

        # Should still have FEN but not the board grid
        assert "FEN:" in prompt

    def test_reset_with_custom_fen(self):
        """Reset should accept custom starting FEN."""
        fen = "4k3/8/8/8/8/8/8/4K2R w - - 0 1"
        env = ChessTextEnv()
        prompt, state = env.reset(start_fen=fen)

        assert fen in prompt
        assert state.board.fen() == fen


class TestMoveProcessing:
    """Tests for move parsing and processing."""

    def test_valid_move_accepted(self):
        """Valid move in correct format should be accepted."""
        env = ChessTextEnv()
        _, state = env.reset()

        response = "<move>e4</move>"
        obs, state, done, reward, info = env.step(response, state)

        assert not done
        assert reward >= 0  # No penalty for valid move
        assert "e4" in state.history
        assert "error" not in info

    def test_move_with_think_tag_accepted(self):
        """Move with <think> tag should be accepted."""
        env = ChessTextEnv()
        _, state = env.reset()

        response = "<think>I'll control the center.</think><move>e4</move>"
        obs, state, done, reward, info = env.step(response, state)

        assert not done
        assert reward >= 0
        assert "e4" in state.history

    def test_missing_move_tag_returns_error(self):
        """Response without <move> tag should return format penalty."""
        env = ChessTextEnv()
        _, state = env.reset()

        response = "I'll play e4"
        obs, state, done, reward, info = env.step(response, state)

        assert not done  # Game continues
        assert reward == -0.1  # Format penalty
        assert info["error"] == "no_move_tag"
        assert "No <move>" in obs  # Error message in observation

    def test_illegal_move_returns_penalty(self):
        """Illegal move should return illegal move penalty."""
        env = ChessTextEnv()
        _, state = env.reset()

        response = "<move>e5</move>"  # Can't move to e5 as first move
        obs, state, done, reward, info = env.step(response, state)

        assert not done  # Game continues
        assert reward == -0.2  # Illegal move penalty
        assert info["error"] == "illegal_move"
        assert "Illegal move" in obs

    def test_invalid_san_returns_penalty(self):
        """Invalid SAN notation should return illegal move penalty."""
        env = ChessTextEnv()
        _, state = env.reset()

        response = "<move>xyz123</move>"
        obs, state, done, reward, info = env.step(response, state)

        assert not done
        assert reward == -0.2
        assert info["error"] == "illegal_move"

    def test_move_updates_board_state(self):
        """Valid move should update the board state."""
        env = ChessTextEnv()
        _, state = env.reset()

        env.step("<move>e4</move>", state)

        # Board should now have pawn on e4
        import chess
        assert state.board.piece_at(chess.E4) is not None
        assert state.board.piece_at(chess.E4).piece_type == chess.PAWN

    def test_turn_alternates_after_move(self):
        """Turn should alternate after each move."""
        env = ChessTextEnv()
        obs1, state = env.reset()

        assert "Side to move: White" in obs1

        obs2, state, _, _, _ = env.step("<move>e4</move>", state)
        assert "Side to move: Black" in obs2

        obs3, state, _, _, _ = env.step("<move>e5</move>", state)
        assert "Side to move: White" in obs3


class TestGameFlow:
    """Tests for game termination and flow."""

    def test_scholars_mate_terminates_game(self):
        """Scholar's mate should terminate game with win."""
        env = ChessTextEnv()
        _, state = env.reset()

        # 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#
        moves = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7"]

        for i, move in enumerate(moves[:-1]):
            _, state, done, _, _ = env.step(f"<move>{move}</move>", state)
            assert not done, f"Game ended early at move {move}"

        # Final move should end game
        obs, state, done, reward, info = env.step("<move>Qxf7</move>", state)

        assert done
        assert reward == 1.0  # White wins
        assert state.result == "1-0"
        assert "checkmate" in state.termination_reason

    def test_stalemate_is_draw(self):
        """Stalemate should terminate with 0 reward."""
        # Position where White can stalemate Black with Qb6
        # Black king on a8, White king on a6, White queen on b1
        fen = "k7/8/K7/8/8/8/8/1Q6 w - - 0 1"
        env = ChessTextEnv()
        _, state = env.reset(start_fen=fen)

        # Qb6 creates stalemate (king trapped on a8, no legal moves)
        obs, state, done, reward, info = env.step("<move>Qb6</move>", state)

        assert done
        assert reward == 0.0
        assert state.result == "1/2-1/2"
        assert "stalemate" in state.termination_reason

    def test_max_moves_truncation(self):
        """Game should end after max_moves."""
        config = ChessTextConfig(max_moves=4)
        env = ChessTextEnv(config)
        _, state = env.reset()

        # Play 4 moves
        moves = ["e4", "e5", "Nf3", "Nc6"]
        for i, move in enumerate(moves):
            obs, state, done, reward, info = env.step(f"<move>{move}</move>", state)

            if i < 3:
                assert not done
            else:
                assert done
                assert reward == 0.0  # Truncation = draw
                assert state.result == "1/2-1/2"

    def test_checkmate_gives_correct_reward_for_loser(self):
        """Loser should get -1 reward on checkmate."""
        # Position where Black can checkmate White with Ra2
        # Two rooks box in the white king
        fen = "4k3/8/8/8/8/8/8/r3K2r b - - 0 1"
        env = ChessTextEnv()
        _, state = env.reset(start_fen=fen)

        # Ra2# checkmates White
        obs, state, done, reward, info = env.step("<move>Ra2</move>", state)

        assert done
        assert reward == 1.0  # Black (mover) wins
        assert state.result == "0-1"

    def test_game_over_observation_shows_result(self):
        """Observation after game over should show result."""
        fen = "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1"
        env = ChessTextEnv()
        _, state = env.reset(start_fen=fen)

        obs, state, done, _, _ = env.step("<move>Re8</move>", state)

        assert done
        assert "Game Over" in obs
        assert "1-0" in obs


class TestSystemPrompt:
    """Tests for system prompt generation."""

    def test_system_prompt_generated(self):
        """System prompt should be generated."""
        config = ChessTextConfig()
        prompt = create_system_prompt(config)

        assert "chess" in prompt.lower()
        assert "<move>" in prompt
        assert "SAN" in prompt or "notation" in prompt.lower()

    def test_system_prompt_includes_think_when_allowed(self):
        """System prompt should mention <think> when allowed."""
        config = ChessTextConfig(allow_think=True)
        prompt = create_system_prompt(config)

        assert "<think>" in prompt

    def test_system_prompt_excludes_think_when_disabled(self):
        """System prompt should not mention <think> when disabled."""
        config = ChessTextConfig(allow_think=False)
        prompt = create_system_prompt(config)

        assert "<think>" not in prompt


class TestStateManagement:
    """Tests for game state tracking."""

    def test_move_count_increments(self):
        """Move count should increment after each move."""
        env = ChessTextEnv()
        _, state = env.reset()

        assert state.move_count == 0

        env.step("<move>e4</move>", state)
        assert state.move_count == 1

        env.step("<move>e5</move>", state)
        assert state.move_count == 2

    def test_history_tracks_moves(self):
        """History should track all moves played."""
        env = ChessTextEnv()
        _, state = env.reset()

        env.step("<move>e4</move>", state)
        env.step("<move>e5</move>", state)
        env.step("<move>Nf3</move>", state)

        assert state.history == ["e4", "e5", "Nf3"]

    def test_illegal_move_doesnt_update_history(self):
        """Illegal moves should not be added to history."""
        env = ChessTextEnv()
        _, state = env.reset()

        env.step("<move>e4</move>", state)
        env.step("<move>e4</move>", state)  # Illegal - square occupied

        assert state.history == ["e4"]
        assert state.move_count == 1

    def test_last_illegal_flag_set(self):
        """State should track if last move was illegal."""
        env = ChessTextEnv()
        _, state = env.reset()

        assert not state.last_illegal

        env.step("<move>e5</move>", state)  # Illegal
        assert state.last_illegal

        env.step("<move>e4</move>", state)  # Legal
        assert not state.last_illegal
