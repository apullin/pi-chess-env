"""
Tests for the ChessTensorEnv Gymnasium environment.

These tests verify:
1. Environment follows Gymnasium API correctly
2. Observations and actions are valid
3. Game mechanics work (checkmate, draw, etc.)
4. Deterministic replay produces same results
5. Info dict contains expected keys
"""

import chess
import numpy as np
import pytest

from chess_env.env import ChessTensorEnv
from chess_env.encoding import ACTION_SPACE_SIZE, OBSERVATION_SHAPE


class TestEnvironmentBasics:
    """Basic Gymnasium compliance tests."""

    def test_observation_space(self):
        """Observation space should match expected shape."""
        env = ChessTensorEnv()
        obs, _ = env.reset()

        assert env.observation_space.shape == OBSERVATION_SHAPE
        assert obs.shape == OBSERVATION_SHAPE

    def test_action_space(self):
        """Action space should have 4672 discrete actions."""
        env = ChessTensorEnv()

        assert env.action_space.n == ACTION_SPACE_SIZE
        assert env.action_space.n == 4672

    def test_reset_returns_valid_observation(self):
        """Reset should return observation and info dict."""
        env = ChessTensorEnv()
        result = env.reset()

        assert len(result) == 2
        obs, info = result

        assert isinstance(obs, np.ndarray)
        assert obs.shape == OBSERVATION_SHAPE
        assert isinstance(info, dict)

    def test_step_returns_correct_tuple(self):
        """Step should return (obs, reward, terminated, truncated, info)."""
        env = ChessTensorEnv()
        obs, info = env.reset()

        # Get a legal action
        legal_actions = np.where(info["action_mask"])[0]
        action = legal_actions[0]

        result = env.step(action)

        assert len(result) == 5
        obs, reward, terminated, truncated, info = result

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_observation_in_valid_range(self):
        """Observations should be in [0, 1] range."""
        env = ChessTensorEnv()
        obs, _ = env.reset()

        assert obs.min() >= 0.0
        assert obs.max() <= 1.0


class TestGameMechanics:
    """Tests for chess game rules and mechanics."""

    def test_legal_moves_in_info(self):
        """Info should contain action mask and legal moves."""
        env = ChessTensorEnv()
        _, info = env.reset()

        assert "action_mask" in info
        assert "legal_moves_san" in info
        assert "legal_moves_uci" in info

        # Start position has 20 legal moves
        assert len(info["legal_moves_san"]) == 20
        assert info["action_mask"].sum() == 20

    def test_checkmate_terminates_game(self):
        """Game should terminate on checkmate with correct reward."""
        env = ChessTensorEnv()
        env.reset()

        # Play Scholar's mate
        # 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#
        moves_san = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7"]

        for san in moves_san[:-1]:
            action = env.san_to_action(san)
            assert action is not None, f"Could not find action for {san}"
            obs, reward, terminated, truncated, info = env.step(action)
            assert not terminated

        # Final move should be checkmate
        action = env.san_to_action("Qxf7")
        obs, reward, terminated, truncated, info = env.step(action)

        assert terminated
        assert reward == 1.0  # White wins
        assert info["termination_reason"] == "checkmate"
        assert info["game_result"] == "1-0"

    def test_stalemate_is_draw(self):
        """Stalemate should terminate with 0 reward."""
        # Position where White can stalemate Black with one move
        # Black king on a8, White king on a6, White queen can stalemate
        fen = "k7/8/K7/8/8/8/8/1Q6 w - - 0 1"
        env = ChessTensorEnv(start_fen=fen)
        env.reset()

        # Qb6 creates stalemate (king trapped on a8, no legal moves)
        action = env.san_to_action("Qb6")
        obs, reward, terminated, truncated, info = env.step(action)

        assert terminated, f"Game should have ended, got terminated={terminated}"
        assert reward == 0.0, f"Stalemate should give 0 reward, got {reward}"
        assert info["termination_reason"] == "stalemate", f"Got {info.get('termination_reason')}"

    def test_max_moves_truncation(self):
        """Game should truncate after max_moves."""
        env = ChessTensorEnv(max_moves=4)
        env.reset()

        # Play 4 arbitrary legal moves
        for i in range(4):
            _, info = env.reset() if i == 0 else (None, info)
            if i == 0:
                _, info = env.reset()

            mask = info["action_mask"]
            action = np.where(mask)[0][0]
            obs, reward, terminated, truncated, info = env.step(action)

            if i < 3:
                assert not truncated
            else:
                assert truncated
                assert reward == 0.0  # Truncation treated as draw

    def test_turn_alternates(self):
        """Turn should alternate between White and Black."""
        env = ChessTensorEnv()
        _, info = env.reset()

        assert info["turn"] == "white"

        action = np.where(info["action_mask"])[0][0]
        _, _, _, _, info = env.step(action)

        assert info["turn"] == "black"

        action = np.where(info["action_mask"])[0][0]
        _, _, _, _, info = env.step(action)

        assert info["turn"] == "white"


class TestActionConversions:
    """Tests for action <-> SAN conversion utilities."""

    def test_san_to_action(self):
        """SAN notation should convert to valid action."""
        env = ChessTensorEnv()
        env.reset()

        action = env.san_to_action("e4")
        assert action is not None
        assert 0 <= action < ACTION_SPACE_SIZE

    def test_action_to_san(self):
        """Action should convert back to SAN notation."""
        env = ChessTensorEnv()
        env.reset()

        action = env.san_to_action("e4")
        san = env.action_to_san(action)

        assert san == "e4"

    def test_get_legal_actions(self):
        """get_legal_actions should return list of valid actions."""
        env = ChessTensorEnv()
        env.reset()

        legal = env.get_legal_actions()

        assert len(legal) == 20  # Start position
        for action in legal:
            assert 0 <= action < ACTION_SPACE_SIZE

    def test_invalid_san_returns_none(self):
        """Invalid SAN should return None."""
        env = ChessTensorEnv()
        env.reset()

        assert env.san_to_action("invalid") is None
        assert env.san_to_action("e9") is None
        assert env.san_to_action("Qxf7") is None  # Not legal from start


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_game_same_result(self):
        """Playing the same moves should produce the same observations."""
        moves = ["e4", "e5", "Nf3", "Nc6", "Bb5"]

        def play_game():
            env = ChessTensorEnv()
            observations = []
            env.reset()

            for san in moves:
                obs, _ = env.reset() if len(observations) == 0 else (obs, None)
                if len(observations) == 0:
                    obs, _ = env.reset()
                observations.append(obs.copy())

                action = env.san_to_action(san)
                obs, _, _, _, _ = env.step(action)

            observations.append(obs.copy())
            return observations

        obs1 = play_game()
        obs2 = play_game()

        assert len(obs1) == len(obs2)
        for o1, o2 in zip(obs1, obs2):
            np.testing.assert_array_equal(o1, o2)

    def test_reset_with_fen(self):
        """Reset with specific FEN should produce consistent state."""
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        env = ChessTensorEnv()

        obs1, info1 = env.reset(options={"fen": fen})
        obs2, info2 = env.reset(options={"fen": fen})

        np.testing.assert_array_equal(obs1, obs2)
        assert info1["fen"] == info2["fen"]

    def test_reset_with_moves(self):
        """Reset with preset moves should apply them."""
        env = ChessTensorEnv()
        obs, info = env.reset(options={"moves": ["e4", "e5", "Nf3"]})

        assert info["fullmove_number"] == 2
        assert info["turn"] == "black"


class TestClone:
    """Tests for environment cloning."""

    def test_clone_produces_identical_state(self):
        """Cloned environment should have identical state."""
        env = ChessTensorEnv()
        env.reset()

        # Play a few moves
        for san in ["e4", "e5", "Nf3"]:
            action = env.san_to_action(san)
            env.step(action)

        cloned = env.clone()

        # Should have same FEN
        assert env.board.fen() == cloned.board.fen()

        # Should have same move count
        assert env.move_count == cloned.move_count

        # Observations should be identical
        obs1 = env.obs_encoder.encode(env.board)
        obs2 = cloned.obs_encoder.encode(cloned.board)
        np.testing.assert_array_equal(obs1, obs2)

    def test_clone_is_independent(self):
        """Changes to clone should not affect original."""
        env = ChessTensorEnv()
        env.reset()

        cloned = env.clone()

        # Make move in clone
        action = cloned.san_to_action("e4")
        cloned.step(action)

        # Original should be unchanged
        assert env.board.fen() == chess.STARTING_FEN
        assert cloned.board.fen() != chess.STARTING_FEN


class TestRender:
    """Tests for rendering."""

    def test_ansi_render(self):
        """ANSI render should return string."""
        env = ChessTensorEnv(render_mode="ansi")
        env.reset()

        output = env.render()

        assert isinstance(output, str)
        assert "r n b q k b n r" in output.lower() or "♖" in output or "R" in output


class TestEdgeCases:
    """Tests for edge cases and unusual positions."""

    def test_insufficient_material(self):
        """Insufficient material should be a draw."""
        # King vs King
        fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
        env = ChessTensorEnv(start_fen=fen)
        obs, info = env.reset()

        # Game should already be over
        assert env.board.is_game_over()
        assert env.board.is_insufficient_material()

    def test_position_with_many_legal_moves(self):
        """Position with many legal moves should work."""
        # Queen in center with lots of moves
        fen = "4k3/8/8/3Q4/8/8/8/4K3 w - - 0 1"
        env = ChessTensorEnv(start_fen=fen)
        _, info = env.reset()

        # Queen has many moves, king has some
        assert info["num_legal_moves"] > 20

    def test_en_passant(self):
        """En passant should work correctly."""
        # Position where en passant is possible
        fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
        env = ChessTensorEnv(start_fen=fen)
        env.reset()

        # En passant capture should be legal
        action = env.san_to_action("exd6")
        assert action is not None

        obs, _, _, _, info = env.step(action)
        # The d5 pawn should be gone
        assert env.board.piece_at(chess.D5) is None
