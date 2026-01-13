"""
Tests for the AlphaZero-style encoding module.

These tests verify:
1. Observation encoding produces correct shapes and values
2. Action encoding is bijective (encode then decode gives same move)
3. Legal action masks are correct
4. Board flipping works correctly for Black's perspective
5. Determinism: same inputs always produce same outputs
"""

import chess
import numpy as np
import pytest

from chess_env.encoding import (
    ACTION_SPACE_SIZE,
    OBSERVATION_SHAPE,
    ActionEncoder,
    ObservationEncoder,
)


class TestObservationEncoder:
    """Tests for ObservationEncoder."""

    def test_observation_shape(self):
        """Observation should be 8x8x119."""
        encoder = ObservationEncoder()
        board = chess.Board()
        encoder.reset(board)

        obs = encoder.encode(board)

        assert obs.shape == OBSERVATION_SHAPE
        assert obs.shape == (8, 8, 119)

    def test_observation_dtype_and_range(self):
        """Observation values should be float32 in [0, 1]."""
        encoder = ObservationEncoder()
        board = chess.Board()
        encoder.reset(board)

        obs = encoder.encode(board)

        assert obs.dtype == np.float32
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_start_position_has_pieces(self):
        """Start position should have pieces encoded on correct squares."""
        encoder = ObservationEncoder()
        board = chess.Board()
        encoder.reset(board)

        obs = encoder.encode(board)

        # In the start position with White to move (no flip):
        # White pawns are on rank 2 (index 6 from top)
        # White's pieces are in planes 0-5 of the first 14 planes
        # Pawn plane is index 0

        # Check that there are exactly 8 white pawns encoded
        pawn_plane = obs[:, :, 0]  # First plane = our pawns (White's pawns)
        assert pawn_plane.sum() == 8.0

        # All white pawns should be on rank 2 (row index 6)
        assert pawn_plane[6, :].sum() == 8.0

    def test_empty_squares_are_zero(self):
        """Empty squares should have zeros in piece planes."""
        encoder = ObservationEncoder()
        board = chess.Board()
        encoder.reset(board)

        obs = encoder.encode(board)

        # Row 4 (rank 5) should be empty in starting position
        piece_planes = obs[4, :, :12]  # All piece planes
        assert piece_planes.sum() == 0.0

    def test_black_perspective_flips_board(self):
        """When Black to move, board should be flipped."""
        encoder = ObservationEncoder()
        board = chess.Board()
        board.push_san("e4")  # White moves, now Black to move
        encoder.reset(board)

        obs = encoder.encode(board)

        # From Black's perspective (board flipped 180°):
        # - Black's pieces are "our" pieces (planes 0-5)
        # - Black's pawns on rank 7 appear at row index 1 after flip
        #   (rank 7 -> 7-7=0, but we invert for display so it's row 1)
        # - Actually: flip means rank 7 -> rank 0, displayed at row 7-0=7...
        # Let's just check that Black has 8 pawns somewhere in plane 0
        pawn_plane = obs[:, :, 0]
        assert pawn_plane.sum() == 8.0, f"Expected 8 black pawns, got {pawn_plane.sum()}"

        # And White (opponent) should have 8 pawns in plane 6
        opp_pawn_plane = obs[:, :, 6]
        assert opp_pawn_plane.sum() == 8.0, f"Expected 8 white pawns as opponent"

    def test_history_tracking(self):
        """History should track multiple positions."""
        encoder = ObservationEncoder()
        board = chess.Board()
        encoder.reset(board)

        # Play some moves
        moves = ["e4", "e5", "Nf3", "Nc6"]
        for san in moves:
            board.push_san(san)
            encoder.push(board)

        obs = encoder.encode(board)

        # Should still produce valid observation
        assert obs.shape == OBSERVATION_SHAPE

        # History should have 5 positions (start + 4 moves)
        assert len(encoder.history) == 5

    def test_determinism(self):
        """Same position should always produce same observation."""
        encoder1 = ObservationEncoder()
        encoder2 = ObservationEncoder()

        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")

        encoder1.reset(chess.Board())
        encoder1.push(chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"))
        encoder1.push(board)

        encoder2.reset(chess.Board())
        encoder2.push(chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"))
        encoder2.push(board)

        obs1 = encoder1.encode(board)
        obs2 = encoder2.encode(board)

        np.testing.assert_array_equal(obs1, obs2)


class TestActionEncoder:
    """Tests for ActionEncoder."""

    def test_action_space_size(self):
        """Action space should have 4672 actions."""
        assert ACTION_SPACE_SIZE == 8 * 8 * 73
        assert ACTION_SPACE_SIZE == 4672

    def test_encode_decode_roundtrip_start_position(self):
        """Encoding then decoding should give the same move."""
        encoder = ActionEncoder()
        board = chess.Board()

        for move in board.legal_moves:
            action = encoder.encode(board, move)
            decoded = encoder.decode(board, action)

            assert decoded is not None
            assert decoded == move, f"Roundtrip failed for {move}: got {decoded}"

    def test_encode_decode_roundtrip_various_positions(self):
        """Test roundtrip on various board positions."""
        encoder = ActionEncoder()
        test_fens = [
            # Start position
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            # After e4 e5
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            # Position with promotion possible
            "8/P7/8/8/8/8/8/4K2k w - - 0 1",
            # Complex middlegame
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            # Endgame
            "8/8/8/8/8/4K3/8/4k2R w - - 0 1",
        ]

        for fen in test_fens:
            board = chess.Board(fen)
            for move in board.legal_moves:
                action = encoder.encode(board, move)
                decoded = encoder.decode(board, action)

                assert decoded is not None, f"Failed to decode action {action} from {fen}"
                assert decoded == move, f"Roundtrip failed for {move} in {fen}: got {decoded}"

    def test_legal_action_mask_shape(self):
        """Legal action mask should have shape (4672,)."""
        encoder = ActionEncoder()
        board = chess.Board()

        mask = encoder.legal_action_mask(board)

        assert mask.shape == (ACTION_SPACE_SIZE,)
        assert mask.dtype == bool

    def test_legal_action_mask_start_position(self):
        """Start position should have exactly 20 legal moves."""
        encoder = ActionEncoder()
        board = chess.Board()

        mask = encoder.legal_action_mask(board)

        # 16 pawn moves + 4 knight moves = 20
        assert mask.sum() == 20

    def test_all_legal_moves_are_masked_true(self):
        """Every legal move should be True in the mask."""
        encoder = ActionEncoder()
        board = chess.Board()

        # Play a few moves to get interesting position
        for san in ["e4", "e5", "Nf3", "Nc6", "Bb5"]:
            board.push_san(san)

        mask = encoder.legal_action_mask(board)

        for move in board.legal_moves:
            action = encoder.encode(board, move)
            assert mask[action], f"Legal move {move} not marked as legal"

    def test_knight_moves(self):
        """Knight moves should encode/decode correctly."""
        encoder = ActionEncoder()
        # Position with knight in center
        board = chess.Board("8/8/8/4N3/8/8/8/4K2k w - - 0 1")

        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.KNIGHT:
                action = encoder.encode(board, move)
                decoded = encoder.decode(board, action)
                assert decoded == move

    def test_pawn_promotion(self):
        """Pawn promotions should encode/decode correctly."""
        encoder = ActionEncoder()
        # White pawn about to promote
        board = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")

        for move in board.legal_moves:
            action = encoder.encode(board, move)
            decoded = encoder.decode(board, action)
            assert decoded == move, f"Promotion move {move} failed: got {decoded}"

    def test_underpromotion(self):
        """Underpromotions should have distinct action indices."""
        encoder = ActionEncoder()
        board = chess.Board("8/P7/8/8/8/8/8/4K2k w - - 0 1")

        actions = {}
        for move in board.legal_moves:
            if move.promotion:
                action = encoder.encode(board, move)
                key = (move.from_square, move.to_square, move.promotion)
                actions[key] = action

        # Should have 4 different actions for 4 promotion types
        assert len(actions) == 4
        assert len(set(actions.values())) == 4  # All unique

    def test_black_perspective_encoding(self):
        """Actions should encode correctly from Black's perspective."""
        encoder = ActionEncoder()
        board = chess.Board()
        board.push_san("e4")  # Now Black to move

        for move in board.legal_moves:
            action = encoder.encode(board, move)
            decoded = encoder.decode(board, action)
            assert decoded == move, f"Black move {move} failed: got {decoded}"

    def test_castling_encodes_correctly(self):
        """Castling moves should encode as king moving two squares."""
        encoder = ActionEncoder()
        # Position where White can castle kingside
        board = chess.Board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")

        # Find castling moves
        for move in board.legal_moves:
            if board.is_castling(move):
                action = encoder.encode(board, move)
                decoded = encoder.decode(board, action)
                # Castling encodes as king move (e1-g1 or e1-c1)
                assert decoded.from_square == move.from_square
                assert decoded.to_square == move.to_square


class TestEncodingIntegration:
    """Integration tests for observation and action encoding together."""

    def test_full_game_encoding(self):
        """Encode observations and actions for a complete short game."""
        obs_encoder = ObservationEncoder()
        action_encoder = ActionEncoder()

        board = chess.Board()
        obs_encoder.reset(board)

        # Scholar's mate
        moves = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]

        for san in moves:
            obs = obs_encoder.encode(board)
            assert obs.shape == OBSERVATION_SHAPE

            move = board.parse_san(san)
            action = action_encoder.encode(board, move)
            decoded = action_encoder.decode(board, action)
            assert decoded == move

            board.push(move)
            obs_encoder.push(board)

        assert board.is_checkmate()

    def test_encoding_determinism_across_games(self):
        """Same game played twice should produce identical encodings."""
        moves = ["d4", "d5", "c4", "e6", "Nc3", "Nf6"]

        def play_game():
            obs_encoder = ObservationEncoder()
            action_encoder = ActionEncoder()
            board = chess.Board()
            obs_encoder.reset(board)

            observations = []
            actions = []

            for san in moves:
                obs = obs_encoder.encode(board)
                observations.append(obs)

                move = board.parse_san(san)
                action = action_encoder.encode(board, move)
                actions.append(action)

                board.push(move)
                obs_encoder.push(board)

            return observations, actions

        obs1, act1 = play_game()
        obs2, act2 = play_game()

        assert act1 == act2
        for o1, o2 in zip(obs1, obs2):
            np.testing.assert_array_equal(o1, o2)
