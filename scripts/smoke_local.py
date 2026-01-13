#!/usr/bin/env python3
"""
Local smoke test for the chess environment.

This script demonstrates that the environment works correctly without
any external dependencies (no model APIs, no cloud services).

It runs through several test scenarios:
1. Basic environment creation and reset
2. Playing a known game (Scholar's mate)
3. Self-play with random policy
4. Observation and action encoding verification

Run with:
    uv run python scripts/smoke_local.py
    # or
    python scripts/smoke_local.py
"""

import sys
import time

import numpy as np


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def test_basic_environment():
    """Test basic environment creation and API."""
    print_section("Test 1: Basic Environment")

    from chess_env import ChessTensorEnv, OBSERVATION_SHAPE, ACTION_SPACE_SIZE

    print(f"Creating environment...")
    env = ChessTensorEnv()

    print(f"  Observation shape: {OBSERVATION_SHAPE}")
    print(f"  Action space size: {ACTION_SPACE_SIZE}")

    print(f"\nResetting environment...")
    obs, info = env.reset()

    print(f"  Observation dtype: {obs.dtype}")
    print(f"  Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"  Turn: {info['turn']}")
    print(f"  Legal moves: {info['num_legal_moves']}")
    print(f"  FEN: {info['fen']}")

    print("\n✓ Basic environment test passed!")
    return True


def test_scholars_mate():
    """Play through Scholar's mate to test game mechanics."""
    print_section("Test 2: Scholar's Mate")

    from chess_env import ChessTensorEnv

    env = ChessTensorEnv(render_mode="ansi")
    env.reset()

    # Scholar's mate: 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#
    moves = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7"]

    print("Playing Scholar's Mate:")
    print("-" * 40)

    for i, san in enumerate(moves):
        player = "White" if i % 2 == 0 else "Black"
        move_num = i // 2 + 1

        action = env.san_to_action(san)
        if action is None:
            print(f"ERROR: Could not find action for {san}")
            return False

        obs, reward, terminated, truncated, info = env.step(action)

        if i % 2 == 0:
            print(f"{move_num}. {san}", end=" ")
        else:
            print(f"{san}")

        if terminated:
            print()
            print("-" * 40)
            print(env.render())
            print(f"\nGame over: {info['game_result']}")
            print(f"Reason: {info['termination_reason']}")
            print(f"Final reward (for White): {reward}")

    if not terminated or info['termination_reason'] != 'checkmate':
        print("ERROR: Game should have ended in checkmate")
        return False

    print("\n✓ Scholar's Mate test passed!")
    return True


def test_self_play_random():
    """Test self-play with random policy."""
    print_section("Test 3: Random Self-Play")

    from chess_env import ChessTensorEnv
    from chess_env.self_play import SelfPlayWrapper, create_random_policy

    env = ChessTensorEnv(max_moves=200)
    wrapper = SelfPlayWrapper(env)
    policy = create_random_policy()

    print("Playing 3 random games...")
    print("-" * 40)

    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    total_moves = 0
    start = time.time()

    for i in range(3):
        result = wrapper.play_game(policy)
        results[result.result] += 1
        total_moves += result.num_moves
        print(f"Game {i+1}: {result.result} in {result.num_moves} moves ({result.termination_reason})")

    elapsed = time.time() - start

    print("-" * 40)
    print(f"Results: W={results['1-0']}, B={results['0-1']}, D={results['1/2-1/2']}")
    print(f"Total moves: {total_moves}")
    print(f"Time: {elapsed:.2f}s ({total_moves/elapsed:.1f} moves/sec)")

    print("\n✓ Random self-play test passed!")
    return True


def test_encoding_roundtrip():
    """Test that encoding/decoding is bijective."""
    print_section("Test 4: Encoding Roundtrip")

    from chess_env import ChessTensorEnv
    from chess_env.encoding import ActionEncoder, ObservationEncoder
    import chess

    print("Testing action encoding roundtrip...")

    encoder = ActionEncoder()
    test_positions = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "8/P7/8/8/8/8/8/4K2k w - - 0 1",  # Promotion position
    ]

    total_moves = 0
    errors = 0

    for fen in test_positions:
        board = chess.Board(fen)
        for move in board.legal_moves:
            total_moves += 1
            action = encoder.encode(board, move)
            decoded = encoder.decode(board, action)

            if decoded != move:
                print(f"  ERROR: {move} -> {action} -> {decoded}")
                errors += 1

    print(f"  Tested {total_moves} moves across {len(test_positions)} positions")
    print(f"  Errors: {errors}")

    if errors > 0:
        return False

    print("\n✓ Encoding roundtrip test passed!")
    return True


def test_observation_determinism():
    """Test that observations are deterministic."""
    print_section("Test 5: Observation Determinism")

    from chess_env import ChessTensorEnv

    moves = ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5"]

    def play_and_collect():
        env = ChessTensorEnv()
        env.reset()
        observations = []

        for san in moves:
            obs = env.obs_encoder.encode(env.board)
            observations.append(obs.copy())
            action = env.san_to_action(san)
            env.step(action)

        return observations

    print("Playing same game twice and comparing observations...")

    obs1 = play_and_collect()
    obs2 = play_and_collect()

    all_match = True
    for i, (o1, o2) in enumerate(zip(obs1, obs2)):
        if not np.array_equal(o1, o2):
            print(f"  Mismatch at move {i}")
            all_match = False

    if not all_match:
        print("ERROR: Observations are not deterministic!")
        return False

    print(f"  All {len(obs1)} observations match perfectly")
    print("\n✓ Observation determinism test passed!")
    return True


def test_trajectory_collection():
    """Test trajectory collection and GAE computation."""
    print_section("Test 6: Trajectory Collection")

    from chess_env import ChessTensorEnv
    from chess_env.self_play import SelfPlayWrapper, create_random_policy

    env = ChessTensorEnv(max_moves=50)
    wrapper = SelfPlayWrapper(env)
    policy = create_random_policy()

    print("Collecting trajectories from 2 games...")

    trajectories = wrapper.collect_trajectories(policy, num_games=2)

    print(f"  Collected {len(trajectories)} trajectories")

    for i, traj in enumerate(trajectories):
        returns, advantages = traj.compute_returns_and_advantages()
        print(f"  Trajectory {i+1} ({traj.color}): {len(traj)} steps, "
              f"result={traj.game_result}, "
              f"mean_return={returns.mean():.3f}")

    print("\n✓ Trajectory collection test passed!")
    return True


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("  CHESS ENVIRONMENT SMOKE TEST")
    print("=" * 60)

    tests = [
        ("Basic Environment", test_basic_environment),
        ("Scholar's Mate", test_scholars_mate),
        ("Random Self-Play", test_self_play_random),
        ("Encoding Roundtrip", test_encoding_roundtrip),
        ("Observation Determinism", test_observation_determinism),
        ("Trajectory Collection", test_trajectory_collection),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\n❌ Some tests failed!")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
