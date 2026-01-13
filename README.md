# Chess Environment for RL Training

AlphaZero-style chess environment for reinforcement learning, compatible with Gymnasium and Prime Intellect's Environments Hub.

## Features

- **Tensor Observations** (8×8×119): AlphaZero-style board encoding with 8-step history
- **Discrete Actions** (4672): Full AlphaZero action space encoding
- **Self-Play Ready**: Built-in wrapper for self-play training with trajectory collection
- **Deterministic**: Same inputs always produce same outputs
- **PI-Compatible**: Designed for Prime Intellect's RL infrastructure

## Installation

```bash
# With uv (recommended)
uv pip install -e .

# With pip
pip install -e .
```

## Quick Start

```python
from chess_env import ChessTensorEnv

env = ChessTensorEnv()
obs, info = env.reset()

# Get legal actions
legal_actions = info["action_mask"]

# Take a random legal action
import numpy as np
action = np.random.choice(np.where(legal_actions)[0])
obs, reward, terminated, truncated, info = env.step(action)

# Convert between SAN and action indices
action = env.san_to_action("e4")
san = env.action_to_san(action)
```

## Self-Play Training

```python
from chess_env import ChessTensorEnv
from chess_env.self_play import SelfPlayWrapper, create_random_policy

env = ChessTensorEnv()
wrapper = SelfPlayWrapper(env)
policy = create_random_policy()  # Replace with your policy

# Play a game
result = wrapper.play_game(policy)
print(f"Result: {result.result}, Moves: {result.num_moves}")

# Collect training data
trajectories = wrapper.collect_trajectories(policy, num_games=100)
```

## Running Tests

```bash
uv run pytest
uv run python scripts/smoke_local.py
```

## Architecture

```
ChessTensorEnv (Gymnasium)
├── ObservationEncoder (8×8×119 tensor)
├── ActionEncoder (4672 discrete actions)
└── python-chess (game logic)
        │
        ▼
SelfPlayWrapper
├── Trajectory collection
├── GAE computation
└── Batch preparation for PPO
```

## License

MIT
