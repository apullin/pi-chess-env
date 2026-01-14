# Chess Environment for RL Training

AlphaZero-style chess environment for reinforcement learning, compatible with Gymnasium and Prime Intellect's Environments Hub.

## What & Intent

1. **Tensor environment first** - `ChessTensorEnv` uses AlphaZero-style encoding (8x8x119 observations, 4672 discrete actions). This gives us a verifiably correct chess game that RL can interact with.

2. **Naive PPO doesn't work** - Vanilla PPO with Stable-Baselines3 barely beats random play. Learning chess from scratch with sparse rewards is hard when your action space is 4672 and games last 100+ moves.

3. **MCTS makes it work** - AlphaZero-style tree search provides much better training signal. The network learns from visit counts, not just win/loss. This gives us a "zero-knowledge" learning setup that actually learns.

4. **Text/LLM interface is next** - The same game core will get a text wrapper (`ChessTextEnv`) for LLM agents and `prime-rl` integration.

**An interesting TODO:** PPO might actually work in the LLM setting. Unlike a randomly-initialized CNN, an LLM already knows what chess *is*. PPO fine-tuning could work where it failed on tensors because the model has priors, the "action space" is just generating short text, and it can reason about positions. Worth testing.

## Status

**v0.1** - Tensor-based environment for CNN/RL training (AlphaZero-style).

| Component | Status |
|-----------|--------|
| `ChessTensorEnv` | Complete - 8x8x119 obs, 4672 actions |
| PPO Training | Complete - via Stable-Baselines3 |
| MCTS Training | Complete - parallel self-play |
| `ChessTextEnv` | Experimental - LLM interface for `prime-rl` planned for v0.2 |

## Features

- **Tensor Observations** (8x8x119): AlphaZero-style board encoding with 8-step history
- **Discrete Actions** (4672): Full AlphaZero action space encoding
- **MCTS Self-Play**: AlphaZero-style tree search training
- **SB3 Integration**: PPO training with Stable-Baselines3
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

## Training

### MCTS Self-Play (AlphaZero-style)

```bash
# Quick test
uv run python scripts/train_mcts.py --iterations 10 --games 5 --simulations 50 --network tiny

# Longer training
uv run python scripts/train_mcts.py --iterations 100 --games 20 --simulations 100 --network small
```

### PPO with Stable-Baselines3

```bash
uv run python scripts/train.py --timesteps 100000 --network small --n-envs 12
```

### Benchmarking

```bash
# Benchmark MCTS model
uv run python scripts/benchmark_mcts.py checkpoints/mcts_final.pt --simulations 100

# Benchmark PPO model
uv run python scripts/benchmark.py checkpoints/final_model.zip --games 20
```

## Running Tests

```bash
uv run pytest
uv run python scripts/smoke_local.py
```

## Architecture

```
ChessTensorEnv (Gymnasium)
├── ObservationEncoder (8x8x119 tensor)
├── ActionEncoder (4672 discrete actions)
└── python-chess (game logic)
        │
        ├── MCTS Training
        │   ├── Tree search with UCB/PUCT
        │   ├── Parallel self-play
        │   └── Policy + Value targets
        │
        └── PPO Training (SB3)
            ├── CNN feature extractor
            └── VecNormalize
```

## Roadmap

- [x] v0.1: TensorEnv + MCTS + PPO training
- [ ] v0.2: ChessTextEnv for LLM agents (`prime-rl` integration)
- [ ] v0.3: Curriculum learning (MateIn1, KQK endgames)

## LLM Interface Concept (v0.2 Planning)

The text/LLM interface opens up several possible directions:

**Benchmarking (no training)** - Evaluate how well various LLMs play chess out of the box. Metrics: legal move rate, win rate vs random, win rate vs Stockfish. Answers "how good is GPT-4 at chess?"

**RL Fine-tuning** - Take a base LLM (Llama 8B, Qwen, etc.) and RL fine-tune it to play better. LoRA is the typical approach - efficient, doesn't destroy base capabilities. This is what `prime-rl` / `vf-rl` are designed for.

**Curriculum** - Makes RL tractable. Start with "just output legal moves" (high reward for legality), then "solve mate-in-1", then full games. Prevents the death spiral where everything is illegal and there's no learning signal.

### The Format

`ChessTextEnv` presents the board as text:
```
=== Chess ===
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . P . . .
. . . . . . . .
P P P P . P P P
R N B Q K B N R

FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
Side to move: Black
Legal moves (20): a6, a5, b6, b5, ... Nf6, Nh6

Respond with your move in <move>SAN</move> format.
```

The LLM responds with `<move>e5</move>`. Rewards: +1 win, -1 loss, -0.2 illegal move.

### Workflow

1. `prime env install apullin/chess`
2. Configure `prime-rl` with ChessTextEnv
3. Pick base model (e.g., `meta-llama/Llama-3-8B-Instruct`)
4. Run RL training with LoRA
5. Model learns legal moves first, then strategy

## License

MIT
