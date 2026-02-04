# Hosted RL Training Plan for Chess

## Overview

This document outlines the plan to integrate Prime Intellect's Hosted RL Training platform with our chess environment to fine-tune LLMs for chess play.

**Key insight from benchmarking:** Current LLMs show a clear hierarchy of difficulty:
1. **Format compliance** - outputting `<move>X</move>` tags (~40-100% depending on model)
2. **Move legality** - choosing a move that's actually legal (~20-100%)
3. **Tactical awareness** - finding mate-in-1, avoiding blunders (~0-20%)
4. **Strategic play** - winning games against even random opponents (~0-50%)

RL training can target each level progressively.

---

## Phase 0: Environment Preparation

### Branch Setup
```bash
git checkout -b hosted-rl
```

### Required Changes to ChessTextEnv

Our `ChessTextEnv` is already built on `verifiers`, but we need to ensure it exposes the right interface for PI's platform.

**File: `chess_env/verifiers_env.py`**

Verify `load_environment()` function exists and returns proper `Environment` object:

```python
def load_environment(
    max_moves: int = 200,
    include_legal_moves: bool = True,
    # ... other config options
) -> vf.Environment:
    """Load chess environment for PI Hosted Training."""
    config = ChessTextConfig(
        max_moves=max_moves,
        include_legal_moves=include_legal_moves,
    )
    return ChessTextEnv(config)
```

### Rubric Design

The rubric is the reward function. We need multiple rubrics for different training objectives:

**Rubric 1: Legal Move Training**
```python
async def legal_move_reward(completion, answer, state) -> float:
    """Reward for producing legal moves in correct format."""
    response = completion[-1]['content']
    move = parse_move_from_response(response)

    if move is None:
        return -0.5  # Format error

    try:
        board = chess.Board(state['fen'])
        parsed = board.parse_san(move)
        if parsed in board.legal_moves:
            return 1.0  # Legal move
        return -0.2  # Illegal move (but correct format)
    except:
        return -0.2  # Unparseable move
```

**Rubric 2: Game Outcome Training**
```python
async def game_outcome_reward(completion, answer, state) -> float:
    """Reward based on game result."""
    # Sparse reward at end of game
    if state['result'] == '1-0' and state['llm_color'] == 'white':
        return 1.0
    elif state['result'] == '0-1' and state['llm_color'] == 'black':
        return 1.0
    elif state['result'] in ['1/2-1/2', '*']:
        return 0.0
    else:
        return -1.0
```

**Rubric 3: Tactical Training (Puzzles)**
```python
async def puzzle_reward(completion, answer, state) -> float:
    """Reward for solving chess puzzles correctly."""
    response = completion[-1]['content']
    move = parse_move_from_response(response)

    if move == answer:  # Exact match to puzzle solution
        return 1.0

    # Partial credit for moves that are also checkmate
    if move and is_checkmate(state['fen'], move):
        return 0.8

    return 0.0
```

---

## Phase 1: Legal Move Training (First Win)

**Goal:** Take a model from ~20-40% legality to 90%+ legality.

**Why this first:**
- Clear, measurable improvement
- Fast feedback loop (single-turn evaluation)
- Foundation for all subsequent training

### Dataset

Create a dataset of random chess positions:

```python
# scripts/generate_training_data.py
def generate_legality_dataset(n_positions: int = 10000):
    """Generate random positions for legality training."""
    positions = []
    for _ in range(n_positions):
        board = chess.Board()
        # Play 5-40 random moves
        for _ in range(random.randint(5, 40)):
            if board.is_game_over():
                break
            board.push(random.choice(list(board.legal_moves)))

        if not board.is_game_over():
            positions.append({
                'fen': board.fen(),
                'legal_moves': [board.san(m) for m in board.legal_moves],
            })
    return positions
```

### Training Config

```toml
# configs/lab/chess-legality.toml
model = "Qwen/Qwen3-4B-Instruct-2507"
max_steps = 500
batch_size = 256
rollouts_per_example = 8

[sampling]
max_tokens = 128  # Chess moves are short

[[env]]
id = "apullin/chess"
args = { rubric = "legality", max_moves = 1 }  # Single-turn for legality
```

### Success Criteria
- Legality rate: 20% → 90%+
- Format compliance: maintain or improve
- Inference speed: no significant degradation

---

## Phase 2: Game Play Training

**Goal:** Model can win games against random opponent.

**Prerequisites:** Phase 1 complete (high legality rate)

### Dataset

Full games against random opponent:

```python
def generate_game_dataset(n_games: int = 1000):
    """Generate starting positions for full games."""
    # Mix of:
    # - Standard starting position (60%)
    # - Random opening positions after 4-10 moves (30%)
    # - Endgame positions (10%)
    pass
```

### Training Config

```toml
# configs/lab/chess-games.toml
model = "Qwen/Qwen3-4B-Instruct-2507"  # Or checkpoint from Phase 1
max_steps = 1000
batch_size = 128
rollouts_per_example = 4

[sampling]
max_tokens = 128

[[env]]
id = "apullin/chess"
args = { rubric = "game_outcome", max_moves = 100, opponent = "random" }
```

### Success Criteria
- Win rate vs random: 0% → 50%+
- Average game length: should decrease (faster wins)
- Illegal move rate during games: <5%

---

## Phase 3: Tactical Training

**Goal:** Model can solve simple tactics (mate-in-1, winning captures).

### Dataset

Curated puzzle dataset:

```python
# Use existing MATE_IN_1_PUZZLES from benchmark
# Expand with:
# - Lichess puzzle database (filtered by rating < 1200)
# - Custom generated tactical positions
```

### Training Config

```toml
# configs/lab/chess-tactics.toml
model = "Qwen/Qwen3-4B-Instruct-2507"  # Or checkpoint from Phase 2
max_steps = 500
batch_size = 256
rollouts_per_example = 8

[sampling]
max_tokens = 256  # Allow some thinking

[[env]]
id = "apullin/chess"
args = { rubric = "puzzle", dataset = "tactics" }
```

### Success Criteria
- Mate-in-1 accuracy: 0% → 60%+
- Simple tactics (forks, pins): measurable improvement

---

## The Small Model Question

> "If using a really small LLM... isn't it just a barely roundabout tensor model through an ASCII interface?"

This is a valid philosophical question. Here's the framework for thinking about it:

### What Small Models Bring to the Table

Even a 0.5B parameter model has learned from pretraining:
- Chess move notation (SAN format)
- Basic chess vocabulary ("checkmate", "castle", piece names)
- Some game structure (turns, winning conditions)
- Pattern matching on text that looks like chess

### What RL Would Add

RL training would teach the model to:
- Always output valid format
- Map board states to legal moves
- Recognize winning patterns
- Avoid losing patterns

### The Experimental Value

Training a small model answers: **"Is there ANY value in the LLM foundation that RL can amplify?"**

Possible outcomes:
1. **Small model learns quickly** → LLM pretraining provides useful foundation
2. **Small model plateaus at low skill** → Need more capacity for chess reasoning
3. **Small model matches larger models** → Chess doesn't need scale, just RL

This is cheap to test with PI's free tier and would inform whether to invest in larger model training.

### Recommended Approach

Run parallel experiments:
- **Qwen3-4B**: "Real" LLM with reasoning capability
- **Smallest available**: Test the "tensor model through ASCII" hypothesis

Compare:
- Learning speed (steps to convergence)
- Final performance ceiling
- Generalization (positions not in training set)

---

## Implementation Checklist

### Branch: `hosted-rl`

- [ ] Verify `load_environment()` interface in `verifiers_env.py`
- [ ] Add rubric functions for each training phase
- [ ] Create `configs/lab/` directory with training configs
- [ ] Generate training datasets
- [ ] Add `prime` CLI commands to README
- [ ] Create evaluation scripts for checkpoints

### Local Testing (Before PI Platform)

```bash
# Install prime CLI
uv tool install prime

# Login
prime login

# Test environment locally
prime eval run ./environments/chess -m openai/gpt-4o-mini --num-examples 10
```

### Push to Environments Hub

```bash
# After local validation
prime env push --path ./chess_env
```

### Start Training Run

```bash
prime rl run configs/lab/chess-legality.toml
```

---

## Incorporating H100 Benchmark Results

The benchmark tests running on the H100 will provide baseline data:

| Model | Legality | Mate-in-1 | Win Rate | Notes |
|-------|----------|-----------|----------|-------|
| Qwen2.5-7B | TBD | TBD | TBD | vLLM baseline |
| (parallel) | TBD | TBD | TBD | Throughput comparison |

**How to use this data:**

1. **Model Selection**: Choose base model with highest legality but room for improvement
2. **Reward Calibration**: Set reward thresholds based on observed distributions
3. **Difficulty Filtering**: Focus training on positions where models currently fail
4. **Evaluation Baseline**: Compare fine-tuned model against these benchmarks

---

## Low-Hanging Fruit Opportunities

Based on prior benchmarking insights:

### 1. Reasoning Models Need Un-Learning
DeepSeek-R1 showed 20% legality vs standard models' 96%. Reasoning models:
- Output plausible-looking notation
- Don't understand board state
- Might benefit from RL that penalizes "hallucinated" moves

### 2. Format Compliance is Cheap
Models that fail on format (`<move>...</move>`) but know chess:
- Quick win with supervised fine-tuning
- Or RL with heavy format reward

### 3. Endgame Specialization
Models might learn endgames faster than full games:
- Smaller state space
- Clearer winning conditions
- Train endgame specialist, then full game

### 4. Opening Book Integration
Instead of training openings:
- Use book moves for first N moves
- Train on middlegame/endgame only
- Reduces training complexity

---

## Timeline Estimates

| Phase | Training Steps | ~Time (estimate) | Compute Cost |
|-------|---------------|------------------|--------------|
| Phase 1: Legality | 500 | Hours | Free tier |
| Phase 2: Games | 1000 | Hours-Days | Free tier |
| Phase 3: Tactics | 500 | Hours | Free tier |

Note: PI Hosted Training is currently free during private beta.

---

## Open Questions

1. **Multi-turn vs Single-turn**: Should we train on full games (slow) or individual positions (fast)?

2. **Opponent Curriculum**: Start with random opponent, graduate to stronger? Or vice versa?

3. **Reward Shaping**: Dense rewards (every move) vs sparse (game end only)?

4. **Checkpoint Strategy**: When to save? How to evaluate intermediate checkpoints?

5. **Failure Modes**: What if model learns to "game" the reward without actually playing better chess?

---

## Next Steps (After H100 Tests Complete)

1. Review benchmark results from H100
2. Select base model for training
3. Create branch `hosted-rl`
4. Implement rubric functions
5. Generate training datasets
6. Test locally with `prime eval`
7. Push environment to Hub
8. Start Phase 1 training
9. Evaluate and iterate

---

## References

- [Prime Intellect Lab Documentation](https://primeintellect.notion.site/Lab-Hosted-RL-Training-Private-Beta-2ca72940136f808a8242f53fd2cc9e75)
- [Verifiers Library](https://github.com/PrimeIntellect-ai/verifiers)
- [GRPO Paper](https://arxiv.org/abs/2402.03300) (DeepSeek)
- [INTELLECT-2 Technical Report](https://arxiv.org/abs/2505.07291)
