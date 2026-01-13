# Prime Intellect “Chess” Environment + RL Demo Guide (local-first)

> **Goals (keep it tight):**
> 1) Make a Chess environment that runs locally (deterministic, verifiable)  
> 2) Prove it can be used for RL **locally** (small demo)  
> 3) Publish it to the Prime Intellect Environments Hub  
> 4) Run RL “the PI way” (prime-rl / vf-rl)  
> 5) Stretch: LLM plays chess (LoRA / RL fine-tune)

This document is written to be **Codex-friendly**: lots of concrete steps, checklists, and “smallest thing that works” milestones.

---

## 0) What you’re building (pick the minimum viable route)

### Two environment “shapes” (you can ship both, but start with one)

**A) `ChessText-v0` (recommended first)**  
- Observation: ASCII board + FEN + “side to move” + short status (optional list of legal SAN moves)  
- Action: the model outputs **SAN** inside `<move>...</move>`  
- Best for: LLM policies, `vf-eval`, `prime env eval`, `prime-rl` / `vf-rl`  

**B) `ChessTensor-v0` (for CNN/Mamba “AlphaZero-ish”)**  
- Observation: `8×8×P` planes (or flattened)  
- Action: 4,672 logits (AlphaZero mapping) + illegal mask  
- Best for: small convnets / Mamba + custom RL loop (and potentially later plugging into PI’s RL stack if you wrap it)

**Reality check on “from NOTHING”**  
Full chess from scratch (random init) is compute-hungry. For a **mini demo** that learns visibly, use a curriculum:
- `MateIn1-v0` (single move wins immediately; fast learning)
- `KQK-v0` (King+Queen vs King endgame)
- then `ChessText-v0` full games

You still get the “zero-knowledge RL” vibe without needing a cluster.

---

## 1) Prereqs

### Local tooling
- Python 3.11+ recommended
- `uv` (fast Python env manager)
- Git
- Optional GPU if you plan to run `prime-rl` locally (inference server + trainer)

### Prime Intellect tooling (only needed once you publish / evaluate with hosted inference)
- Prime CLI (`prime`)
- Logged in: `prime login`
- Username set in PI profile

PI docs for creating environments cover `prime env init`, local editable installs, and `prime env push`.  
Docs also mention `vf-eval` for quick local evaluation runs.  
(See: https://docs.primeintellect.ai/tutorials-environments/create)

---

## 2) Create the environment package (local-only)

### Option A (recommended): start from PI template
```bash
# install prime CLI (one suggested method)
uv tool install prime

prime login

# scaffold a new env package
prime env init chess

cd environments/chess
```

This gives you:
- `README.md` (shown on Hub)
- `pyproject.toml` (deps/metadata/tags/version)
- `chess.py` (stub `load_environment()` that returns a `vf.Environment`)

### Repo structure (suggested)
```
environments/chess/
  chess_env.py              # the actual environment implementation
  pyproject.toml
  README.md
  tests/
    test_chess_env.py
  scripts/
    smoke_local.py          # runs a few moves without any API calls
```

---

## 3) Authoring the environment (design & verification)

### Design principles (keep it easy to verify)
- **Deterministic**: given initial FEN + sequence of moves, the outcome is fixed.
- **No side effects**: no filesystem writes, no network calls in env logic.
- **State is explicit**: store board/FEN, move history, and terminal status in environment `state`.

### Chess rules engine
Use `python-chess`:
- legality checks: `board.parse_san(...)` or `board.legal_moves`
- terminal: `board.is_game_over()`, `board.result()`

### Format contract (for LLM friendliness)
- The model’s move must be in `<move>...</move>`
- Optional `<think>...</think>`
- Keep generation short: **max_tokens ~ 8–32** for chess moves

### Rewards (simple and verifiable)
For `ChessText-v0`:
- `+1` if the last mover checkmates and wins
- `0` draw
- `-1` loss
- `-0.2` illegal/empty move (episode continues)

You **do not** need to “learn the reward function.”  
You typically learn a **value function** (critic) to estimate expected win/draw/loss from positions; advantage is computed from returns minus that baseline.

---

## 4) Implement `ChessText-v0` as a Verifiers `MultiTurnEnv`

Verifiers recommends:
- `SingleTurnEnv` for one-shot tasks  
- `MultiTurnEnv` for interactive games (chess is interactive)

A `MultiTurnEnv` implements:
- `env_response(messages, state) -> (messages, state)` (async)
- optionally `is_completed(...)` to end episodes when the game ends

### Minimal behaviors to implement
- `setup_state`: initialize board to start FEN
- `env_response`: parse last assistant message, apply move or mark illegal, then return updated observation as a **user** message
- `is_completed`: return `True` when game over or max plies reached
- `Rubric`: combine reward components (format, legality, terminal outcome)

---

## 5) Local testing (no cloud)

### 5.1 Install your env editable
```bash
uv pip install -e .
```

### 5.2 Unit tests (you should ship them)
Create `tests/test_chess_env.py` and cover:
- reset produces valid start position
- legal SAN move updates board correctly
- illegal move sets `state["illegal"]=True` and doesn’t crash
- checkmate sequence ends episode
- deterministic replay: FEN + SAN sequence leads to same final result

Run:
```bash
uv run python -m unittest
# or
uv run pytest -q
```

### 5.3 Local “smoke script” without any model
Make a `scripts/smoke_local.py` that:
- calls `load_environment()`
- runs a known legal sequence (e.g., Scholar’s Mate)
- prints observation snapshots

This proves the env works without *any* API keys or PI infrastructure.

### 5.4 Optional: local eval harnesses (may call external inference)
- `uv run vf-eval chess` — great for end-to-end testing, but by default it uses a model API.
- `prime env eval ...` — typically uses Prime Inference or another OpenAI-compatible endpoint.

If you want “model-in-loop” locally without cloud, run a local OpenAI-compatible server (e.g., vLLM) and point the evaluator to it.

---

## 6) Publishing to the Environments Hub (PI cloud)

Only do this after local tests are clean.

```bash
prime env push
# or to a team namespace:
prime env push --team <team-slug>
```

After each push, PI runs “Environment Actions” automatically:
- build in a fresh container
- install your wheel
- run your test suite

You can inspect pass/fail logs in the Hub UI under the environment’s **Actions** tab.

---

## 7) RL demo plan (keep it scoped)

You want:
- proof that the env can be used for RL
- visible learning from “nothing”
- a path to PI’s `prime-rl`

### Recommended two-track approach
**Track 1 (fast + visual): tiny CNN on `MateIn1-v0` (local PyTorch)**
- Shows real RL learning quickly
- Avoids LLM complexity
- Demonstrates policy/value and advantage clearly

**Track 2 (PI-native): LLM policy on `ChessText-v0` via `vf-rl` or `prime-rl`**
- Uses PI’s intended RL stack
- Demonstrates LoRA / RL finetune
- “Stretch”: full chess play as text

---

## 8) “How do I set up the NN to play chess?” (practical starter)

### 8.1 CNN policy + critic (classic)
A minimal AlphaZero-style network is:

- **Backbone**: small conv stack / mini-ResNet over `8×8×P`
- **Policy head**: outputs logits over actions (e.g., 4,672)
- **Value head**: outputs a scalar in `[-1, 1]` predicting expected outcome

> You do *not* need a separate “reward model” for chess; you learn **value**, and use it for advantage.

#### Separate vs shared networks?
- Most modern setups share the backbone and have two heads (policy + value).
- Keeping a separate critic network is fine, but usually unnecessary at this scale.

### 8.2 Advantage estimation (the “critic’s job”)
In PPO-style training, advantage is roughly:
- `A_t = (return_t) - V(s_t)`
or using GAE for lower variance.

### 8.3 Why full chess from scratch is hard
- Long horizon (many plies)
- Sparse reward (only at end)
- Huge branching factor

That’s why your demo should start with a simplified environment like:
- **MateIn1** (win in one move) — a few thousand positions is enough
- **KQK** (mate conversion) — learns endgame skill
- then step up

---

## 9) Local RL Demo (CNN) — recommended minimal experiment

### 9.1 Build `MateIn1-v0`
Make a **SingleTurnEnv** where:
- Prompt contains a FEN and “find the winning move”
- Model outputs `<move>...</move>` (SAN)
- Reward:
  - `1.0` if move checkmates
  - `0.0` otherwise
  - plus small format reward

This is fast to learn and doesn’t require multi-turn episodes.

### 9.2 Training loop choices
**Option A: simple supervised bootstrapping**
- Generate “mate-in-1” labels by scanning legal moves with python-chess
- Train a small CNN to predict the correct move (cross-entropy)
- Then switch to RL fine-tuning (optional)

**Option B: true RL from scratch**
- Sample FENs, pick actions from policy, reward is sparse
- Use PPO/A2C with value head
- This works for MateIn1 because episodes are length 1 → very clean signal

For a “look ma, RL!” demo, Option B is excellent.

---

## 10) PI-native RL (LLM plays ChessText-v0)

PI’s Verifiers training docs recommend using:
- `prime-rl` for production-grade scaling
- `vf-rl` as a smaller “hackable” trainer

They provide a bootstrap tool:
```bash
uv add 'verifiers[rl]'
uv run vf-setup
```

This clones and installs `prime-rl` and produces example configs, then you can run:
```bash
uv run prime-rl @ configs/prime-rl/wiki-search.toml
```

### Adapting that to chess
1) Install your environment locally (editable) or from the hub:
```bash
# local
uv pip install -e path/to/environments/chess

# or hub
prime env install <owner>/chess
```

2) Edit the generated prime-rl config:
- set `[[orchestrator.env]] id = "<owner>/chess"` (or local name)
- set `[model] name = "<HF model name>"` (a small instruct model to start)
- set generation parameters:
  - `max_tokens` small (8–32)
  - low temperature (0–0.7)

3) Run prime-rl:
```bash
uv run prime-rl @ configs/prime-rl/chess.toml
```

**Best practice for chess-text RL:**  
Start with a “legality curriculum”:
- Phase 1: reward mostly for valid SAN and legal moves
- Phase 2: shift weight to win/loss outcome

That prevents the “all illegal moves → zero reward forever” trap.

---

## 11) “LLM plays chess” stretch goal (recommended scope)

Don’t aim for “LLM becomes Stockfish.” Aim for:
- it learns to output **legal moves** reliably
- it learns a couple tactics in a small curriculum
- it improves win rate vs a weak opponent (random / material-greedy)

### Two sane LLM routes
**Route 1: LoRA RL fine-tune (PI-native)**
- Take a small instruct model
- Use `ChessText-v0` with legality + sparse outcome
- Run `vf-rl` (quick) or `prime-rl` (robust)

**Route 2: SFT then RL**
- SFT on “legal move generation” for random positions (dense supervision)
- Then RL on outcome-based reward

Route 2 usually gets you to “working agent” faster.

---

## 12) Acceptance checklist (so this project doesn’t sprawl)

### Milestone A — Environment works locally
- [ ] `uv pip install -e .` works
- [ ] `python -m unittest` passes
- [ ] smoke script plays a known legal sequence
- [ ] deterministic replay test passes

### Milestone B — Environment is publishable
- [ ] `pyproject.toml` has correct metadata + deps + tags
- [ ] `README.md` explains protocol (<move> SAN)
- [ ] `prime env push` succeeds
- [ ] Actions tab shows tests passing

### Milestone C — RL demo (local)
- [ ] `MateIn1-v0` exists and trains from random init
- [ ] training curve moves upward within minutes
- [ ] saved checkpoint can solve some held-out positions

### Milestone D — PI-native RL
- [ ] `vf-setup` + config points at your chess env
- [ ] `vf-rl` or `prime-rl` run starts without errors
- [ ] reward metrics/logs show legality improves

### Milestone E — Stretch LLM plays chess
- [ ] Legal-move rate > 95%
- [ ] Win rate vs random opponent improves

---

## Appendix A — Reference code (optional)

This appendix can include full code blocks (env + tests).  
Use it as a reference; Codex can refactor and split into files.

### A.1 Minimal ChessTextEnv (non-async “local harness” style)
> Useful for quick local debugging; for PI Hub you’ll still want a `vf.MultiTurnEnv` implementation.

```python
# (Optional) put this in scripts/ or a local module for smoke testing.
# For Hub publication, prefer vf.MultiTurnEnv (async) + rubric rewards.

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import chess
import verifiers as vf

THINK_MOVE_SYSTEM_PROMPT = "Output one SAN move in <move>...</move> (optional <think>...</think>)."
NOTHINK_MOVE_SYSTEM_PROMPT = "Output one SAN move in <move>...</move>."

def make_parser(use_think: bool = True) -> vf.XMLParser:
    return vf.XMLParser(fields=["think", "move"], answer_field="move") if use_think else vf.XMLParser(fields=["move"], answer_field="move")

@dataclass
class ChessConfig:
    include_legal_san: bool = True
    max_halfmoves: int = 400
    start_fen: Optional[str] = None

class ChessTextEnvLocal:
    def __init__(self, cfg: ChessConfig, parser: vf.XMLParser):
        self.cfg = cfg
        self.parser = parser
        self.board = chess.Board()
        if cfg.start_fen:
            self.board.set_fen(cfg.start_fen)
        self.ply = 0
        self.status = "Game start."

    def reset(self) -> str:
        self.board = chess.Board()
        if self.cfg.start_fen:
            self.board.set_fen(self.cfg.start_fen)
        self.ply = 0
        self.status = "Game start."
        return self.obs()

    def step(self, completion: List[Dict[str, Any]]) -> Tuple[str, float, bool, Dict[str, Any]]:
        try:
            san = self.parser.parse_answer(completion).strip()
        except Exception:
            san = ""

        illegal = False
        try:
            move = self.board.parse_san(san) if san else None
        except Exception:
            move = None
            illegal = True

        reward, done = 0.0, False
        if illegal or move is None:
            reward = -0.2
            self.status = f"Illegal/empty move: '{san}'"
        else:
            mover_was_white = self.board.turn
            self.board.push(move)
            self.ply += 1
            if self.board.is_game_over() or self.ply >= self.cfg.max_halfmoves:
                result = self.board.result() if self.board.is_game_over() else "1/2-1/2"
                if result == "1-0":
                    reward = +1.0 if mover_was_white else -1.0
                elif result == "0-1":
                    reward = +1.0 if (not mover_was_white) else -1.0
                else:
                    reward = 0.0
                done = True
                self.status = f"Game over: {result}"
            else:
                self.status = f"You played {san}"

        return self.obs(), reward, done, {"fen": self.board.fen(), "illegal": illegal}

    def obs(self) -> str:
        lines = [
            "=== Chess ===",
            str(self.board),
            f"FEN: {self.board.fen()}",
            f"Side to move: {'White' if self.board.turn else 'Black'}",
            f"Status: {self.status}",
        ]
        if self.cfg.include_legal_san and not self.board.is_game_over():
            legal = sorted({self.board.san(m) for m in self.board.legal_moves})
            lines.append("Legal SAN (sample): " + ", ".join(legal[:60]) + (" ..." if len(legal) > 60 else ""))
        lines.append("Respond with exactly one SAN move in <move>...</move>.")
        return "\n".join(lines)
```

### A.2 Minimal unit test sketch
```python
import unittest
import verifiers as vf
from chess_env_local import ChessTextEnvLocal, ChessConfig, make_parser

def mk(move_xml: str):
    return [{"role": "assistant", "content": move_xml}]

class TestChessLocal(unittest.TestCase):
    def test_legal_and_illegal(self):
        env = ChessTextEnvLocal(ChessConfig(), make_parser(use_think=False))
        env.reset()
        _, r, done, info = env.step(mk("<move>e4</move>"))
        self.assertFalse(done)
        self.assertGreaterEqual(r, 0.0)
        _, r2, _, info2 = env.step(mk("<move>e9</move>"))
        self.assertLess(r2, 0.0)
        self.assertTrue(info2["illegal"])

if __name__ == "__main__":
    unittest.main()
```

---

## Appendix B — Links you’ll keep open while implementing

- Create & Upload Environment: https://docs.primeintellect.ai/tutorials-environments/create  
- Install & Use Environment: https://docs.primeintellect.ai/tutorials-environments/install  
- Environment Actions: https://docs.primeintellect.ai/tutorials-environments/environment-actions  
- Verifiers environments guide (MultiTurnEnv patterns): https://docs.primeintellect.ai/verifiers/source/environments  
- Verifiers training guide (`vf-setup`, `vf-rl`, prime-rl configs): https://docs.primeintellect.ai/verifiers/source/training  
- PRIME-RL entrypoints (orchestrator/trainer/inference): https://docs.primeintellect.ai/prime-rl/entrypoints  

---

## What I recommend you do next (very concrete)

1) Implement **ChessText-v0** first (MultiTurnEnv + rubric).  
2) Add unit tests + a smoke script.  
3) Push to Hub; confirm Actions pass.  
4) Build **MateIn1-v0** (SingleTurnEnv) and train a tiny CNN locally (fast win).  
5) Only then attempt `vf-rl` / `prime-rl` on `ChessText-v0` (LLM route).  

When you’re ready, tell me which of these you want Codex to implement first:
- “ChessText-v0 MultiTurnEnv” (Hub-ready), or
- “MateIn1-v0 + tiny CNN PPO” (RL demo), or
- Both in one repo (cleaner long-term).
