# Questions for Chess RL Environment Project

## 1. Scope & Priority

**Q1.1: Which environment shape to start with?**
The guide recommends `ChessText-v0` (LLM-friendly, text-based) first. Are you aligned with this, or do you prefer `ChessTensor-v0` (CNN/AlphaZero-style) first?

Options:
- [ ] ChessText-v0 first (recommended by guide)
- [x] ChessTensor-v0 first
- [ ] Both in parallel

Let's get NORMAL chess working first. THen we can extend to do "LLMs can play chess" part of the project later on. Assuming it is not a huge rework of the whole thing? Do we actually need a different environment altogether for the LLM approach? It can't just be a little frontend that turns the board encoding into a text encoding, and the reverse of that to execute a move?

**Q1.2: Which milestone is your first target?**
- [ ] Milestone A only (local env works)
- [ ] Milestones A + B (publishable to Hub)
- [ ] Milestones A + B + C (local RL demo with MateIn1)
- [x] All milestones through D (PI-native RL)
- [ ] Full stretch goal E (LLM plays chess)

We can do the LLM part as a second chapter. But it's unclear to me if that's the SAME env or a different env.

**Q1.3: Do you want MateIn1-v0 curriculum environment?**
The guide strongly recommends this for a fast "visible learning" demo. Should I build this alongside ChessText-v0?

- [ ] Yes, build MateIn1-v0
- [x] No, skip curriculum for now

I don't think I want this? It seems like a "shortcut" or "trick" - which is likely valid and valuable, but the real attrction here is if we can RL actual chess strategy and capability from total naive start. Obviously, there will be LESS late games to learn from, which is what the cirriculum is trying to do.

We can tackle this later UNLESS: you KNOW ahead of time that my "naive" plan is just NOT goingn to work.

---

## 2. Technical Decisions

**Q2.1: Thinking tokens?**
Should the LLM be allowed `<think>...</think>` before `<move>...</move>`?

- [x] Yes, include thinking (more tokens, potentially better reasoning)
- [ ] No, just `<move>` (faster, simpler)

Definitely! that's the point - leverage a thinking model, and see if it can already "think" in chess terms.

**Q2.2: Include legal moves in observation?**
Showing legal SAN moves makes it easier for the LLM but reduces the "learning" aspect.

- [ ] Yes, include legal moves list
- [ ] No, model must figure it out
- [x] Configurable (both options available)

I am in the dark here. I actually don't know how RL environments deal with it ... illegal move just terminates immediately? Make it na option, Can't be too hard.

**Q2.3: Self-play or opponent?**
For multi-turn games:
- [x] Self-play (same model plays both sides)
- [ ] Fixed opponent (random, Stockfish at low depth, greedy)
- [ ] Configurable opponent

Self-play is the main idea. Structing it so we can swap in any opponent would be the obvious intent, but that opponent can be frozen from 10 epochs ago (which is how the self-play networks wokr, I think?)

**Q2.4: Max game length?**
Default is 400 half-moves. Any preference?

- [x] 400 (default, covers essentially all games)
- [ ] Shorter (e.g., 200) for faster episodes
- [ ] Custom: ___

I think a 400 step rollout should be fine ? Let's do it.

---

## 3. Infrastructure & Tools

**Q3.1: Do you have GPU access locally?**
This affects whether we can run local RL training or need to use PI's cloud infrastructure.

- [ ] Yes, local GPU (what card?)
- [x] No, CPU only locally
- [ ] Will use PI cloud compute

On OSX. But obviously I'd LIKE to be able to switch over to CUDA with a rented machine, when we move to cloud training.

**Q3.2: Team namespace for Hub push?**
Will you push to your personal namespace or a team?

- [x] Personal namespace
- [ ] Team: _______________

**Q3.3: Base model for LLM experiments?**
If we get to the LLM route, which base model to start with?

- [x] Let me (Claude) suggest based on what PI supports
- [ ] Specific model: _______________

---

## 4. Testing & Verification

**Q4.1: Determinism verification approach?**
How strict do you want determinism tests?

- [ ] Basic: same moves → same result
- [ ] Strict: bit-for-bit reproducible with seeds
- [x] Very strict: include move generation order

I am not sure actually. I think we need "very strict"? dunno.

**Q4.2: Integration with existing code?**
Is there any existing code in this repo I should be aware of, or starting fresh?

- [ ] Starting completely fresh
- [ ] There's existing code at: _______________

Do a web search for what you might need first.

---

## 5. Timeline & Iteration

**Q5.1: Preferred iteration style?**
- [ ] Build everything, then test
- [x] Incremental: small working pieces, test each
- [x] Show me plans before implementing

Charge ahead - but you CAN pause and ask me for input, do another questions2.md, etc.

**Q5.2: How much explanation do you want?**
- [ ] Minimal - just code and essential comments
- [ ] Moderate - explain key design decisions
- [x] Detailed - explain everything as we go

To be honest, I essentially want you to write up a long technique guide about each piece, pretty much. If it helps you to write that as you go, all the better!

---

## Notes / Additional Context

(Add anything else I should know here)

There's a ton about RL I just don't know. Really leaning on you here.

One major intent here is:
1) Just get a CHESS environemtn that I can contribute to PI itself! get some recognition!
2) But I'll need a demo that it WORKS
3) the "zero" demo is such a simple proposal and is nominally proven to work, so it has elegance to it
4) The LLM-chess is there because LLM's apparently can do anything, e.g. VLM/VLA for robots.

---

## My Recommendations (pending your answers)

1. **Start with ChessText-v0** as a `vf.MultiTurnEnv` - this is the PI-native path
2. **Build MateIn1-v0** alongside for fast RL demo validation
3. **Include `<think>` support** but make it optional via config
4. **Show legal moves** initially (can disable later for harder challenge)
5. **Use configurable opponent** - start with random, add Stockfish option
6. **Incremental approach** - get Milestone A working first, then push to Hub

Once you answer these, I can start implementing immediately.
