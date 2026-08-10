# DQN_play_yakuza

Yakuza 6 reinforcement-learning control experiment.

## The model plans, it doesn't replay macros

The default agent (`--arch ctm`) is a **Continuous Thought Machine planner**
(after [Sakana AI's CTM](https://arxiv.org/abs/2505.05522)). Two things make it
different from a plain DQN:

- **It outputs a sequence, not an action.** One decision yields a plan
  `[a₀, a₁, …, a_{L-1}]` — "what I intend to do for the next L steps". Combos are
  therefore *planned by the model*, not read from a recorded macro file. The old
  `combos.json` / `ComboManager` path is gone.
- **It thinks before answering, and knows when it's done thinking.** Each decision
  runs `iterations` internal ticks over the same frame; every tick emits a plan and
  a self-assessed certainty, and the most-certain tick wins. **How much of the plan
  to commit** is then set by how decisive that plan is — confident means finish the
  combo, unsure means one step at a time. Taking damage mid-plan discards the rest
  immediately.

Decisiveness is measured as the *relative action gap* — (best Q − runner-up Q) /
(best Q − worst Q), averaged over slots — not as the entropy of the Q-softmax. Entropy
tracks the absolute scale of Q, which grows with training: measured on random Q
values, σ=2.0 noise scores 0.42 "certainty" while a genuinely decisive σ=0.3 policy
scores 0.03, so an entropy-driven agent would commit long combos out of noise. The
relative gap is scale-free: a flat 0.147 whenever no action is preferred, ~0.72 when
one clearly is. (Entropy is still used *inside* a forward pass to pick the tick, where
all ticks share one Q scale and the comparison is valid.)

So the model can change its mind: the plan issued at step *t+1* is free to
contradict what step *t* intended. How often it does is logged as `Plan/Drift`.

Training: slot 0 is grounded by ordinary QR-DQN (double, n-step, PER); slots 1…L-1
learn by bootstrapping off the target network's plan one step ahead
(`Q_j(s) ← Q_{j-1}^target(s')`), so the whole plan stays value-anchored.

The previous architecture (Conv3D + FPN + GRU + QR-DQN, single action per step) is
still available as `--arch pro` for A/B comparison.

## Architecture

```
ctm_components.py     # vendored CTM primitives (NLM / SynapseUNET), Apache-2.0
ctm_planner.py        # CTMPlannerNet — image+HP -> plan of quantile Q-values
ctm_agent.py          # CTMPlannerAgent — TD on slot 0 + plan-consistency on slots 1..L-1
network_components.py # legacy ProNet (--arch pro)

core/                 # OS-agnostic: protocol, DecisionClient, CommandServer, game loop
backends/windows/     # SendInput, MSS/BitBlt/DXCAM, NtSuspend
backends/linux/       # uinput, PipeWire/XShm, SIGSTOP (Proton/Wayland)
```

Root-level modules (`game_node.py`, `directkeys.py`, …) are thin compatibility shims.

- [DISTRIBUTED.md](DISTRIBUTED.md) — two-node AI/game runtime  
- [LINUX.md](LINUX.md) — Proton + Wayland game-side setup  

## Dependencies

### Windows game machine

```bash
pip install -r requirements-game.txt
```

### Linux game machine (Proton / Wayland)

```bash
pip install -r requirements-game-linux.txt
```

### AI / training machine

```bash
pip install -r requirements-ai.txt
```

## Quick start

```bash
# AI (CTM planner is the default)
python main.py --mode ai-node --host 0.0.0.0 --port 15001 --model-profile large
# baseline for comparison:
python main.py --mode ai-node --arch pro --model-profile large

# Game client (auto-selects windows/linux backend)
python main.py --mode game-node --decision-host <AI_IP> --decision-port 15001
# optional: --capture auto|pipewire|x11shm|mss|bitblt|dxcam
```

Checkpoints are per-architecture and are **not** interchangeable
(`ctm_planner*.pth` vs `dqn_optimized*.pth`); loading the wrong one fails loudly
instead of silently initialising half a network.

## Checks

```bash
python test_ctm_planner.py      # shapes, commit logic, one training step, plan convergence
python tools/bench_ctm.py       # decision latency vs the --game-fps budget
python test_health_detection.py # health-bar regression against samples/
```

