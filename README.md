# rl-evo-lab

A reproduction and extension of **EDER** (Evolutionary Distributed Experience Replay), originally developed for an MSc thesis: *"Improving Exploration in Evolutionary Reinforcement Learning through Novelty Search"* — NUI Galway, 2021.

---

## Background

Standard deep RL exploration (epsilon-greedy, entropy bonuses) is local and reactive. EDER replaces it entirely with an **Evolution Strategy (ES) actor population** that drives diversity at the *parameter* level, coupled with an **intrinsic novelty reward** that incentivises visiting unexplored states.

Key inspirations:
- **ERL** (Khadka & Tumer, 2018) — EA populations improve DRL exploration, but purely extrinsic fitness is contradictory for an exploration mechanism
- **NGU** (Badia et al., 2020) — episodic + lifelong intrinsic rewards improve exploration in DRL
- **ES** (Salimans et al., 2017) — scalable gradient-free optimisation via parameter-space noise

---

## Algorithm: EDER

```
┌─────────────────────────────────────────────────────────────┐
│  Actor (Evolution Strategy)                                  │
│                                                              │
│  Each episode:                                               │
│    1. Sample N noisy policies: θᵢ = θ + σεᵢ, ε ~ N(0,I)    │
│    2. Score each on augmented reward: rₐ = rₑ + β·rᵢ        │
│    3. Update θ toward high-scoring directions                │
│    4. Push transitions (extrinsic only) → Replay Buffer      │
│    5. Periodically sync θ ← learner weights                  │
│                                                              │
│  rᵢ = KNN distance over episodic memory of                   │
│       controllable-state embeddings (reset each episode)     │
└───────────────────────┬─────────────────────────────────────┘
                        │ replay buffer
┌───────────────────────▼─────────────────────────────────────┐
│  Learner (DQN)                                               │
│                                                              │
│  - Trains on extrinsic reward only                           │
│  - Never interacts with env during training                  │
│  - Periodically broadcasts weights → Actor                   │
└─────────────────────────────────────────────────────────────┘
```

The replay buffer is the **only interface** between actor and learner — both components are independently swappable.

---

## Results (original thesis, CartPole-v1)

| Method | Episodes to max reward |
|---|---|
| DQN (baseline) | ~3,500 |
| EDER (extrinsic) | ~75 |
| EDER (augmented) | ~150, stable |

Extrinsic EDER was fastest but suffered catastrophic forgetting as the buffer became homogeneous. Augmented EDER traded some speed for stability — intrinsic diversity kept the buffer varied.

---

## Structure

```
src/rl_evo_lab/
  actor/      # Evolution Strategy population
  learner/    # DQN (→ SAC/DDPG)
  buffer/     # Shared replay buffer
  intrinsic/  # Episodic KNN + lifelong RND novelty
  utils/      # Seeding, logging, config
```

## Setup

```bash
poetry install                    # core: torch, gymnasium, numpy
poetry install --extras mujoco    # + MuJoCo environments
poetry run pytest                 # tests
poetry run ruff check src/        # lint
```

---

## References

- Khadka & Tumer (2018) — [ERL: Evolution-Guided Policy Gradient](https://arxiv.org/abs/1805.07917)
- Salimans et al. (2017) — [ES as a Scalable Alternative to RL](https://arxiv.org/abs/1703.03864)
- Badia et al. (2020) — [Never Give Up](https://arxiv.org/abs/2002.06038)
- Mnih et al. (2015) — [DQN](https://www.nature.com/articles/nature14236)
- Lillicrap et al. (2015) — [DDPG](https://arxiv.org/abs/1509.02971)
- Lehman & Stanley (2011) — [Novelty Search](https://dl.acm.org/doi/10.1145/1830483.1830503)
