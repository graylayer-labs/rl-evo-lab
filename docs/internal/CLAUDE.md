# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reproduction and extension of MSc thesis: *"Improving Exploration in Evolutionary Reinforcement Learning through Novelty Search"* (NUI Galway, 2021).

The core algorithm is **EDER** (Evolutionary Distributed Experience Replay): a gradient-based DRL learner (DQN) combined with an Evolution Strategy (ES) actor population. The ES fills the replay buffer with diverse experiences; the learner trains purely from that buffer — no epsilon-greedy exploration.

- Python `>=3.12`, managed with [UV](https://docs.astral.sh/uv/)

## Common Commands

```bash
uv sync                 # Sync dependencies from uv.lock
uv add <package>        # Add a dependency
uv add --dev <package>  # Add a dev dependency
uv run pytest           # Run tests
uv run pytest tests/test_foo.py::test_bar  # Run single test
```

## Architecture

### Key design invariants
- **Learner and actor are independently swappable** — the replay buffer is the only interface between them
- **Augmented reward is internal to the actor** — the learner always trains on extrinsic reward only
- **ES rollouts fill the buffer exclusively** — the learner does not store its own episodes

### Learner (DQN → SAC/DDPG)
Standard DQN (policy net + target net). Only interacts with the environment for evaluation. Updated via batches sampled from the shared replay buffer (MSE loss + Adam).

### Actor (Evolution Strategy — Salimans et al. 2017)
Each episode: generates N noisy parameter vectors (θ + σε, ε ~ N(0,1)), scores each on augmented reward, updates θ toward best-performing directions. Periodically syncs back to learner weights to prevent divergence.

### Augmented Reward
```
rₐ = rₑ + β · rᵢ
```
- `rᵢ` = KNN distance over an **episodic** memory of controllable-state embeddings (reset each episode)
- Controllable states: embeddings from a network trained to predict the action taken between (sₜ, sₜ₊₁)
- Only extrinsic transitions are pushed to the replay buffer

### Key hyperparameters
| Param | Value | Notes |
|-------|-------|-------|
| σ | 0.06 | ES noise std dev (baseline default) |
| β | 0.02 | Intrinsic reward weight (baseline default) |
| N workers | 50 | ES population size |
| Sync freq τ | 25 eps | Actor → learner weight sync |

## Logging
Track these separately — they tell very different stories:
- Actor (augmented) reward per episode
- Learner evaluation reward
- Mean worker reward across population
- Replay buffer diversity metric

Always seed everything and log σ and β per run.

## Current Roadmap
1. Clean EDER reproduction (DQN + ES, CartPole)
2. Swap DQN learner for SAC or DDPG (continuous action spaces)
3. Test on MuJoCo locomotion / sparse reward tasks
4. Add lifelong novelty module (RND-based) alongside episodic KNN

## Experiment Design & Fair Comparison (Phase 3.1)

### The Compute Parity Problem

When comparing EDER, ES+DQN, and DQN at the same `total_episodes` ceiling, the **environment-step budgets are wildly unequal**:

- **DQN**: One episode = one env step. 2000 episodes → ~2000 env steps.
- **ES+DQN / EDER**: One actor episode = one *generation* of 50 workers (default `es_n_workers=50`). Each worker runs one full env episode. So 2000 actor episodes → ~100,000 env steps.

**This means ES methods get ~50× more environment interaction than DQN at the same episode ceiling.** Plotting by episode *confounds* the comparison and appears to favor ES methods unfairly.

### Solution: Plot by Environment Steps

All experiment comparisons now default to `x_axis="env_steps"` (not `"episode"`). This ensures:
- **Honest comparison**: All methods measured against the same compute budget
- **Reproducibility**: Results in README always use fair axes
- **Clarity**: The plot's x-axis directly reflects the actual cost to run each algorithm

Example:
```bash
# Old behavior (biased): 2000 episodes
#  DQN: 2000 env steps
#  EDER: 100,000 env steps (50× more!)

# New behavior (fair): x_axis="env_steps"
#  DQN: reaches 248 reward at ~2000 steps
#  EDER: reaches 217 reward at ~2000 steps
#  Direct 1:1 cost comparison
```

### Implementation Details

- `Experiment.run()` and `Experiment.plot()` now default to `x_axis="env_steps"`
- `compare()` function defaults to `x_col="total_env_steps"`
- CLI `--x-axis` flag allows override if needed: `--x-axis episode`
- Metrics CSV always contains `total_env_steps` column (logged per rollout)
- Aggregation uses NaN-padding (not truncation) to preserve longer runs

### Backward Compatibility

To compare using episode count (not recommended):
```bash
python experiments/cartpole_efficiency.py --x-axis episode --show
```

## Key References
- Khadka & Tumer (2018) — ERL: Evolution-Guided Policy Gradient
- Salimans et al. (2017) — ES as scalable alternative to RL
- Badia et al. (2020) — Never Give Up (NGU)
- Mnih et al. (2015) — DQN
- Lillicrap et al. (2015) — DDPG

---

## Local RL infrastructure

This repository is self-contained. Keep its replay buffer, DQN network, flat
parameter helpers, seeding, experiment lifecycle, and logging under
`src/rl_evo_lab`.

These implementations are intentionally tailored to EDER:
- `ReplayBuffer` preserves integer actions and exposes `diversity_metric()`.
- `QNetwork` exposes flat parameters for the ES actor.
- `RunLogger` owns the experiment-specific CSV, progress, and status lifecycle.

Do not add a shared RL-library dependency unless another active project needs the
same stable abstraction.
