# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reproduction and extension of MSc thesis: *"Improving Exploration in Evolutionary Reinforcement Learning through Novelty Search"* (NUI Galway, 2021).

The core algorithm is **EDER** (Evolutionary Distributed Experience Replay): a gradient-based DRL learner (DQN) combined with an Evolution Strategy (ES) actor population. The ES fills the replay buffer with diverse experiences; the learner trains purely from that buffer — no epsilon-greedy exploration.

- Python `>=3.12`, managed with [Poetry](https://python-poetry.org/)

## Common Commands

```bash
poetry install          # Install dependencies
poetry add <package>    # Add a dependency
poetry run pytest       # Run tests
poetry run pytest tests/test_foo.py::test_bar  # Run single test
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
| σ | 0.06 | ES noise std dev — best from original experiments |
| β | 0.02 | Intrinsic reward weight |
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

## Key References
- Khadka & Tumer (2018) — ERL: Evolution-Guided Policy Gradient
- Salimans et al. (2017) — ES as scalable alternative to RL
- Badia et al. (2020) — Never Give Up (NGU)
- Mnih et al. (2015) — DQN
- Lillicrap et al. (2015) — DDPG
