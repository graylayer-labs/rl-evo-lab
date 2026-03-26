# rl-evo-lab

Reproduction and extension of **EDER** (Evolutionary Distributed Experience Replay), originally developed as part of an MSc thesis: *"Improving Exploration in Evolutionary Reinforcement Learning through Novelty Search"* (NUI Galway, 2021).

## What is EDER?

EDER combines a gradient-based deep RL learner (DQN) with an Evolution Strategy (ES) actor population. The ES generates diverse experiences that fill the learner's replay buffer, replacing epsilon-greedy exploration entirely. Augmented reward `rₐ = rₑ + β·rᵢ` drives the ES toward novel states via a KNN-based intrinsic reward over episodic memory.

## Setup

```bash
poetry install                        # core deps (torch, gymnasium, numpy)
poetry install --extras mujoco        # + MuJoCo environments
```

## Usage

```bash
poetry run pytest                     # run tests
poetry run ruff check src/            # lint
poetry run ruff format src/           # format
```

## Package Structure

```
src/rl_evo_lab/
  actor/      # Evolution Strategy population
  learner/    # DQN → SAC/DDPG
  buffer/     # Shared replay buffer (actor/learner interface)
  intrinsic/  # Episodic KNN + lifelong RND novelty modules
  utils/      # Seeding, logging, config
```

## Roadmap

- [x] Repo setup
- [ ] Replay buffer
- [ ] ES actor (Salimans et al. 2017)
- [ ] DQN learner
- [ ] Episodic KNN intrinsic reward
- [ ] EDER integration + CartPole reproduction
- [ ] SAC/DDPG learner swap
- [ ] MuJoCo experiments
- [ ] Lifelong RND module
