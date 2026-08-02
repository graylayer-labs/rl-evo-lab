# Current State of rl-evo-lab

## Code

The codebase contains:
- 5 bug fixes in core algorithm (dqn.py, es_actor.py, es_worker.py, train.py)
- 11 unit tests validating these fixes
- 30 existing tests still passing
- No performance claims anywhere

## Experiments

Three experiment scripts exist:
- `cartpole_normal.py` — CartPole-v1 with 3 algorithm conditions across 3 seeds
- `lunarlander_normal.py` — LunarLander-v3 with 3 algorithm conditions across 3 seeds  
- `acrobot_exploration.py` — Acrobot-v1 with 3 algorithm conditions across 3 seeds

Run data exists in `runs/` with metrics.csv files for EDER conditions only.

## Algorithm

EDER (Evolutionary Distributed Experience Replay) is implemented with:
- ES Actor: population-based parameter exploration
- DQN Learner: Q-network trained from shared replay buffer
- IDN Novelty: KNN-based intrinsic reward signal

Config system supports environment-specific hyperparameters via presets.

## Known Status

Learning curves show mixed results: some seeds converge (eval reward ≥ threshold), others plateau. No pattern across seeds or environments yet established.

