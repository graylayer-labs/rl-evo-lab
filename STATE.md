# Current State

## Code

- 5 bugs fixed in: dqn.py, es_actor.py, es_worker.py, train.py
- 11 unit tests validating the fixes
- 30 existing tests passing
- No performance claims in codebase

## Experiments

Three experiments defined:
- cartpole_normal.py
- lunarlander_normal.py  
- acrobot_exploration.py

Each experiment runs 3 algorithm conditions × 3 seeds:
- EDER (ES + DQN + Novelty)
- ES+DQN (ES + DQN, no novelty)
- DQN (epsilon-greedy baseline)

Run metrics exist in `runs/` for EDER condition only.

## Algorithm

EDER implementation contains:
- ES Actor with antithetic sampling and parameter perturbation
- DQN Learner with double Q-network and target net
- IDN Novelty signal with episodic KNN memory

Environment-specific hyperparameters defined via config presets.

## Results to Date

EDER runs show convergence on some seeds per environment. No complete experimental comparison across all conditions. No validated performance claims.

