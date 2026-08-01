# rl-evo-lab Handoff

## Project Overview

**EDER** (Evolutionary Distributed Experience Replay): A hybrid RL system combining:
- **ES Actor:** Population-based parameter exploration (N workers generate noisy parameter vectors)
- **DQN Learner:** Trains purely from replay buffer (no env interaction during training)
- **Novelty Signal:** KNN-based intrinsic reward over learned state embeddings to drive exploration

**Origin:** Reproduction + extension of 2021 MSc thesis on ES + novelty-driven exploration.

**Current State:** 
- Clean codebase with actor/learner/buffer separation
- Multi-seed experiment runner with automated comparisons
- Config system supports environment-specific hyperparameters
- Tests exist in `tests/test_algorithms.py`

## What Needs Doing

### 1. Review the Models

- **`src/rl_evo_lab/actor/es_actor.py`** — Evolution Strategy implementation
  - Check: parameter perturbation logic, fitness ranking, weight updates
  
- **`src/rl_evo_lab/learner/dqn.py`** — DQN learner
  - Check: Double DQN structure (policy net selects, target net evaluates)
  - Check: target net update frequency, loss computation
  
- **`src/rl_evo_lab/learner/idn.py`** — Inverse Dynamics Network for novelty
  - Check: embedding training, KNN novelty computation
  - Check: episodic memory reset logic

Verify these implementations are sound. No specific fixes expected; this is just code review to understand what's there.

### 2. Run Initial Validation Experiments

Run three baseline experiments to establish where things stand:

**CartPole-v1 (discrete, dense reward, short episodes):**
```bash
poetry run python experiments/cartpole_normal.py
```
- Three conditions: EDER, ES+DQN (no novelty), DQN (pure epsilon-greedy)
- Three seeds: 7, 42, 123
- 500 episodes each
- Output: `runs/cartpole_normal/comparison.png`

**LunarLander-v3 (continuous, dense reward, medium episodes):**
```bash
poetry run python experiments/lunarlander_normal.py
```
- Same three conditions, three seeds
- 500 episodes each
- Output: `runs/lunarlander_normal/comparison.png`

**Acrobot-v1 (discrete, sparse reward, medium-long episodes):**
```bash
poetry run python experiments/acrobot_exploration.py
```
- Same three conditions, three seeds
- 500 episodes each
- Output: `runs/acrobot_exploration/comparison.png`

### 3. Document Findings

After experiments complete:
1. Collect final learner_eval_reward means for each condition across seeds
2. Note which method wins on each environment
3. Record any anomalies (e.g., crashes, divergence, unexpected behavior)
4. Update `EXPERIMENT_LOG.md` with raw numbers and observations

## Key Files

| File | Purpose |
|------|---------|
| `src/rl_evo_lab/experiment.py` | Experiment harness + seed management |
| `src/rl_evo_lab/train.py` | Main training loop |
| `src/rl_evo_lab/utils/config.py` | Config + environment presets |
| `experiments/*.py` | Experiment definitions (CartPole, LunarLander, etc.) |
| `tests/test_algorithms.py` | Algorithm correctness tests |

## Environment Info

- Python ≥3.12
- Dependencies: torch, gymnasium, numpy, matplotlib, pandas
- Install: `poetry install`
- Run: `poetry run python <script>`

## Notes

- Experiments run multi-threaded; use `--workers N` to limit parallelism
- Results are cached: re-run with `--force` to override
- Config system supports environment categories (discrete_dense_short, continuous_dense_medium, etc.)
- Early stopping enabled to avoid wasted compute after convergence

---

**Next Step:** Code review, then run the three experiments, then document findings.
