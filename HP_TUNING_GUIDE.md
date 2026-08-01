# Hyperparameter Management Guide

## Principles

This project uses **static, category-based hyperparameters** rather than dynamic/adaptive tuning. This approach prioritizes:
- **Clarity**: You know exactly what HPs are running
- **Reproducibility**: Same env = same HPs across runs
- **Debuggability**: Failures are in algorithm, not adaptation
- **Scalability**: New environments use proven category defaults

## Environment Categories

Environments are grouped by properties:

| Category | Properties | Example | HPs |
|----------|-----------|---------|-----|
| `discrete_dense_short` | Discrete actions, dense reward, short episodes | CartPole | β=0.1, σ=0.1, ramp=200 |
| `continuous_dense_medium` | Continuous actions, dense reward, medium episodes | LunarLander | β=0.15, σ=0.12, ramp=250 |
| `discrete_sparse_long` | Discrete actions, sparse reward, long episodes | Acrobot | β=0.2, σ=0.15, ramp=300 |
| `continuous_sparse_long` | Continuous actions, sparse reward, long episodes | MountainCar | β=0.25, σ=0.18, ramp=350 |

## Workflow

### For established environments (CartPole, LunarLander, etc.)

```bash
# 1. Run experiment directly (uses preset HPs)
python experiments/cartpole_normal.py

# 2. Results are in runs/cartpole_normal/ with comparison plots
```

### For new environments

```bash
# 1. Create experiment file (examples: experiments/cartpole_normal.py)
# 2. Assign to a category in src/rl_evo_lab/utils/config.py
# 3. Run diagnostic to verify category defaults work
python scripts/run_hp_diagnostic.py --env your_env --episodes 100

# 4. Review diagnostic results:
#    - If learner_reward >= baseline category avg: use category defaults
#    - If learner_reward < baseline: run full diagnostic sweep
#    - If IDN loss high: increase idn_updates_per_episode or idn_lr

# 5. If diagnostic looks good, run full experiment
python experiments/your_env.py
```

## Tuning a New Environment (if diagnostic suggests it)

When a new environment doesn't match its category's typical performance:

```bash
# 1. Run diagnostic sweeps (like we did for CartPole)
# Sigma sweep:
python scripts/run_hp_diagnostic.py --env your_env --episodes 50  # baseline
# ... manually test σ in {0.05, 0.1, 0.15, 0.2}

# Beta sweep:
# ... manually test β in {0.05, 0.1, 0.15, 0.2}

# Novelty ramp sweep:
# ... manually test novelty_ramp in {0, 100, 200, 300}

# 2. Update presets in src/rl_evo_lab/utils/config.py with findings
# 3. Update category defaults if discovering a pattern
# 4. Document reasoning in config comments
```

## Key HPs and Their Effects

### β (novelty weight)
- **What it does**: Controls how much intrinsic (novelty) reward influences exploration
- **Too low** (0.02): Novelty signal is noise, doesn't guide exploration effectively
- **Too high** (0.5+): Novelty dominates extrinsic reward, can override task objective
- **Sweet spot**: 0.1-0.2 depending on reward density and horizon
- **How to detect issues**:
  - If EDER < ES+DQN: β probably too low
  - If learner_eval crashes mid-training: β might be too high or ramp too aggressive

### σ (ES mutation noise)
- **What it does**: Controls how much we perturb parameters in ES actor population
- **Too low** (0.01): Population converges quickly but to local optima
- **Too high** (0.3+): Population is too noisy, can't converge
- **Sweet spot**: 0.1-0.15 for most tasks
- **How to detect issues**:
  - If all workers get similar low rewards: σ might be too small relative to problem scale
  - If worker diversity metric is high but rewards are bad: σ might be too high

### novelty_ramp_episodes
- **What it does**: How many episodes before novelty weight reaches full strength
- **Too low** (50): Sharp reward landscape shift destabilizes learner
- **Too high** (400+): Novelty takes forever to engage, ES won't explore
- **Sweet spot**: 200-300 for most tasks
- **How to detect issues**:
  - If learner_eval crashes at ep 100-200: ramp too aggressive, increase it
  - If EDER never beats ES+DQN: ramp might be too slow, or β is actually the issue

## Validation Checklist

Before considering a new environment "tuned":

- [ ] Diagnostic run (50-100 eps) completes without errors
- [ ] IDN loss decreases (embeddings are learning)
- [ ] Learner eval increases monotonically (no mid-training crashes)
- [ ] EDER outperforms ES+DQN (novelty is helping exploration)
- [ ] Full 500+ episode run matches diagnostic trajectory
- [ ] 3-seed run shows reasonable variance (not wild swings)

## Decision Log

Record HP tuning decisions here:

```
2026-07-31 · CartPole · β=0.02→0.1, σ=0.06→0.1, ramp=100→200
  Why: Diagnostic sweeps showed weak novelty signal, aggressive ramp destabilizing
  Result: EDER 365.4 mean (was ~190), beats ES+DQN by 87%

[Record future tuning decisions below]
```

## Future Work

- [ ] Validate lunarlander category defaults (needs similar sweep)
- [ ] Validate acrobot category defaults
- [ ] Validate mountaincar category defaults
- [ ] Consider meta-learning approach if 10+ environments need tuning
- [ ] Build automated sweep pipeline to reduce manual diagnostic work
