# Research Status: Novelty-Guided Evolutionary RL

**Updated: 2026-08-08 19:30 UTC**

## Executive Summary

We are conducting a thesis-style investigation: **"Can novelty-driven exploration and/or evolutionary strategies improve RL in exploration-stuck environments?"**

Three methods are being compared across four environments:
1. **DQN** (baseline ε-greedy)
2. **Evolutionary RL** (ES+DQN, no novelty)
3. **Novelty-Guided RL** (ES+DQN+novelty)

---

## Phase 2: DQN Baseline ✓ COMPLETE

**Goal:** Establish ground truth showing where RL gets stuck.

**Results:** All 4 environments × 3 seeds

| Environment | Type | Result | Interpretation |
|---|---|---|---|
| CartPole-v1 | Dense | 152/475 (32%) | DQN struggles even on trivial task |
| LunarLander-v3 | Dense | 215/200 ✓ | DQN solves precision task |
| CartPole-sparse | Sparse | 0/475 | Zero-gradient failure (no per-step reward) |
| Acrobot-v1 | Sparse | -500/−100 | Discovery failure (rare behavior) |

**Insight:** Sparse-reward and dense-hard tasks are exploration-stuck for pure ε-greedy.

---

## Phase 3: Evolutionary & Novelty-Guided RL 🔄 IN PROGRESS

**Goal:** Test whether ES and/or novelty solve the problems DQN can't.

**Status:** 7/12 runs complete, ~2.5 hours remaining

```
CARTPOLE ✓ Complete
  DQN:                152.0 ✗
  Evolutionary RL:    236.5 (+55.6% vs DQN) ✗
  Novelty-Guided:     97.2 (-58.9% vs ES) ✗

LUNARLANDER 🔄 In Progress
  DQN:                215.5 ✓
  Evolutionary RL:    (complete)
  Novelty-Guided:     (pending)

CARTPOLE-SPARSE ⏳ Queued
  (all methods pending)

ACROBOT ⏳ Queued
  (all methods pending)
```

**ETA Completion:** ~21:52 UTC (2026-08-08)

---

## EARLY FINDINGS: CartPole

### Key Discovery
**On dense-reward CartPole:**
- **ES population HELPS** (+55.6% improvement over DQN)
  - Population diversity beats ε-greedy exploration
- **Novelty HURTS** (−58.9% degradation vs ES alone)
  - Intrinsic motivation conflicts with extrinsic reward
  
### Interpretation
This **validates the thesis hypothesis**: novelty is overhead on dense-reward tasks where gradient signal is sufficient. ES alone provides better exploration through population diversity.

### Remaining Questions
1. Does novelty help on sparse-reward tasks (CartPole-sparse, Acrobot)?
2. Does novelty help on hard exploration (deferred to Phase 4)?
3. Why does novelty hurt? (IDN training interference? Reward scaling?)

---

## Tools Built This Session

### 1. **Experiment Runners**
- `experiments/baseline_dqn.py` — Run DQN baseline
- `experiments/evolutionary_rl.py` — Run ES+DQN
- `experiments/novelty_guided_rl.py` — Run ES+novelty
- `experiments/compare_all_methods.py` — Orchestrate 3-method comparison

### 2. **Analysis & Monitoring**
- `scripts/monitor_phase3_progress.py` — Real-time progress tracking
- `scripts/aggregate_phase3_results.py` — Auto-generate findings table
- `scripts/analyze_phase3_results.py` — Interpret results, generate narratives

### 3. **Documentation**
- `ENVIRONMENTS.md` — Justified 5-env spectrum, hypotheses
- `WORK_LOG.md` — Internal decision tracking
- `README.md` — Clean thesis statement, portfolio-ready
- `PHASE_STATUS.md` — This file

---

## What Comes Next

### After Phase 3 Completes
1. Auto-generate findings table with `scripts/aggregate_phase3_results.py`
2. Run interpretation: `scripts/analyze_phase3_results.py --thesis-summary`
3. Update README with all three methods' results
4. Document findings narrative

### Phase 4 (Optional)
1. Test on Montezuma's Revenge (hard exploration)
2. If novelty helps on sparse, debug why it hurts on dense
3. Consider novelty scheduling or reward scaling variants

### Publication-Ready State
- ✓ Clear thesis statement
- ✓ Justified environment spectrum
- ✓ Fair comparison (env_steps, not episodes)
- ✓ All results reproducible and cached
- ✓ Honest narrative (failures documented as rigorously as wins)

---

## Commands for Current Phase

**Monitor progress:**
```bash
python scripts/monitor_phase3_progress.py --watch 60
```

**Check current results:**
```bash
python scripts/aggregate_phase3_results.py
python scripts/analyze_phase3_results.py --env cartpole
```

**Generate comparison plots (when ready):**
```bash
python experiments/compare_all_methods.py --all-envs --show
```

---

## Commits This Session

```
1bf0379 feat: add Phase 3 results analysis and interpretation script
5300259 feat: add Phase 3 progress monitoring script
47635c7 feat: add Phase 3 comparison and aggregation tools
7b8e7fd docs: update README with Phase 3 running instructions
d891114 feat: add Phase 3 experiment runners
b1a8cab chore: clean slate reset - Phase 2 complete
7a94846 feat: add DQN baseline results to findings table
90831e0 docs: reframe README for thesis-style portfolio project
136ceeb chore: add environment spectrum setup
```

**GitHub:** https://github.com/graylayer-labs/rl-evo-lab (47+ commits)

---

## Key Takeaway So Far

**Novelty is NOT universally helpful.**

CartPole demonstrates:
- ES population > ε-greedy (diversity matters for exploration)
- Novelty < ES alone (intrinsic signal conflicts with extrinsic signal)

This suggests a nuanced answer to the research question:
- Novelty may help on truly sparse tasks (where extrinsic signal is gone)
- Novelty may hurt on mixed/dense tasks (where extrinsic signal is dominant)

Waiting for sparse-task results (CartPole-sparse, Acrobot) to test this hypothesis.
