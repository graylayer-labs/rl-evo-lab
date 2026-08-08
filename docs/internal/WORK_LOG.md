# Work Log — Internal Session Notes

**Not intended for readers.** This file tracks decisions and progress through research phases.

---

## Phase 2: DQN Baseline (COMPLETE ✓)

**Completed 2026-08-08**

### Goals
- Establish ground truth: where does RL get stuck?
- Create clean, justified environment spectrum
- Remove all stale Phase 1 results

### What Was Done
1. ✅ ENVIRONMENTS.md: 5-env spectrum with hypotheses and rationale
2. ✅ README.md: Reframed as "Novelty-Guided Evolutionary RL" thesis project
3. ✅ WORK_LOG.md: This file (process tracking, not for readers)
4. ✅ Infrastructure:
   - CartPoleSparseWrapper in envs.py (sparse reward signal)
   - cartpole_sparse + montezuma presets in config.py
   - baseline_dqn.py experiment runner
5. ✅ DQN Baselines (3 seeds each):
   - CartPole-v1: 151.7 mean (fails to reach 475 threshold)
   - LunarLander-v3: 215.5 mean (solves at 200 threshold) ✅
   - CartPole-sparse: 0.0 mean (fails completely)
   - Acrobot-v1: -500 mean (fails completely)
6. ✅ Cleanup: Deleted all Phase 1 results and scratch files

### Key Insight from Baseline
Dense-reward tasks are still hard for ε-greedy (CartPole: 32% of solved).
Sparse-reward tasks are impossible (CartPole-sparse, Acrobot: 0% progress).
This shows exactly where ES and/or novelty would need to prove value.

---

## Phase 3: Evolutionary RL vs. Novelty-Guided RL (NEXT)

### Goals
- Run ES+DQN on same 4 environments (shows ES impact in isolation)
- Run Novelty-Guided RL on same 4 environments (shows novelty impact)
- Compare all three fairly using environment steps

### Plan
1. Create experiments/evolutionary_rl.py (ES+DQN on all 4 envs)
2. Create experiments/novelty_guided_rl.py (Novelty-Guided on all 4 envs)
3. Run both in parallel with DQN baselines available for comparison
4. Fill README findings table with all three approaches
5. Interpret: which technique(s) actually solve the exploration problem?

### Success Criteria
- All three methods compared on same environments
- Comparisons use env_steps (fair compute budget)
- Clear narrative: "Here's where each technique helps or hurts"
- Montezuma's deferred until Phase 4 (optional hard test)
