# Work Log — Internal Session Notes

**Not intended for readers.** This file tracks decisions, experiments, and debugging notes during active development.

---

## Session: 2026-08-08 — Thesis Restructure + Phase 2 Setup

### Context
Phase 1 tested EDER vs ES+DQN vs DQN on 3 envs (CartPole, LunarLander, Acrobot). Result: EDER lost across the board, even on Acrobot (supposedly "hard exploration"). Raises critical question: is novelty actually useful, or is it overhead?

### Decisions Made
1. **Environment spectrum locked**: 5 envs from trivial → dense → sparse → hard
   - CartPole-v1 (dense baseline)
   - LunarLander-v3 (dense precision test)
   - CartPole-sparse (sparse variant, new wrapper)
   - Acrobot-v1 (sparse discovery)
   - Montezuma's-ram-v5 (hard exploration, 1 seed capped budget)

2. **Documentation restructure**:
   - README: clean thesis statement + findings table (DQN baseline only, others "pending")
   - ENVIRONMENTS.md: 5 envs with explicit hypotheses for each
   - WORK_LOG.md: this file
   - Root cleanup: move scratch scripts to scripts/ or delete

3. **DQN baseline first**: before touching ES+DQN or EDER again, establish ground truth for "does exploration matter here"

### Implementation Plan
1. ✅ ENVIRONMENTS.md written
2. ✅ README restructured (clean thesis, findings table skeleton)
3. ✅ WORK_LOG.md created
4. ⏳ CartPoleSparseWrapper implementation
5. ⏳ Atari setup (ale-py, autorom, montezuma preset)
6. ⏳ baseline_dqn.py experiment runner
7. ⏳ Run DQN baseline across all 5 envs
8. ⏳ Fill findings table with results

### Next Steps
- Implement CartPoleSparseWrapper in envs.py
- Add cartpole_sparse preset to config.py
- Test sparse wrapper produces 0-reward-until-terminal trajectories
