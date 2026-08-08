# Environment Selection & Justification

This document explains why we chose these 5 environments for the baseline validation study, what each tests, and what we expect to learn from each.

## The Problem

Phase 1 tested EDER (Evolutionary + DQN + Novelty) against simpler baselines on CartPole, LunarLander, and Acrobot. Result: EDER lost across the board, even on Acrobot (which should be "hard exploration"). This raises a critical question: **Is novelty-driven exploration actually useful, or is it overhead on the kinds of tasks this codebase targets?**

We can't answer this without testing EDER on tasks where exploration is *actually* the bottleneck. Phase 1 tested only dense-reward or trivially-sparse tasks. This phase tests a spectrum.

## Environment Spectrum

### Tier A: Dense-Reward Baseline (Proof of Concept)

These are the "should-work-trivially" tasks. DQN should solve them easily. EDER should also solve them but use more compute. If EDER can't solve these, something is fundamentally broken.

#### CartPole-v1
- **Reward**: +1 per step, up to 500 steps (episode length limit)
- **Solved**: ≥475 reward in eval (20-episode window)
- **Why**: Industry standard, trivial exploration problem
- **What we learn**: Baseline cost of EDER's complexity; proof that DQN works
- **Expected DQN outcome**: Solves in ~100 episodes, 3/3 seeds
- **Expected EDER outcome**: Also solves, but slower (population overhead)

#### LunarLander-v3
- **Reward**: Dense (fuel penalty, landing bonus, crash penalty)
- **Solved**: ≥200 reward in eval
- **Why**: Continuous control, precision-critical (novelty should hurt here)
- **What we learn**: Whether novelty adds overhead on tasks requiring precision
- **Expected DQN outcome**: Solves in ~300 episodes, 3/3 seeds
- **Expected EDER outcome**: Likely fails or underperforms ES+DQN (novelty pushes toward crashes instead of landings)

---

### Tier B: Sparse-Reward Discovery (Exploration Matters)

These introduce sparsity — reward signal becomes rare. Gradient-based methods struggle; exploration becomes critical. This is where novelty *should* help if it helps anywhere.

#### CartPole-v1 (Sparse Variant)
- **Reward**: 0 per step, +500 only if episode reaches step limit without terminating early
- **Solved**: ≥475 reward
- **Why**: Minimal sparse-reward problem; same dynamics as CartPole-v1 but no per-step signal
- **What we learn**: Does ES provide enough exploration without per-step reward? Does novelty help?
- **Expected DQN outcome**: Struggles (maybe 30-50% solve rate); no gradient signal until terminal
- **Expected ES+DQN outcome**: Should help via population diversity; better than DQN
- **Expected EDER outcome**: If novelty helps, should beat ES+DQN here

#### Acrobot-v1
- **Reward**: -1 per step (goal: reach -100 or better), max 500 steps
- **Solved**: ≤-100 reward
- **Why**: Sparse discrete control, discovery of swing-up strategy; published benchmark
- **What we learn**: Can ES + novelty discover hard control sequences in sparse settings?
- **Expected DQN outcome**: Varies (50-70% solve rate via brute-force luck)
- **Expected ES+DQN outcome**: Should improve via population; consistent exploration
- **Expected EDER outcome**: Critical test — novelty should help find swing-up, or it doesn't work

---

### Tier C: Hard Exploration (Falsification Test)

This is where EDER's thesis either holds or breaks. Montezuma's Revenge is the canonical sparse-reward benchmark in the novelty-search literature (RND, NGU). If EDER can't find *any* signal here, novelty isn't solving the right problem.

#### Montezuma's Revenge (RAM Observation Variant)
- **Variant**: `ALE/MontezumaRevenge-ram-v5` (128-byte flat RAM obs, not images)
- **Reward**: Sparse (score only on item collection / room discovery)
- **Solved**: No fixed threshold; we're documenting failure/success modes
- **Why**: Proven sparse-reward benchmark where simple RL fails completely
- **What we learn**: Does EDER + novelty find *any* meaningful exploration strategy?
- **Expected DQN outcome**: Fails (score ≈ 0; no reward signal to guide learning)
- **Expected ES+DQN outcome**: May find 1-2 items via random exploration; unlikely to progress
- **Expected EDER outcome**: Novelty should guide ES population toward high-novelty states, eventually finding items; if it doesn't, novelty mechanism is broken

---

## Hypothesis Summary

| Tier | Env | DQN | ES+DQN | EDER | Insight if true |
|---|---|---|---|---|---|
| A | CartPole-v1 | ✅ solves | ✅ solves | ✅ solves | Baseline working; EDER adds cost |
| A | LunarLander-v3 | ✅ solves | ✅ solves | ❌ fails | Novelty hurts precision tasks |
| B | CartPole-sparse | ❌ struggles | ✅ OK | ✅+ better | ES helps sparse, novelty refines |
| B | Acrobot-v1 | ~50% | ✅ better | ✅✅ best | Novelty helps on discovery tasks |
| C | Montezuma's | ❌ fails | ❌ fails | ✅ finds items | EDER solves hard exploration |

**If this pattern holds:** EDER = overhead on dense-reward, helpful on sparse/discovery. Thesis: "Novelty is task-dependent; exploration-hard tasks reward the cost."

**If this pattern breaks (e.g., EDER fails on Montezuma's):** Novelty mechanism (KNN state diversity) is misaligned with actual RL exploration needs. Thesis: "Intrinsic motivation as currently implemented doesn't solve sparse-reward RL."

---

## Compute Budget

| Env | Seeds | Eps/Seed | Time/Algo | Notes |
|---|---|---|---|---|
| CartPole-v1 | 3 | 2000 | ~5 min | Fast; parallelizable |
| LunarLander-v3 | 3 | 3000 | ~30 min | Moderate |
| CartPole-sparse | 3 | 2000 | ~5 min | Same length as CartPole-v1 |
| Acrobot-v1 | 3 | 1000 | ~1 hr | Longer convergence; watch early stopping |
| Montezuma's | 1 | 500 capped | ~2 hrs | 1 seed only (DQN expected to fail anyway); capped budget to save compute |

**Total**: ~5 hours per algorithm × 3 algos (DQN, ES+DQN, EDER) in parallel = ~5 hours wall-clock for full comparison.

DQN baseline phase: focus on CartPole, LunarLander, CartPole-sparse, Acrobot (4 envs, all 3 seeds). Montezuma's separate, 1 seed, after others confirm viability.

---

## What We're Not Testing

- **MuJoCo continuous control** (HalfCheetah, Walker2d): deferred to Phase 3 if baseline pattern holds
- **Image-based environments** (full Atari, not RAM): Montezuma's-ram-v5 is flat-obs workaround; image models = future work
- **Reward shaping variants**: fixed envs, no hand-tuned rewards
- **Hyperparameter sweeps**: baseline uses repo defaults; tuning comes after understanding

---

## References

- **Montezuma's Revenge in novelty literature**: 
  - Burda et al. (2019) — RND: "Exploration by Random Network Distillation" achieves score 13000+ on Montezuma's
  - Badia et al. (2020) — NGU: "Never Give Up" achieves score 20000+ on Montezuma's
- **Acrobot as exploration benchmark**:
  - Standard Gym env; sparse reward but solvable by well-tuned RL algorithms
- **Sparse-reward CartPole**:
  - Custom variant; validates sparse-reward principle on a simple env
