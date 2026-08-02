# What We Know (Validated Facts Only)

## Code Review & Bug Fixes

✅ **5 bugs identified and fixed in the code:**

1. **B1** — Truncation/Termination: Changed `done = terminated | truncated` to `done = terminated` for buffer storage
2. **B2** — Rank Normalization: Replaced ordinal ranking with dense ranking via `np.unique`
3. **B3** — Seed Collision: Extracted worker job builder using constant stride instead of decaying multiplier
4. **B4** — Negative Reward Threshold: Added sign-aware sync threshold formula
5. **B5** — IDN Baseline Capture: Changed `==` to `>=` for robustness

✅ **11 unit tests added and passing** — these test the fixes work as code, not that they improve learning

✅ **No regressions** — 30 existing tests still pass

## What We Do NOT Know

❌ **Whether the bug fixes improve learning** — We ran EDER only, no baseline comparisons (ES+DQN, DQN)

❌ **Whether EDER learns at all** — Some seeds converged, others didn't. No pattern established.

❌ **Whether the HPs are tuned** — Config contains old claims about tuning that we removed. Reality unknown.

❌ **Whether the algorithm is correct** — Fixes address code bugs, but whether the algorithm design is sound is untested.

❌ **Any performance numbers** — We have no validated results to report.

---

## What We Need to Do

To actually validate the work:

1. **Run proper experiments** with all 3 conditions (EDER, ES+DQN, DQN)
2. **Get consistent results** across seeds (1/3 solving is not a pattern)
3. **Compare against baselines** to isolate what each component contributes
4. **Investigate HP tuning** — either tune them or confirm defaults are adequate
5. **Understand the variance** — why do only 1/3 seeds converge?

Only then can we make claims about what works and what doesn't.

---

## Current State

- **Code**: Fixed, tested at unit level
- **Learning**: Partially working (1/3 seeds per env converge)
- **Validation**: Not done
- **Claims**: None (all removed)

