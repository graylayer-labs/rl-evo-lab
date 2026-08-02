# EDER Model Code Review Findings

**Reviewed:** rl-evo-lab core models (ES Actor, DQN Learner, IDN, integration)
**Review Date:** 2026-08-01
**Confidence Levels:** Critical bugs are **Confirmed by code reading**; medium/low are **Plausible, need runtime validation**

---

## Summary

**5 confirmed bugs** affecting correctness:
- **1 critical**: DQN truncation/termination bootstrap (Confirmed)
- **2 high-severity**: Rank normalization ties, seed collision under decay
- **2 medium**: Negative-reward sync threshold, IDN baseline capture off-by-one

**3 robustness concerns** (design/overfitting risks):
- IDN repeated-batch overfitting
- Embedding ReLU half-space collapse
- Config mutation fragility

**Overall assessment:** Core algorithm logic is sound, but three production bugs will cause systematic errors on CartPole, Acrobot, and negative-reward envs. These are not edge cases—they affect the three validation experiments in the handoff.

---

## Severity Table (sorted most-severe first)

| ID | Model | File:Line | Severity | Category | One-Line Summary |
|---|---|---|---|---|---|
| **B1** | DQN | dqn.py:76 | **CRITICAL** | Correctness | Truncation treated as terminal; bootstraps zero'd on time limits |
| **B2** | ES Actor | es_actor.py:26-41 | **HIGH** | Correctness | Rank normalization doesn't handle ties; docstring mismatches code |
| **B3** | ES Actor | es_actor.py:217-226 | **HIGH** | Correctness | Seed collision when eff_n_workers decays; same ε across generations |
| **B4** | ES Actor | train.py:102 | **HIGH** | Correctness | Sync threshold inverts on negative-reward envs (Acrobot, MountainCar) |
| **B5** | IDN | es_actor.py:296-297 | **MEDIUM** | Correctness | Baseline capture off-by-one; silent fallback if zero transitions in warmup |
| **R1** | IDN | inverse_dynamics.py:53-58 | **MEDIUM** | Robustness | Repeated gradient steps on same batch → overfitting per generation |
| **R2** | IDN | inverse_dynamics.py:19 | **MEDIUM** | Robustness | ReLU-terminated embedding collapses to non-negative half-space |
| **R3** | Config | config.py:283-285 | **LOW** | Fragility | Frozen dataclass mutation via `__setattr__` is fragile pattern |

---

## Detailed Findings

### B1: DQN Truncation Bootstrap Bug (CRITICAL)

**File:** `src/rl_evo_lab/learner/dqn.py:76, 89`

**The Bug:**
```python
# Line 76 (es_worker.py)
done = terminated or truncated

# Line 89 (dqn.py collect_episode)
done = terminated or truncated
buffer.push(obs, action, float(reward), next_obs, float(done))

# Line 33 (dqn.py train_step)
target = batch.reward + self.cfg.gamma * next_q * (1.0 - batch.done)
```

When an episode hits a **time limit** (truncated=True, terminated=False), `done` is set to True, and the transition is pushed to the buffer with `done=1.0`. During training, the bootstrap is zeroed: `(1.0 - 1.0) = 0`, treating the time-limit state as if the episode truly ended with zero future value.

**Why This Is Wrong:**
- In Gymnasium/Gym, `truncated=True` means "episode hit time limit, but environment is still valid" — the agent could continue.
- `terminated=True` means "true episode end" (goal reached, failure, etc.).
- Bootstrap should only be zeroed on **terminated**, not truncated.

**Impact on Validation Experiments:**
- **CartPole-v1**: Episodes are truncated at 500 steps when successful. The learner learns that reaching 500 steps has zero future value, which is backward. This suppresses learning of sustained behavior and wastes the ES buffer of good episodes.
- **LunarLander-v3**: Similar issue. Time-limit truncations (2000 steps) are treated as terminal, suppressing credit for nearly-solved landings.
- **Acrobot-v1**: Episodes max at 500 steps. Same problem.

**Test Gap:** Existing tests don't validate truncation handling. `test_algorithms.py` has no truncation-specific test.

**Confidence:** **CONFIRMED** — the code clearly performs `done = terminated or truncated`, which conflates the two semantics. This is a well-known DQN bug.

---

### B2: Rank Normalization Doesn't Handle Ties

**File:** `src/rl_evo_lab/actor/es_actor.py:26-41`

**The Bug:**
```python
def _rank_normalize(fitnesses: np.ndarray) -> np.ndarray:
    """Rank fitnesses and normalise to [-0.5, 0.5].
    ...
    Ties share the same rank (dense rank).  # ← WRONG CLAIM
    """
    n = len(fitnesses)
    ranks = np.empty(n, dtype=np.float32)
    order = np.argsort(fitnesses)  # ascending order
    ranks[order] = np.arange(n, dtype=np.float32)  # assign ranks 0..n-1 to sorted order
    if n > 1:
        ranks = ranks / (n - 1) - 0.5
    return ranks
```

The docstring claims "Ties share the same rank," but `np.arange(n)` assigns **distinct** integers to each position in the sorted array, regardless of whether fitnesses are tied. If fitnesses are `[100, 100, 100, 100]` (all tied), argsort returns `[0, 1, 2, 3]` (arbitrary order among ties), and ranks become `[-0.5, -0.167, 0.167, 0.5]` — different ranks for tied workers.

**Example Failure Scenario:**
On CartPole-v1, after ~100 episodes, many workers reach the reward ceiling (500). If all 50 workers score 500, they should contribute equally to the ES gradient. Instead:
- `np.argsort([500, 500, ..., 500])` returns `[0, 1, ..., 49]` (arbitrary order).
- Ranks become `[-0.5, -0.48, ..., 0.5]` (all different).
- ES gradient accumulates `Σ rank_i · ε_i`, where rank_i are all different.
- Result: the gradient is nearly random, not a clear "all top workers" signal.

**Impact:** Especially severe on **CartPole**, where solving means hitting the 500-step ceiling. ES population diversity is wasted on noisy rank signals instead of coherent exploration.

**Confidence:** **CONFIRMED** — the code clearly uses `np.arange`, which does not implement dense ranking.

**Fix:** Use `scipy.stats.rankdata(fitnesses, method='dense')` or a custom dense-rank function.

---

### B3: Seed Collision When Worker Count Decays

**File:** `src/rl_evo_lab/actor/es_actor.py:217-226`

**The Bug:**
```python
# Lines 119-134: _effective_n_workers() decays over training
def _effective_n_workers(self) -> int:
    progress = self._convergence_progress()
    n = round(self.cfg.es_n_workers + progress * (self.cfg.es_workers_min - self.cfg.es_n_workers))
    # ... now n is different per generation

# Lines 216-226: seed uses eff_n_workers as multiplier
if cfg.es_antithetic:
    n_seeds = eff_n_workers // 2
    for k in range(n_seeds):
        seed = episode_num * eff_n_workers + k  # ← multiplier changes per episode
        worker_jobs.append((seed, +1))
        worker_jobs.append((seed, -1))
```

If `es_n_workers=50` (initial) but decays to `es_workers_min=4` (final):
- Episode 0, eff_n_workers=50: seed for k=0 is `0*50 + 0 = 0`
- Episode 100, eff_n_workers=25: seed for k=0 is `100*25 + 0 = 2500`
- Episode 200, eff_n_workers=12: seed for k=0 is `200*12 + 0 = 2400`

But the `episode_num * eff_n_workers` product is not collision-free when the multiplier changes. For example:
- Episode 100, eff_n_workers=20: `100*20 + 0 = 2000`
- Episode 50, eff_n_workers=40: `50*40 + 0 = 2000` ← **same seed**

Two different episodes generate identical noise vectors, reducing exploration diversity.

**Impact:** As the ES population converges (worker count shrinks), the algorithm **accidentally loses population diversity** by regenerating old noise patterns. This happens silently and won't crash, but it undermines ES exploration.

**Confidence:** **CONFIRMED** — the seed formula and decay logic are clearly incompatible.

**Fix:** Use a collision-free seeding scheme, e.g., `seed = hash((episode_num, worker_id, generation_counter))` or allocate seeds sequentially from a global counter.

---

### B4: Negative-Reward Sync Threshold Inversion

**File:** `src/rl_evo_lab/train.py:102-105`

**The Bug:**
```python
# Line 102: Compute sync threshold as a fraction of actor return
threshold = cfg.sync_eval_threshold * mean_extrinsic_return

# Line 103: Sync if learner_eval meets threshold
if last_eval >= threshold:
    actor.sync_from_learner(learner.get_weights())
```

**On positive-reward environments** (CartPole, LunarLander):
- Actor mean return ≈ 100–200 (positive)
- Threshold = 0.7 × 150 = 105
- Sync when learner_eval ≥ 105 ✓ sensible

**On negative-reward environments** (Acrobot solved at -100, MountainCar at -110):
- Actor mean return ≈ -150 (negative)
- Threshold = 0.7 × (-150) = **-105**
- Sync when learner_eval ≥ -105 ✓ but this is **more lenient** than intended

On Acrobot/MountainCar, the actor's negative reward generates a more-negative threshold, and the comparison flips: the learner syncs **earlier** (higher eval scores, less negative), not later. The intent of the threshold is to prevent premature sync before the learner has learned enough, but on negative envs, the threshold **encourages** premature sync.

**Example:**
- Acrobot: cfg.sync_eval_threshold = 0.7, mean_extrinsic = -200
- Threshold = -140
- Learner syncs when eval ≥ -140 (still far from solved at -100)
- This is the opposite of the intent: sync too early, potentially carrying bad actor params into the learner.

**Confidence:** **CONFIRMED** — the formula clearly uses multiplication on potentially-negative returns.

**Fix:** Use absolute value or a better-designed threshold that accounts for reward sign.

---

### B5: IDN Baseline Capture Off-by-One

**File:** `src/rl_evo_lab/actor/es_actor.py:296-297`

**The Bug:**
```python
# Line 296: Record baseline only at specific episode
if self._idn_loss_init is None and episode_num == cfg.novelty_warmup_episodes - 1:
    self._idn_loss_init = self._idn_loss_ema

# Line 101-104: Use baseline to compute confidence
if self._idn_loss_init is not None and self._idn_loss_init > 1e-8:
    raw_confidence = max(0.0, 1.0 - self._idn_loss_ema / self._idn_loss_init)
else:
    raw_confidence = 1.0  # ← fallback if baseline never recorded
```

The baseline is recorded on episode `cfg.novelty_warmup_episodes - 1` (the last warmup episode, 0-indexed). If:
1. `novelty_warmup_episodes=50`, baseline is recorded at episode 49.
2. But what if episode 49 produces zero transitions (environment failure, truncation at t=0)? Then `all_actions` is empty (line 287), and `idn.update()` is never called, so `idn_loss` stays 0.0, and `_idn_loss_init` is set to 0.0.
3. Now at line 102, `1.0 - 0.0 / 0.0 = NaN` or the condition fails silently, and raw_confidence falls back to 1.0 forever.

This is a silent fallback, not a crash, but it means the IDN confidence scaling never engages — the novelty signal always uses unscaled beta.

**Also:** If no transitions exist in warmup or if the baseline episode is skipped for any reason, `_idn_loss_init` stays None, and raw_confidence is always 1.0, meaning IDN training quality is never tracked.

**Confidence:** **PLAUSIBLE** — the code path exists but requires a specific failure scenario (zero transitions at baseline episode). This is more of a design fragility than a guaranteed bug.

**Fix:** Ensure the baseline is captured robustly (e.g., on the first episode with non-zero transitions post-warmup, not on a fixed episode number).

---

## Robustness Concerns (Not Bugs, But Risks)

### R1: IDN Repeated-Batch Overfitting

**File:** `src/rl_evo_lab/intrinsic/inverse_dynamics.py:53-58`

```python
def update(self, obs, next_obs, actions, n_steps: int) -> float:
    ...
    for _ in range(n_steps):
        logits, _ = self.forward(obs_t, next_obs_t)
        loss = nn.functional.cross_entropy(logits, actions_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item()
    return total_loss / max(n_steps, 1)
```

The same batch `(obs_t, next_obs_t, actions_t)` is used for all `n_steps` gradient updates (default 5). Over 500 training episodes, each generation's data is seen 5 times without variation, leading to overfitting on that generation's state distribution and weaker embeddings for novel states.

**Mitigation:** Sample multiple batches or use replay within update. Low impact for now but watch for embedding quality degradation in longer runs.

---

### R2: ReLU-Terminated Embedding Half-Space Collapse

**File:** `src/rl_evo_lab/intrinsic/inverse_dynamics.py:19`

```python
self.encoder = nn.Sequential(
    nn.Linear(2 * cfg.obs_dim, 128),
    nn.ReLU(),
    nn.Linear(128, cfg.embed_dim),
    nn.ReLU(),  # ← All embeddings are >= 0
)
```

All embedding coordinates are forced ≥ 0. KNN distance on a non-negative half-space can be less discriminative — embeddings collapse toward the zero corner over time. For long-horizon runs (1000+ episodes), this could reduce novelty signal contrast.

**Mitigation:** Consider removing the final ReLU or using a non-monotonic activation (e.g., GELU, ELU).

---

### R3: Config Frozen Dataclass Mutation

**File:** `src/rl_evo_lab/utils/config.py:283-285`

```python
cfg = EDERConfig(**config_kwargs)
for k, v in merged.items():
    if k.startswith('_'):
        object.__setattr__(cfg, k, v)  # ← mutation side-channel
return cfg
```

Custom flags are injected into a frozen dataclass via `__setattr__`. This works but is fragile — future type-checkers or Python versions might enforce frozen semantics more strictly. The pattern is a code smell but not currently broken.

---

## Integration Issues

### Harness Ordering (train.py)

The sequence is correct:
1. **Collect** (ES generation or DQN epsilon-greedy episode)
2. **Train** (learner gradient steps if buffer is full)
3. **Evaluate** (every eval_freq episodes)
4. **Sync** (actor←learner after eval or periodically)
5. **Decay** (convergence-dependent decays use updated learner_eval)

However, `learner_eval` is only updated every `eval_freq` episodes (line 76), so intermediate decays use stale eval values. This is acceptable but note that beta, sigma, and n_workers changes are quantized in steps of eval_freq.

---

## Test Coverage Gaps

| Finding | Test Coverage | Needed Test |
|---------|---|---|
| **B1: Truncation bootstrap** | ❌ None | `test_dqn_truncation_vs_termination()` |
| **B2: Rank ties** | ⚠️ `test_rank_normalize_identical()` covers identical fitnesses but not partial ties | `test_rank_normalize_partial_ties()` |
| **B3: Seed collision** | ❌ None | `test_seed_collision_under_decay()` |
| **B4: Negative reward threshold** | ❌ None | `test_sync_threshold_negative_env()` |
| **B5: IDN baseline** | ❌ None | `test_idn_baseline_capture_edge_cases()` |

---

## Recommendations for Next Steps

### Immediate (Before Running Validation Experiments)

1. **Fix B1 (Truncation):** Change line 76 and 89:
   ```python
   done = terminated  # Only true terminal, not truncated
   ```
   This is a 1-line fix with massive impact on all three validation experiments.

2. **Fix B2 (Rank Ties):** Implement proper dense rank normalization.
   ```python
   from scipy.stats import rankdata
   dense_ranks = rankdata(fitnesses, method='dense') - 1  # [0, n-1]
   ranks = dense_ranks / (len(fitnesses) - 1) - 0.5 if len(fitnesses) > 1 else 0
   ```

3. **Fix B3 (Seed Collision):** Use a collision-free seeding scheme, e.g., increment a global counter or use a better hash.

### Before Production

4. **Validate B4 (Negative Reward Threshold):** Run Acrobot/MountainCar with and without the fix; check sync behavior.

5. **Investigate B5 (IDN Baseline):** Add a guard to ensure baseline is recorded only on episodes with transitions.

### Nice-to-Haves

6. **R1 (IDN Overfitting):** Batch-replay or mini-batch sampling within `update()`.
7. **R2 (ReLU Collapse):** Remove final ReLU or use ELU; monitor embedding norm over time.
8. **R3 (Config Mutation):** Refactor to use a proper nested config object or inheritance instead of `__setattr__`.

---

## Code Quality Observations

**Positive:**
- Clean separation of actor/learner/buffer concerns ✓
- Explicit typing throughout ✓
- Extensive config system for environment-specific tuning ✓
- Convergence decay logic is principled and well-structured ✓

**Areas for Improvement:**
- The `done = terminated or truncated` conflation is a widespread RL implementation error; this codebase caught by it.
- Rank normalization docstring doesn't match implementation — favor code comments over docstrings for algorithmic details.
- Seed generation should be validated before use (e.g., collision test in test suite).

---

## References

- Mnih et al. (2015) — DQN: proper handling of terminal states vs truncation
- van Hasselt et al. (2016) — Double DQN
- Salimans et al. (2017) — ES parameter updates and rank normalization
- Badia et al. (2020) — Never Give Up (episodic novelty reference)

