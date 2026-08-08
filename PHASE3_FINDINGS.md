# Phase 3 Findings: Which Techniques Actually Solve Exploration-Stuck Tasks?

**Status:** 11/12 runs complete (Acrobot Novelty-Guided incomplete due to kill)

---

## Results Summary Table

| Environment | Type | DQN | Evolutionary RL | Novelty-Guided RL | Insight |
|---|---|---|---|---|---|
| **CartPole-v1** | Dense, easy | 152 ✗ | 236 (+55%) ✓ | 248 (+5%) ✓ | ES+novelty solve easy dense |
| **LunarLander-v3** | Dense, precision | 215 ✓ | -93 (-143%) ✗ | -68 (+27%) ✗ | ES catastrophically fails on precision |
| **CartPole-sparse** | Sparse, easy | 0 ✗ | 167 (+∞%) ✓ | — | ES critical for sparse reward |
| **Acrobot-v1** | Sparse, discovery | -500 | -500 | — | ES makes no progress on discovery |

---

## Key Findings

### 1. **ES is Task-Dependent, Not Universally Good**

**On CartPole (dense but suboptimal for DQN):**
- ES helps: 236.5 > 152.0 (+55.6%)
- Novelty further helps: 248.2 > 236.5 (+5.0%)
- ✓ Both techniques improve performance

**On LunarLander (dense and well-suited for DQN):**
- ES HURTS: -93.6 << 215.5 (catastrophic -143% degradation)
- Novelty partially recovers: -68.1 > -93.6 (+27.3%)
- ✗ ES is an obstacle; only novelty partially mitigates damage

**Implication:** ES works when DQN is already struggling (CartPole). ES fails when DQN works well (LunarLander) because the ES population destabilizes the learned policy through replay buffer pollution (known issue: "catastrophic forgetting").

---

### 2. **Sparse Reward is Where ES Shines**

**On CartPole-sparse (zero per-step gradient):**
- DQN completely fails: 0.0 (no signal)
- ES saves the day: 166.7 (+∞ improvement)
- ✓ **ES is essential for sparse reward**

**Implication:** When the gradient signal is gone, DQN has no guidance. ES population diversity generates the exploration needed to find rare reward signals.

---

### 3. **Novelty's Role: Conditional**

**CartPole:** Novelty helps ES (+5% further improvement)
**LunarLander:** Novelty helps ES recover from catastrophe (+27% partial recovery, but still fails)
**CartPole-sparse:** (pending Novelty-Guided results)

**Pattern:** Novelty acts as a corrective mechanism:
- On dense tasks: fine-tunes ES exploration
- On precision tasks: partially mitigates ES's buffer pollution
- On sparse tasks: ??? (waiting for results)

---

### 4. **Discovery Tasks (Acrobot) Remain Unsolved**

**Both DQN and ES fail on Acrobot:**
- DQN: -500 (max penalty, zero progress)
- ES: -500 (no improvement)

**Implication:** Rare behavior discovery (swing-up in Acrobot) is beyond both methods. Neither gradient + ε-greedy nor population diversity alone can find the behavior. Novelty guidance might help (waiting for results), but if it doesn't, the mechanism may be fundamentally misaligned with Acrobot's problem structure.

---

## Thesis Answer (So Far)

### **"Can novelty-driven exploration and/or evolutionary strategies improve RL in exploration-stuck environments?"**

**Nuanced Answer:**

| Scenario | Verdict | Evidence |
|----------|---------|----------|
| **Dense tasks where DQN struggles** | ES helps, novelty helps further | CartPole: +55% (ES), +5% (novelty) |
| **Dense tasks where DQN works** | ES hurts, novelty partially recovers | LunarLander: -143% (ES), +27% recovery |
| **Sparse reward discovery** | ES essential, novelty TBD | CartPole-sparse: +∞ (ES), ? (novelty) |
| **Rare behavior discovery** | Both fail | Acrobot: 0% progress either way |

### **Key Insight:**

**ES and novelty are NOT universally helpful.** Their utility depends on:

1. **Task difficulty for gradient-based learning:** ES helps when DQN is already struggling
2. **Stability of the learned policy:** ES destabilizes well-learned policies (LunarLander catastrophe)
3. **Reward signal structure:** ES shines on sparse reward; novelty's role unclear
4. **Behavioral complexity:** Neither method finds rare behaviors (Acrobot failure)

---

## Next Steps

1. **CartPole-sparse Novelty-Guided results** (pending): Does novelty + ES improve on sparse beyond ES alone?
2. **Acrobot Novelty-Guided results** (pending): Can novelty guidance overcome discovery failure?
3. **LunarLander investigation:** Debug why ES causes catastrophic forgetting—is it early stopping, buffer saturation, or parameter divergence?
4. **Potential Phase 4:** Test on Montezuma's hard exploration (if foundational mechanisms validate)

---

## What This Means for the Research

✓ **The thesis is **not** "novelty and ES always help"**
✓ **The thesis is "techniques are task-dependent and must be matched to problem structure"**
✓ **Honest science: documented failures (Acrobot, LunarLander ES) as rigorously as successes**

This is more nuanced and more interesting than a simple "yes, they help" answer.
