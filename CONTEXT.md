# Project Context: Evolutionary Reinforcement Learning

## Background

This repo is a reproduction and extension of my MSc thesis:
**"Improving Exploration in Evolutionary Reinforcement Learning through Novelty Search"**
NUI Galway, 2021. First-class MSc in Computer Science (AI).

---

## Original Algorithm: EDER (Evolutionary Distributed Experience Replay)

### Core Idea
EDER combines a gradient-based DRL learner with an Evolution Strategy (ES)
actor population. The ES generates diverse experiences for the learner's replay
buffer, replacing epsilon-greedy exploration entirely.

### Motivation
- ERL (Khadka & Tumer, 2018) showed EA populations improve DRL exploration
  but its EA was purely extrinsic/goal-driven — contradictory for an exploration mechanism
- NGU (Badia et al., 2020) showed intrinsic rewards (curiosity) improve exploration in DRL
- EDER combines both: an ES population scored on augmented reward (extrinsic + intrinsic)

### Architecture

**Learner (DQN)**
- Standard DQN: policy network + target network
- Only interacts with environment to evaluate performance
- Does NOT store its own rollouts — replay buffer filled exclusively by the actor
- Updated via sampled batches from replay buffer using MSE loss + Adam

**Actor (Evolution Strategy — Salimans et al. 2017)**
- Worker network with same architecture as learner policy net
- Each episode: generates N noisy agents (θ + σε, ε ~ N(0,1))
- Scores each agent on augmented reward F(θ + σε)
- Updates worker parameters toward best-performing directions
- Periodically syncs back to learner policy weights (anchor to prevent divergence)

**Reward Function**
```
rₐ = rₑ + β · rᵢ
```
- `rₑ` = extrinsic reward from environment
- `rᵢ` = intrinsic reward: KNN distance over episodic memory of controllable states
- `β` = scalar weighting exploration vs exploitation
- Controllable states: learned embeddings from a network trained to predict
  action taken between (sₜ, sₜ₊₁) — captures agent-controllable aspects only
- Episodic memory M reset each episode; only extrinsic transitions pushed to replay buffer

**Key hyperparameters**
- σ (noise std dev): exploration breadth in parameter space. Too small = no diversity,
  too large = divergence. σ=0.06 best in experiments.
- β: intrinsic reward weight. Set to 0.02 in augmented experiments.
- N workers: 50
- Sync frequency τ: every 25 episodes

### Results (CartPole-v1)
- Extrinsic EDER: reached max reward ~75 episodes vs ~3500 for DQN (same update count)
  but suffered catastrophic forgetting as replay buffer became homogeneous
- Augmented EDER: slower to peak but stable — intrinsic reward kept buffer diverse,
  mitigating catastrophic forgetting
- Both EDER variants significantly outperformed standalone DQN on sample efficiency
- Standalone ES (no DQN): volatile, performance degraded with larger σ

### Known Limitations from Original Work
- Only tested on CartPole — too simple to draw strong conclusions
- Catastrophic forgetting not fully solved, sidestepped via intrinsic diversity
- Sample efficiency comparison vs DQN complicated by parallel actor setup
- No lifelong novelty module (only episodic) — NGU uses both
- Only DQN tested as learner — DDPG, SAC, R2D2 not explored
- Atari benchmark not reached due to compute constraints

---

## Current Direction: Reproduction + Extension

### Immediate Goals
1. Clean reproduction of EDER in a modern codebase
2. Swap DQN learner for SAC or DDPG (continuous action spaces, MuJoCo targets)
3. Test on harder environments: MuJoCo locomotion, sparse reward tasks
4. Add lifelong novelty module (RND-based, per NGU) alongside episodic KNN

### Longer-Term Ideas
- Replace KNN intrinsic reward with learned curiosity (e.g. RND, ICM)
- Explore MAP-Elites or QD (Quality Diversity) as the EA component instead of ES
- Multi-objective fitness: novelty + extrinsic as a Pareto front rather than scalar blend
- Population heterogeneity: different β values per worker to get a spectrum from
  pure explorer to pure exploiter within the same population
- Potential application to embodied AI / robotics (drone control, robotic arm sorting)

### Design Principles for This Repo
- Modular: learner and actor should be swappable independently
- The replay buffer is the interface between actor and learner — keep it clean
- Augmented reward is internal to the actor; learner always trains on extrinsic only
- Logging: track actor reward, learner reward, mean worker reward, and buffer diversity
  separately — the original thesis showed these tell very different stories
- Reproducibility: seed everything, log σ and β per run

---

---

## GPU / MPS Roadmap

### Current State
All training runs on CPU. Networks are tiny (2-layer MLP, hidden_dim=128) and batch
sizes are small (64). For these workloads, CPU outperforms MPS because:
- MPS dispatch overhead > actual computation for small tensors
- ES workers do single-sample inference (batch=1) — GPU is always slower here
- The real bottleneck is the Python/gymnasium env loop, not the network

### When MPS Becomes Worth It
| Trigger | Why |
|---|---|
| SAC/DDPG learner (continuous control) | Larger networks, actor+critic+value, more gradient steps |
| hidden_dim ≥ 512 | Matrix multiply large enough to amortise MPS dispatch |
| batch_size ≥ 256 | GPU parallelism starts to dominate per-sample overhead |
| CNN policy (pixel obs / Atari) | Conv operations are where MPS genuinely shines |
| Brax / MuJoCo MJX envs | Entire env step runs as a GPU kernel — no Python loop |

### Recommended Migration Path
1. Add SAC learner (next roadmap item) — hidden_dim ≥ 256, actor+critic networks
2. Move to Brax (JAX-based physics, runs on MPS via JAX Metal backend)
   - Envs: Ant, HalfCheetah, Hopper, Humanoid
   - ES population episodes become batched GPU operations, not a Python for-loop
3. At that point: keep ES workers on CPU (batch=1 inference), move learner training to MPS

### Note on ThreadPoolExecutor + MPS
Multiple threads queueing small operations to MPS causes contention and is slower than
CPU. Current design (threads for ES workers) intentionally keeps worker inference on CPU.
Only the learner training loop (large batch, single call) should move to MPS when the
time comes.

---

## Key References
- Khadka & Tumer (2018) — ERL: Evolution-Guided Policy Gradient
- Salimans et al. (2017) — ES as scalable alternative to RL
- Badia et al. (2020) — Never Give Up (NGU): episodic + lifelong intrinsic reward
- Mnih et al. (2015) — DQN
- Lehman & Stanley (2011) — Novelty Search
- Lillicrap et al. (2015) — DDPG

---

## Author Background
ML Engineer, 3+ years experience. Strong background in deep learning, MLOps,
AWS/SageMaker. Prior work in deep learning hardware (Intel). First-class MSc CS/AI.
This project sits at the intersection of research reproduction and original contribution.
