# Results

Mean ± std of the post-training `final_eval` (fresh held-out episodes,
greedy policy) across seeds. `Solved?` compares the mean against each
environment's `solved_reward`.

## cartpole

| Condition | Mean | Std | Seeds | Eval Episodes | Solved? |
|---|---|---|---|---|---|
| DDQN | 500.0 | 0.0 | 3 | 100 | ✓ |
| DDQN+ES | 500.0 | 0.0 | 3 | 100 | ✓ |
| DDQN+Novelty | 500.0 | 0.0 | 3 | 100 | ✓ |
| DDQN+ES+Novelty | 493.1 | 9.8 | 3 | 100 | ✓ |

## lunarlander

| Condition | Mean | Std | Seeds | Eval Episodes | Solved? |
|---|---|---|---|---|---|
| DDQN | 259.0 | 6.4 | 3 | 50 | ✓ |
| DDQN+ES | -123.6 | 105.0 | 3 | 50 | ✗ |
| DDQN+Novelty | 254.5 | 8.3 | 3 | 50 | ✓ |
| DDQN+ES+Novelty | -56.1 | 104.2 | 3 | 50 | ✗ |

## cartpole_sparse

| Condition | Mean | Std | Seeds | Eval Episodes | Solved? |
|---|---|---|---|---|---|
| DDQN | 500.0 | 0.0 | 3 | 100 | ✓ |
| DDQN+ES | 500.0 | 0.0 | 3 | 100 | ✓ |
| DDQN+Novelty | 500.0 | 0.0 | 3 | 100 | ✓ |
| DDQN+ES+Novelty | 498.3 | 2.4 | 3 | 100 | ✓ |

## acrobot

| Condition | Mean | Std | Seeds | Eval Episodes | Solved? |
|---|---|---|---|---|---|
| DDQN | -286.8 | 137.3 | 3 | 50 | ✗ |
| DDQN+ES | -500.0 | 0.0 | 3 | 50 | ✗ |
| DDQN+Novelty | -412.8 | 107.6 | 3 | 50 | ✗ |
| DDQN+ES+Novelty | -500.0 | 0.0 | 3 | 50 | ✗ |

## montezuma

| Condition | Mean | Std | Seeds | Eval Episodes | Solved? |
|---|---|---|---|---|---|
| DDQN | 0.0 | 0.0 | 3 | 20 | ✗ |
| DDQN+ES | — | — | — | — | not run |
| DDQN+Novelty | — | — | — | — | not run |
| DDQN+ES+Novelty | — | — | — | — | not run |

