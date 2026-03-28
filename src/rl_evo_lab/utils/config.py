from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EDERConfig:
    # Environment
    env_id: str = "CartPole-v1"
    obs_dim: int = 4
    act_dim: int = 2

    # ES Actor
    use_es: bool = True               # False = pure DQN with ε-greedy, no ES population
    es_sigma: float = 0.06
    es_n_workers: int = 50
    es_lr: float = 0.01
    es_weight_decay: float = 0.005
    es_antithetic: bool = True
    sync_freq: int = 25
    sync_eval_threshold: float = 0.7  # only sync if learner_eval >= threshold * mean_actor_ext
    use_novelty: bool = True          # False = ES+DQN baseline, no intrinsic reward

    # DQN Learner
    hidden_dim: int = 128
    dqn_lr: float = 1e-3
    gamma: float = 0.99
    target_update_freq: int = 100
    batch_size: int = 64
    grad_clip: float = 10.0

    # ε-greedy (pure DQN mode only)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 200

    # Replay Buffer
    buffer_capacity: int = 100_000

    # Intrinsic reward
    beta: float = 0.02
    knn_k: int = 5
    embed_dim: int = 64
    idn_lr: float = 1e-3
    idn_updates_per_episode: int = 5
    # Novelty schedule: zero for warmup eps, then linear ramp, then IDN-confidence-scaled
    novelty_warmup_episodes: int = 50
    novelty_ramp_episodes: int = 100
    # Minimum fraction of beta kept on after warmup so novelty can't be silenced permanently
    # by a slow-learning IDN. E.g. 0.1 means beta never drops below beta * 0.1 post-warmup.
    novelty_beta_floor: float = 0.1
    # Cross-generation global novelty buffer (0 = disabled, episodic-only)
    global_novelty_capacity: int = 2_000
    # Convergence decay: as learner_eval crosses novelty_decay_start_reward → solved_reward,
    # a shared progress signal [0, 1] drives beta, sigma, and n_workers all toward their
    # minimum values. This prevents the ES population from destabilising a solved learner.
    solved_reward: float = 475.0           # environment's "solved" reward threshold
    novelty_solve_decay: bool = True       # enable convergence decay for beta/sigma/workers
    novelty_decay_start_reward: float = 400.0  # learner eval at which decay begins
    es_sigma_min: float = 0.005           # sigma floor when fully converged (finer exploration)
    es_workers_min: int = 4               # worker count floor when fully converged

    # Training
    total_episodes: int = 500           # hard ceiling — early stopping usually ends runs sooner
    learner_updates_per_episode: int = 20
    min_buffer_size: int = 1_000

    # Early stopping — prevents wasted compute after solving or stagnating.
    # Checked at every eval_freq episode. Both conditions are evaluated independently.
    early_stop_solved_window: int = 5   # stop after this many consecutive evals >= solved_reward
    early_stop_patience: int = 30       # stop if best eval doesn't improve by min_delta for this many evals
    early_stop_min_delta: float = 2.0   # minimum reward improvement to reset the patience counter

    # Eval / logging
    seed: int = 42
    eval_freq: int = 10
    eval_episodes: int = 20
    use_wandb: bool = False
    wandb_project: str = "rl-evo-lab"


# ---------------------------------------------------------------------------
# Environment presets
# ---------------------------------------------------------------------------

ENV_PRESETS: dict[str, dict[str, Any]] = {
    # CartPole-v1 — solved at 475. Short episodes (~200 steps), fast to fill buffer.
    "cartpole": {
        "env_id": "CartPole-v1", "obs_dim": 4, "act_dim": 2,
        "total_episodes": 2000,
        "buffer_capacity": 50_000,
        "min_buffer_size": 500,
        "es_n_workers": 6,
        "target_update_freq": 100,
        "epsilon_decay_episodes": 200,
        "solved_reward": 475.0,
        "novelty_decay_start_reward": 400.0,
    },
    # LunarLander-v3 — solved at 200. Longer episodes (~400 steps), sparse early signal.
    # Larger network (256) and conservative lr (5e-4) for harder dynamics.
    # Longer novelty warmup/ramp: IDN needs more data before embeddings are reliable.
    "lunarlander": {
        "env_id": "LunarLander-v3", "obs_dim": 8, "act_dim": 4,
        "total_episodes": 3000,
        "buffer_capacity": 100_000,
        "min_buffer_size": 5_000,
        "es_n_workers": 10,
        "eval_freq": 25,
        "sync_freq": 50,
        "learner_updates_per_episode": 50,
        "epsilon_decay_episodes": 800,
        "target_update_freq": 200,
        "hidden_dim": 256,
        "dqn_lr": 5e-4,
        "batch_size": 128,
        "embed_dim": 128,
        "novelty_warmup_episodes": 100,
        "novelty_ramp_episodes": 200,
        "solved_reward": 200.0,
        "novelty_decay_start_reward": 150.0,
    },
    # Acrobot-v1 — solved at -100. Medium episodes (~200-500 steps).
    "acrobot": {
        "env_id": "Acrobot-v1", "obs_dim": 6, "act_dim": 3,
        "total_episodes": 1000,
        "buffer_capacity": 50_000,
        "min_buffer_size": 1_000,
        "epsilon_decay_episodes": 400,
        "solved_reward": -100.0,
        "novelty_decay_start_reward": -130.0,
    },
    # MountainCar-v0 — solved at -110. Dense negative reward (-1/step), but agent
    # must discover momentum-building behaviour to reach the goal — ε-greedy DQN
    # almost never manages this unaided. EDER's novelty drives population to explore
    # (position, velocity) space, eventually discovering the swing strategy.
    "mountaincar": {
        "env_id": "MountainCar-v0", "obs_dim": 2, "act_dim": 3,
        "total_episodes": 1000,      # ceiling; early stopping usually triggers first
        "buffer_capacity": 30_000,   # episodes are max 200 steps; 30k is plenty
        "min_buffer_size": 1_000,
        "es_n_workers": 10,          # 5 antithetic pairs — more than cartpole, task needs exploration
        "epsilon_decay_episodes": 300,
        "beta": 0.05,                # higher novelty weight for sparse-signal env
        "novelty_warmup_episodes": 50,
        "novelty_ramp_episodes": 75,
        "solved_reward": -110.0,
        "novelty_decay_start_reward": -150.0,
    },
}


def make_config(env: str = "cartpole", **overrides: Any) -> EDERConfig:
    """Build an EDERConfig from an env preset name with optional overrides.

    Example::

        cfg = make_config("lunarlander", total_episodes=1000, seed=7)
    """
    preset = ENV_PRESETS.get(env)
    if preset is None:
        raise ValueError(f"Unknown env preset {env!r}. Available: {list(ENV_PRESETS)}")
    return EDERConfig(**{**preset, **overrides})
