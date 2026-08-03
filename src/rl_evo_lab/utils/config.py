from dataclasses import dataclass
from typing import Any

# Environment categories for principled HP selection
# Categories based on: action_space, reward_density, horizon_length, exploration_difficulty
ENV_CATEGORIES = {
    "discrete_dense_short": {
        "description": "Discrete actions, dense reward, short episodes (CartPole-like)",
        "es_sigma": 0.1,
        "beta": 0.1,
        "novelty_ramp_episodes": 200,
        "es_n_workers": 6,
    },
    "continuous_dense_medium": {
        "description": "Continuous actions, dense reward, medium episodes (LunarLander-like)",
        "es_sigma": 0.12,
        "beta": 0.15,
        "novelty_ramp_episodes": 250,
        "es_n_workers": 10,
    },
    "discrete_sparse_long": {
        "description": "Discrete actions, sparse reward, long episodes (Acrobot-like)",
        "es_sigma": 0.15,
        "beta": 0.2,
        "novelty_ramp_episodes": 300,
        "es_n_workers": 12,
    },
    "continuous_sparse_long": {
        "description": "Continuous actions, sparse reward, long episodes (MountainCar-like)",
        "es_sigma": 0.18,
        "beta": 0.25,
        "novelty_ramp_episodes": 350,
        "es_n_workers": 15,
    },
}


@dataclass(frozen=True)
class EDERConfig:
    # Environment
    env_id: str = "CartPole-v1"
    obs_dim: int = 4
    act_dim: int = 2

    # ES Actor
    use_es: bool = True  # False = pure DQN with ε-greedy, no ES population
    es_sigma: float = 0.06
    es_n_workers: int = 50
    es_lr: float = 0.01
    es_weight_decay: float = 0.005
    es_antithetic: bool = True
    sync_freq: int = 25
    sync_eval_threshold: float = 0.7  # only sync if learner_eval >= threshold * mean_actor_ext
    use_novelty: bool = True  # False = ES+DQN baseline, no intrinsic reward

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
    solved_reward: float = 475.0  # environment's "solved" reward threshold
    novelty_solve_decay: bool = True  # enable convergence decay for beta/sigma/workers
    novelty_decay_start_reward: float = 400.0  # learner eval at which decay begins
    es_sigma_min: float = 0.005  # sigma floor when fully converged (finer exploration)
    es_workers_min: int = 4  # worker count floor when fully converged

    # Training
    total_episodes: int = 500  # hard ceiling — early stopping usually ends runs sooner
    learner_updates_per_episode: int = 20
    min_buffer_size: int = 1_000

    # Early stopping — prevents wasted compute after solving or stagnating.
    # Checked at every eval_freq episode. Both conditions are evaluated independently.
    early_stop_solved_window: int = 5  # stop after this many consecutive evals >= solved_reward
    early_stop_patience: int = (
        30  # stop if best eval doesn't improve by min_delta for this many evals
    )
    early_stop_min_delta: float = 2.0  # minimum reward improvement to reset the patience counter

    # Buffer push filtering — selectively gate which worker episodes enter the buffer.
    # None = push everything (backward compatible default).
    buffer_push_alpha: float | None = None
    # alpha=0.5: equal weight fitness+novelty. alpha=1.0: fitness only. alpha=0.0: novelty only.

    buffer_push_top_k: int | None = None
    # Push only top-K workers by combined score. None = push all that pass the floor.
    # Recommended: set to ~60-70% of es_n_workers (e.g. 4 out of 6 for cartpole).

    buffer_novelty_floor: float = 0.2
    # Top fraction of workers by raw novelty always enter buffer regardless of combined score.
    # Ensures genuine exploration of new state regions is never filtered out.

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
    # CartPole-v1 — Category: discrete_dense_short
    # HPs selected from category defaults with environment-specific tuning
    "cartpole": {
        "env_id": "CartPole-v1",
        "obs_dim": 4,
        "act_dim": 2,
        "category": "discrete_dense_short",
        "total_episodes": 2000,
        "buffer_capacity": 50_000,
        "min_buffer_size": 500,
        "es_sigma": 0.1,
        "beta": 0.1,
        "novelty_ramp_episodes": 200,
        "target_update_freq": 100,
        "epsilon_decay_episodes": 200,
        "solved_reward": 475.0,
        "novelty_decay_start_reward": 400.0,
    },
    # CartPole-Tough: Category: discrete_dense_short
    # Random starting position/angle + stricter termination (12° vs 24°) + 1000 step limit
    "cartpole_tough": {
        "env_id": "CartPole-v1",
        "obs_dim": 4,
        "act_dim": 2,
        "category": "discrete_dense_short",
        "total_episodes": 2000,
        "buffer_capacity": 50_000,
        "min_buffer_size": 500,
        "es_sigma": 0.1,
        "beta": 0.1,
        "novelty_ramp_episodes": 200,
        "target_update_freq": 100,
        "epsilon_decay_episodes": 200,
        "solved_reward": 10000.0,  # high threshold so only full 1000-step episodes count
        "novelty_decay_start_reward": 8000.0,
        # Custom flags for tough variant (applied in train.py)
        "_tough": True,
        "_cartpole_angle_limit": 0.209,  # 12° in radians
        "_cartpole_position_limit": 2.0,  # tighter than default 2.4
        "_episode_limit": 1000,
        "_random_start": True,
    },
    # LunarLander-v3 — Category: continuous_dense_medium
    "lunarlander": {
        "env_id": "LunarLander-v3",
        "obs_dim": 8,
        "act_dim": 4,
        "category": "continuous_dense_medium",
        "total_episodes": 3000,
        "buffer_capacity": 100_000,
        "min_buffer_size": 5_000,
        "es_n_workers": 10,
        "es_sigma": 0.12,  # category default, needs validation
        "beta": 0.15,  # category default, needs validation
        "novelty_ramp_episodes": 250,  # category default, needs validation
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
        "solved_reward": 200.0,
        "novelty_decay_start_reward": 150.0,
    },
    # LunarLander-Tough: random start + tight landing zone (±0.1 pad center) + 2000 step limit
    "lunarlander_tough": {
        "env_id": "LunarLander-v3",
        "obs_dim": 8,
        "act_dim": 4,
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
        "solved_reward": 250.0,  # higher for tight landing requirement
        "novelty_decay_start_reward": 200.0,
        # Custom flags for tough variant (applied in train.py)
        "_tough": True,
        "_random_start_every_episode": True,
        "_landing_zone_radius": 0.1,  # ±0.1 of pad center
        "_episode_limit": 2000,
    },
    # Acrobot-v1 — solved at -100. Discrete actions, sparse reward, long episodes.
    # Category: discrete_sparse_long (sparse signal needs stronger novelty + faster ramp)
    "acrobot": {
        "env_id": "Acrobot-v1",
        "obs_dim": 6,
        "act_dim": 3,
        "category": "discrete_sparse_long",
        "total_episodes": 1000,
        "buffer_capacity": 50_000,
        "min_buffer_size": 1_000,
        "epsilon_decay_episodes": 400,
        # Category-specific HPs: applied via make_config() from ENV_CATEGORIES
        "es_sigma": 0.15,
        "beta": 0.2,
        "es_n_workers": 12,
        "novelty_ramp_episodes": 300,
        "solved_reward": -100.0,
        "novelty_decay_start_reward": -130.0,
    },
    # MountainCar-v0 — solved at -110. Continuous actions, sparse reward, long episodes.
    # Dense negative reward (-1/step), but agent must discover momentum-building behaviour.
    # ε-greedy DQN almost never manages this unaided. EDER's novelty drives exploration.
    # Category: continuous_sparse_long (sparse + continuous need strong novelty + large pop)
    "mountaincar": {
        "env_id": "MountainCar-v0",
        "obs_dim": 2,
        "act_dim": 3,
        "category": "continuous_sparse_long",
        "total_episodes": 1000,  # ceiling; early stopping usually triggers first
        "buffer_capacity": 30_000,  # episodes are max 200 steps; 30k is plenty
        "min_buffer_size": 1_000,
        # Category-specific HPs: applied via make_config() from ENV_CATEGORIES
        "es_sigma": 0.18,
        "beta": 0.25,
        "es_n_workers": 15,
        "novelty_ramp_episodes": 350,
        "epsilon_decay_episodes": 300,
        "novelty_warmup_episodes": 50,
        "solved_reward": -110.0,
        "novelty_decay_start_reward": -150.0,
    },
}


def make_config(env: str = "cartpole", **overrides: Any) -> EDERConfig:
    """Build an EDERConfig from an env preset name with optional overrides.

    Merges in this order (higher priority wins):
    1. Dataclass defaults (EDERConfig)
    2. Category defaults (ENV_CATEGORIES[category_name])
    3. Preset values (ENV_PRESETS[env])
    4. Overrides (caller-provided kwargs)

    Example::

        cfg = make_config("lunarlander", total_episodes=1000, seed=7)
    """
    preset = ENV_PRESETS.get(env)
    if preset is None:
        raise ValueError(f"Unknown env preset {env!r}. Available: {list(ENV_PRESETS)}")

    # Extract category from preset, if specified
    category_name = preset.get("category")
    category = ENV_CATEGORIES.get(category_name) if category_name else {}

    # Merge: category defaults < preset < overrides
    # This ensures overrides win, preset > category > dataclass defaults
    merged = {**category, **preset, **overrides}

    # Filter out metadata and custom flags that are not EDERConfig parameters
    config_kwargs = {
        k: v for k, v in merged.items()
        if not k.startswith("_") and k not in {"category", "description"}
    }
    cfg = EDERConfig(**config_kwargs)

    # Attach custom flags to config for use by environment wrappers
    for k, v in merged.items():
        if k.startswith("_"):
            object.__setattr__(cfg, k, v)
    return cfg
