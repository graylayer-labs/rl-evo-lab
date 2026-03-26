from dataclasses import dataclass


@dataclass(frozen=True)
class EDERConfig:
    # Environment
    env_id: str = "CartPole-v1"
    obs_dim: int = 4
    act_dim: int = 2

    # ES Actor
    es_sigma: float = 0.06
    es_n_workers: int = 50
    es_lr: float = 0.01
    es_weight_decay: float = 0.005
    es_antithetic: bool = True
    sync_freq: int = 25
    sync_eval_threshold: float = 0.7  # only sync if learner_eval >= threshold * mean_actor_ext
    use_novelty: bool = True          # False = ES+DQN baseline, no intrinsic reward

    # DQN Learner
    dqn_lr: float = 1e-3
    gamma: float = 0.99
    target_update_freq: int = 100
    batch_size: int = 64
    grad_clip: float = 10.0

    # Replay Buffer
    buffer_capacity: int = 100_000

    # Intrinsic reward
    beta: float = 0.02
    knn_k: int = 5
    embed_dim: int = 64
    idn_lr: float = 1e-3
    idn_updates_per_episode: int = 5

    # Training
    total_episodes: int = 500
    learner_updates_per_episode: int = 20
    min_buffer_size: int = 1_000

    # Eval / logging
    seed: int = 42
    eval_freq: int = 10
    eval_episodes: int = 20
    use_wandb: bool = False
    wandb_project: str = "rl-evo-lab"
