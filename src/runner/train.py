from __future__ import annotations

from multiprocessing import Queue
from pathlib import Path

import torch

from actor.es_actor import ESActor
from buffer.replay_buffer import ReplayBuffer
from envs.wrappers import make_env_with_config
from infra.config import EDERConfig
from infra.logging import EpisodeLog, RunLogger
from infra.seeding import seed_everything
from intrinsic.inverse_dynamics import InverseDynamicsNetwork
from learner.dqn import DQNLearner


def _compute_sync_threshold(sync_eval_threshold: float, mean_extrinsic_return: float) -> float:
    """Compute the learner evaluation threshold for actor-learner synchronisation.

    The intent is to sync only when the learner has reached a performance fraction of the actor.
    For positive rewards: threshold = threshold_fraction * mean_return (e.g., 0.7 * 100 = 70).
    For negative rewards: mirror symmetrically so the tolerance is equivalent in magnitude.
        E.g., 0.7 * -100 should give -130 (learner can be 30% worse, or 30 more negative).
    """
    if mean_extrinsic_return >= 0:
        return sync_eval_threshold * mean_extrinsic_return
    else:
        # For negative rewards: invert so a 30% tolerance means 30% worse (more negative).
        # threshold = mean * (2 - sync_eval_threshold), e.g., -100 * (2 - 0.7) = -130
        return mean_extrinsic_return * (2.0 - sync_eval_threshold)


def train(
    cfg: EDERConfig | None = None,
    log_dir: str = "runs",
    verbose: bool = True,
    progress_queue: Queue | None = None,
    run_dir: Path | None = None,
) -> None:
    if cfg is None:
        cfg = EDERConfig()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.obs_dim)
    learner = DQNLearner(cfg, device)
    idn = InverseDynamicsNetwork(cfg, device) if cfg.use_es else None
    actor = ESActor(cfg, device) if cfg.use_es else None
    logger = RunLogger(
        cfg, log_dir=log_dir, verbose=verbose, progress_queue=progress_queue, run_dir=run_dir
    )

    def env_fn():
        return make_env_with_config(cfg.env_id, cfg)

    collect_env = make_env_with_config(
        cfg.env_id, cfg
    )  # used by DQN collect_episode when use_es=False
    eval_env = make_env_with_config(cfg.env_id, cfg)

    last_loss = 0.0
    last_eval = 0.0
    cumulative_env_steps = 0

    # Early stopping state — success-only. See EDERConfig.early_stop_solved_window.
    _solved_streak = 0  # consecutive eval windows at or above solved_reward

    for episode in range(cfg.total_episodes):
        eval_reward = None
        diversity = None
        did_sync = False
        idn_loss = 0.0

        if cfg.use_es:
            stats = actor.run_generation(env_fn, idn, buffer, episode)
            mean_extrinsic_return = stats.mean_extrinsic_return
            mean_augmented_fitness = stats.mean_augmented_fitness
            idn_loss = stats.idn_loss
            effective_beta = stats.effective_beta
            cumulative_env_steps += stats.total_env_steps
        else:
            ep_return, ep_steps = learner.collect_episode(collect_env, buffer, episode)
            mean_extrinsic_return = ep_return
            mean_augmented_fitness = ep_return
            effective_beta = 0.0
            cumulative_env_steps += ep_steps

        if len(buffer) >= cfg.min_buffer_size:
            for _ in range(cfg.learner_updates_per_episode):
                last_loss = learner.train_step(buffer)

        # Evaluate at eval_freq for all conditions — keeps chart resolution consistent.
        if episode % cfg.eval_freq == 0:
            eval_reward = learner.evaluate(eval_env, cfg.eval_episodes)
            last_eval = eval_reward
            diversity = buffer.diversity_metric()
            if cfg.use_es:
                actor.update_learner_eval(last_eval)
                # Sync immediately when solved: anchors ES to the working policy
                # every eval cycle, cutting off the forgetting cycle at its root.
                if last_eval >= cfg.solved_reward:
                    actor.sync_from_learner(learner.get_weights())
                    did_sync = True

            # Early stopping tracker — updated every eval window
            if last_eval >= cfg.solved_reward:
                _solved_streak += 1
            else:
                _solved_streak = 0

        # Periodic pre-solve sync: pulls ES toward learner before it fully solves,
        # preventing the actor from diverging too far. Skipped when solved (handled above).
        if cfg.use_es and not did_sync and episode % cfg.sync_freq == 0:
            threshold = _compute_sync_threshold(cfg.sync_eval_threshold, mean_extrinsic_return)
            if last_eval >= threshold:
                actor.sync_from_learner(learner.get_weights())
                did_sync = True

        logger.log(
            EpisodeLog(
                episode=episode,
                total_env_steps=cumulative_env_steps,
                actor_augmented_reward=mean_augmented_fitness,
                actor_extrinsic_reward=mean_extrinsic_return,
                learner_loss=last_loss,
                learner_eval_reward=eval_reward,
                buffer_diversity=diversity,
                idn_loss=idn_loss,
                effective_beta=effective_beta,
                buffer_size=len(buffer),
                sync=did_sync,
            )
        )

        # Check early stopping after logging so the final episode is always in the CSV
        if episode % cfg.eval_freq == 0 and _solved_streak >= cfg.early_stop_solved_window:
            break

    # Authoritative post-training evaluation — fresh held-out episodes, distinct
    # seed range from the periodic training-time eval above (which uses +10_000).
    final_eval_rewards = learner.evaluate_episodes(
        eval_env, cfg.final_eval_episodes, seed_offset=20_000
    )
    logger.close(final_eval_rewards=final_eval_rewards)
    collect_env.close()
    eval_env.close()

    if run_dir is not None:
        policy_path = Path(run_dir) / "policy.pt"
        learner.policy_net.cpu()
        torch.save(learner.policy_net.state_dict(), policy_path)


if __name__ == "__main__":
    train()
