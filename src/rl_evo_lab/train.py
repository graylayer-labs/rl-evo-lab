import gymnasium as gym
import torch

from rl_evo_lab.actor.es_actor import ESActor
from rl_evo_lab.buffer.replay_buffer import ReplayBuffer
from rl_evo_lab.intrinsic.inverse_dynamics import InverseDynamicsNetwork
from rl_evo_lab.learner.dqn import DQNLearner
from rl_evo_lab.utils.config import EDERConfig
from rl_evo_lab.utils.logging import EpisodeLog, RunLogger
from rl_evo_lab.utils.seeding import seed_everything


def train(cfg: EDERConfig = EDERConfig(), log_dir: str = "runs") -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.obs_dim)
    learner = DQNLearner(cfg, device)
    idn = InverseDynamicsNetwork(cfg, device)
    actor = ESActor(cfg, device)
    logger = RunLogger(cfg, log_dir=log_dir)

    env_fn = lambda: gym.make(cfg.env_id)
    eval_env = gym.make(cfg.env_id)

    last_loss = 0.0
    last_eval = 0.0  # tracks most recent learner eval for sync gating

    for episode in range(cfg.total_episodes):
        stats = actor.run_generation(env_fn, idn, buffer, episode)

        eval_reward = None
        diversity = None
        did_sync = False

        if len(buffer) >= cfg.min_buffer_size:
            for _ in range(cfg.learner_updates_per_episode):
                last_loss = learner.train_step(buffer)

        # Dynamic sync: evaluate at every sync candidate, only sync if learner
        # has reached sync_eval_threshold × mean actor extrinsic reward.
        if episode % cfg.sync_freq == 0:
            sync_eval = learner.evaluate(eval_env, cfg.eval_episodes)
            last_eval = sync_eval
            eval_reward = sync_eval
            diversity = buffer.diversity_metric()

            threshold = cfg.sync_eval_threshold * stats.mean_extrinsic_return
            if sync_eval >= threshold:
                actor.sync_from_learner(learner.get_weights())
                did_sync = True

        elif episode % cfg.eval_freq == 0:
            eval_reward = learner.evaluate(eval_env, cfg.eval_episodes)
            last_eval = eval_reward
            diversity = buffer.diversity_metric()

        logger.log(EpisodeLog(
            episode=episode,
            actor_augmented_reward=stats.mean_augmented_fitness,
            actor_extrinsic_reward=stats.mean_extrinsic_return,
            learner_loss=last_loss,
            learner_eval_reward=eval_reward,
            buffer_diversity=diversity,
            idn_loss=stats.idn_loss,
            buffer_size=len(buffer),
            sync=did_sync,
        ))

    logger.close()
    eval_env.close()


if __name__ == "__main__":
    train()
