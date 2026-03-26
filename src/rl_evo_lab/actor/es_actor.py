from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rl_evo_lab.actor.es_worker import WorkerResult, run_worker_episode
from rl_evo_lab.buffer.replay_buffer import ReplayBuffer
from rl_evo_lab.intrinsic.episodic_novelty import EpisodicNovelty
from rl_evo_lab.intrinsic.inverse_dynamics import InverseDynamicsNetwork
from rl_evo_lab.learner.network import QNetwork
from rl_evo_lab.utils.config import EDERConfig


@dataclass
class ActorStats:
    mean_augmented_fitness: float
    mean_extrinsic_return: float
    idn_loss: float


def _rank_normalize(fitnesses: np.ndarray) -> np.ndarray:
    """Rank fitnesses and normalise to [-0.5, 0.5].

    The lowest-fitness worker receives -0.5, the highest +0.5.
    Ties share the same rank (dense rank).
    """
    n = len(fitnesses)
    # argsort gives indices that would sort ascending; assign ranks 0..n-1
    ranks = np.empty(n, dtype=np.float32)
    order = np.argsort(fitnesses)          # ascending: worst → best
    ranks[order] = np.arange(n, dtype=np.float32)
    if n > 1:
        ranks = ranks / (n - 1) - 0.5     # map [0, n-1] → [-0.5, 0.5]
    else:
        ranks[:] = 0.0
    return ranks


class ESActor:
    """Evolution Strategy actor that fills the replay buffer with diverse experience."""

    def __init__(self, cfg: EDERConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

        # Initialise theta_base from a fresh QNetwork
        _init_net = QNetwork(cfg.obs_dim, cfg.act_dim)
        self.theta_base: np.ndarray = _init_net.get_flat_params().copy()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_generation(
        self,
        env_fn,
        idn: InverseDynamicsNetwork,
        buffer: ReplayBuffer,
        episode_num: int,
    ) -> ActorStats:
        """Run one ES generation.

        Each worker runs one episode with a perturbed copy of theta_base.
        Fitness scores are rank-normalised, then used to compute the ES
        gradient update.  All extrinsic transitions are pushed to the buffer.
        The IDN is updated on all collected transitions.
        """
        cfg = self.cfg

        # ---- Build list of (seed, sign) pairs -------------------------
        worker_jobs: list[tuple[int, int]] = []
        if cfg.es_antithetic:
            n_seeds = cfg.es_n_workers // 2
            for k in range(n_seeds):
                seed = episode_num * cfg.es_n_workers + k
                worker_jobs.append((seed, +1))
                worker_jobs.append((seed, -1))
            # If n_workers is odd, add one extra +1 worker
            if cfg.es_n_workers % 2 != 0:
                seed = episode_num * cfg.es_n_workers + n_seeds
                worker_jobs.append((seed, +1))
        else:
            for k in range(cfg.es_n_workers):
                seed = episode_num * cfg.es_n_workers + k
                worker_jobs.append((seed, +1))

        # ---- Run each worker episode ----------------------------------
        results: list[WorkerResult] = []
        for seed, sign in worker_jobs:
            env = env_fn()
            # Each worker gets its own EpisodicNovelty instance so that
            # episodes are truly independent.
            novelty = EpisodicNovelty(cfg.knn_k)
            result = run_worker_episode(
                base_params=self.theta_base,
                noise_seed=seed,
                sigma=cfg.es_sigma,
                env=env,
                cfg=cfg,
                idn=idn,
                novelty=novelty,
                beta=cfg.beta,
                noise_sign=sign,
                device=self.device,
            )
            env.close()
            results.append(result)

        # ---- Rank-normalise fitness scores ---------------------------
        fitnesses = np.array([r.fitness for r in results], dtype=np.float32)
        rank_weights = _rank_normalize(fitnesses)

        # ---- ES gradient update -------------------------------------
        # delta = Σ (rank_w_i * sign_i * noise_i) / (N * sigma)
        n_workers_actual = len(results)
        delta = np.zeros_like(self.theta_base)
        for i, result in enumerate(results):
            delta += rank_weights[i] * result.noise_sign * result.noise_vector

        delta /= n_workers_actual * cfg.es_sigma
        self.theta_base = (
            self.theta_base
            + cfg.es_lr * delta
            - cfg.es_weight_decay * self.theta_base
        )

        # ---- Push all transitions to buffer --------------------------
        for result in results:
            for obs, action, reward, next_obs, done in result.transitions:
                buffer.push(obs, action, reward, next_obs, done)

        # ---- Train IDN on collected transitions ----------------------
        all_obs = []
        all_next_obs = []
        all_actions = []
        for result in results:
            for obs, action, _, next_obs, _ in result.transitions:
                all_obs.append(obs)
                all_next_obs.append(next_obs)
                all_actions.append(action)

        idn_loss = 0.0
        if cfg.use_novelty and all_obs:
            obs_arr = np.stack(all_obs).astype(np.float32)
            next_obs_arr = np.stack(all_next_obs).astype(np.float32)
            actions_arr = np.array(all_actions, dtype=np.int64)
            idn_loss = idn.update(obs_arr, next_obs_arr, actions_arr, cfg.idn_updates_per_episode)

        return ActorStats(
            mean_augmented_fitness=float(fitnesses.mean()),
            mean_extrinsic_return=float(np.mean([r.extrinsic_return for r in results])),
            idn_loss=idn_loss,
        )

    # ------------------------------------------------------------------
    # Weight synchronisation
    # ------------------------------------------------------------------

    def sync_from_learner(self, params: np.ndarray) -> None:
        """Replace theta_base with the learner's current weights."""
        self.theta_base = params.copy()

    def get_base_params(self) -> np.ndarray:
        return self.theta_base.copy()
