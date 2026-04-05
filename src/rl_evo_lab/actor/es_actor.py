from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    total_env_steps: int  # sum of all worker episode lengths this generation
    effective_beta: float = 0.0


def _rank_normalize(fitnesses: np.ndarray) -> np.ndarray:
    """Rank fitnesses and normalise to [-0.5, 0.5].

    The lowest-fitness worker receives -0.5, the highest +0.5.
    Ties share the same rank (dense rank).
    """
    n = len(fitnesses)
    # argsort gives indices that would sort ascending; assign ranks 0..n-1
    ranks = np.empty(n, dtype=np.float32)
    order = np.argsort(fitnesses)  # ascending: worst → best
    ranks[order] = np.arange(n, dtype=np.float32)
    if n > 1:
        ranks = ranks / (n - 1) - 0.5  # map [0, n-1] → [-0.5, 0.5]
    else:
        ranks[:] = 0.0
    return ranks


class ESActor:
    """Evolution Strategy actor that fills the replay buffer with diverse experience."""

    def __init__(self, cfg: EDERConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

        # Initialise theta_base from a fresh QNetwork
        _init_net = QNetwork(cfg.obs_dim, cfg.act_dim, hidden=cfg.hidden_dim)
        self.theta_base: np.ndarray = _init_net.get_flat_params().copy()

        # IDN loss EMA — used to gauge embedding quality for adaptive beta
        self._idn_loss_ema: float = 1.0  # start high (worst case)
        self._idn_loss_init: float | None = None  # recorded at end of warmup

        # Learner eval tracking — used to decay beta once the learner converges
        self._learner_eval: float = -float("inf")

        # Cross-generation global novelty buffer (None = disabled)
        cap = cfg.global_novelty_capacity if cfg.use_novelty else 0
        self._global_novelty: EpisodicNovelty | None = (
            EpisodicNovelty(cfg.knn_k, capacity=cap) if cap > 0 else None
        )

    def _convergence_progress(self) -> float:
        """How far the learner has converged toward solved_reward.

        Returns 0.0 while learner_eval < novelty_decay_start_reward (not yet converging),
        rising linearly to 1.0 when learner_eval >= solved_reward (fully solved).

        This single signal drives all three convergence decays — beta, sigma, n_workers —
        so they always move in lockstep from the same threshold.
        """
        cfg = self.cfg
        if not cfg.novelty_solve_decay:
            return 0.0
        if self._learner_eval < cfg.novelty_decay_start_reward:
            return 0.0
        span = abs(cfg.solved_reward - cfg.novelty_decay_start_reward)
        if span < 1e-8:
            return 1.0
        return min(1.0, abs(self._learner_eval - cfg.novelty_decay_start_reward) / span)

    def _effective_beta(self, episode: int) -> float:
        """Novelty weight for this episode.

        Phase 1 — warmup:     beta = 0, IDN trains silently.
        Phase 2 — ramp:       beta linearly increases to target.
        Phase 3 — confident:  beta scaled by IDN confidence (loss relative to warmup baseline).
        Phase 4 — converging: beta decays to 0 via _convergence_progress().
        """
        cfg = self.cfg
        if not cfg.use_novelty or episode < cfg.novelty_warmup_episodes:
            return 0.0

        ramp = min(1.0, (episode - cfg.novelty_warmup_episodes) / max(cfg.novelty_ramp_episodes, 1))

        if self._idn_loss_init is not None and self._idn_loss_init > 1e-8:
            raw_confidence = max(0.0, 1.0 - self._idn_loss_ema / self._idn_loss_init)
        else:
            raw_confidence = 1.0
        confidence = max(cfg.novelty_beta_floor, raw_confidence)

        return cfg.beta * ramp * confidence * (1.0 - self._convergence_progress())

    def _effective_sigma(self) -> float:
        """ES noise std dev for this generation.

        Decays from es_sigma → es_sigma_min as the learner converges.
        Smaller perturbations when converged = less noisy buffer data per worker,
        while still maintaining a minimal exploration footprint.
        """
        progress = self._convergence_progress()
        return self.cfg.es_sigma + progress * (self.cfg.es_sigma_min - self.cfg.es_sigma)

    def _effective_n_workers(self) -> int:
        """Number of ES workers for this generation.

        Decays from es_n_workers → es_workers_min as the learner converges.
        Fewer workers when converged = fewer noisy transitions pushed to the buffer
        per training episode, reducing the rate of buffer pollution.
        Always returns an even number when antithetic sampling is enabled.
        """
        progress = self._convergence_progress()
        n = round(
            self.cfg.es_n_workers + progress * (self.cfg.es_workers_min - self.cfg.es_n_workers)
        )
        n = max(self.cfg.es_workers_min, n)
        if self.cfg.es_antithetic and n % 2 != 0:
            n = max(self.cfg.es_workers_min, n - 1)
        return n

    def update_learner_eval(self, reward: float) -> None:
        """Notify the actor of the learner's latest eval reward.

        Called from train.py after each evaluation so convergence-decay methods
        have an up-to-date signal to work from.
        """
        self._learner_eval = reward

    # ------------------------------------------------------------------
    # Buffer push filtering
    # ------------------------------------------------------------------

    def _select_workers_to_push(
        self,
        results: list[WorkerResult],
        rank_weights: np.ndarray,
    ) -> list[int]:
        """Return indices of workers whose transitions should be pushed to the buffer.

        If buffer_push_alpha is None, returns all indices (backward compatible).
        Otherwise scores each worker episode by:
            push_score = alpha * fitness_rank + (1-alpha) * novelty_rank
        and selects top-K by combined score, plus a novelty floor override.
        """
        cfg = self.cfg
        if cfg.buffer_push_alpha is None:
            return list(range(len(results)))

        n = len(results)
        alpha = cfg.buffer_push_alpha

        # Fitness rank [0,1]: remap existing [-0.5, 0.5] ranks
        fitness_ranks = rank_weights + 0.5

        # Novelty rank [0,1]: rank-normalise mean_novelty scores across workers
        novelty_scores = np.array([r.mean_novelty for r in results], dtype=np.float32)
        novelty_ranks = _rank_normalize(novelty_scores) + 0.5

        # Combined score
        combined = alpha * fitness_ranks + (1.0 - alpha) * novelty_ranks

        # Novelty floor: top buffer_novelty_floor fraction always enters
        n_floor = max(1, int(n * cfg.buffer_novelty_floor))
        floor_idx = set(np.argsort(novelty_scores)[-n_floor:].tolist())

        # Top-K by combined score (default: all workers)
        top_k = cfg.buffer_push_top_k if cfg.buffer_push_top_k is not None else n
        top_k_idx = set(np.argsort(combined)[-top_k:].tolist())

        return sorted(top_k_idx | floor_idx)

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

        # ---- Compute convergence-decayed quantities for this episode --
        eff_beta = self._effective_beta(episode_num)
        eff_sigma = self._effective_sigma()
        eff_n_workers = self._effective_n_workers()

        # ---- Build list of (seed, sign) pairs -------------------------
        worker_jobs: list[tuple[int, int]] = []
        if cfg.es_antithetic:
            n_seeds = eff_n_workers // 2
            for k in range(n_seeds):
                seed = episode_num * eff_n_workers + k
                worker_jobs.append((seed, +1))
                worker_jobs.append((seed, -1))
            if eff_n_workers % 2 != 0:
                seed = episode_num * eff_n_workers + n_seeds
                worker_jobs.append((seed, +1))
        else:
            for k in range(eff_n_workers):
                seed = episode_num * eff_n_workers + k
                worker_jobs.append((seed, +1))

        # ---- Run worker episodes in parallel (threads) ----------------
        # Workers are read-only w.r.t. IDN and global_novelty during collection;
        # each has its own env and episodic novelty — no shared mutable state.
        # torch.no_grad() + numpy release the GIL so threads get real concurrency.
        def _run_one(job: tuple[int, int]) -> WorkerResult:
            seed, sign = job
            env = env_fn()
            novelty = EpisodicNovelty(cfg.knn_k)
            result = run_worker_episode(
                base_params=self.theta_base,
                noise_seed=seed,
                sigma=eff_sigma,
                env=env,
                cfg=cfg,
                idn=idn,
                novelty=novelty,
                effective_beta=eff_beta,
                noise_sign=sign,
                device=self.device,
                global_novelty=self._global_novelty,
            )
            env.close()
            return result

        with ThreadPoolExecutor(max_workers=len(worker_jobs)) as pool:
            # Collect in submission order for reproducibility
            results: list[WorkerResult] = list(pool.map(_run_one, worker_jobs))

        # ---- Rank-normalise fitness scores ---------------------------
        fitnesses = np.array([r.fitness for r in results], dtype=np.float32)
        rank_weights = _rank_normalize(fitnesses)

        # ---- ES gradient update -------------------------------------
        # delta = Σ (rank_w_i * sign_i * noise_i) / (N * sigma)
        n_workers_actual = len(results)
        delta = np.zeros_like(self.theta_base)
        for i, result in enumerate(results):
            delta += rank_weights[i] * result.noise_sign * result.noise_vector

        delta /= n_workers_actual * eff_sigma
        self.theta_base = (
            self.theta_base + cfg.es_lr * delta - cfg.es_weight_decay * self.theta_base
        )

        # ---- Push selected transitions to buffer ---------------------
        push_indices = self._select_workers_to_push(results, rank_weights)
        for i in push_indices:
            for obs, action, reward, next_obs, done in results[i].transitions:
                buffer.push(obs, action, reward, next_obs, done)

        # ---- Train IDN on collected transitions ----------------------
        all_obs, all_next_obs, all_actions = [], [], []
        for result in results:
            for obs, action, _, next_obs, _ in result.transitions:
                all_obs.append(obs)
                all_next_obs.append(next_obs)
                all_actions.append(action)

        idn_loss = 0.0
        if cfg.use_novelty and all_actions:
            obs_arr = np.stack(all_obs).astype(np.float32)
            next_obs_arr = np.stack(all_next_obs).astype(np.float32)
            actions_arr = np.array(all_actions, dtype=np.int64)
            idn_loss = idn.update(obs_arr, next_obs_arr, actions_arr, cfg.idn_updates_per_episode)

            # Update IDN loss EMA (α=0.05 for slow tracking)
            self._idn_loss_ema = 0.95 * self._idn_loss_ema + 0.05 * idn_loss
            # Record baseline IDN loss at the end of warmup
            if self._idn_loss_init is None and episode_num == cfg.novelty_warmup_episodes - 1:
                self._idn_loss_init = self._idn_loss_ema

            # Add this generation's embeddings to the global novelty buffer.
            # Workers cached their embeddings in WorkerResult so we reuse them —
            # no recomputation needed. Workers only query() during the episode;
            # we add() here after all workers finish so within-generation ordering
            # doesn't affect scores.
            if self._global_novelty is not None:
                for result in results:
                    for emb in result.embeddings:
                        self._global_novelty.add(emb)

        return ActorStats(
            mean_augmented_fitness=float(fitnesses.mean()),
            mean_extrinsic_return=float(np.mean([r.extrinsic_return for r in results])),
            idn_loss=idn_loss,
            total_env_steps=sum(len(r.transitions) for r in results),
            effective_beta=eff_beta,
        )

    # ------------------------------------------------------------------
    # Weight synchronisation
    # ------------------------------------------------------------------

    def sync_from_learner(self, params: np.ndarray) -> None:
        """Replace theta_base with the learner's current weights."""
        self.theta_base = params.copy()

    def get_base_params(self) -> np.ndarray:
        return self.theta_base.copy()
