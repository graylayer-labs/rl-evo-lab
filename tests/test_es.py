from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_evo_lab.actor.es_actor import ESActor, _rank_normalize
from rl_evo_lab.actor.es_worker import WorkerResult, run_worker_episode
from rl_evo_lab.buffer.replay_buffer import ReplayBuffer
from rl_evo_lab.intrinsic.episodic_novelty import EpisodicNovelty
from rl_evo_lab.intrinsic.inverse_dynamics import InverseDynamicsNetwork
from rl_evo_lab.learner.network import QNetwork
from rl_evo_lab.utils.config import EDERConfig

# ---------------------------------------------------------------------------
# 1. Rank normalisation
# ---------------------------------------------------------------------------


def test_rank_normalization():
    fitnesses = np.array([10.0, 5.0, 1.0, 8.0], dtype=np.float32)
    ranks = _rank_normalize(fitnesses)

    # Min must be -0.5, max must be +0.5
    assert pytest.approx(ranks.min(), abs=1e-6) == -0.5
    assert pytest.approx(ranks.max(), abs=1e-6) == 0.5

    # The rank order must match the fitness order:
    # fitness sorted ascending: [1, 5, 8, 10] → indices [2, 1, 3, 0]
    # so ranks[2] < ranks[1] < ranks[3] < ranks[0]
    assert ranks[2] < ranks[1] < ranks[3] < ranks[0]


# ---------------------------------------------------------------------------
# 2. Worker episode produces valid WorkerResult
# ---------------------------------------------------------------------------


def test_worker_episode_returns_transitions():
    cfg = EDERConfig(es_n_workers=2)
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    idn = InverseDynamicsNetwork(cfg, device)
    novelty = EpisodicNovelty(cfg.knn_k)

    # Build base params from a fresh network matching the config's hidden_dim
    net = QNetwork(cfg.obs_dim, cfg.act_dim, hidden=cfg.hidden_dim)
    base_params = net.get_flat_params()

    result = run_worker_episode(
        base_params=base_params,
        noise_seed=42,
        sigma=cfg.es_sigma,
        env=env,
        cfg=cfg,
        idn=idn,
        novelty=novelty,
        effective_beta=0.0,  # augmented == extrinsic when beta=0
        noise_sign=+1,
        device=device,
    )
    env.close()

    # Non-empty episode
    assert len(result.transitions) > 0

    # fitness is a plain Python float (or numpy scalar that behaves like one)
    assert isinstance(result.fitness, float)

    # With beta=0, augmented and extrinsic returns must be equal
    assert pytest.approx(result.fitness, abs=1e-5) == result.extrinsic_return

    # Noise vector must match the parameter count
    assert result.noise_vector.shape == base_params.shape

    # Transitions have the expected 5-tuple structure
    obs, action, reward, next_obs, done = result.transitions[0]
    assert obs.shape == (cfg.obs_dim,)
    assert next_obs.shape == (cfg.obs_dim,)
    assert isinstance(action, int)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


# ---------------------------------------------------------------------------
# 3. ESActor updates theta_base after one generation
# ---------------------------------------------------------------------------


def test_es_actor_updates_params():
    cfg = EDERConfig(
        es_n_workers=4,
        es_antithetic=True,  # 2 seed pairs → 4 workers
        idn_updates_per_episode=1,
    )
    device = torch.device("cpu")

    actor = ESActor(cfg, device)
    idn = InverseDynamicsNetwork(cfg, device)
    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.obs_dim)

    initial_params = actor.get_base_params().copy()

    env_fn = lambda: gym.make("CartPole-v1")
    stats = actor.run_generation(env_fn, idn, buffer, episode_num=0)

    updated_params = actor.get_base_params()

    # Parameters must have changed (ES update is virtually guaranteed to move theta)
    assert not np.allclose(initial_params, updated_params), (
        "theta_base did not change after one ES generation"
    )

    # Buffer must have received transitions
    assert len(buffer) > 0

    # Stats have plausible values
    assert isinstance(stats.mean_augmented_fitness, float)
    assert isinstance(stats.mean_extrinsic_return, float)
    assert isinstance(stats.idn_loss, float)


# ---------------------------------------------------------------------------
# 4. _select_workers_to_push selection logic
# ---------------------------------------------------------------------------


def _make_result(fitness: float, mean_novelty: float) -> WorkerResult:
    dummy_noise = np.zeros(1, dtype=np.float32)
    return WorkerResult(
        noise_vector=dummy_noise,
        noise_sign=+1,
        fitness=fitness,
        extrinsic_return=fitness,
        mean_novelty=mean_novelty,
    )


def test_select_workers_backward_compat():
    """buffer_push_alpha=None returns all indices unchanged."""
    cfg = EDERConfig(buffer_push_alpha=None)
    actor = ESActor(cfg, torch.device("cpu"))
    results = [_make_result(f, n) for f, n in [(10, 0.1), (5, 0.9), (1, 0.5)]]
    rank_weights = _rank_normalize(np.array([r.fitness for r in results]))
    selected = actor._select_workers_to_push(results, rank_weights)
    assert selected == [0, 1, 2]


def test_select_workers_top_k_filters():
    """With top_k=2 and alpha=1.0 (fitness only), only top-2 fitness workers + floor pass."""
    # Workers: high-fitness, mid-fitness, low-fitness/high-novelty, low-fitness/low-novelty
    results = [
        _make_result(fitness=100.0, mean_novelty=0.1),  # 0: high fitness, low novelty
        _make_result(fitness=80.0, mean_novelty=0.2),  # 1: mid fitness, low novelty
        _make_result(fitness=10.0, mean_novelty=0.95),  # 2: low fitness, HIGH novelty → floor
        _make_result(fitness=5.0, mean_novelty=0.05),  # 3: low fitness, low novelty → excluded
    ]
    cfg = EDERConfig(buffer_push_alpha=1.0, buffer_push_top_k=2, buffer_novelty_floor=0.25)
    actor = ESActor(cfg, torch.device("cpu"))
    rank_weights = _rank_normalize(np.array([r.fitness for r in results]))
    selected = actor._select_workers_to_push(results, rank_weights)

    # Top-2 by fitness: workers 0 and 1
    assert 0 in selected
    assert 1 in selected
    # High-novelty worker passes via floor (top 25% = 1 worker by novelty = worker 2)
    assert 2 in selected
    # Low-fitness/low-novelty worker is excluded
    assert 3 not in selected


def test_select_workers_novelty_floor_overrides_combined():
    """A worker with the highest novelty but lowest fitness passes via the floor,
    even when alpha=1.0 (pure fitness gate) and top_k excludes it by score."""
    results = [
        _make_result(fitness=100.0, mean_novelty=0.01),  # 0: best fitness, worst novelty
        _make_result(fitness=90.0, mean_novelty=0.02),  # 1: good fitness, low novelty
        _make_result(fitness=1.0, mean_novelty=0.99),  # 2: worst fitness, best novelty → floor
    ]
    cfg = EDERConfig(buffer_push_alpha=1.0, buffer_push_top_k=2, buffer_novelty_floor=0.33)
    actor = ESActor(cfg, torch.device("cpu"))
    rank_weights = _rank_normalize(np.array([r.fitness for r in results]))
    selected = actor._select_workers_to_push(results, rank_weights)

    assert 2 in selected  # floor override: top novelty always enters


def test_select_workers_balanced_alpha():
    """With alpha=0.5, a high-novelty/low-fitness worker can outscore a low-novelty/mid-fitness one."""
    results = [
        _make_result(fitness=50.0, mean_novelty=0.9),  # 0: mid fitness, high novelty
        _make_result(fitness=80.0, mean_novelty=0.1),  # 1: high fitness, low novelty
        _make_result(fitness=5.0, mean_novelty=0.05),  # 2: low fitness, low novelty → excluded
    ]
    cfg = EDERConfig(buffer_push_alpha=0.5, buffer_push_top_k=2, buffer_novelty_floor=0.0)
    actor = ESActor(cfg, torch.device("cpu"))
    rank_weights = _rank_normalize(np.array([r.fitness for r in results]))
    selected = actor._select_workers_to_push(results, rank_weights)

    # Worker 0 (high novelty) and 1 (high fitness) should both be in top-2 combined
    assert 0 in selected
    assert 1 in selected
    assert 2 not in selected


# ---------------------------------------------------------------------------
# 5. Seed collision detection under worker decay (B3)
# ---------------------------------------------------------------------------


def test_seed_collision_free_under_decay():
    """Seeds should be disjoint across episodes even when eff_n_workers decays.

    Bug B3: old code used seed = episode_num * eff_n_workers + k, where eff_n_workers
    decays per generation. This causes collisions across episodes: e.g., ep10×50workers=500
    and ep50×10workers=500 → same noise vectors reused → loss of diversity.
    New code uses a constant stride (cfg.es_n_workers), guaranteeing disjoint seed ranges
    per episode. Note: within an episode, antithetic pairs intentionally share seeds.
    """
    cfg = EDERConfig(
        es_n_workers=50,
        es_antithetic=True,
        novelty_solve_decay=True,
        solved_reward=475.0,
        novelty_decay_start_reward=400.0,
        es_workers_min=4,
    )
    device = torch.device("cpu")
    actor = ESActor(cfg, device)

    # Simulate a decaying schedule: collect job seeds across multiple episodes
    episode_seed_ranges = {}
    for episode_num in [0, 50, 100, 150, 200]:
        # Simulate convergence progress
        actor._learner_eval = 400.0 + (episode_num / 200.0) * (475.0 - 400.0)
        eff_n_workers = actor._effective_n_workers()
        jobs = actor._build_worker_jobs(episode_num, eff_n_workers)
        unique_seeds = sorted(set(seed for seed, _ in jobs))
        episode_seed_ranges[episode_num] = unique_seeds

    # Verify no seed collisions ACROSS EPISODES (disjoint ranges)
    all_episode_seeds = []
    for ep, seeds in episode_seed_ranges.items():
        all_episode_seeds.extend(seeds)

    # Check: all unique seeds across episodes should be unique
    # (allowing within-episode duplication for antithetic pairs)
    assert len(all_episode_seeds) == len(set(all_episode_seeds)), (
        f"Seed collision detected across episodes: "
        f"{[(ep, seeds) for ep, seeds in episode_seed_ranges.items()]}"
    )

    # Also verify the ranges are non-overlapping
    seed_ranges_list = list(episode_seed_ranges.values())
    for i, range_i in enumerate(seed_ranges_list):
        for j, range_j in enumerate(seed_ranges_list):
            if i < j:
                overlap = set(range_i) & set(range_j)
                assert not overlap, (
                    f"Episodes {list(episode_seed_ranges.keys())[i]} and "
                    f"{list(episode_seed_ranges.keys())[j]} have overlapping seed ranges: {overlap}"
                )


# ---------------------------------------------------------------------------
# 6. IDN baseline capture robustness (B5)
# ---------------------------------------------------------------------------


def test_idn_baseline_captured_after_warmup():
    """IDN loss baseline should be captured at/after warmup, on first episode with transitions.

    Bug B5: old code used == to gate the baseline capture on an exact episode, failing
    silently if that episode had zero transitions. New code uses >= so it captures on
    the first non-empty episode at/after the warmup boundary.
    """
    from rl_evo_lab.utils.config import make_config

    cfg = make_config("cartpole", novelty_warmup_episodes=3, es_n_workers=2)
    device = torch.device("cpu")
    actor = ESActor(cfg, device)
    idn = InverseDynamicsNetwork(cfg, device)
    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.obs_dim)

    env_fn = lambda: gym.make("CartPole-v1")

    # Run a generation at the boundary episode (episode 2 = warmup_episodes - 1)
    # This should trigger baseline capture (or at least be ready to capture on next non-empty episode)
    stats = actor.run_generation(env_fn, idn, buffer, episode_num=2)

    # After running generation 2 (at boundary), baseline should have been captured
    # (because CartPole episodes are rarely empty; we get transitions)
    if stats.idn_loss > 0.0:  # only check if IDN actually trained
        assert actor._idn_loss_init is not None, (
            "IDN baseline should be captured at/after warmup episode. "
            "Got _idn_loss_init=None."
        )

    # Continue to post-warmup: baseline should persist, not reset
    initial_baseline = actor._idn_loss_init
    stats2 = actor.run_generation(env_fn, idn, buffer, episode_num=3)
    assert actor._idn_loss_init == initial_baseline, (
        "Baseline should not change after first capture"
    )


def test_idn_beta_uses_baseline():
    """Effective beta should scale with IDN confidence once baseline is captured."""
    from rl_evo_lab.utils.config import make_config

    cfg = make_config(
        "cartpole",
        novelty_warmup_episodes=2,
        novelty_ramp_episodes=50,
        beta=0.1,
        es_n_workers=2,
    )
    device = torch.device("cpu")
    actor = ESActor(cfg, device)

    # Test at episode 3 (after warmup, during ramp phase)
    # At episode 3: ramp = (3-2)/50 = 0.02 → should see non-zero beta

    # With good IDN (loss < baseline): confidence > 0 → beta should be scaled
    actor._idn_loss_init = 0.5
    actor._idn_loss_ema = 0.3  # better than baseline
    beta_with_good_idn = actor._effective_beta(3)

    # With bad IDN (loss > baseline): confidence → 0 → beta should be suppressed
    actor._idn_loss_ema = 1.0  # worse than baseline
    beta_with_bad_idn = actor._effective_beta(3)

    # Good IDN should produce higher beta than bad IDN
    assert beta_with_good_idn > beta_with_bad_idn, (
        f"Good IDN (loss 0.3) should produce higher beta than bad IDN (loss 1.0). "
        f"Got {beta_with_good_idn} vs {beta_with_bad_idn}"
    )
    # Both should be positive during ramp (if baseline is set)
    assert beta_with_good_idn > 0.0, (
        f"With baseline set and ramp phase, beta should be > 0. Got {beta_with_good_idn}"
    )
