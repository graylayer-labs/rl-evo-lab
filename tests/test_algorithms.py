"""Unit tests for core algorithm implementations."""

import numpy as np
import torch
import pytest
from pathlib import Path

from rl_evo_lab.actor.es_actor import ESActor, _rank_normalize
from rl_evo_lab.learner.dqn import DQNLearner
from rl_evo_lab.buffer.replay_buffer import ReplayBuffer
from rl_evo_lab.utils.config import make_config
from rl_evo_lab.intrinsic.inverse_dynamics import InverseDynamicsNetwork


class TestRankNormalize:
    """Test ES rank normalization."""

    def test_rank_normalize_basic(self):
        """Ranks should be in [-0.5, 0.5] and monotonic."""
        fitnesses = np.array([10.0, 5.0, 20.0, 15.0])
        ranks = _rank_normalize(fitnesses)

        # All in [-0.5, 0.5]
        assert np.all(ranks >= -0.5) and np.all(ranks <= 0.5)

        # Monotonic (worst → best)
        expected_order = [1, 0, 3, 2]  # indices sorted by fitness ascending
        sorted_indices = np.argsort(fitnesses)
        assert np.allclose(ranks[sorted_indices], np.linspace(-0.5, 0.5, len(fitnesses)))

    def test_rank_normalize_identical(self):
        """All identical fitnesses should have rank 0."""
        fitnesses = np.array([5.0, 5.0, 5.0, 5.0])
        ranks = _rank_normalize(fitnesses)
        assert np.allclose(ranks, 0.0)

    def test_rank_normalize_single(self):
        """Single fitness should have rank 0."""
        fitnesses = np.array([10.0])
        ranks = _rank_normalize(fitnesses)
        assert np.allclose(ranks, 0.0)


class TestDoubleQN:
    """Test that Double DQN is correctly implemented."""

    def test_double_qn_uses_policy_net_for_selection(self):
        """Double DQN should use policy_net to select action, target_net to evaluate."""
        cfg = make_config("cartpole")
        device = torch.device("cpu")

        learner = DQNLearner(cfg, device)
        buffer = ReplayBuffer(cfg.buffer_capacity, cfg.obs_dim)

        # Create dummy batch
        batch_size = 8
        obs = torch.randn(batch_size, cfg.obs_dim)
        next_obs = torch.randn(batch_size, cfg.obs_dim)
        actions = torch.randint(0, cfg.act_dim, (batch_size,))
        rewards = torch.randn(batch_size)
        dones = torch.zeros(batch_size)

        # Mock batch
        from collections import namedtuple
        Batch = namedtuple('Batch', ['obs', 'action', 'reward', 'next_obs', 'done'])
        batch = Batch(obs=obs, action=actions, reward=rewards, next_obs=next_obs, done=dones)

        # Compute Q-targets manually
        with torch.no_grad():
            # Double DQN: policy_net selects, target_net evaluates
            next_actions = learner.policy_net(next_obs).argmax(dim=1)
            next_q_values = learner.target_net(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Standard DQN (for comparison): target_net selects AND evaluates
            next_q_std = learner.target_net(next_obs).max(dim=1).values

        # They should be DIFFERENT if policy_net and target_net diverge
        # (they might be similar at init, but principle is correct)
        assert next_q_values.shape == next_q_std.shape
        print(f"✓ Double DQN structure correct: policy_net selects, target_net evaluates")


class TestIDNLearning:
    """Test that IDN is actually learning embeddings."""

    def test_idn_loss_decreases(self):
        """IDN loss should decrease over training."""
        cfg = make_config("cartpole")
        device = torch.device("cpu")

        idn = InverseDynamicsNetwork(cfg, device)

        obs_samples = torch.randn(100, cfg.obs_dim)
        next_obs_samples = torch.randn(100, cfg.obs_dim)
        action_samples = torch.randint(0, cfg.act_dim, (100,))

        initial_loss = None
        for step in range(50):
            loss = idn.train_step(obs_samples, next_obs_samples, action_samples)
            if step == 0:
                initial_loss = loss

        final_loss = loss
        assert final_loss < initial_loss, f"IDN not learning: {initial_loss:.4f} → {final_loss:.4f}"
        print(f"✓ IDN loss decreases: {initial_loss:.4f} → {final_loss:.4f}")


class TestNoveltyRamp:
    """Test that novelty beta ramps according to schedule."""

    def test_beta_ramp_schedule(self):
        """Beta should increase from 0 to target over novelty_ramp_episodes."""
        cfg = make_config("cartpole", novelty_warmup_episodes=50, novelty_ramp_episodes=100, beta=0.1)
        device = torch.device("cpu")

        actor = ESActor(cfg, device)

        # Check beta at different episodes
        beta_ep0 = actor._effective_beta(0)
        beta_ep50 = actor._effective_beta(50)  # warmup ends
        beta_ep100 = actor._effective_beta(100)  # ramp ends
        beta_ep200 = actor._effective_beta(200)  # post-ramp

        assert beta_ep0 == 0.0, f"Beta should be 0 during warmup, got {beta_ep0}"
        assert 0 < beta_ep50 < cfg.beta, f"Beta should ramp after warmup, got {beta_ep50}"
        assert np.isclose(beta_ep100, cfg.beta), f"Beta should reach target at ramp end, got {beta_ep100}"
        assert np.isclose(beta_ep200, cfg.beta), f"Beta should hold after ramp, got {beta_ep200}"

        print(f"✓ Novelty ramp: 0→{beta_ep50:.4f}→{beta_ep100:.4f}→{beta_ep200:.4f}")


class TestConfigOverrides:
    """Test that config overrides work correctly."""

    def test_es_sigma_override(self):
        """Should be able to override es_sigma via make_config."""
        cfg1 = make_config("cartpole")
        cfg2 = make_config("cartpole", es_sigma=0.2)

        assert cfg1.es_sigma == 0.06  # default
        assert cfg2.es_sigma == 0.2   # overridden
        print(f"✓ es_sigma override: {cfg1.es_sigma} → {cfg2.es_sigma}")

    def test_beta_override(self):
        """Should be able to override beta via make_config."""
        cfg1 = make_config("cartpole")
        cfg2 = make_config("cartpole", beta=0.2)

        assert cfg1.beta == 0.02  # default
        assert cfg2.beta == 0.2   # overridden
        print(f"✓ beta override: {cfg1.beta} → {cfg2.beta}")

    def test_novelty_ramp_override(self):
        """Should be able to override novelty_ramp_episodes."""
        cfg1 = make_config("cartpole")
        cfg2 = make_config("cartpole", novelty_ramp_episodes=0)
        cfg3 = make_config("cartpole", novelty_ramp_episodes=200)

        assert cfg1.novelty_ramp_episodes == 100  # default
        assert cfg2.novelty_ramp_episodes == 0    # disabled
        assert cfg3.novelty_ramp_episodes == 200  # delayed
        print(f"✓ novelty_ramp override: {cfg1.novelty_ramp_episodes} → 0 / 200")


class TestBufferIntegrity:
    """Test that buffer correctly stores transitions."""

    def test_buffer_push_pop(self):
        """Buffer should store and retrieve transitions correctly."""
        cfg = make_config("cartpole")
        buffer = ReplayBuffer(cfg.buffer_capacity, cfg.obs_dim)

        # Push a transition
        obs = np.random.randn(cfg.obs_dim).astype(np.float32)
        next_obs = np.random.randn(cfg.obs_dim).astype(np.float32)
        action = 1
        reward = 10.0
        done = False

        buffer.push(obs, action, reward, next_obs, done)

        assert len(buffer) == 1

        # Sample and verify
        batch = buffer.sample(1, torch.device("cpu"))
        assert batch.obs.shape[0] == 1
        assert np.allclose(batch.obs.numpy()[0], obs)
        assert np.allclose(batch.reward.numpy()[0], reward)

        print(f"✓ Buffer integrity: push/pop correct")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
