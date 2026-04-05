import numpy as np
import torch

from rl_evo_lab.buffer.replay_buffer import ReplayBuffer


def test_push_and_sample():
    buf = ReplayBuffer(capacity=100, obs_dim=4)
    for i in range(10):
        buf.push(np.ones(4) * i, i % 2, float(i), np.ones(4) * (i + 1), i == 9)
    assert len(buf) == 10
    batch = buf.sample(5, torch.device("cpu"))
    assert batch.obs.shape == (5, 4)
    assert batch.action.shape == (5,)


def test_overflow():
    buf = ReplayBuffer(capacity=5, obs_dim=4)
    for i in range(10):
        buf.push(np.ones(4) * i, 0, 1.0, np.ones(4) * (i + 1), False)
    assert len(buf) == 5


def test_diversity_metric():
    buf = ReplayBuffer(capacity=100, obs_dim=4)
    for i in range(50):
        buf.push(np.random.randn(4), 0, 1.0, np.random.randn(4), False)
    d = buf.diversity_metric()
    assert isinstance(d, float)
    assert d > 0.0
