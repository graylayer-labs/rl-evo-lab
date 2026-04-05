import numpy as np
import torch

from rl_evo_lab.intrinsic.episodic_novelty import EpisodicNovelty
from rl_evo_lab.intrinsic.inverse_dynamics import InverseDynamicsNetwork
from rl_evo_lab.utils.config import EDERConfig


def test_episodic_novelty_resets():
    nov = EpisodicNovelty(k=3)
    emb = np.random.randn(64)
    for _ in range(5):
        nov.score(emb)
    nov.reset()
    assert nov.score(emb) == 0.0  # after reset, memory is empty


def test_episodic_novelty_increases():
    nov = EpisodicNovelty(k=2)
    # fill memory with fixed embeddings
    base = np.zeros(8)
    for _ in range(3):
        nov.score(base)
    # a very different embedding should get high novelty
    novel = np.ones(8) * 100.0
    score = nov.score(novel)
    assert score > 0.0


def test_idn_output_shapes():
    cfg = EDERConfig()
    device = torch.device("cpu")
    idn = InverseDynamicsNetwork(cfg, device)
    obs = np.random.randn(4).astype(np.float32)
    next_obs = np.random.randn(4).astype(np.float32)
    emb = idn.embed(obs, next_obs)
    assert emb.shape == (cfg.embed_dim,)
