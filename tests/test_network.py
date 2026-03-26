import numpy as np
import pytest
from rl_evo_lab.learner.network import QNetwork


def test_flat_param_roundtrip():
    net = QNetwork(obs_dim=4, act_dim=2)
    original = net.get_flat_params().copy()
    net.set_flat_params(original * 2)
    assert not np.allclose(net.get_flat_params(), original)
    net.set_flat_params(original)
    assert np.allclose(net.get_flat_params(), original)


def test_n_params():
    net = QNetwork(obs_dim=4, act_dim=2)
    assert net.n_params > 0
    assert net.n_params == len(net.get_flat_params())
