from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Shared MLP architecture for DQN learner and ES actor."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([p.data.cpu().numpy().ravel() for p in self.parameters()])

    def set_flat_params(self, params: np.ndarray) -> None:
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(params[idx : idx + n].reshape(p.shape)))
            idx += n

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
