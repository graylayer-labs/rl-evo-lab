from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    obs: torch.Tensor        # (batch, obs_dim)
    action: torch.Tensor     # (batch,) long
    reward: torch.Tensor     # (batch,)
    next_obs: torch.Tensor   # (batch, obs_dim)
    done: torch.Tensor       # (batch,) float


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int) -> None:
        self._capacity = capacity
        self._ptr = 0
        self._size = 0

        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros(capacity, dtype=np.int64)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._done = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self._obs[self._ptr] = obs
        self._action[self._ptr] = action
        self._reward[self._ptr] = reward
        self._next_obs[self._ptr] = next_obs
        self._done[self._ptr] = float(done)
        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, device: torch.device) -> Transition:
        idx = np.random.randint(0, self._size, size=batch_size)
        return Transition(
            obs=torch.from_numpy(self._obs[idx]).to(device),
            action=torch.from_numpy(self._action[idx]).to(device),
            reward=torch.from_numpy(self._reward[idx]).to(device),
            next_obs=torch.from_numpy(self._next_obs[idx]).to(device),
            done=torch.from_numpy(self._done[idx]).to(device),
        )

    def diversity_metric(self, subsample: int = 256) -> float:
        """Mean pairwise L2 distance over a random subsample of stored observations."""
        n = min(self._size, subsample)
        if n < 2:
            return 0.0
        idx = np.random.choice(self._size, size=n, replace=False)
        obs = torch.from_numpy(self._obs[idx])  # (n, obs_dim)
        dists = torch.cdist(obs, obs)  # (n, n)
        # upper triangle only, excluding diagonal
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        return dists[mask].mean().item()

    def __len__(self) -> int:
        return self._size
