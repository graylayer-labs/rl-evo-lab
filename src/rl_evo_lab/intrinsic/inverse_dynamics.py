from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_evo_lab.utils.config import EDERConfig


class InverseDynamicsNetwork(nn.Module):
    def __init__(self, cfg: EDERConfig, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(2 * cfg.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, cfg.embed_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(cfg.embed_dim, cfg.act_dim)
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=cfg.idn_lr)

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, next_obs], dim=-1)
        embedding = self.encoder(x)
        logits = self.head(embedding)
        return logits, embedding

    def embed(self, obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            o = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            no = torch.from_numpy(next_obs).float().unsqueeze(0).to(self.device)
            _, emb = self.forward(o, no)
        return emb.squeeze(0).cpu().numpy()

    def update(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        actions: np.ndarray,
        n_steps: int,
    ) -> float:
        """Train on a batch of transitions for n_steps gradient steps."""
        obs_t = torch.from_numpy(obs).float().to(self.device)
        next_obs_t = torch.from_numpy(next_obs).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)

        total_loss = 0.0
        for _ in range(n_steps):
            logits, _ = self.forward(obs_t, next_obs_t)
            loss = nn.functional.cross_entropy(logits, actions_t)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(n_steps, 1)
