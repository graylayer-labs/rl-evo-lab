from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_evo_lab.buffer.replay_buffer import ReplayBuffer
from rl_evo_lab.learner.network import QNetwork
from rl_evo_lab.utils.config import EDERConfig


class DQNLearner:
    def __init__(self, cfg: EDERConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.policy_net = QNetwork(cfg.obs_dim, cfg.act_dim, hidden=cfg.hidden_dim).to(device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.dqn_lr)
        self._step = 0

    def train_step(self, buffer: ReplayBuffer) -> float:
        batch = buffer.sample(self.cfg.batch_size, self.device)

        with torch.no_grad():
            next_q = self.target_net(batch.next_obs).max(dim=1).values
            target = batch.reward + self.cfg.gamma * next_q * (1.0 - batch.done)

        current_q = self.policy_net(batch.obs).gather(1, batch.action.unsqueeze(1)).squeeze(1)
        loss = nn.functional.huber_loss(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        self._step += 1
        if self._step % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def evaluate(self, env: gym.Env, n_episodes: int) -> float:
        self.policy_net.eval()
        total = 0.0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    action = self.policy_net(obs_t).argmax(dim=1).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total += reward
        self.policy_net.train()
        return total / n_episodes

    def collect_episode(self, env: gym.Env, buffer: ReplayBuffer, episode: int) -> tuple[float, int]:
        """Run one ε-greedy episode and push transitions to buffer.

        Returns (total_extrinsic_return, n_steps).
        """
        decay = self.cfg.epsilon_decay_episodes
        frac = min(1.0, episode / max(decay, 1))
        epsilon = self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        n_steps = 0
        self.policy_net.eval()
        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                    action = self.policy_net(obs_t).argmax(dim=1).item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, float(reward), next_obs, float(done))
            obs = next_obs
            total_reward += float(reward)
            n_steps += 1
        self.policy_net.train()
        return total_reward, n_steps

    def get_weights(self) -> np.ndarray:
        return self.policy_net.get_flat_params()

    def load_weights(self, params: np.ndarray) -> None:
        self.policy_net.set_flat_params(params)
