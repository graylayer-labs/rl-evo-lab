from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from rl_evo_lab.intrinsic.episodic_novelty import EpisodicNovelty
from rl_evo_lab.intrinsic.inverse_dynamics import InverseDynamicsNetwork
from rl_evo_lab.learner.network import QNetwork
from rl_evo_lab.utils.config import EDERConfig


@dataclass
class WorkerResult:
    noise_vector: np.ndarray
    noise_sign: int  # +1 or -1
    fitness: float  # sum of augmented rewards
    extrinsic_return: float  # sum of extrinsic rewards
    transitions: list[tuple] = field(default_factory=list)
    # Each tuple: (obs, action, reward_extrinsic, next_obs, done)


def run_worker_episode(
    base_params: np.ndarray,
    noise_seed: int,
    sigma: float,
    env,
    cfg: EDERConfig,
    idn: InverseDynamicsNetwork,
    novelty: EpisodicNovelty,
    beta: float,
    noise_sign: int,
    device: torch.device,
) -> WorkerResult:
    """Run a single ES worker episode with perturbed parameters.

    The noise_vector is derived from np.random.RandomState(noise_seed) so it is
    reproducible given the seed without transmitting the full vector.
    """
    # Generate noise vector from seed (reproducible)
    rng = np.random.RandomState(noise_seed)
    noise_vector = rng.randn(len(base_params)).astype(np.float32)

    # Perturb parameters
    worker_params = base_params + noise_sign * sigma * noise_vector

    # Load into a fresh QNetwork (CPU; no gradients needed)
    net = QNetwork(cfg.obs_dim, cfg.act_dim)
    net.set_flat_params(worker_params)
    net.eval()

    # Reset episodic novelty
    novelty.reset()

    obs, _ = env.reset()
    done = False

    fitness = 0.0
    extrinsic_return = 0.0
    transitions: list[tuple] = []

    while not done:
        obs_t = torch.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            action = net(obs_t).argmax(dim=1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        obs_arr = np.array(obs, dtype=np.float32)
        next_obs_arr = np.array(next_obs, dtype=np.float32)

        if cfg.use_novelty:
            embedding = idn.embed(obs_arr, next_obs_arr)
            intrinsic_reward = novelty.score(embedding)
        else:
            intrinsic_reward = 0.0

        augmented_reward = float(reward) + beta * intrinsic_reward
        fitness += augmented_reward
        extrinsic_return += float(reward)

        # Only extrinsic reward goes into the transition / replay buffer
        transitions.append((obs_arr, int(action), float(reward), next_obs_arr, bool(done)))

        obs = next_obs

    return WorkerResult(
        noise_vector=noise_vector,
        noise_sign=noise_sign,
        fitness=fitness,
        extrinsic_return=extrinsic_return,
        transitions=transitions,
    )
