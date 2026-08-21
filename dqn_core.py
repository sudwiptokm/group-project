"""Pure, SUMO-free DQN math for the corridor's independent per-agent learners.

Unlike ppo_core.py's ActorCritic (one network shared across all agents), SP5's
IDQN gives each corridor signal its own QNetwork/ReplayBuffer/target network.
This module builds the pieces one agent needs; train_corridor_dqn.py
instantiates three independent copies, one per ts_id. Kept dependency-light
(torch/numpy only) and free of any SUMO/env code so the math is unit-tested in
isolation, same discipline as ppo_core.py.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(sizes, activation=nn.ReLU) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """obs -> one Q-value per discrete action. One instance per agent."""

    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64)):
        super().__init__()
        self.net = _mlp([obs_dim, *hidden, act_dim])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ReplayBuffer:
    """Fixed-size, uniform-sampling replay buffer. One instance per agent --
    IDQN agents never share experience, unlike IPPO's pooled rollout buffer."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._act = np.zeros(capacity, dtype=np.int64)
        self._rew = np.zeros(capacity, dtype=np.float32)
        self._done = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._pos

    def add(self, obs, action: int, reward: float, next_obs, done: float) -> None:
        self._obs[self._pos] = obs
        self._next_obs[self._pos] = next_obs
        self._act[self._pos] = action
        self._rew[self._pos] = reward
        self._done[self._pos] = done
        self._pos += 1
        if self._pos == self.capacity:
            self._pos = 0
            self._full = True

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self), size=batch_size)
        return (
            torch.as_tensor(self._obs[idx]),
            torch.as_tensor(self._act[idx]),
            torch.as_tensor(self._rew[idx]),
            torch.as_tensor(self._next_obs[idx]),
            torch.as_tensor(self._done[idx]),
        )


class EpsilonSchedule:
    """Linear decay from 1.0 to exploration_final_eps over
    exploration_fraction * total_steps steps, then flat.

    Matches SB3 DQN's own schedule (stable_baselines3.common.utils.get_linear_fn
    applied over training progress): at elapsed fraction f = step/total_steps,
    eps = 1.0 if f <= 0, exploration_final_eps if f >= exploration_fraction,
    else linearly interpolated between them.
    """

    def __init__(self, total_steps: int, exploration_fraction: float,
                 exploration_final_eps: float):
        self.decay_steps = max(1, int(exploration_fraction * total_steps))
        self.final_eps = exploration_final_eps

    def value(self, step: int) -> float:
        if step >= self.decay_steps:
            return self.final_eps
        frac = step / self.decay_steps
        return 1.0 + frac * (self.final_eps - 1.0)


def dqn_loss(q_network: QNetwork, target_network: QNetwork, obs: torch.Tensor,
             actions: torch.Tensor, rewards: torch.Tensor, next_obs: torch.Tensor,
             dones: torch.Tensor, gamma: float) -> Tuple[torch.Tensor, dict]:
    """Standard DQN TD loss: smooth-L1 (Huber, beta=1 -- SB3 DQN's default)
    between Q(s,a) and r + gamma * (1 - done) * max_a' Q_target(s', a').

    actions must be int64; the terminal mask (dones) zeroes the bootstrap term
    so a terminal transition's target is the reward alone, matching
    ppo_core.compute_gae's non_terminal masking of its own bootstrap.
    """
    q_values = q_network(obs).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_q = target_network(next_obs).max(dim=1).values
        target = rewards + gamma * (1.0 - dones) * next_q
    loss = F.smooth_l1_loss(q_values, target)
    return loss, {"q_mean": q_values.mean().item(), "target_mean": target.mean().item()}
