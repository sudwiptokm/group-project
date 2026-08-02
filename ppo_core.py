"""Pure, SUMO-free PPO math for the corridor IPPO/MAPPO agents.

ActorCritic keeps SEPARATE actor and critic networks so the critic's input
(`state`) can widen to a joint state in SP3 (MAPPO) without touching the actor.
For IPPO, callers pass the agent's local observation as both obs and state.
Kept dependency-light (torch only) and free of any SUMO/env code so the math is
unit-tested in isolation.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp(sizes, activation=nn.Tanh) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """One network shared across all (homogeneous) agents.

    actor:  local obs  -> action logits
    critic: state      -> scalar value   (state == local obs for IPPO)
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 state_dim: Optional[int] = None, hidden=(256, 256)):
        super().__init__()
        state_dim = obs_dim if state_dim is None else state_dim
        self.actor = _mlp([obs_dim, *hidden, act_dim])
        self.critic = _mlp([state_dim, *hidden, 1])

    def policy(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(obs))

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.policy(obs)
        action = dist.sample()
        return action, dist.log_prob(action)
