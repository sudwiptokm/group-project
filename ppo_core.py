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


def compute_gae(rewards, values, dones, gamma: float, lam: float,
                last_value: float = 0.0):
    """Generalised advantage estimation over one rollout segment.

    rewards/values/dones are equal-length sequences; values[t] = V(s_t);
    dones[t] = 1.0 if s_{t+1} is terminal. last_value bootstraps V(s_T) for the
    final non-terminal step. Returns (advantages, returns) as plain float lists.
    """
    n = len(rewards)
    advantages = [0.0] * n
    gae = 0.0
    next_value = last_value
    for t in reversed(range(n)):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        advantages[t] = gae
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def ppo_loss(dist, actions, old_log_prob, advantages, values, returns,
             clip: float, ent_coef: float, vf_coef: float = 0.5):
    """PPO clipped surrogate + value loss - entropy bonus.

    dist: current Categorical over `actions`. Returns (loss, info_dict).
    """
    log_prob = dist.log_prob(actions)
    ratio = torch.exp(log_prob - old_log_prob)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = ((values - returns) ** 2).mean()
    entropy = dist.entropy().mean()
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return loss, {
        "pg": policy_loss.item(),
        "vf": value_loss.item(),
        "ent": entropy.item(),
    }
