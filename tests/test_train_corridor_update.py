"""Fast, SUMO-free guard that a PPO update actually MOVES the shared policy.

The slow learning-check (greedy argmax at a micro training budget) can pass even
if training is a silent no-op — trained and untrained argmax often coincide. This
test closes that gap: it runs one real `update()` on a synthetic per-agent buffer
and asserts the policy parameters change, catching a broken gradient flow or a
no-op optimizer without needing SUMO.
"""
import copy

import torch

import ppo_core as pc
import train_corridor as tc


def _synthetic_per_agent_buffer(obs_dim, n=32):
    """Two agents, each a short trajectory whose reward correlates with the action
    so the policy has a real gradient signal to fit."""
    per = {}
    for agent in ("C1", "C2"):
        acts = [torch.tensor(i % 2) for i in range(n)]
        per[agent] = {
            "obs": [torch.randn(obs_dim) for _ in range(n)],
            "act": acts,
            "logp": [torch.tensor(-0.6931) for _ in range(n)],  # ln(0.5)
            "val": [0.0 for _ in range(n)],
            "rew": [1.0 if int(acts[i]) == 1 else 0.0 for i in range(n)],
            "done": [0.0] * (n - 1) + [1.0],
        }
    return per


def test_update_changes_policy_parameters():
    torch.manual_seed(0)
    obs_dim, act_dim = 19, 2
    policy = pc.ActorCritic(obs_dim, act_dim)
    optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
    hp = {"gamma": 0.99, "gae_lambda": 0.95, "n_epochs": 4,
          "batch_size": 16, "clip_range": 0.2, "ent_coef": 0.0}

    before = copy.deepcopy(policy.state_dict())
    per = _synthetic_per_agent_buffer(obs_dim)
    last_obs = {a: torch.randn(obs_dim) for a in per}  # bootstrap obs per agent
    tc.update(policy, optim, per, hp, last_obs)
    after = policy.state_dict()

    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed, "update() moved no policy parameter — gradient not flowing?"
