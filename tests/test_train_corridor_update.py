"""Fast, SUMO-free guard that a PPO update MOVES the shared policy — for both the
IPPO (local-obs critic) and MAPPO (joint-state critic) paths. Catches a broken
gradient flow / no-op optimizer without needing SUMO.
"""
import copy

import torch

import ppo_core as pc
import train_corridor as tc

OBS_DIM = 19
N_AGENTS = 2  # two synthetic agents


def _buffer(state_dim, n=32):
    """Per-agent buffer with both `obs` (actor input, OBS_DIM) and `state` (critic
    input, state_dim). Reward correlates with action so there is a gradient signal."""
    per = {}
    for agent in ("C1", "C2"):
        acts = [torch.tensor(i % 2) for i in range(n)]
        per[agent] = {
            "obs": [torch.randn(OBS_DIM) for _ in range(n)],
            "state": [torch.randn(state_dim) for _ in range(n)],
            "act": acts,
            "logp": [torch.tensor(-0.6931) for _ in range(n)],
            "val": [0.0 for _ in range(n)],
            "rew": [1.0 if int(acts[i]) == 1 else 0.0 for i in range(n)],
            "done": [0.0] * (n - 1) + [1.0],
        }
    return per


def _run_update(state_dim):
    torch.manual_seed(0)
    policy = pc.ActorCritic(OBS_DIM, 2, state_dim=state_dim)
    optim = torch.optim.Adam(policy.parameters(), lr=1e-3)
    hp = {"gamma": 0.99, "gae_lambda": 0.95, "n_epochs": 4,
          "batch_size": 16, "clip_range": 0.2, "ent_coef": 0.0}
    per = _buffer(state_dim)
    last_states = {a: torch.randn(state_dim) for a in per}
    before = copy.deepcopy(policy.state_dict())
    tc.update(policy, optim, per, hp, last_states)
    after = policy.state_dict()
    return any(not torch.equal(before[k], after[k]) for k in before)


def test_update_moves_policy_ippo_local_state():
    assert _run_update(state_dim=OBS_DIM), "IPPO update moved no parameter"


def test_update_moves_policy_mappo_joint_state():
    assert _run_update(state_dim=OBS_DIM * N_AGENTS), "MAPPO update moved no parameter"
