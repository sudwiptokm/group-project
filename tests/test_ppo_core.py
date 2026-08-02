"""Unit tests for the SUMO-free PPO core."""
import torch

import ppo_core as pc


def test_actorcritic_output_shapes():
    ac = pc.ActorCritic(obs_dim=19, act_dim=2)
    obs = torch.zeros((5, 19))
    dist = ac.policy(obs)
    assert dist.logits.shape == (5, 2)
    value = ac.value(obs)
    assert value.shape == (5,)


def test_actorcritic_act_returns_action_and_logprob():
    ac = pc.ActorCritic(obs_dim=19, act_dim=2)
    obs = torch.zeros((3, 19))
    actions, logp = ac.act(obs)
    assert actions.shape == (3,)
    assert logp.shape == (3,)
    assert actions.dtype == torch.int64


def test_critic_state_dim_can_differ_from_obs_dim():
    # the SP3 seam: critic input may be wider than the actor's obs (joint state)
    ac = pc.ActorCritic(obs_dim=19, act_dim=2, state_dim=57)  # 3*19
    state = torch.zeros((4, 57))
    assert ac.value(state).shape == (4,)
    # actor still consumes local obs of width 19
    assert ac.policy(torch.zeros((4, 19))).logits.shape == (4, 2)
