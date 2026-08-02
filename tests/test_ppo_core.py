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


def test_compute_gae_known_values():
    # rewards all 1, zero values, last step terminal, gamma=lam=1
    # hand-derivation: adv = [3, 2, 1], returns = adv + values = [3, 2, 1]
    adv, ret = pc.compute_gae(
        rewards=[1.0, 1.0, 1.0], values=[0.0, 0.0, 0.0],
        dones=[0.0, 0.0, 1.0], gamma=1.0, lam=1.0, last_value=0.0)
    assert adv == [3.0, 2.0, 1.0]
    assert ret == [3.0, 2.0, 1.0]


def test_compute_gae_terminal_blocks_bootstrap():
    # a terminal at t=0 must stop the value bootstrap from leaking backward
    adv, ret = pc.compute_gae(
        rewards=[5.0], values=[2.0], dones=[1.0],
        gamma=0.99, lam=0.95, last_value=100.0)
    # delta = 5 + 0.99*100*(1-1) - 2 = 3 ; gae = 3
    assert abs(adv[0] - 3.0) < 1e-9
    assert abs(ret[0] - 5.0) < 1e-9
