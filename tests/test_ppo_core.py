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


def test_ppo_loss_zero_policy_gradient_when_ratio_one():
    # old_logp == new logp -> ratio 1; with vf_coef=0, ent_coef=0 and values==returns,
    # loss = -mean(advantages)
    ac = pc.ActorCritic(obs_dim=4, act_dim=2)
    obs = torch.zeros((6, 4))
    dist = ac.policy(obs)
    actions = dist.sample()
    old_logp = dist.log_prob(actions).detach()
    advantages = torch.ones(6)
    values = torch.zeros(6)
    returns = torch.zeros(6)
    loss, info = pc.ppo_loss(dist, actions, old_logp, advantages, values, returns,
                             clip=0.1, ent_coef=0.0, vf_coef=0.0)
    assert abs(info["pg"] - (-1.0)) < 1e-5   # -mean(adv) = -1
    assert abs(loss.item() - (-1.0)) < 1e-5


def test_ppo_loss_entropy_bonus_lowers_loss():
    ac = pc.ActorCritic(obs_dim=4, act_dim=2)
    obs = torch.zeros((6, 4))
    dist = ac.policy(obs)
    actions = dist.sample()
    old_logp = dist.log_prob(actions).detach()
    adv = torch.zeros(6)
    vals = torch.zeros(6)
    rets = torch.zeros(6)
    loss_no_ent, _ = pc.ppo_loss(dist, actions, old_logp, adv, vals, rets,
                                 clip=0.1, ent_coef=0.0, vf_coef=0.0)
    loss_ent, _ = pc.ppo_loss(dist, actions, old_logp, adv, vals, rets,
                              clip=0.1, ent_coef=0.5, vf_coef=0.0)
    assert loss_ent.item() < loss_no_ent.item()  # entropy subtracted from loss
