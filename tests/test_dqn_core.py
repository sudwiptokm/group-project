"""Unit tests for the SUMO-free DQN core."""
import copy

import numpy as np
import torch

import dqn_core as dc


def test_qnetwork_output_shape():
    net = dc.QNetwork(obs_dim=19, act_dim=4)
    obs = torch.zeros((5, 19))
    q = net(obs)
    assert q.shape == (5, 4)


def test_replay_buffer_add_and_sample_shapes():
    buf = dc.ReplayBuffer(capacity=10, obs_dim=3)
    rng = np.random.default_rng(0)
    for i in range(5):
        buf.add(np.full(3, i, dtype=np.float32), action=i % 2, reward=float(i),
                next_obs=np.full(3, i + 1, dtype=np.float32), done=0.0)
    assert len(buf) == 5
    obs, act, rew, nobs, done = buf.sample(4, rng)
    assert obs.shape == (4, 3)
    assert act.shape == (4,)
    assert rew.shape == (4,)
    assert nobs.shape == (4, 3)
    assert done.shape == (4,)


def test_replay_buffer_wraps_at_capacity():
    buf = dc.ReplayBuffer(capacity=3, obs_dim=1)
    for i in range(5):
        buf.add(np.array([float(i)], dtype=np.float32), action=0, reward=0.0,
                next_obs=np.array([0.0], dtype=np.float32), done=0.0)
    assert len(buf) == 3  # capacity, not total adds


def test_epsilon_schedule_boundary_values():
    sched = dc.EpsilonSchedule(total_steps=100, exploration_fraction=0.5,
                               exploration_final_eps=0.1)
    assert abs(sched.value(0) - 1.0) < 1e-9
    assert abs(sched.value(25) - 0.55) < 1e-9   # 1.0 + 0.5*(0.1-1.0)
    assert abs(sched.value(50) - 0.1) < 1e-9
    assert abs(sched.value(75) - 0.1) < 1e-9    # past decay -> floor


def _const_qnet(bias_values):
    """A one-layer QNetwork whose output is a fixed vector regardless of obs,
    for hand-computable dqn_loss values."""
    net = dc.QNetwork(obs_dim=1, act_dim=len(bias_values), hidden=())
    linear = net.net[0]
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.copy_(torch.tensor(bias_values))
    return net


def test_dqn_loss_known_value_nonterminal():
    q_net = _const_qnet([1.0, 2.0])
    target_net = _const_qnet([3.0, 4.0])
    obs = torch.zeros((1, 1))
    actions = torch.tensor([1])
    rewards = torch.tensor([5.0])
    next_obs = torch.zeros((1, 1))
    dones = torch.tensor([0.0])
    loss, info = dc.dqn_loss(q_net, target_net, obs, actions, rewards, next_obs,
                             dones, gamma=0.9)
    # Q(s,1)=2.0 ; target = 5 + 0.9*1*max(3,4) = 5 + 3.6 = 8.6 ; diff=6.6
    # smooth_l1 (beta=1): |diff|>1 -> |diff|-0.5 = 6.1
    assert abs(loss.item() - 6.1) < 1e-4
    assert abs(info["q_mean"] - 2.0) < 1e-6
    assert abs(info["target_mean"] - 8.6) < 1e-4


def test_dqn_loss_terminal_blocks_bootstrap():
    q_net = _const_qnet([1.0, 2.0])
    target_net = _const_qnet([3.0, 4.0])
    obs = torch.zeros((1, 1))
    actions = torch.tensor([1])
    rewards = torch.tensor([5.0])
    next_obs = torch.zeros((1, 1))
    dones = torch.tensor([1.0])
    loss, info = dc.dqn_loss(q_net, target_net, obs, actions, rewards, next_obs,
                             dones, gamma=0.9)
    # done=1 -> bootstrap masked -> target = reward alone = 5.0
    # Q=2.0 ; diff=3.0 -> loss = 3.0 - 0.5 = 2.5
    assert abs(loss.item() - 2.5) < 1e-4
    assert abs(info["target_mean"] - 5.0) < 1e-4


def test_dqn_loss_gradient_updates_qnetwork_parameters():
    # fast, SUMO-free guard that a real training step actually moves the
    # Q-network -- mirrors test_train_corridor_update.py's role for ppo_core
    torch.manual_seed(0)
    q_net = dc.QNetwork(obs_dim=4, act_dim=2)
    target_net = dc.QNetwork(obs_dim=4, act_dim=2)
    target_net.load_state_dict(q_net.state_dict())
    optim = torch.optim.Adam(q_net.parameters(), lr=1e-2)

    before = copy.deepcopy(q_net.state_dict())
    obs = torch.randn(16, 4)
    actions = torch.randint(0, 2, (16,))
    rewards = torch.randn(16)
    next_obs = torch.randn(16, 4)
    dones = torch.zeros(16)
    loss, _ = dc.dqn_loss(q_net, target_net, obs, actions, rewards, next_obs,
                          dones, gamma=0.99)
    optim.zero_grad()
    loss.backward()
    optim.step()
    after = q_net.state_dict()

    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed, "one dqn_loss gradient step moved no Q-network parameter"
