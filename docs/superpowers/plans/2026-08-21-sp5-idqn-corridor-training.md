# SP5 IDQN Corridor Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a true-independent DQN (matching RESCO's IDQN, not this project's parameter-shared IPPO) for the corridor, validate it against SB3 DQN, then run it at SP4's confirmatory budget (100k steps) — gated by a 3-seed pilot — and report whether it closes the +4.38s gap to `green_wave` that IPPO could not.

**Architecture:** `dqn_core.py` (pure, SUMO-free DQN math: `QNetwork`, `ReplayBuffer`, `EpsilonSchedule`, `dqn_loss`) mirrors `ppo_core.py`'s purity discipline. `train_corridor_dqn.py` instantiates 3 fully independent `(QNetwork, target, buffer, optimizer, epsilon)` tuples, one per corridor signal, and trains them by stepping `make_corridor_env` jointly each timestep — unlike `train_corridor.py`'s IPPO, there is no shared policy and no pooled buffer. `analysis/validate_dqn_core.py` (mirrors `analysis/validate_ppo_core.py`) checks `dqn_core` against SB3 `DQN` on the single intersection before any corridor number is trusted. `analysis/idqn_sweep.py` (mirrors `analysis/ippo_sweep.py`) drives the pilot and full sweep, pairing against both `green_wave` (`analysis/corridor_sweep.csv`) and `ippo` (`analysis/ippo_sweep.csv`).

**Tech Stack:** Python 3.11 (venv), PyTorch 2.8.0, sumo-rl, stable-baselines3, pandas, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-21-sp5-idqn-corridor-design.md`.

## Global Constraints

- Run everything in the venv: `source venv/bin/activate` first; `SUMO_HOME` comes from the venv's activate script.
- Work directly on `main` — the project consolidated off the per-SP feature-branch model after SP4 (`docs/HANDOFF_2026-08-21.md`: `feature/corridor-ippo`/`feature/corridor-integrate` merged and deleted), and this plan's spec commits already landed on `main`.
- `min_green` must be passed explicitly at every call site this plan touches — never rely on `resolve_min_green`'s fallback. All corridor runs in this plan use `min_green=10`, matching SP4's calibrated floor (`docs/FINDINGS_2026-08-18-sp4-ippo-vs-corrected-bar.md`).
- Ranking metric is delay per completed trip from tripinfo (`analysis/tripinfo.reduce_tripinfo`'s `trip_time_loss_mean`), never `system_mean_waiting_time` — see `docs/FINDINGS_2026-08-12.md` §1.
- The 3 disclosed DQN hyperparameters (`lr`=2.3195e-05, `learning_starts`=5000, `target_update_interval`=5000) come from `docs/FINDINGS_2026-08-12.md`'s "Training attempted" section; every other DQN hyperparameter comes from `algos.ALGOS['dqn']['defaults']()` — never from SB3's raw library internals directly. This is a disclosed, best-effort reconstruction (spec §4), not the original tuned config.
- Correctness ordering is not optional: `dqn_core`'s unit tests and the real `analysis/validate_dqn_core.py` run (Task 2) must pass before Task 7's pilot runs, and the pilot's decision rule (Task 7) gates whether Task 8's full sweep runs at all.
- No `Co-Authored-By` / Claude / Anthropic attribution in any commit message in this plan.
- Do not commit anything under `logs/`, `models/`, or `params/` — all three are gitignored. `analysis/idqn_sweep.csv` is the one run-output file this plan tracks (matching `analysis/ippo_sweep.csv`'s precedent).

---

## File Structure

| File | Create/Modify | Responsibility |
|------|---------------|-----------------|
| `dqn_core.py` | Create | Pure, SUMO-free DQN math: `QNetwork`, `ReplayBuffer`, `EpsilonSchedule`, `dqn_loss` |
| `tests/test_dqn_core.py` | Create | Unit tests for `dqn_core.py` (shapes, known-value TD loss, gradient-flow guard) |
| `analysis/validate_dqn_core.py` | Create | Single-agent head-to-head: hand-rolled `dqn_core` vs SB3 `DQN`, matched hyperparameters/budget |
| `tests/test_validate_dqn_core.py` | Create | Matched-hyperparameter contract test |
| `train_corridor_dqn.py` | Create | 3-independent-agent training/eval driver over the corridor env |
| `tests/test_idqn_hp.py` | Create | Fast, SUMO-free tests for `train_corridor_dqn.py`'s pure-logic pieces |
| `tests/test_train_corridor_dqn.py` | Create | Slow SUMO smoke test: short IDQN train + eval writes a readable CSV |
| `analysis/idqn_sweep.py` | Create | Staged pilot/full sweep driver, resumable, pairs against `green_wave` and `ippo` |
| `tests/test_idqn_sweep.py` | Create | Unit tests for the pairing logic, on synthetic data (no SUMO) |
| `compare.py` | Modify | Add `idqn` alongside `ippo` in the corridor comparison block |
| `docs/FINDINGS_2026-08-21-sp5-idqn-vs-corrected-bar.md` | Create | The written verdict |

---

## Task 1: `dqn_core.py` — pure DQN math

**Files:**
- Create: `dqn_core.py`
- Test: `tests/test_dqn_core.py`

**Interfaces:**
- Produces: `QNetwork(obs_dim, act_dim, hidden=(64,64))` (`nn.Module`, `.forward(obs) -> Tensor[N, act_dim]`); `ReplayBuffer(capacity, obs_dim)` with `.add(obs, action, reward, next_obs, done)`, `.sample(batch_size, rng) -> (obs, act, rew, next_obs, done)` tensors, `__len__`; `EpsilonSchedule(total_steps, exploration_fraction, exploration_final_eps)` with `.value(step) -> float`; `dqn_loss(q_network, target_network, obs, actions, rewards, next_obs, dones, gamma) -> (loss: Tensor, info: dict)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dqn_core.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dqn_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dqn_core'`.

- [ ] **Step 3: Implement `dqn_core.py`**

Create `dqn_core.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dqn_core.py -v`
Expected: 7 PASS.

- [ ] **Step 5: Fast suite regression**

Run: `pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add dqn_core.py tests/test_dqn_core.py
git commit -m "feat: add dqn_core, pure SUMO-free DQN math for independent per-agent learners"
```

---

## Task 2: Validate `dqn_core` against SB3 DQN on the single intersection

**Files:**
- Create: `analysis/validate_dqn_core.py`
- Test: `tests/test_validate_dqn_core.py`

**Interfaces:**
- Consumes: `dqn_core.QNetwork`, `dqn_core.ReplayBuffer`, `dqn_core.EpsilonSchedule`, `dqn_core.dqn_loss` (Task 1); `algos.ALGOS['dqn']['defaults']()`, `algos.build`; `env_common.make_env`, `env_common.tripinfo_path`; `analysis.tripinfo.reduce_tripinfo`.
- Produces: `matched_hp() -> dict`; not consumed by later tasks, but establishes the "matched to SB3 defaults" convention `train_corridor_dqn.py` (Task 3) explicitly diverges from and discloses.

This is the top technical risk in the stack (same class of risk `validate_ppo_core.py` exists to surface for `ppo_core`), and per the Global Constraints it must pass before Task 7's pilot spends any corridor-training compute.

- [ ] **Step 1: Write the failing unit test (pure logic, no SUMO)**

Create `tests/test_validate_dqn_core.py`:
```python
"""Unit test for validate_dqn_core.py's matched-hyperparameter contract --
the same "must be byte-for-byte SB3's defaults" guard validate_ppo_core.py
has for PPO."""
import pytest

vdc = pytest.importorskip("analysis.validate_dqn_core")


def test_matched_hyperparameters_come_from_algos_dqn_defaults():
    from algos import ALGOS
    sb3_defaults = ALGOS["dqn"]["defaults"]()
    hp = vdc.matched_hp()
    assert hp["lr"] == sb3_defaults["learning_rate"]
    assert hp["buffer_size"] == sb3_defaults["buffer_size"]
    assert hp["learning_starts"] == sb3_defaults["learning_starts"]
    assert hp["batch_size"] == sb3_defaults["batch_size"]
    assert hp["gamma"] == sb3_defaults["gamma"]
    assert hp["train_freq"] == sb3_defaults["train_freq"]
    assert hp["target_update_interval"] == sb3_defaults["target_update_interval"]
    assert hp["exploration_fraction"] == sb3_defaults["exploration_fraction"]
    assert hp["exploration_final_eps"] == sb3_defaults["exploration_final_eps"]
    assert hp["hidden"] == tuple(sb3_defaults["policy_kwargs"]["net_arch"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validate_dqn_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.validate_dqn_core'`.

- [ ] **Step 3: Implement `analysis/validate_dqn_core.py`**

Create `analysis/validate_dqn_core.py`:
```python
"""Head-to-head: hand-rolled dqn_core vs SB3 DQN, matched hyperparameters and
step budget, on the single-agent intersection env (scenario 'base').

dqn_core/train_corridor_dqn were built for the multi-agent corridor's
independent-per-agent training. The single-intersection env is single-agent
sumo-rl (Gymnasium-style: obs, reward, terminated, truncated), so this file
adapts dqn_core's QNetwork/ReplayBuffer/dqn_loss to that API directly rather
than reusing train_corridor_dqn's multi-agent training loop.

Not a pass/fail gate: a from-scratch reimplementation is not expected to
exactly match a mature library's sample efficiency. It reports held-out
tripinfo delay and wall-clock timing for both so the risk this comparison
exists to surface -- "is dqn_core's gradient step actually equivalent to
SB3's" -- has evidence either way before any corridor number is trusted.

    python -m analysis.validate_dqn_core --steps 100000 --seed 0
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

import dqn_core as dc
from algos import ALGOS, build
from analysis.tripinfo import reduce_tripinfo
from env_common import make_env, tripinfo_path

MIN_GREEN = 60  # env_common.DEFAULT_MIN_GREEN -- explicit here for the same
                # reason it must be explicit everywhere else in this project


def matched_hp() -> dict:
    """dqn_core's hyperparameter names <- algos.ALGOS['dqn']['defaults']().

    Plain SB3 DQN defaults, NOT train_corridor_dqn._hp()'s 3 disclosed
    overrides -- this validation isolates "is dqn_core's math right" from
    "are these the hyperparameters SP5's real corridor run uses."
    """
    d = ALGOS["dqn"]["defaults"]()
    return {
        "lr": d["learning_rate"], "buffer_size": d["buffer_size"],
        "learning_starts": d["learning_starts"], "batch_size": d["batch_size"],
        "gamma": d["gamma"], "train_freq": d["train_freq"],
        "target_update_interval": d["target_update_interval"],
        "exploration_fraction": d["exploration_fraction"],
        "exploration_final_eps": d["exploration_final_eps"],
        "hidden": tuple(d["policy_kwargs"]["net_arch"]),
    }


def train_dqn_core(seed: int, steps: int) -> str:
    """Single-agent DQN training loop using dqn_core, matched to SB3 DQN's
    hyperparameters."""
    hp = matched_hp()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env(seed=seed, scenario="base", min_green=MIN_GREEN)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.n
    q_net = dc.QNetwork(obs_dim, act_dim, hidden=hp["hidden"])
    target_net = dc.QNetwork(obs_dim, act_dim, hidden=hp["hidden"])
    target_net.load_state_dict(q_net.state_dict())
    optim = torch.optim.Adam(q_net.parameters(), lr=hp["lr"])
    buffer = dc.ReplayBuffer(hp["buffer_size"], obs_dim)
    eps_sched = dc.EpsilonSchedule(steps, hp["exploration_fraction"],
                                   hp["exploration_final_eps"])

    obs, _ = env.reset()
    for t in range(steps):
        eps = eps_sched.value(t)
        if rng.random() < eps:
            action = int(rng.integers(act_dim))
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = int(q_net(obs_t).argmax(dim=-1).item())
        nobs, reward, terminated, truncated, _ = env.step(action)
        done = float(terminated or truncated)
        buffer.add(obs, action, float(reward), nobs, done)
        obs = nobs
        if done:
            obs, _ = env.reset()

        if t >= hp["learning_starts"] and t % hp["train_freq"] == 0 \
                and len(buffer) >= hp["batch_size"]:
            b_obs, b_act, b_rew, b_nobs, b_done = buffer.sample(hp["batch_size"], rng)
            loss, _ = dc.dqn_loss(q_net, target_net, b_obs, b_act, b_rew, b_nobs,
                                  b_done, hp["gamma"])
            optim.zero_grad()
            loss.backward()
            optim.step()

        if t > 0 and t % hp["target_update_interval"] == 0:
            target_net.load_state_dict(q_net.state_dict())
    env.close()

    os.makedirs("models", exist_ok=True)
    path = f"models/validate_dqn_core_seed{seed}.pt"
    torch.save({"state_dict": q_net.state_dict(), "hidden": hp["hidden"]}, path)
    return path


def eval_dqn_core(model_path: str, seed: int) -> float:
    out_csv = f"logs/eval_validate_dqn_core_seed{seed}"
    env = make_env(seed=seed, scenario="base", min_green=MIN_GREEN,
                   out_csv=out_csv, tripinfo=True)
    ckpt = torch.load(model_path, weights_only=True)
    q_net = dc.QNetwork(env.observation_space.shape[0], env.action_space.n,
                        hidden=tuple(ckpt["hidden"]))
    q_net.load_state_dict(ckpt["state_dict"])
    q_net.eval()
    obs, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            q = q_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        obs, _, terminated, truncated, _ = env.step(int(q.argmax(dim=-1)[0]))
        done = terminated or truncated
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    return reduce_tripinfo(tripinfo_path(out_csv))["trip_time_loss_mean"]


def train_eval_sb3(seed: int, steps: int) -> float:
    params = ALGOS["dqn"]["defaults"]()
    env = Monitor(make_env(seed=seed, scenario="base", min_green=MIN_GREEN))
    model = build("dqn", env, params, seed=seed, tb_log="logs/tb")
    model.learn(total_timesteps=steps)
    env.close()

    out_csv = f"logs/eval_validate_sb3_dqn_seed{seed}"
    eval_env = make_env(seed=seed, scenario="base", min_green=MIN_GREEN,
                        out_csv=out_csv, tripinfo=True)
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
    eval_env.save_csv(eval_env.out_csv_name, eval_env.episode)
    eval_env.close()
    return reduce_tripinfo(tripinfo_path(out_csv))["trip_time_loss_mean"]


if __name__ == "__main__":
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.monotonic()
    dqn_core_model = train_dqn_core(args.seed, args.steps)
    dqn_core_delay = eval_dqn_core(dqn_core_model, seed=args.seed + 1000)
    t1 = time.monotonic()
    sb3_delay = train_eval_sb3(args.seed, args.steps)
    t2 = time.monotonic()

    print(f"\ndqn_core:  delay/trip={dqn_core_delay:.1f}s  wall={t1 - t0:.0f}s")
    print(f"sb3 DQN:   delay/trip={sb3_delay:.1f}s  wall={t2 - t1:.0f}s")
    print(f"dqn_core - sb3: {dqn_core_delay - sb3_delay:+.1f}s")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_validate_dqn_core.py -v`
Expected: PASS.

- [ ] **Step 5: Fast suite regression**

Run: `pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Run the real validation (slow, SUMO) — the correctness gate**

Run:
```bash
source venv/bin/activate
python -m analysis.validate_dqn_core --steps 100000 --seed 0
```
Expected: prints both delay/trip numbers, wall-clock for each, and the difference. Record all four numbers — they go into the findings doc in Task 9. This is evidence-gathering, not a scripted pass/fail: if `dqn_core` is dramatically worse (e.g. >2x delay, or it never brings the policy off a random baseline), stop and debug `dqn_core`/`train_dqn_core` before spending any further compute on corridor training — that would mean the top technical risk materialised, exactly as `validate_ppo_core.py`'s equivalent gate was framed for PPO.

- [ ] **Step 7: Commit**

```bash
git add analysis/validate_dqn_core.py tests/test_validate_dqn_core.py
git commit -m "feat: validate dqn_core against SB3 DQN, matched HPs, single intersection"
```

---

## Task 3: `train_corridor_dqn.py` — 3 independent agents on the corridor

**Files:**
- Create: `train_corridor_dqn.py`
- Test: `tests/test_idqn_hp.py` (fast, no SUMO)
- Test: `tests/test_train_corridor_dqn.py` (slow, SUMO)

**Interfaces:**
- Consumes: `dqn_core.QNetwork`, `dqn_core.ReplayBuffer`, `dqn_core.EpsilonSchedule`, `dqn_core.dqn_loss` (Task 1); `algos.ALGOS['dqn']['defaults']()`; `env_common.CORRIDOR_SCENARIOS`, `env_common.make_corridor_env`.
- Produces: `CORRIDOR_TS_IDS = ("C1", "C2", "C3")`; `_hp() -> dict`; `_tag(scenario, lam, seed, min_green, steps) -> str`; `_model_path(agent_id, scenario, lam, seed, min_green, steps) -> str`; `train(scenario, lam, seed, steps, min_green) -> dict[str, str]` (ts_id -> model path); `evaluate(scenario, lam, seed, min_green, steps, tripinfo=False) -> str` (eval CSV path). All four consumed by Task 4's `analysis/idqn_sweep.py`.

- [ ] **Step 1: Write the failing fast tests (pure logic, no SUMO)**

Create `tests/test_idqn_hp.py`:
```python
"""Unit tests for train_corridor_dqn.py's pure-logic pieces (no SUMO)."""
import train_corridor_dqn as tcd
from algos import ALGOS


def test_corridor_ts_ids_constant():
    assert tcd.CORRIDOR_TS_IDS == ("C1", "C2", "C3")


def test_tag_includes_min_green_and_steps():
    assert tcd._tag("corridor_peak", 0.5, seed=0, min_green=10, steps=100000) == \
        "corridor_peak_lam05_seed0_mg10_s100000"


def test_model_path_includes_agent_id():
    p = tcd._model_path("C1", "corridor_peak", 0.5, 0, 10, 100000)
    assert p == "models/idqn_C1_corridor_peak_lam05_seed0_mg10_s100000.pt"


def test_model_path_agents_do_not_collide():
    paths = {a: tcd._model_path(a, "corridor_peak", 0.5, 0, 10, 100000)
             for a in tcd.CORRIDOR_TS_IDS}
    assert len(set(paths.values())) == 3


def test_hp_uses_disclosed_values_and_algos_defaults():
    d = ALGOS["dqn"]["defaults"]()
    hp = tcd._hp()
    assert hp["lr"] == 2.3195e-05
    assert hp["learning_starts"] == 5000
    assert hp["target_update_interval"] == 5000
    assert hp["buffer_size"] == d["buffer_size"]
    assert hp["batch_size"] == d["batch_size"]
    assert hp["gamma"] == d["gamma"]
    assert hp["train_freq"] == d["train_freq"]
    assert hp["exploration_fraction"] == d["exploration_fraction"]
    assert hp["exploration_final_eps"] == d["exploration_final_eps"]
    assert hp["hidden"] == tuple(d["policy_kwargs"]["net_arch"])


def test_train_and_evaluate_require_min_green_kwarg():
    import inspect
    assert "min_green" in inspect.signature(tcd.train).parameters
    assert "min_green" in inspect.signature(tcd.evaluate).parameters
    assert inspect.signature(tcd.train).parameters["min_green"].default is \
        inspect.Parameter.empty
    assert inspect.signature(tcd.evaluate).parameters["min_green"].default is \
        inspect.Parameter.empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_idqn_hp.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_corridor_dqn'`.

- [ ] **Step 3: Implement `train_corridor_dqn.py`**

Create `train_corridor_dqn.py`:
```python
"""True-independent DQN on the multi-agent corridor env.

Unlike train_corridor.py's parameter-shared IPPO (one policy, pooled
transitions), SP5's IDQN gives each corridor signal its own QNetwork, target
network, replay buffer and optimizer (dqn_core.py). Agents interact only by
stepping the shared SUMO env jointly each timestep; each stores its own
transition and trains independently once its own learning_starts threshold is
hit. This matches RESCO's actual IDQN rather than this project's own
parameter-shared "IPPO" -- see
docs/superpowers/specs/2026-08-21-sp5-idqn-corridor-design.md.
"""
import argparse
import os

import numpy as np
import torch

import dqn_core as dc
from algos import ALGOS
from env_common import CORRIDOR_SCENARIOS, make_corridor_env

# The corridor net's 3 traffic-signal ids (corridor.net.xml), fixed by the
# network SP1 built. Known ahead of instantiating an env so idqn_sweep.py's
# resumability check can test for existing checkpoints without starting SUMO.
CORRIDOR_TS_IDS = ("C1", "C2", "C3")

# The single-intersection DQN pilot's tuned-for-100k config (params/*.json) is
# unrecoverable -- gitignored, cloud-only, never committed (confirmed via
# `git log --all -- params/`). Only 3 of its values survive, disclosed in
# prose in docs/FINDINGS_2026-08-12.md's "Training attempted" section. The
# rest come from algos.ALGOS['dqn']['defaults'] -- this project's own
# canonical "SB3 defaults" source, the same one validate_ppo_core.py's
# matched_hp() already reads for PPO. Best-effort reconstruction, not the
# original tuned config -- disclosed limitation, see the SP5 design spec §4.
_DISCLOSED = {
    "lr": 2.3195e-05,
    "learning_starts": 5000,
    "target_update_interval": 5000,
}


def _hp() -> dict:
    d = ALGOS["dqn"]["defaults"]()
    return {
        "lr": _DISCLOSED["lr"],
        "buffer_size": d["buffer_size"],
        "learning_starts": _DISCLOSED["learning_starts"],
        "batch_size": d["batch_size"],
        "gamma": d["gamma"],
        "train_freq": d["train_freq"],
        "target_update_interval": _DISCLOSED["target_update_interval"],
        "exploration_fraction": d["exploration_fraction"],
        "exploration_final_eps": d["exploration_final_eps"],
        "hidden": tuple(d["policy_kwargs"]["net_arch"]),
    }


def _obs_act_dims(env):
    tid = env.ts_ids[0]
    return env.observation_spaces(tid).shape[0], env.action_spaces(tid).n


def _tag(scenario: str, lam: float, seed: int, min_green: int, steps: int) -> str:
    """Filename tag shared by train (models) and evaluate (eval CSV), same
    convention as train_corridor._tag -- see that function's docstring for why
    min_green and steps are both folded in."""
    return f"{scenario}_lam{str(lam).replace('.', '')}_seed{seed}_mg{min_green}_s{steps}"


def _model_path(agent_id: str, scenario: str, lam: float, seed: int,
                min_green: int, steps: int) -> str:
    """Where one agent's checkpoint lives. Unlike train_corridor.py's single
    shared-policy path, IDQN has one file per agent -- agent_id is folded in
    right after the algo name so the 3 files for one run sort together and
    never collide with each other or with a different run's tag."""
    return f"models/idqn_{agent_id}_{_tag(scenario, lam, seed, min_green, steps)}.pt"


def train(scenario: str, lam: float, seed: int, steps: int, min_green: int) -> dict:
    """Train 3 fully independent DQN agents. Returns {ts_id: model_path}."""
    hp = _hp()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_corridor_env(seed=seed, scenario=scenario, lam=lam, min_green=min_green)
    ids = env.ts_ids
    obs_dim, act_dim = _obs_act_dims(env)

    agents = {}
    for i in ids:
        q_net = dc.QNetwork(obs_dim, act_dim, hidden=hp["hidden"])
        target_net = dc.QNetwork(obs_dim, act_dim, hidden=hp["hidden"])
        target_net.load_state_dict(q_net.state_dict())
        agents[i] = {
            "q": q_net,
            "target": target_net,
            "optim": torch.optim.Adam(q_net.parameters(), lr=hp["lr"]),
            "buffer": dc.ReplayBuffer(hp["buffer_size"], obs_dim),
            "eps": dc.EpsilonSchedule(steps, hp["exploration_fraction"],
                                      hp["exploration_final_eps"]),
        }

    obs = env.reset()
    for t in range(steps):
        actions = {}
        for i in ids:
            a = agents[i]
            if rng.random() < a["eps"].value(t):
                actions[i] = int(rng.integers(act_dim))
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                    actions[i] = int(a["q"](obs_t).argmax(dim=-1).item())
        nobs, rewards, dones, _ = env.step(actions)
        done_all = float(dones["__all__"])
        for i in ids:
            agents[i]["buffer"].add(obs[i], actions[i], float(rewards[i]), nobs[i],
                                    done_all)
        obs = nobs
        if done_all:
            obs = env.reset()

        if t >= hp["learning_starts"] and t % hp["train_freq"] == 0:
            for i in ids:
                a = agents[i]
                if len(a["buffer"]) < hp["batch_size"]:
                    continue
                b_obs, b_act, b_rew, b_nobs, b_done = a["buffer"].sample(
                    hp["batch_size"], rng)
                loss, _ = dc.dqn_loss(a["q"], a["target"], b_obs, b_act, b_rew,
                                      b_nobs, b_done, hp["gamma"])
                a["optim"].zero_grad()
                loss.backward()
                a["optim"].step()

        if t > 0 and t % hp["target_update_interval"] == 0:
            for i in ids:
                agents[i]["target"].load_state_dict(agents[i]["q"].state_dict())
    env.close()

    os.makedirs("models", exist_ok=True)
    paths = {}
    for i in ids:
        path = _model_path(i, scenario, lam, seed, min_green, steps)
        torch.save({"state_dict": agents[i]["q"].state_dict(), "hidden": hp["hidden"]},
                   path)
        paths[i] = path
    print(f"idqn models saved: {list(paths.values())}")
    return paths


def evaluate(scenario: str, lam: float, seed: int, min_green: int, steps: int,
            tripinfo: bool = False) -> str:
    """Run all 3 agents' greedy policies on a held-out seed, writing one eval
    CSV in the SafetyLoggingEnv format so compare.py reads it as `idqn`. With
    tripinfo=True also writes the per-trip XML analysis/idqn_sweep.py reduces.

    Loads each agent's checkpoint by reconstructing its path from _model_path
    -- callers never pass paths directly, so train() and evaluate() can never
    disagree about where a checkpoint lives."""
    os.makedirs("logs", exist_ok=True)
    tag = _tag(scenario, lam, seed, min_green, steps)
    out_csv = f"logs/eval_idqn_{tag}"
    env = make_corridor_env(seed=seed, scenario=scenario, lam=lam,
                            min_green=min_green, out_csv=out_csv, tripinfo=tripinfo)
    ids = env.ts_ids
    obs_dim, act_dim = _obs_act_dims(env)
    policies = {}
    for i in ids:
        ckpt = torch.load(_model_path(i, scenario, lam, seed, min_green, steps),
                          weights_only=True)
        q_net = dc.QNetwork(obs_dim, act_dim, hidden=tuple(ckpt["hidden"]))
        q_net.load_state_dict(ckpt["state_dict"])
        q_net.eval()
        policies[i] = q_net

    obs = env.reset()
    done = False
    while not done:
        actions = {}
        for i in ids:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs[i], dtype=torch.float32).unsqueeze(0)
                actions[i] = int(policies[i](obs_t).argmax(dim=-1).item())
        obs, _, dones, _ = env.step(actions)
        done = dones["__all__"]
    env.save_csv(env.out_csv_name, env.episode)
    env.close()
    out = f"logs/eval_idqn_{tag}_conn{env.label}_ep{env.episode}.csv"
    print(f"idqn eval written: {out}")
    return out


if __name__ == "__main__":
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="corridor_offpeak",
                   choices=list(CORRIDOR_SCENARIOS))
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--min-green", type=int, required=True,
                   help="explicit -- this script never falls back to $MIN_GREEN/DEFAULT_MIN_GREEN")
    p.add_argument("--eval", action="store_true",
                   help="evaluate existing checkpoints instead of training")
    p.add_argument("--tripinfo", action="store_true",
                   help="also write the per-trip XML (only meaningful with --eval)")
    args = p.parse_args()
    if args.eval:
        evaluate(args.scenario, args.lam, args.seed, args.min_green, args.steps,
                 tripinfo=args.tripinfo)
    else:
        train(args.scenario, args.lam, args.seed, args.steps, args.min_green)
```

- [ ] **Step 4: Run fast tests to verify they pass**

Run: `pytest tests/test_idqn_hp.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Write the slow smoke test**

Create `tests/test_train_corridor_dqn.py`:
```python
"""Slow smoke: short IDQN train + eval must write a readable eval CSV. Requires SUMO."""
import os

import pytest

pytestmark = pytest.mark.slow

import train_corridor_dqn as tcd


@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_idqn_trains_and_evaluates(monkeypatch):
    monkeypatch.setenv("EPISODE_SECONDS", "200")
    paths = tcd.train("corridor_offpeak", lam=0.5, seed=0, steps=600, min_green=10)
    assert set(paths.keys()) == set(tcd.CORRIDOR_TS_IDS)
    for p in paths.values():
        assert os.path.exists(p)

    csv = tcd.evaluate("corridor_offpeak", lam=0.5, seed=42, min_green=10, steps=600)
    assert os.path.exists(csv)
    import pandas as pd
    df = pd.read_csv(csv)
    assert df["system_mean_speed"].mean() > 0
```

- [ ] **Step 6: Run the slow test (SUMO required)**

Run: `pytest tests/test_train_corridor_dqn.py -v`
Expected: PASS (skipped if `SUMO_HOME` is unset, matching `test_train_corridor.py`'s own skip behaviour).

- [ ] **Step 7: Fast suite regression**

Run: `pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add train_corridor_dqn.py tests/test_idqn_hp.py tests/test_train_corridor_dqn.py
git commit -m "feat: train_corridor_dqn, 3 independent DQN agents on the corridor"
```

---

## Task 4: `analysis/idqn_sweep.py` — staged pilot/full sweep, paired against green_wave and ippo

**Files:**
- Create: `analysis/idqn_sweep.py`
- Test: `tests/test_idqn_sweep.py`

**Interfaces:**
- Consumes: `train_corridor_dqn.CORRIDOR_TS_IDS`, `train_corridor_dqn._model_path`, `train_corridor_dqn._tag`, `train_corridor_dqn.train`, `train_corridor_dqn.evaluate` (Task 3); `analysis.tripinfo.reduce_tripinfo`; `env_common.CORRIDOR_SCENARIOS`, `env_common.tripinfo_path`.
- Produces: `run_one(...) -> dict`; `sweep(...) -> pd.DataFrame`; `paired_vs(idqn_df, bar_df, bar_name) -> dict`; `load_green_wave_bar(scenario, min_green) -> pd.DataFrame`; `load_ippo_bar(scenario, min_green) -> pd.DataFrame`. Consumed directly (as a script) by Task 7's pilot and Task 8's full sweep.

- [ ] **Step 1: Write the failing tests (pure logic, no SUMO)**

Create `tests/test_idqn_sweep.py`:
```python
"""Unit tests for analysis/idqn_sweep.py's pairing logic (no SUMO)."""
import pandas as pd
import pytest

idq = pytest.importorskip("analysis.idqn_sweep")


def _rows(controller, scenario, min_green, delays):
    return pd.DataFrame([
        {"controller": controller, "scenario": scenario, "seed": 42 + i,
         "min_green": min_green, "delay_per_trip": d, "trips": 2900, "wall_s": 1.0}
        for i, d in enumerate(delays)
    ])


def test_paired_vs_seed_alignment():
    baseline = _rows("green_wave", "corridor_peak", 10, [13.0, 14.0, 13.5])
    idqn = _rows("idqn", "corridor_peak", 10, [12.0, 15.0, 13.0])
    d = idq.paired_vs(idqn, baseline, "green_wave")
    # idqn - green_wave per seed: [-1.0, 1.0, -0.5]
    assert d["n"] == 3
    assert d["wins"] == 2          # negative diff = idqn wins (lower delay)
    assert abs(d["mean"] - (-1.0 / 6)) < 1e-9
    assert d["vs"] == "green_wave"


def test_paired_vs_requires_matching_seeds():
    baseline = _rows("ippo", "corridor_peak", 10, [13.0, 14.0])
    idqn = _rows("idqn", "corridor_peak", 10, [12.0])  # only seed 42 present
    d = idq.paired_vs(idqn, baseline, "ippo")
    assert d["n"] == 1             # only the overlapping seed is paired


def test_paired_vs_wrong_scenario_raises():
    baseline = _rows("green_wave", "corridor_tidal", 10, [14.0])
    idqn = _rows("idqn", "corridor_peak", 10, [12.0])
    with pytest.raises(ValueError):
        idq.paired_vs(idqn, baseline, "green_wave")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_idqn_sweep.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.idqn_sweep'`.

- [ ] **Step 3: Implement `analysis/idqn_sweep.py`**

Create `analysis/idqn_sweep.py`:
```python
"""IDQN training/eval at one explicit floor, reduced to delay-per-completed-trip
and paired against both analysis/corridor_sweep.csv's green_wave rows and
analysis/ippo_sweep.csv's ippo rows.

Same methodology as analysis/ippo_sweep.py (tripinfo reduction, seed set
42-51, resumable "reuse what's on disk" design) applied to the true-
independent DQN driver (train_corridor_dqn.py) instead of the parameter-
shared IPPO one.

    python -m analysis.idqn_sweep --scenario corridor_peak --min-green 10 \
        --seeds 42 43 44 --lam 0.5 --steps 100000
"""
import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import pandas as pd

import train_corridor_dqn as tcd
from analysis.tripinfo import reduce_tripinfo
from env_common import CORRIDOR_SCENARIOS, tripinfo_path

OUT_CSV = os.path.join(REPO, "analysis", "idqn_sweep.csv")
CORRIDOR_SWEEP_CSV = os.path.join(REPO, "analysis", "corridor_sweep.csv")
IPPO_SWEEP_CSV = os.path.join(REPO, "analysis", "ippo_sweep.csv")

os.environ.setdefault("TIME_TO_TELEPORT", "300")


def run_one(scenario: str, seed: int, min_green: int, lam: float, steps: int,
           force: bool = False) -> dict:
    """Train (if any of the 3 agent checkpoints is missing) + eval one seed,
    reduced to the ranking metric. Resumable: existing model/tripinfo files
    are reused."""
    paths = [tcd._model_path(a, scenario, lam, seed, min_green, steps)
             for a in tcd.CORRIDOR_TS_IDS]
    if force or not all(os.path.exists(p) for p in paths):
        t0 = time.monotonic()
        tcd.train(scenario, lam, seed, steps, min_green)
        took = time.monotonic() - t0
    else:
        took = float("nan")
    tcd.evaluate(scenario, lam, seed, min_green, steps, tripinfo=True)
    trip = tripinfo_path(f"logs/eval_idqn_{tcd._tag(scenario, lam, seed, min_green, steps)}")
    row = reduce_tripinfo(trip)
    return {
        "controller": "idqn", "scenario": scenario, "seed": seed,
        "min_green": min_green, "delay_per_trip": row["trip_time_loss_mean"],
        "trips": row["trips_completed"], "wall_s": took,
    }


def sweep(scenario: str, seeds, min_green: int, lam: float, steps: int,
         force: bool = False) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        rows.append(run_one(scenario, seed, min_green, lam, steps, force))
        r = rows[-1]
        took = "reused" if pd.isna(r["wall_s"]) else f"{r['wall_s']:.0f}s"
        print(f"[{len(rows)}/{len(seeds)}] idqn seed{seed} "
              f"delay/trip={r['delay_per_trip']:7.1f}s trips={r['trips']:5d}  ({took})",
              flush=True)
    return pd.DataFrame(rows)


def paired_vs(idqn_df: pd.DataFrame, bar_df: pd.DataFrame, bar_name: str) -> dict:
    """idqn - bar_df per seed, paired. Both dataframes must be one (scenario,
    min_green) already -- raises if they disagree, the same cross-scenario
    guard ippo_sweep.paired_vs_green_wave relies on. bar_name labels which
    reference (green_wave / ippo) this call compares against."""
    i_scen = set(idqn_df["scenario"])
    b_scen = set(bar_df["scenario"])
    if i_scen != b_scen or len(i_scen) != 1:
        raise ValueError(f"scenario mismatch: idqn={i_scen} {bar_name}={b_scen}")
    wide = pd.merge(
        idqn_df[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": "idqn"}),
        bar_df[["seed", "delay_per_trip"]].rename(columns={"delay_per_trip": bar_name}),
        on="seed", how="inner")
    d = wide["idqn"] - wide[bar_name]
    return {
        "scenario": idqn_df["scenario"].iloc[0], "vs": bar_name,
        "mean": float(d.mean()),
        "sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
        "wins": int((d < 0).sum()), "n": int(len(d)),
    }


def load_green_wave_bar(scenario: str, min_green: int) -> pd.DataFrame:
    """green_wave rows already in analysis/corridor_sweep.csv for this
    (scenario, min_green)."""
    df = pd.read_csv(CORRIDOR_SWEEP_CSV)
    return df[(df["controller"] == "green_wave") & (df["scenario"] == scenario) &
              (df["min_green"] == min_green)]


def load_ippo_bar(scenario: str, min_green: int) -> pd.DataFrame:
    """ippo rows already in analysis/ippo_sweep.csv for this
    (scenario, min_green) -- SP4's result, the second reference this sweep
    pairs against."""
    df = pd.read_csv(IPPO_SWEEP_CSV)
    return df[(df["scenario"] == scenario) & (df["min_green"] == min_green)]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=list(CORRIDOR_SCENARIOS))
    p.add_argument("--min-green", type=int, required=True)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if not os.environ.get("SUMO_HOME"):
        raise SystemExit("SUMO_HOME not set")

    df = sweep(args.scenario, args.seeds, args.min_green, args.lam, args.steps, args.force)
    if os.path.exists(OUT_CSV):
        prior = pd.read_csv(OUT_CSV)
        df = pd.concat([prior, df]).drop_duplicates(
            subset=["scenario", "seed", "min_green"], keep="last")
    df.to_csv(OUT_CSV, index=False)

    this_run = df[(df["scenario"] == args.scenario) & (df["min_green"] == args.min_green)]

    gw_bar = load_green_wave_bar(args.scenario, args.min_green)
    if gw_bar.empty:
        print(f"no green_wave rows in {CORRIDOR_SWEEP_CSV} for "
              f"{args.scenario}/mg{args.min_green} -- cannot pair")
    else:
        result = paired_vs(this_run, gw_bar, "green_wave")
        print(f"\nidqn - green_wave, {args.scenario} mg{args.min_green}: "
              f"{result['mean']:+.2f} +/- {result['sd']:.2f} s, "
              f"idqn wins {result['wins']}/{result['n']}")

    ippo_bar = load_ippo_bar(args.scenario, args.min_green)
    if ippo_bar.empty:
        print(f"no ippo rows in {IPPO_SWEEP_CSV} for "
              f"{args.scenario}/mg{args.min_green} -- cannot pair")
    else:
        result = paired_vs(this_run, ippo_bar, "ippo")
        print(f"idqn - ippo, {args.scenario} mg{args.min_green}: "
              f"{result['mean']:+.2f} +/- {result['sd']:.2f} s, "
              f"idqn wins {result['wins']}/{result['n']}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_idqn_sweep.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Fast suite regression**

Run: `pytest -q -m "not slow"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add analysis/idqn_sweep.py tests/test_idqn_sweep.py
git commit -m "feat: tripinfo-based IDQN sweep, paired against green_wave and ippo"
```

---

## Task 5: Wire `idqn` into `compare.py`

**Files:**
- Modify: `compare.py:223-228`

`compare.py` already has a corridor block that adds `ippo` rows (added when IPPO's own SP2 was built — see the existing lines below); SP5 mirrors it for `idqn` so both learned controllers show up in the same comparison table without a separate script.

- [ ] **Step 1: Add the `idqn` block**

In `compare.py`, find:
```python
    corridor_lambdas = ["00", "05", "10"]
    for scenario in corridor_scenarios:
        for lam in corridor_lambdas:
            df = _run_means(args.logs, "ippo", scenario, lam=lam)
            if not df.empty:
                rows.append(_summarise(df, "ippo", scenario, lam))
```
Replace with:
```python
    corridor_lambdas = ["00", "05", "10"]
    for scenario in corridor_scenarios:
        for lam in corridor_lambdas:
            df = _run_means(args.logs, "ippo", scenario, lam=lam)
            if not df.empty:
                rows.append(_summarise(df, "ippo", scenario, lam))
            df = _run_means(args.logs, "idqn", scenario, lam=lam)
            if not df.empty:
                rows.append(_summarise(df, "idqn", scenario, lam))
```

- [ ] **Step 2: Fast suite regression**

Run: `pytest -q -m "not slow"`
Expected: all pass — `tests/test_compare_corridor.py` exercises `_run_means`/`_summarise` generically and is unaffected by which entity name is passed.

- [ ] **Step 3: Commit**

```bash
git add compare.py
git commit -m "feat: add idqn alongside ippo in compare.py's corridor comparison"
```

---

## Task 6: Measure IDQN throughput

**Files:**
- None (measurement only)

Same discipline SP4's Task 6 used: measure `agent-steps/s` for one short IDQN run before committing to the pilot's wall-clock, rather than trusting the spec's extrapolated estimate (15-18h for the full sweep) blind. True-independent DQN has 3x the per-step network/buffer overhead IPPO's shared policy does, so this number may differ meaningfully from IPPO's measured throughput.

- [ ] **Step 1: Timed short run**

Run:
```bash
source venv/bin/activate
python -c "
import time
import train_corridor_dqn as tcd
t0 = time.monotonic()
tcd.train('corridor_peak', lam=0.5, seed=0, steps=2000, min_green=10)
dt = time.monotonic() - t0
print(f'{2000/dt:.1f} agent-steps/s, {dt:.1f}s for 2000 steps')
"
for a in C1 C2 C3; do rm -f "models/idqn_${a}_corridor_peak_lam05_seed0_mg10_s2000.pt"; done
```

- [ ] **Step 2: Record the number**

Write the measured throughput down — it goes at the top of `docs/FINDINGS_2026-08-21-sp5-idqn-vs-corrected-bar.md` (created in Task 9) alongside the actual pilot/full-sweep wall-clock, same "measure before trusting" discipline the corridor calibration and SP4 both used. If the measured throughput implies the full 20-run sweep would take dramatically longer than the spec's 15-18h estimate (e.g. >30h), flag it before Task 8 rather than after — the pilot's own wall-clock in Task 7 will confirm or correct this either way.

No commit for this task — it produces a number that feeds later tasks, not a code change.

---

## Task 7: Pilot — `corridor_peak`, seeds 42/43/44, 100k steps, decision gate

**Files:**
- None (run + decision only; the findings doc is written in Task 9 regardless of outcome)

This is the gate from the spec's §5: the exact subset (`corridor_peak`, seeds 42/43/44, 100k steps) SP4's own confirmatory check used, so the gap-to-`green_wave` numbers line up row-for-row against SP4's **+4.38s**.

- [ ] **Step 1: Run the pilot**

Run:
```bash
source venv/bin/activate
python -m analysis.idqn_sweep --scenario corridor_peak --min-green 10 --lam 0.5 \
    --steps 100000 --seeds 42 43 44
```
Expected: 3 lines of per-seed output, then two paired summary lines (`idqn - green_wave` and `idqn - ippo`). This can be interrupted and re-run — `run_one` reuses existing checkpoints/tripinfo, same as `ippo_sweep.py`.

- [ ] **Step 2: Apply the decision rule**

Compare the printed `idqn - green_wave` mean against SP4's `+4.38s` (`docs/FINDINGS_2026-08-18-sp4-ippo-vs-corrected-bar.md`'s IPPO-at-100k-on-this-same-3-seed-subset number):

- **Ties or meaningfully closes the gap further** (e.g. the mean is below roughly half of +4.38s, or crosses to negative/a win) → proceed to Task 8's full sweep.
- **Comparable to or worse than +4.38s** → do not run Task 8. Go directly to Task 9 and write up the pilot result as a third replication of "fixed plan beats learned control here," on the 3-seed subset only.

This is a qualitative call, not a scripted threshold — same framing SP4's Task 7 verdict used ("state the result plainly, in either direction"). Record the actual numbers (mean, sd, wins/n, both vs `green_wave` and vs `ippo`) — they are needed verbatim in Task 9 regardless of which branch is taken.

- [ ] **Step 3: No commit for this task**

`analysis/idqn_sweep.csv` is not committed yet — Task 9 commits it once, after whichever of Task 7/Task 8 is the last to run, so there is exactly one commit of run output instead of one per stage.

---

## Task 8: Full sweep (only if Task 7's decision rule says proceed)

**Files:**
- None (run only)

Skip this task entirely if Task 7's decision rule said stop — go to Task 9. If it said proceed, this runs the remaining 17 seed-runs (the ~85% of compute the pilot gate exists to withhold until there is a signal): `corridor_peak` seeds 45-51 (7 more, completing all 10) and `corridor_tidal` seeds 42-51 (10, not yet run at all).

- [ ] **Step 1: Complete `corridor_peak`**

Run:
```bash
source venv/bin/activate
python -m analysis.idqn_sweep --scenario corridor_peak --min-green 10 --lam 0.5 \
    --steps 100000 --seeds 42 43 44 45 46 47 48 49 50 51
```
Expected: the 3 pilot seeds are reused (`(reused)` in the per-seed output), 7 new seeds train, then updated paired summary lines over all 10.

- [ ] **Step 2: Run `corridor_tidal`**

Run:
```bash
python -m analysis.idqn_sweep --scenario corridor_tidal --min-green 10 --lam 0.5 \
    --steps 100000 --seeds 42 43 44 45 46 47 48 49 50 51
```

- [ ] **Step 3: No commit for this task**

Same reasoning as Task 7 Step 3 — Task 9 commits `analysis/idqn_sweep.csv` once.

---

## Task 9: Write the findings doc

**Files:**
- Create: `docs/FINDINGS_2026-08-21-sp5-idqn-vs-corrected-bar.md`

Fill in every `<...>` with the actual numbers from Tasks 2, 6, 7, and (if it ran) 8 — no placeholders left in the committed version, same discipline SP4's Task 7 Step 3 used. The doc's shape differs depending on whether Task 8 ran; both branches are written out below.

- [ ] **Step 1: Write the findings doc**

Create `docs/FINDINGS_2026-08-21-sp5-idqn-vs-corrected-bar.md`. **If Task 8 did NOT run** (pilot-only stop), use:
```markdown
# IDQN vs the corrected corridor bar — pilot result

Written 2026-08-21. Executes SP5
(docs/superpowers/specs/2026-08-21-sp5-idqn-corridor-design.md): test whether
a true-independent DQN (matching RESCO's IDQN, not this project's
parameter-shared IPPO) closes the gap to green_wave that IPPO could not.

## Config note

The single-intersection DQN pilot's tuned-for-100k config (params/*.json) is
unrecoverable (gitignored, cloud-only, never committed). This run uses the 3
disclosed values (lr=2.3195e-05, learning_starts=5000,
target_update_interval=5000) plus algos.ALGOS['dqn']['defaults']() for
everything else — a best-effort reconstruction, not the original tuned
config.

## dqn_core vs SB3 DQN (single intersection, matched HPs, correctness gate)

| | delay/trip (s) | wall-clock |
|---|---:|---:|
| dqn_core | <DQN_CORE_DELAY> | <DQN_CORE_WALL>s |
| SB3 DQN  | <SB3_DELAY> | <SB3_WALL>s |

<one or two sentences: is dqn_core competitive, and is that enough confidence
to trust the corridor numbers below>

## Throughput

Measured <AGENT_STEPS_PER_S> agent-steps/s on this machine (Task 6). The
3-seed pilot took <PILOT_WALL_HOURS>h wall-clock.

## IDQN vs green_wave and IPPO, paired, corridor_peak, min_green=10, seeds 42-44 (pilot only)

| vs | idqn (mean +/- sd) | bar (mean +/- sd) | paired idqn - bar | wins |
|---|---:|---:|---:|---:|
| green_wave | <IDQN_MEAN> +/- <IDQN_SD> s | <GW_MEAN> +/- <GW_SD> s | <DIFF_GW> +/- <DIFF_GW_SD> s | <WINS_GW>/3 |
| ippo (100k, same 3 seeds) | <IDQN_MEAN> +/- <IDQN_SD> s | <IPPO_MEAN> +/- <IPPO_SD> s | <DIFF_IPPO> +/- <DIFF_IPPO_SD> s | <WINS_IPPO>/3 |

## Decision

Pilot gap to green_wave: <PILOT_GAP>. SP4's IPPO gap on this same 3-seed
subset at 100k steps was +4.38s. <State plainly whether IDQN tied, closed,
or did not close the gap relative to IPPO, and that per the pilot's decision
rule (docs/superpowers/plans/2026-08-21-sp5-idqn-corridor-training.md Task
7) the full 10-seed/2-scenario sweep was NOT run because
<comparable-to-or-worse-than-+4.38s, quote the actual numbers>.>

## Verdict

<Does the pilot result change the project's consolidation recommendation
(docs/HANDOFF_2026-08-21.md)? If IDQN's pilot result is also a loss or a
statistical tie, state plainly that this is now the THIRD algorithm-scale
replication (single-intersection DQN tie, corridor IPPO loss, corridor IDQN
pilot loss/tie) of "a competently-timed fixed plan beats learned control
here," and that consolidation stands. If the pilot showed real promise but
the gate still stopped short of a full sweep, say that explicitly as the
open thread for anyone who wants to spend the remaining compute later.>
```

**If Task 8 DID run** (full sweep), use instead:
```markdown
# IDQN vs the corrected corridor bar

Written 2026-08-21. Executes SP5
(docs/superpowers/specs/2026-08-21-sp5-idqn-corridor-design.md): test whether
a true-independent DQN (matching RESCO's IDQN, not this project's
parameter-shared IPPO) closes the gap to green_wave that IPPO could not.

## Config note

The single-intersection DQN pilot's tuned-for-100k config (params/*.json) is
unrecoverable (gitignored, cloud-only, never committed). This run uses the 3
disclosed values (lr=2.3195e-05, learning_starts=5000,
target_update_interval=5000) plus algos.ALGOS['dqn']['defaults']() for
everything else — a best-effort reconstruction, not the original tuned
config.

## dqn_core vs SB3 DQN (single intersection, matched HPs, correctness gate)

| | delay/trip (s) | wall-clock |
|---|---:|---:|
| dqn_core | <DQN_CORE_DELAY> | <DQN_CORE_WALL>s |
| SB3 DQN  | <SB3_DELAY> | <SB3_WALL>s |

<one or two sentences: is dqn_core competitive, and is that enough confidence
to trust the corridor numbers below>

## Throughput and pilot gate

Measured <AGENT_STEPS_PER_S> agent-steps/s on this machine (Task 6). The
3-seed pilot (corridor_peak, seeds 42-44) showed <PILOT_GAP> vs green_wave,
which <ties/closes> SP4's IPPO gap of +4.38s on the same 3 seeds, so the
decision rule (Task 7) proceeded to the full 10-seed/2-scenario sweep. Full
sweep took <FULL_WALL_HOURS>h wall-clock.

## IDQN vs green_wave, paired, min_green=10, seeds 42-51

| scenario | idqn (mean +/- sd) | green_wave (mean +/- sd, from analysis/corridor_sweep.csv) | paired idqn - gw | wins |
|---|---:|---:|---:|---:|
| corridor_peak  | <IDQN_PEAK_MEAN> +/- <IDQN_PEAK_SD> s | 13.46 +/- 0.22 s | <DIFF_PEAK> +/- <DIFF_PEAK_SD> s | <WINS_PEAK>/10 |
| corridor_tidal | <IDQN_TIDAL_MEAN> +/- <IDQN_TIDAL_SD> s | 13.96 +/- 0.34 s | <DIFF_TIDAL> +/- <DIFF_TIDAL_SD> s | <WINS_TIDAL>/10 |

## IDQN vs IPPO (SP4, 100k steps), paired, same seeds

| scenario | idqn (mean +/- sd) | ippo (mean +/- sd, from analysis/ippo_sweep.csv) | paired idqn - ippo | wins |
|---|---:|---:|---:|---:|
| corridor_peak  | <IDQN_PEAK_MEAN> +/- <IDQN_PEAK_SD> s | <IPPO_PEAK_MEAN> +/- <IPPO_PEAK_SD> s | <DIFF_IPPO_PEAK> +/- <DIFF_IPPO_PEAK_SD> s | <WINS_IPPO_PEAK>/10 |
| corridor_tidal | <IDQN_TIDAL_MEAN> +/- <IDQN_TIDAL_SD> s | <IPPO_TIDAL_MEAN> +/- <IPPO_TIDAL_SD> s | <DIFF_IPPO_TIDAL> +/- <DIFF_IPPO_TIDAL_SD> s | <WINS_IPPO_TIDAL>/10 |

## Verdict

<Does IDQN clear the green_wave bar on either scenario? State the result
plainly, in either direction. If it does not (paired mean >= 0, wins < 5/10)
on both scenarios: state that explicitly, and that this is now the fourth
replication of "a competently-timed fixed plan beats learned control here"
across algorithm families and network scales — RESCO's own best-converging
algorithm, at RESCO's own convergence-scale budget, still did not close it.
If it does clear the bar on one or both: say which, by how much, and that
this reopens the consolidation recommendation in
docs/HANDOFF_2026-08-21.md.>
```

- [ ] **Step 2: Commit**

```bash
git add analysis/idqn_sweep.csv docs/FINDINGS_2026-08-21-sp5-idqn-vs-corrected-bar.md
git commit -m "docs: IDQN vs the corrected corridor bar -- the SP5 result"
```

Do NOT commit anything under `logs/` or `models/` (gitignored); `analysis/idqn_sweep.csv` is the only run output this plan tracks.

- [ ] **Step 3: Push**

```bash
git push origin main
```

---

## Self-Review Notes

- **Spec coverage:** §1 (true independent architecture) → Task 3's per-agent `QNetwork`/`buffer`/`optimizer`/`eps` tuples. §2 (all 5 components) → Tasks 1/2/3/4. §4 (hyperparameter reconstruction, disclosed limitation) → Task 3's `_hp()` plus the Global Constraints note plus both findings-doc branches' "Config note" section. §5 (pilot gate before full sweep) → Tasks 7/8's explicit conditional structure. §6 (correctness guard before any corridor training) → Task 2 runs and must pass before Task 7. The spec's out-of-scope items (`corridor_skew`, RESCO's conv architecture, hyperparameter retuning) are not present anywhere in this plan, matching the spec's explicit exclusions.
- **Placeholder scan:** the only bracketed placeholders are in Task 9's two findings-doc templates, explicitly required to be filled with real numbers before commit — the same pattern SP4's Task 7 Step 3 used, not a deferred TODO.
- **Type/name consistency:** `_tag(scenario, lam, seed, min_green, steps)` and `_model_path(agent_id, scenario, lam, seed, min_green, steps)` used identically across Task 3's implementation, Task 3's tests, and Task 4's `idqn_sweep.run_one`. `train(scenario, lam, seed, steps, min_green) -> dict[ts_id, path]` and `evaluate(scenario, lam, seed, min_green, steps, tripinfo=False) -> str` used identically across Task 3 and Task 4. `dqn_loss(q_network, target_network, obs, actions, rewards, next_obs, dones, gamma)` signature matches between Task 1's implementation, Task 1's tests, Task 2's `validate_dqn_core.py`, and Task 3's `train_corridor_dqn.py`.
- **Deferred correctly:** corridor-specific DQN hyperparameter retuning, isolating parameter-sharing as its own variable, and RESCO's conv-per-road architecture are all explicitly out of scope per the spec and do not appear as tasks here — consistent with the spec's "Open decisions deferred to later sub-projects" section.
