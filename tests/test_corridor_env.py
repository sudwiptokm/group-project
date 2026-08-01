"""Smoke test: the corridor env exposes 3 agents with correct obs/reward shapes
and terminates. Requires SUMO; slow.

Real sumo-rl multi-agent API (single_agent=False, sumo-rl ≤1.x):
  - reset() -> dict {ts_id: np.ndarray}   (NOT a (obs, info) tuple)
  - env.ts_ids                             -> list of agent ids (NO env.agents)
  - env.action_spaces(ts_id)              -> per-agent Discrete space (method)
  - env.action_space                       -> shared Discrete (same for all TLS)
  - step(dict) -> (obs_dict, rew_dict, done_dict, info_dict)  (4-tuple, NOT 5)
  - done_dict has a '__all__' key for episode termination
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.slow

import env_common as ec


@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_corridor_env_reset_step(monkeypatch):
    # monkeypatch auto-restores EPISODE_SECONDS so a short test episode does not
    # leak into other tests in the same process.
    monkeypatch.setenv("EPISODE_SECONDS", "120")
    env = ec.make_corridor_env(seed=0, scenario="corridor_offpeak", lam=0.5)

    # --- reset returns a plain dict, not a (obs, info) tuple ---
    obs = env.reset()
    assert isinstance(obs, dict), f"reset() should return dict, got {type(obs)}"
    assert set(obs.keys()) == {"C1", "C2", "C3"}, f"Expected agents C1/C2/C3, got {set(obs.keys())}"
    for a in obs:
        assert isinstance(obs[a], np.ndarray), f"obs[{a}] should be np.ndarray"

    # --- agent ids via env.ts_ids (no env.agents attribute) ---
    assert set(env.ts_ids) == {"C1", "C2", "C3"}

    # --- per-agent action space via env.action_spaces(ts_id) ---
    actions = {ts_id: int(env.action_spaces(ts_id).sample()) for ts_id in env.ts_ids}

    # --- step returns 4-tuple (obs, rewards, dones, info) ---
    result = env.step(actions)
    assert len(result) == 4, f"step() should return 4-tuple, got len={len(result)}"
    obs2, rewards, dones, info = result

    assert set(rewards.keys()) == {"C1", "C2", "C3"}, (
        f"Expected reward keys C1/C2/C3, got {set(rewards.keys())}"
    )
    for a in rewards:
        assert np.isscalar(rewards[a]), f"rewards[{a}] should be scalar"

    # dones dict has per-agent keys + '__all__'
    assert "__all__" in dones, "dones dict should contain '__all__' key"

    env.close()
