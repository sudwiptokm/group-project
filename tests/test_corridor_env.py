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

import env_common as ec
from env_common import SafetyLoggingEnv, make_corridor_env


def test_safety_logging_env_incident_defaults_to_none():
    import inspect
    sig = inspect.signature(SafetyLoggingEnv.__init__)
    assert "incident" in sig.parameters
    assert sig.parameters["incident"].default is None


def test_make_corridor_env_incident_defaults_to_none():
    import inspect
    sig = inspect.signature(make_corridor_env)
    assert "incident" in sig.parameters
    assert sig.parameters["incident"].default is None


@pytest.mark.slow
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


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_incident_closes_and_reopens_lane(monkeypatch):
    monkeypatch.setenv("EPISODE_SECONDS", "20")
    env = make_corridor_env(seed=0, scenario="corridor_offpeak", min_green=10,
                            incident=("C1_C2", 0, 5.0, 10.0))
    env.reset()
    lane_id = "C1_C2_0"
    seen_closed = False
    done = False
    # vehicles observed on the closed lane before the window starts -- used
    # below to scope the occupancy assertion to vehicles that freshly enter
    # during the closure, not ones already on the lane and passing through
    # (that's expected, not a defect).
    ids_before_window = set()
    while not done:
        t = env.sumo.simulation.getTime()
        current_ids = set(env.sumo.lane.getLastStepVehicleIDs(lane_id))
        if t < 5.0:
            ids_before_window |= current_ids
        if 5.0 <= t < 15.0:
            for vc in ("passenger", "motorcycle", "moped"):
                assert vc in env.sumo.lane.getDisallowed(lane_id)
            seen_closed = True
            # no vehicle of type moto/auto/car (vClass motorcycle/moped/
            # passenger) should freshly route onto the closed lane during
            # the window -- this is the check that catches the original
            # defect, since a getDisallowed()-only assertion passes even
            # when moto/auto freely use the lane.
            for veh_id in current_ids - ids_before_window:
                vtype = env.sumo.vehicle.getTypeID(veh_id)
                assert vtype not in ("moto", "auto", "car"), (
                    f"vehicle {veh_id} of type {vtype} freshly entered "
                    f"closed lane {lane_id} at t={t}"
                )
        actions = {i: 0 for i in env.ts_ids}
        _, _, dones, _ = env.step(actions)
        done = dones["__all__"]
    assert seen_closed, "incident window was never reached in a 20s episode"
    assert list(env.sumo.lane.getDisallowed(lane_id)) == []
    env.close()
