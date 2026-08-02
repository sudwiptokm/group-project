"""Unit tests for the critic state builder (SUMO-free)."""
import numpy as np

import train_corridor as tc


def test_ippo_state_is_local_obs():
    obs = {"C1": np.array([1.0, 2.0]), "C2": np.array([3.0, 4.0])}
    states = tc.build_states(obs, ["C1", "C2"], centralized=False)
    assert np.array_equal(states["C1"], np.array([1.0, 2.0]))
    assert np.array_equal(states["C2"], np.array([3.0, 4.0]))


def test_mappo_state_is_joint_concat_in_order():
    obs = {"C1": np.array([1.0, 2.0]), "C2": np.array([3.0, 4.0]), "C3": np.array([5.0, 6.0])}
    states = tc.build_states(obs, ["C1", "C2", "C3"], centralized=True)
    joint = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    for agent in ("C1", "C2", "C3"):
        assert np.array_equal(states[agent], joint)  # same joint state for all agents


def test_mappo_state_respects_ts_ids_order():
    obs = {"C1": np.array([1.0]), "C2": np.array([2.0]), "C3": np.array([3.0])}
    # a different id order must reorder the concatenation
    states = tc.build_states(obs, ["C3", "C1", "C2"], centralized=True)
    assert np.array_equal(states["C1"], np.array([3.0, 1.0, 2.0]))
