"""Unit tests for the per-decision-window safety accumulator (no SUMO needed).

Background: sumo-rl computes rewards and info only at action steps, i.e. after
`_run_steps()` has advanced the simulation `delta_time` seconds.  Because
`yellow_time < delta_time` is enforced by sumo-rl, `ts.is_yellow` is always
False again by then, so an instantaneous safety sample can never observe the
yellow/clearing interval, and sees only 1 of every `delta_time` braking
snapshots.  The window accumulates both sub-terms on every simulation second
instead.
"""
from types import SimpleNamespace

import pytest

import env_common as ec


def _fake_ts(ts_id="C", *, lane_vehicles=None, accel=None, types=None,
             is_yellow=False, internal_lanes=(), internal_vehicles=None,
             link_calls=None):
    """Duck-typed traffic signal; see tests/test_safety_reward.py::_fake_ts."""
    lane_vehicles = lane_vehicles or {}
    accel = accel or {}
    types = types or {}
    internal_vehicles = internal_vehicles or {}

    def get_last_step_vehicle_ids(lane):
        return lane_vehicles.get(lane, internal_vehicles.get(lane, []))

    links = [[("in", "out", via)] for via in internal_lanes]

    def get_controlled_links(_id):
        if link_calls is not None:
            link_calls.append(_id)
        return links

    sumo = SimpleNamespace(
        lane=SimpleNamespace(getLastStepVehicleIDs=get_last_step_vehicle_ids),
        vehicle=SimpleNamespace(
            getAcceleration=lambda v: accel[v],
            getTypeID=lambda v: types[v],
        ),
        trafficlight=SimpleNamespace(getControlledLinks=get_controlled_links),
    )
    return SimpleNamespace(sumo=sumo, id=ts_id, lanes=list(lane_vehicles.keys()),
                           is_yellow=is_yellow)


def _braking_ts(ts_id="C", is_yellow=False):
    # one moto braking at -6 m/s^2 -> brake 1.0 per sampled second
    return _fake_ts(
        ts_id,
        lane_vehicles={"L1": ["m1"]},
        accel={"m1": -6.0},
        types={"m1": "moto", "a1": "auto"},
        is_yellow=is_yellow,
        internal_lanes=[":C_0"],
        internal_vehicles={":C_0": ["a1"]},
    )


# ---------------------------------------------------------------- accumulation


def test_window_sums_brake_over_every_sampled_second():
    w = ec._SafetyWindow()
    for _ in range(3):
        w.accumulate(_braking_ts())
    brake, _ = w.for_ts("C")
    assert brake == pytest.approx(3.0)


def test_window_captures_exposure_from_yellow_seconds_only():
    # the bug this fix exists for: yellow occurs mid-window and is over by the
    # time the reward is computed, so only a per-second sample can see it.
    w = ec._SafetyWindow()
    w.accumulate(_braking_ts(is_yellow=True))    # exposure 0.6
    w.accumulate(_braking_ts(is_yellow=False))   # exposure 0.0
    _, exposure = w.for_ts("C")
    assert exposure == pytest.approx(0.6)


def test_window_reset_clears_accumulated_terms():
    w = ec._SafetyWindow()
    w.accumulate(_braking_ts(is_yellow=True))
    w.reset()
    assert w.for_ts("C") == (0.0, 0.0)


def test_window_keeps_signals_separate():
    w = ec._SafetyWindow()
    w.accumulate(_braking_ts("A"))
    w.accumulate(_braking_ts("B", is_yellow=True))
    assert w.for_ts("A") == pytest.approx((1.0, 0.0))
    assert w.for_ts("B") == pytest.approx((1.0, 0.6))


def test_window_unseen_signal_reads_zero():
    assert ec._SafetyWindow().for_ts("nope") == (0.0, 0.0)


def test_window_totals_sum_over_all_signals():
    w = ec._SafetyWindow()
    w.accumulate(_braking_ts("A"))
    w.accumulate(_braking_ts("B", is_yellow=True))
    brake, exposure = w.totals()
    assert brake == pytest.approx(2.0)
    assert exposure == pytest.approx(0.6)


def test_window_totals_empty_is_zero():
    assert ec._SafetyWindow().totals() == (0.0, 0.0)


# ------------------------------------------------------- internal-lane caching


def test_window_queries_controlled_links_once_per_signal():
    # getControlledLinks is a TraCI round-trip and the junction topology is
    # static; accumulating every simulation second must not re-query it.
    calls = []
    w = ec._SafetyWindow()
    for _ in range(4):
        w.accumulate(_fake_ts(
            lane_vehicles={"L1": []},
            is_yellow=True,
            internal_lanes=[":C_0"],
            internal_vehicles={":C_0": []},
            link_calls=calls,
        ))
    assert len(calls) == 1


# ------------------------------------------------------------- reward wiring


def test_reward_penalty_reads_window_not_instantaneous_sample():
    ts = _braking_ts(is_yellow=False)   # instantaneous penalty would be 1.0
    w = ec._SafetyWindow()
    for _ in range(5):
        w.accumulate(_braking_ts(is_yellow=True))   # 5 * (1.0 + 0.6)
    ts.env = SimpleNamespace(_safety_window=w)
    assert ec._step_safety_penalty(ts) == pytest.approx(8.0)


def test_reward_penalty_falls_back_to_instantaneous_without_window():
    # keeps the function usable on a plain SumoEnvironment / bare stub
    ts = _braking_ts(is_yellow=True)
    assert ec._step_safety_penalty(ts) == pytest.approx(1.6)


def test_reward_subtracts_window_penalty(monkeypatch):
    monkeypatch.setattr(ec, "_efficiency", lambda ts: 7.0)
    monkeypatch.setattr(ec, "_step_safety_penalty", lambda ts: 4.0)
    fn = ec.make_safety_reward_fn(0.5, scale=2.0)
    # 7.0 - 0.5 * (4.0 / 2.0) = 6.0
    assert fn(object()) == pytest.approx(6.0)


def test_reward_lambda_zero_ignores_window(monkeypatch):
    monkeypatch.setattr(ec, "_efficiency", lambda ts: 7.0)
    monkeypatch.setattr(ec, "_step_safety_penalty", lambda ts: 999.0)
    assert ec.make_safety_reward_fn(0.0)(object()) == pytest.approx(7.0)
