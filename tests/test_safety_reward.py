"""Unit tests for the safety-aware reward math (no SUMO needed)."""
from types import SimpleNamespace

import pytest

import env_common as ec


def test_vehicle_vuln_exact_types():
    assert ec._vehicle_vuln("moto") == 1.0
    assert ec._vehicle_vuln("auto") == 0.6
    assert ec._vehicle_vuln("car") == 0.3


def test_vehicle_vuln_distribution_suffix():
    # vType ids may carry a distribution suffix like "moto@0"
    assert ec._vehicle_vuln("moto@0") == 1.0
    assert ec._vehicle_vuln("auto@3") == 0.6


def test_vehicle_vuln_unknown_defaults_to_car():
    assert ec._vehicle_vuln("bus") == ec.DEFAULT_VULN == 0.3


def _fake_ts(*, lane_vehicles, accel, types, is_yellow, internal_lanes, internal_vehicles):
    """Build a duck-typed traffic-signal stand-in for _safety_penalty.

    lane_vehicles      : {lane_id: [veh_id, ...]}   approach lanes
    accel              : {veh_id: acceleration}     (negative = braking)
    types              : {veh_id: type_id}
    internal_lanes     : [lane_id, ...]             junction internal lanes
    internal_vehicles  : {lane_id: [veh_id, ...]}
    """
    def get_last_step_vehicle_ids(lane):
        return lane_vehicles.get(lane, internal_vehicles.get(lane, []))

    # getControlledLinks -> [[(in_lane, out_lane, via_lane)], ...]
    links = [[("in", "out", via)] for via in internal_lanes]

    sumo = SimpleNamespace(
        lane=SimpleNamespace(getLastStepVehicleIDs=get_last_step_vehicle_ids),
        vehicle=SimpleNamespace(
            getAcceleration=lambda v: accel[v],
            getTypeID=lambda v: types[v],
        ),
        trafficlight=SimpleNamespace(getControlledLinks=lambda _id: links),
    )
    return SimpleNamespace(sumo=sumo, id="C", lanes=list(lane_vehicles.keys()),
                           is_yellow=is_yellow)


def test_safety_penalty_counts_hard_braking_weighted():
    # one moto braking hard (accel -5 < -4.5) -> weight 1.0 ; one car cruising -> 0
    ts = _fake_ts(
        lane_vehicles={"L1": ["m1", "c1"]},
        accel={"m1": -5.0, "c1": 0.0},
        types={"m1": "moto", "c1": "car"},
        is_yellow=False,
        internal_lanes=[":C_0"],
        internal_vehicles={},
    )
    assert ec._safety_penalty(ts) == pytest.approx(1.0)


def test_safety_penalty_ignores_soft_braking():
    # accel -3 is above -B_THRESH (-4.5) -> not an emergency brake
    ts = _fake_ts(
        lane_vehicles={"L1": ["m1"]},
        accel={"m1": -3.0},
        types={"m1": "moto"},
        is_yellow=False,
        internal_lanes=[":C_0"],
        internal_vehicles={},
    )
    assert ec._safety_penalty(ts) == pytest.approx(0.0)


def test_safety_penalty_exposure_only_when_yellow():
    # an auto sitting on an internal lane; counts only while is_yellow
    common = dict(
        lane_vehicles={"L1": []},
        accel={},
        types={"a1": "auto"},
        internal_lanes=[":C_0"],
        internal_vehicles={":C_0": ["a1"]},
    )
    assert ec._safety_penalty(_fake_ts(is_yellow=False, **common)) == pytest.approx(0.0)
    assert ec._safety_penalty(_fake_ts(is_yellow=True, **common)) == pytest.approx(0.6)


def test_safety_penalty_sums_brake_and_exposure():
    ts = _fake_ts(
        lane_vehicles={"L1": ["m1"]},
        accel={"m1": -6.0},
        types={"m1": "moto", "a1": "auto"},
        is_yellow=True,
        internal_lanes=[":C_0"],
        internal_vehicles={":C_0": ["a1"]},
    )
    # brake: moto 1.0 ; exposure: auto 0.6
    assert ec._safety_penalty(ts) == pytest.approx(1.6)


def test_internal_lanes_collects_all_connections_per_index():
    # one signal index controlling two connections -> both via lanes must appear
    links = [[("iA", "oA", ":C_0"), ("iB", "oB", ":C_1")]]
    sumo = SimpleNamespace(
        trafficlight=SimpleNamespace(getControlledLinks=lambda _id: links)
    )
    ts = SimpleNamespace(sumo=sumo, id="C")
    assert set(ec._internal_lanes(ts)) == {":C_0", ":C_1"}


def test_reward_lambda_zero_is_pure_efficiency(monkeypatch):
    monkeypatch.setattr(ec, "_efficiency", lambda ts: 7.0)
    # safety must be ignored entirely at lam=0
    monkeypatch.setattr(ec, "_safety_penalty", lambda ts: 999.0)
    fn = ec.make_safety_reward_fn(0.0)
    assert fn(object()) == pytest.approx(7.0)


def test_reward_subtracts_scaled_safety(monkeypatch):
    monkeypatch.setattr(ec, "_efficiency", lambda ts: 7.0)
    monkeypatch.setattr(ec, "_safety_penalty", lambda ts: 4.0)
    fn = ec.make_safety_reward_fn(0.5, scale=2.0)
    # 7.0 - 0.5 * (4.0 / 2.0) = 6.0
    assert fn(object()) == pytest.approx(6.0)


def test_reward_fn_has_unique_name():
    assert ec.make_safety_reward_fn(1.0).__name__ != ec.make_safety_reward_fn(0.0).__name__


def test_safety_components_splits_brake_and_exposure():
    # moto braking hard on approach (brake 1.0) + auto exposed on internal lane during
    # yellow (exposure 0.6); components must split, and their sum == _safety_penalty.
    ts = _fake_ts(
        lane_vehicles={"L1": ["m1"]},
        accel={"m1": -6.0},
        types={"m1": "moto", "a1": "auto"},
        is_yellow=True,
        internal_lanes=[":C_0"],
        internal_vehicles={":C_0": ["a1"]},
    )
    brake, exposure = ec._safety_components(ts)
    assert brake == pytest.approx(1.0)
    assert exposure == pytest.approx(0.6)
    assert ec._safety_penalty(ts) == pytest.approx(brake + exposure)
