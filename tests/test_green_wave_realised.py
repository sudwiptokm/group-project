"""What green_wave actually does in SUMO, not what it intends.

tests/test_green_wave_plan.py pins the plan arithmetic. This runs the controller
against a live signal and measures the switching times it really achieves,
because the defect being fixed here lived entirely in the gap between the two:
the plan looked like it swept ten floors and the simulation ran five.
"""
import os

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("sumo_rl")

import corridor_baseline as cb          # noqa: E402
import corridor_control as cc           # noqa: E402
import env_common as ec                 # noqa: E402

YELLOW, DELTA = 3, 5


def _switch_times(min_green, seconds=300, ts_id="C1"):
    """Sim times at which `ts_id` actually changed green phase."""
    env = ec.make_corridor_env(seed=0, scenario="corridor_peak", lam=0.0,
                               out_csv=f"logs/_test_gw_mg{min_green}",
                               min_green=min_green, tripinfo=False)
    env.reset()
    ts = env.traffic_signals[ts_id]
    switches, last, done = [], ts.green_phase, False
    try:
        while not done:
            _, _, dones, _ = env.step(cb.green_wave_actions(env))
            done = dones["__all__"]
            if ts.green_phase != last:
                switches.append(int(env.sim_step))
                last = ts.green_phase
    finally:
        env.close()
    return switches


def _cycles(switches):
    return {b - a for a, b in zip(switches, switches[1:])}


@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_realised_cycle_matches_the_plan(monkeypatch):
    monkeypatch.setenv("EPISODE_SECONDS", "300")
    for mg in (5, 15, 60):
        expected = cc.plan_phase_seconds(mg, YELLOW, DELTA)
        assert _cycles(_switch_times(mg)) == {expected}, (
            f"mg{mg}: plan says {expected}s per phase")


@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_adjacent_floors_are_no_longer_the_same_plan(monkeypatch):
    """mg5 and mg10 produced bit-identical episodes; so did mg15/mg20 and
    mg25/mg30. Ten labelled floors were five distinct plans."""
    monkeypatch.setenv("EPISODE_SECONDS", "300")
    for lo, hi in ((5, 10), (15, 20), (25, 30)):
        assert _cycles(_switch_times(lo)) != _cycles(_switch_times(hi))


@pytest.mark.skipif(not os.environ.get("SUMO_HOME"), reason="SUMO_HOME not set")
def test_downstream_signals_are_offset_by_the_travel_time(monkeypatch):
    """The progression itself: C2 switches one travel time after C1.

    200 m at 13.89 m/s is 14.4 s, quantised to the 5 s decision grid.
    """
    monkeypatch.setenv("EPISODE_SECONDS", "300")
    mg = 15
    phase_seconds = cc.plan_phase_seconds(mg, YELLOW, DELTA)
    c1 = _switch_times(mg, ts_id="C1")
    c2 = _switch_times(mg, ts_id="C2")
    positions, free_flow_speed = cb._green_wave_inputs("corridor.net.xml")
    offset = (positions[1] - positions[0]) / free_flow_speed
    lag = (c2[0] - c1[0]) % (2 * phase_seconds)
    assert abs(lag - offset) <= DELTA, (
        f"C2 lags C1 by {lag}s, expected ~{offset:.1f}s")
